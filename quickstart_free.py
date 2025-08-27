import os, re, uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# env
from dotenv import load_dotenv
load_dotenv(".env")  # ignored in Railway; Railway uses its own env vars

USE_LOCAL_EMBED = os.getenv("USE_LOCAL_EMBED","true").lower()=="true"
USE_LOCAL_LLM   = os.getenv("USE_LOCAL_LLM","true").lower()=="true"

# --- Embeddings (free) ---
if USE_LOCAL_EMBED:
    from sentence_transformers import SentenceTransformer
    _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    EMB_DIM = 384
    def embed(texts):
        embs = _emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embs]
else:
    from openai import OpenAI
    _openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMB_MODEL = os.getenv("EMB_MODEL","text-embedding-3-small")
    EMB_DIM = 1536 if "small" in EMB_MODEL else 3072
    def embed(texts):
        r = _openai.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in r.data]

# --- Pinecone ---
from pinecone import Pinecone, ServerlessSpec
PC = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX","rag-copilot")

# Ensure index exists with correct dimension; recreate to guarantee match
existing = [i["name"] for i in PC.list_indexes()]
if INDEX_NAME in existing:
    try:
        PC.delete_index(INDEX_NAME)
    except Exception:
        pass
if INDEX_NAME not in [i["name"] for i in PC.list_indexes()]:
    PC.create_index(
        name=INDEX_NAME,
        dimension=EMB_DIM,
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION","us-east-1")),
    )
INDEX = PC.Index(INDEX_NAME)

# --- API ---
app = FastAPI(title="RAG Copilot — Free Mode (Railway)")

class AskReq(BaseModel):
    query: str
    namespace: str = "demo"
    k: int = 8

def chunk_text(text: str, size: int = 1200, overlap: int = 150):
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size]); i += size - overlap
    return out

@app.get("/")
def root():
    return {"ok": True, "mode": {"local_embed": USE_LOCAL_EMBED, "local_llm": USE_LOCAL_LLM}, "endpoints": ["/ingest","/ask","/health"]}

@app.get("/health")
def health():
    issues=[]
    if not os.getenv("PINECONE_API_KEY"): issues.append("PINECONE_API_KEY missing")
    return {"ok": len(issues)==0, "issues": issues, "index": INDEX_NAME}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), namespace: str = Form("demo")):
    from pypdf import PdfReader
    content = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
        tmp = "/tmp/upload.pdf"
        with open(tmp, "wb") as f: f.write(content)
        reader = PdfReader(tmp)
        for p in reader.pages:
            text += p.extract_text() or ""
    else:
        text = content.decode("utf-8", errors="ignore")
    text = re.sub(r"\s+"," ", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted")

    chunks = chunk_text(text)
    vecs = embed(chunks)
    items = [{"id": str(uuid.uuid4()), "values": v, "metadata": {"text": c, "ord": i}} for i, (v,c) in enumerate(zip(vecs, chunks))]
    INDEX.upsert(vectors=items, namespace=namespace)
    return {"ok": True, "chunks": len(items), "namespace": namespace}

@app.post("/ask")
async def ask(req: AskReq):
    qvec = embed([req.query])[0]
    res = INDEX.query(vector=qvec, top_k=req.k, namespace=req.namespace, include_metadata=True)
    matches = res.get("matches", [])
    ctxs = [m["metadata"]["text"] for m in matches]

    if USE_LOCAL_LLM:
        # Free “strict quotes” response
        def quote(t):
            t=t.strip()
            return "“"+(t[:280]+"…" if len(t)>280 else t)+"”"
        lines = [f"• {quote(c)} [C{i+1}]" for i,c in enumerate(ctxs[:3])]
        bullets = "\n".join(lines) if lines else "No context found. Try ingesting a PDF."
        return JSONResponse({
            "answer": f"Strict quotes mode (local). Below are verbatim excerpts:\n\n{bullets}",
            "citations": [{"score": round(m.get('score',0.0),3), "snippet": (m['metadata']['text'][:240] if 'metadata' in m and 'text' in m['metadata'] else '')} for m in matches]
        })
    else:
        # Uses OpenAI for full synthesis (requires credits)
        from openai import OpenAI
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        GEN_MODEL = os.getenv("GEN_MODEL","gpt-4o-mini")
        ctx = "\n\n---\n".join([f"[C{i+1}] "+c[:1500] for i,c in enumerate(ctxs)])
        out = llm.chat.completions.create(
            model=GEN_MODEL, temperature=0.2,
            messages=[{"role":"system","content":"Use ONLY provided context; quote verbatim for key claims."},
                      {"role":"user","content":f"Question: {req.query}\n\nContext:\n{ctx}"}]
        )
        ans = out.choices[0].message.content
        return JSONResponse({
            "answer": ans,
            "citations": [{"score": round(m.get('score',0.0),3), "snippet": m["metadata"]["text"][:240]} for m in matches]
        })
