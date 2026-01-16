import os
os.environ["OMP_NUM_THREADS"] = "64"  # For FAISS
import json
import logging
import uvicorn
import pickle
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datasets import load_dataset
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Data Loading and Indexing ---

class LawDatabase:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", cache_dir=f"{os.path.abspath(os.path.dirname(__file__))}/.cache/nitibench_server"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.law_map = {} # Map law_name -> full content
        self.laws_by_name = {}
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self):
        cache_index_path = os.path.join(self.cache_dir, "law_index.faiss")
        cache_meta_path = os.path.join(self.cache_dir, "law_meta.pkl")

        if os.path.exists(cache_index_path) and os.path.exists(cache_meta_path):
            logger.info(f"Loading cached index from {cache_index_path}")
            try:
                self.index = faiss.read_index(cache_index_path)
                with open(cache_meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.documents = meta["documents"]
                    self.laws_by_name = meta["laws_by_name"]
                logger.info(f"Loaded {len(self.documents)} documents from cache.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding index.")

        logger.info("Loading Nitibench dataset...")
        try:
            # Using "main" config and "train" split as per user's successful inspection
            dataset = load_dataset("VISAI-AI/nitibench", split="ccl")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

        logger.info("Processing laws...")
        seen_docs = set()
        self.laws_by_name = {} # Map law_name -> { section_num -> content }

        # Load WangchanX-Legal-ThaiCCL-RAG
        logger.info("Loading WangchanX-Legal-ThaiCCL-RAG dataset...")
        try:
            rag_dataset = load_dataset("airesearch/WangchanX-Legal-ThaiCCL-RAG", split="train")
            for item in rag_dataset:
                contexts = item.get('positive_contexts', [])
                if not contexts:
                    continue
                for ctx in contexts:
                    metadata = ctx.get('metadata', {})
                    law_name = metadata.get('law_title')
                    section_num = metadata.get('section')
                    content = ctx.get('context')
                    
                    if not content:
                        continue

                    if content not in seen_docs:
                        self.documents.append({
                            "text": content,
                            "law_name": law_name,
                            "section_num": section_num
                        })
                        seen_docs.add(content)
                    
                    if law_name:
                        if law_name not in self.laws_by_name:
                            self.laws_by_name[law_name] = {}
                        self.laws_by_name[law_name][str(section_num)] = content
        except Exception as e:
            logger.warning(f"Failed to load WangchanX-Legal-ThaiCCL-RAG dataset: {e}")

        for item in dataset:
            # Iterate over both relevant_laws and reference_laws
            for field in ['relevant_laws', 'reference_laws']:
                laws = item.get(field, [])
                if not laws:
                    continue
                
                # Ensure laws is a list
                if not isinstance(laws, list):
                    continue

                for law_item in laws:
                    if not isinstance(law_item, dict):
                        continue

                    # Extract fields based on user provided structure
                    law_name = law_item.get('law_name')
                    section_content = law_item.get('section_content')
                    section_num = law_item.get('section_num')
                    
                    if not section_content:
                        continue

                    # Deduplicate for indexing based on content
                    if section_content not in seen_docs:
                        self.documents.append({
                            "text": section_content,
                            "law_name": law_name,
                            "section_num": section_num
                        })
                        seen_docs.add(section_content)
                    
                    # Organize for reading
                    if law_name:
                        if law_name not in self.laws_by_name:
                            self.laws_by_name[law_name] = {}
                        
                        # Use section_num as key, or just append if no section_num
                        key = str(section_num) if section_num else "unknown"
                        self.laws_by_name[law_name][key] = section_content

        logger.info(f"Collected {len(self.documents)} unique law sections.")
        
        # Build FAISS index
        if self.documents:
            logger.info("Encoding documents...")
            texts = [doc["text"] for doc in self.documents]
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            
            # Normalize for Inner Product (Cosine Similarity)
            faiss.normalize_L2(embeddings)
            
            d = embeddings.shape[1]
            ntotal = embeddings.shape[0]
            
            # Choose index type based on size
            if ntotal < 10000:
                logger.info("Dataset small, using FlatIP index.")
                self.index = faiss.IndexFlatIP(d)
                self.index.add(embeddings)
            else:
                # Use IVF-SQ8 for larger datasets
                # Determine number of centroids (nlist)
                # Rule of thumb: 4 * sqrt(ntotal)
                nlist = int(4 * np.sqrt(ntotal))
                nlist = min(nlist, ntotal // 39) # Ensure enough points per cluster
                nlist = max(nlist, 1)
                
                index_string = f"IVF{nlist},SQ8"
                logger.info(f"Building index with factory: {index_string}")
                
                self.index = faiss.index_factory(d, index_string, faiss.METRIC_INNER_PRODUCT)
                
                logger.info("Training index...")
                self.index.train(embeddings)
                logger.info("Adding vectors...")
                self.index.add(embeddings)

            logger.info(f"Index built with {self.index.ntotal} vectors.")
            
            # Save cache
            logger.info("Saving index to cache...")
            faiss.write_index(self.index, cache_index_path)
            with open(cache_meta_path, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "laws_by_name": self.laws_by_name
                }, f)
            logger.info("Cache saved.")
            
        else:
            logger.warning("No documents found to index!")

    def search(self, queries, topk=3):
        if not self.index:
            return [[] for _ in queries]
            
        query_embeddings = self.model.encode(queries, show_progress_bar=False).astype('float32')
        faiss.normalize_L2(query_embeddings)
        
        D, I = self.index.search(query_embeddings, topk)
        
        results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(topk):
                idx = I[i][j]
                if idx != -1:
                    doc = self.documents[idx]
                    # Format as expected by _passages2string in search_r1_like_utils.py
                    # It expects: {"document": {"contents": "Title\nText"}}
                    
                    title = f"{doc.get('law_name', 'Unknown Law')} {doc.get('section_num', '')}".strip()
                    content = f"{title}\n{doc['text']}"
                    
                    query_results.append({
                        "document": {
                            "contents": content
                        },
                        "score": float(D[i][j])
                    })
            results.append(query_results)
        return results

    def read(self, law_name, section_num=None):
        if not law_name:
            return None

        original_law_name = law_name
        law_name = re.sub(r"\s+", " ", str(law_name)).strip()

        # If the caller passed something like "<law name> <section>" or "<law name> มาตรา <section>",
        # recover (law_name, section_num) so the tool can work reliably.
        if law_name not in self.laws_by_name:
            if section_num is None:
                m = re.match(r"^(?P<name>.+?)(?:\s+(?:มาตรา\s*)?|\s+)(?P<section>[0-9]+(?:/[0-9]+)?)$", law_name)
                if m:
                    law_name = m.group("name").strip()
                    section_num = m.group("section").strip()

        if law_name not in self.laws_by_name:
            # Best-effort substring match (pick the longest matching known law name).
            candidates = [k for k in self.laws_by_name.keys() if k and (k in law_name or law_name in k)]
            if candidates:
                law_name = max(candidates, key=len)

        if law_name not in self.laws_by_name:
            return None
            
        law_sections = self.laws_by_name[law_name]
        
        if section_num:
            section_num = re.sub(r"\s+", " ", str(section_num)).strip()
            section_num = re.sub(r"^(?:มาตรา\s*)", "", section_num).strip()
            # Try exact match
            if section_num in law_sections:
                return law_sections[section_num]
            # Maybe try normalizing? (e.g. "132" vs "Section 132")
            return None
        else:
            # Return all sections joined
            sorted_sections = sorted(law_sections.items(), key=lambda x: x[0])
            full_text = f"Law: {law_name}\n\n"
            for sec_num, content in sorted_sections:
                full_text += f"--- Section {sec_num} ---\n{content}\n\n"
            return full_text.strip()

# Initialize DB
db = LawDatabase()


# --- API Models ---

class SearchRequest(BaseModel):
    queries: List[str]
    topk: int = 3
    return_scores: bool = True

class ReadRequest(BaseModel):
    law_name: str
    section_num: Optional[str] = None

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    db.load_data()

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    results = db.search(request.queries, request.topk)
    # The format expected by search_r1_like_utils.py is {"result": [...]}
    return {"result": results}

@app.post("/read")
async def read_endpoint(request: ReadRequest):
    content = db.read(request.law_name, request.section_num)
    if content:
        return {"text": content}
    else:
        raise HTTPException(status_code=404, detail="Law not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8932)))
