import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIR = "storage/chroma"
COLLECTION = "research-papers"
EMBED_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o-mini"  # good price/quality; swap as you prefer

# ----- Optional: Cross-encoder reranker (bge-reranker) -----
USE_RERANKER = True
RERANK_TOP_K = 20
FINAL_K = 6

try:
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) if USE_RERANKER else None
except Exception:
    reranker = None
    USE_RERANKER = False

def load_vectordb():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION
    )

def hybrid_retrieve(vectordb, query: str, k_dense=30) -> List[Document]:
    # Dense hits
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": k_dense,
            "fetch_k": k_dense,
            "lambda_mult": 0.0,  # set >0 if you enabled hybrid in your DB
        }
    )
    hits = retriever.get_relevant_documents(query)
    return hits

def rerank_docs(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not (USE_RERANKER and reranker and len(docs) > top_k):
        return docs[:top_k]
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.compute_score(pairs)  # higher is better
    doc_with_scores = list(zip(docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [d for d, s in doc_with_scores[:top_k]]

SYSTEM_RULES = """
You are an AI Research Assistant. Answer ONLY using the provided context.
- Cite sources inline like [1], [2], using the provided source map.
- If the answer is not in context, say "I don’t know based on the provided papers."
- Prefer quoting or paraphrasing precisely; avoid speculation.
"""

USER_TEMPLATE = """Question: {question}

Context:
{context}

Cite sources with their numeric IDs in square brackets.
"""

prompt = PromptTemplate.from_template(USER_TEMPLATE)

def build_context(blobs: List[Document]) -> (str, Dict[int, Dict[str, Any]]):
    lines = []
    source_map = {}
    for i, d in enumerate(blobs, start=1):
        # Shorten content per chunk if huge
        snippet = d.page_content.strip()
        meta = d.metadata or {}
        title = meta.get("title") or meta.get("source") or f"doc-{i}"
        pg = meta.get("page")
        src = f"{title}" + (f", p.{pg}" if pg is not None else "")
        source_map[i] = {"title": title, "page": pg, "path": meta.get("source")}
        lines.append(f"[{i}] ({src})\n{snippet}\n")
    return "\n\n".join(lines), source_map

def answer_question(query: str) -> Dict[str, Any]:
    vectordb = load_vectordb()
    dense_hits = hybrid_retrieve(vectordb, query, k_dense=RERANK_TOP_K if USE_RERANKER else FINAL_K)

    selected = rerank_docs(query, dense_hits, FINAL_K)

    context, source_map = build_context(selected)
    llm = ChatOpenAI(model=GPT_MODEL, temperature=0.1)
    chain = prompt | llm | StrOutputParser()

    text = chain.invoke({"question": query, "context": context})

    return {
        "answer": text,
        "sources": source_map
    }

# -------- CLI -----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=False, default="Summarize the main contributions of these papers.")
    args = parser.parse_args()

    resp = answer_question(args.query)
    print("\n=== Answer ===\n")
    print(resp["answer"])
    print("\n=== Sources ===")
    print(json.dumps(resp["sources"], indent=2))




# ---- Streamlit UI ----
# streamlit run app.py

import streamlit as st

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research Assistant (RAG + LangChain)")

query = st.text_input("Ask a question about your uploaded papers:", value="What is the key innovation in these works?")
if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        resp = answer_question(query)
    st.markdown("### Answer")
    st.write(resp["answer"])
    st.markdown("### Sources")
    for i, meta in resp["sources"].items():
        st.write(f"[{i}] **{meta['title']}**" + (f", p.{meta['page']}" if meta.get("page") is not None else ""))

