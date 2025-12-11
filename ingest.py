import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "storage/chroma"
EMBED_MODEL = "text-embedding-3-large"  # high quality; use 3-small if cost sensitive

def load_documents(data_dir=DATA_DIR):
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.lower().endswith(".pdf"):
            # Try PyPDF first; fallback to Unstructured if PDFs are funky
            try:
                docs.extend(PyPDFLoader(path).load())
            except Exception:
                docs.extend(UnstructuredFileLoader(path).load())
        elif fname.lower().endswith((".txt", ".md", ".html")):
            docs.extend(UnstructuredFileLoader(path).load())
        # Add more loaders (docx, pptx) as you like
    return docs

def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} raw docs. Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=160, add_start_index=True, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split to {len(chunks)} chunks.")

    # Add useful metadata
    for c in chunks:
        c.metadata.setdefault("source", c.metadata.get("source") or c.metadata.get("file_path"))
        c.metadata.setdefault("title", os.path.basename(c.metadata.get("source", "doc")))
        # you can also attach page numbers if loader provides them (PyPDFLoader puts 'page')

    # print("Embedding & writing to Chroma...")
    # embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        # Load the Sentence-Transformers model (this is a local model)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Embedding & writing to Chroma...")

    # Generate embeddings using the local model
    embeddings = model.encode([doc.page_content for doc in chunks], show_progress_bar=True)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="research-papers",
    )
    vectordb.persist()
    print("Done. Vector store persisted at:", PERSIST_DIR)

if __name__ == "__main__":
    main()
