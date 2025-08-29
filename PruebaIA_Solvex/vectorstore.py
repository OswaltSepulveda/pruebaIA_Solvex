from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs_folder = "docs"
if not os.path.exists(docs_folder):
    raise FileNotFoundError(f"Este directorio no se encuentra: {docs_folder}")

docs = []
for fname in os.listdir(docs_folder):
    with open(os.path.join(docs_folder, fname), "r", encoding="utf-8") as f:
        content = f.read()
    docs.append(Document(page_content=content, metadata={"filename": fname}))

if not docs:
    raise ValueError(f"No se encontraron archivos en: {docs_folder}")

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("index")