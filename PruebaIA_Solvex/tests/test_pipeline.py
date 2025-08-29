import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pipeline import agente_recuperador, agente_respondedor, State

# Fixture para crear un vectorstore ficticio
@pytest.fixture
def vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [
        Document(page_content="Smartphone con 8GB RAM", metadata={"filename": "doc1.txt"}),
        Document(page_content="Laptop con 16GB RAM", metadata={"filename": "doc2.txt"}),
        Document(page_content="Tablet con pantalla grande", metadata={"filename": "doc3.txt"}),
        Document(page_content="Auriculares inalámbricos", metadata={"filename": "doc4.txt"}),
        Document(page_content="Cámara digital de alta resolución", metadata={"filename": "doc5.txt"}),
    ]
    return FAISS.from_documents(docs, embeddings)

# Fixture para inicializar el estado
@pytest.fixture
def state(vectorstore):
    return State(query="smartphone", vectorstore=vectorstore, contexto="", respuesta="")

# Prueba para agente_recuperador
def test_agente_recuperador(state):
    result = agente_recuperador(state)
    assert "contexto" in result
    assert len(result["contexto"]) > 0
    assert "Smartphone" in result["contexto"]  # Verifica que recupera el documento relevante
    assert result["query"] == state["query"]  # Verifica que la consulta no cambia
    assert result["contexto"].startswith("Smartphone")  # Comprueba que "Smartphone" es el primero en el contexto

# Prueba para agente_respondedor
def test_agente_respondedor(state, monkeypatch):
    # Simular el generador
    def mock_generator(prompt):
        return [{"generated_text": f"{prompt}\nRespuesta: Información sobre smartphones"}]
    
    monkeypatch.setattr("pipeline.generator", mock_generator)
    state["contexto"] = "Smartphone con 8GB RAM"
    result = agente_respondedor(state)
    
    assert "respuesta" in result
    assert result["respuesta"] == "Información sobre smartphones"
    assert result["contexto"] == state["contexto"]
    assert result["query"] == state["query"]