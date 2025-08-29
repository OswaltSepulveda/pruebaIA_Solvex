import os
from langgraph.graph import StateGraph, END
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict

HF_TOKEN = "usa tu token aqui para pruebas"

# Define state schema
class State(TypedDict):
    query: str
    vectorstore: FAISS
    contexto: str
    respuesta: str

# Función para inicializar el vectorstore y el modelo
def initialize_pipeline():
    global vectorstore, generator
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    LOCAL_MODEL_PATH = "./models/llama-2-7b-chat-hf"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, token=HF_TOKEN)
        print("Modelo cargado desde el disco.")
    except Exception as e:
        print(f"Error al cargar el modelo local: {e}")
        print("Descargando el modelo desde Hugging Face...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                device_map="auto",
                quantization_config=bnb_config,
                token=HF_TOKEN
            )
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
        except Exception as e:
            raise RuntimeError(f"Error al descargar el modelo: {e}")

    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        pad_token_id=tokenizer.eos_token_id
    )

# Inicializar solo si no estamos en un entorno de pruebas
if "pytest" not in os.environ.get("PYTEST_CURRENT_TEST", ""):
    initialize_pipeline()

def agente_recuperador(state: State) -> State:
    query = state["query"]
    docs = state["vectorstore"].similarity_search(query, k=10)
    state["contexto"] = "\n".join([d.page_content for d in docs])
    return state

def agente_respondedor(state: State) -> State:
    query = state["query"]
    contexto = state["contexto"]
    print(f"Contexto pasado al modelo:\n{contexto}")
    prompt = f"""
    Eres un asistente experto en productos. 
    Usa SOLO la siguiente información para responder. 
    Enumera TODOS los productos mencionados en el contexto, sin omitir ninguno.
    Si no hay información relevante, responde: "No encontré información en los documentos".

    Contexto:
    {contexto}

    Pregunta: {query}

    Respuesta:
    """
    output = generator(prompt)
    salida = output[0]['generated_text']
    respuesta = salida.split("Respuesta:")[-1].strip() if "Respuesta:" in salida else salida.strip()
    state["respuesta"] = respuesta
    return state

graph = StateGraph(State)
graph.add_node("recuperador", agente_recuperador)
graph.add_node("respondedor", agente_respondedor)
graph.add_edge("recuperador", "respondedor")
graph.set_entry_point("recuperador")
graph.add_edge("respondedor", END)