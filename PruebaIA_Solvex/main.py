from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import graph, vectorstore
import uvicorn

app = FastAPI(title="API Consulta de Productos")

class QueryRequest(BaseModel):
    user_id: str
    query: str

@app.get("/")
def root():
    return {"message": "API de consulta de productos funcionando correctamente."}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    state = {"query": request.query, "vectorstore": vectorstore}
    try:
        result = graph.compile().invoke(state)
        return {"user_id": request.user_id, "respuesta": result.get("respuesta", "")}
    except Exception as e:
        return {"user_id": request.user_id, "error": f"Error procesando query: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)