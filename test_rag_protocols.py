from langchain_ollama import OllamaEmbeddings
from src.rag.busca_medquad import inicializar_rag

emb = OllamaEmbeddings(model="nomic-embed-text")
retriever, ok = inicializar_rag(emb)

print("ok:", ok)
print("retriever:", type(retriever))

query = (
    "breast_cancer_screening breast cancer screening mammogram risk factors women "
    "câncer de mama rastreamento mamografia sinais de alerta"
)

docs = retriever.invoke(query) if retriever else []
print("docs:", len(docs))

for i, doc in enumerate(docs, 1):
    print("\n--- DOC", i, "---")
    print("metadata:", doc.metadata)
    print(doc.page_content[:700])