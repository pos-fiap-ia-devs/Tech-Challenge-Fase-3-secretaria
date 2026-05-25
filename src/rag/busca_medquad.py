import csv
import hashlib
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.logger import get_logger

log = get_logger("consultas_medica.rag")

_PALAVRAS_RELEVANTES = [
    "breast", "cervical", "ovarian", "uterine", "endometrial", "vaginal",
    "vulvar", "fallopian", "pregnancy", "prenatal", "maternal", "menopause",
    "gynecolog", "obstetric", "fibroid", "endometriosis", "osteoporosis",
    "women", "female", "breastfeed", "lactation", "contraception", "hpv",
    "chlamydia", "gonorrhea", "syphilis", "alzheimer", "dementia",
    "depression", "anxiety", "diabetes", "heart", "stroke", "cancer",
]

_PALAVRAS_BLOQUEADAS = [
    "prostate", "testicular", "penile", "male breast", "scrotal",
    "erectile", "vasectomy", "testosterone", "urethral", "urachal",
    "childhood", "pediatric", "neonatal", "infant", "newborn",
]


def _eh_relevante(focus_area: str) -> bool:
    f = focus_area.lower()
    if any(bk in f for bk in _PALAVRAS_BLOQUEADAS):
        return False
    return any(kw in f for kw in _PALAVRAS_RELEVANTES)


def _hash_csv(dataset_path: str) -> str:
    h = hashlib.md5()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_MAX_ANSWER_CHARS = 2000


def _carregar_documentos(dataset_path: str) -> list[Document]:
    if not os.path.exists(dataset_path):
        log.warning("MedQuAD CSV não encontrado: %s", dataset_path)
        return []
    docs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()[:_MAX_ANSWER_CHARS]
            cat = row.get("focus_area") or row.get("source") or "MedQuAD"
            if q and a and _eh_relevante(cat):
                docs.append(
                    Document(
                        page_content=f"Category: {cat}\nQuestion: {q}\nAnswer: {a}",
                        metadata={"source": "MedQuAD", "categoria": cat},
                    )
                )
    log.info("MedQuAD | %d documentos relevantes carregados", len(docs))
    return docs


def inicializar_rag(embeddings_model):
    """Inicializa RAG com medquad.csv. Persiste índice FAISS em disco — só reconstrói se CSV mudar."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "medquad.csv")
        faiss_dir = os.path.join(base_dir, "data", "faiss_index")
        hash_file = os.path.join(faiss_dir, "csv.hash")

        current_hash = _hash_csv(dataset_path) if os.path.exists(dataset_path) else ""
        cached_hash = open(hash_file).read().strip() if os.path.exists(hash_file) else ""

        if os.path.exists(faiss_dir) and current_hash == cached_hash:
            log.info("RAG | carregando índice FAISS do disco (%s)", faiss_dir)
            vectorstore = FAISS.load_local(
                faiss_dir, embeddings_model, allow_dangerous_deserialization=True
            )
        else:
            log.info("RAG | construindo índice FAISS (CSV novo ou modificado)...")
            docs = _carregar_documentos(dataset_path)
            vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)
            os.makedirs(faiss_dir, exist_ok=True)
            vectorstore.save_local(faiss_dir)
            with open(hash_file, "w") as f:
                f.write(current_hash)
            log.info("RAG | índice salvo em disco: %d documentos", len(docs))

        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.55},
        ), True

    except Exception as exc:
        log.warning("RAG Engine falhou (%s) — sem RAG disponível", exc)
        return None, False
