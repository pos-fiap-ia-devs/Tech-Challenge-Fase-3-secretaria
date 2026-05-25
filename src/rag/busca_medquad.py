import csv
import glob
import hashlib
import os
import re
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


def _hash_index_sources(csv_path: str, protocols_dir: str) -> str:
    """
    Hash combinado do CSV MedQuAD e dos protocolos Markdown (nome + conteúdo).
    Usado para invalidar o índice FAISS quando qualquer fonte mudar.
    """
    h = hashlib.md5()
    if os.path.exists(csv_path):
        h.update(_hash_csv(csv_path).encode("utf-8"))
    for path in sorted(glob.glob(os.path.join(protocols_dir, "*.md"))):
        h.update(os.path.basename(path).encode("utf-8"))
        h.update(_hash_csv(path).encode("utf-8"))
    return h.hexdigest()


_MAX_ANSWER_CHARS = 2000
_PROTOCOL_SECTION_MAX_CHARS = 1800
_SECTION_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


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


def _split_section_content(content: str, max_chars: int) -> list[str]:
    """Quebra conteúdo longo de uma seção em subchunks preservando parágrafos."""
    content = content.strip()
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) > max_chars:
            for offset in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[offset : offset + max_chars])
            current = ""
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks or [content[:max_chars]]


def _parse_markdown_sections(text: str) -> list[dict[str, str | int]]:
    """
    Divide Markdown em seções por headings #, ## e ###.

    Retorna lista de dicts com title, content e section_index.
    """
    sections: list[dict[str, str | int]] = []
    current_title = "Introdução"
    current_lines: list[str] = []
    section_index = 0

    for line in text.splitlines():
        heading_match = _SECTION_HEADING_RE.match(line.strip())
        if heading_match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(
                        {
                            "title": current_title,
                            "content": body,
                            "section_index": section_index,
                        }
                    )
                    section_index += 1
            current_title = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                {
                    "title": current_title,
                    "content": body,
                    "section_index": section_index,
                }
            )

    return sections


def _carregar_protocolos(protocols_dir: str, base_dir: str) -> list[Document]:
    """
    Carrega arquivos Markdown de data/protocols/ como texto puro (sem execução).

    Cada seção (# / ## / ###) vira um ou mais chunks FAISS com título e metadados.
    """
    if not os.path.isdir(protocols_dir):
        log.warning("Pasta de protocolos não encontrada: %s — continuando só com MedQuAD", protocols_dir)
        return []

    md_paths = sorted(glob.glob(os.path.join(protocols_dir, "*.md")))
    log.info("Protocolos | %d arquivo(s) Markdown encontrado(s) em %s", len(md_paths), protocols_dir)

    if not md_paths:
        log.warning("Nenhum arquivo .md encontrado em %s", protocols_dir)
        return []

    docs: list[Document] = []
    total_sections = 0
    for path in md_paths:
        filename = os.path.basename(path)
        category = os.path.splitext(filename)[0]
        rel_path = os.path.relpath(path, base_dir).replace("\\", "/")

        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
        except OSError as exc:
            log.warning("Falha ao ler protocolo %s — ignorando: %s", filename, exc)
            continue

        if not text:
            log.warning("Protocolo vazio — ignorando: %s", filename)
            continue

        sections = _parse_markdown_sections(text)
        total_sections += len(sections)
        log.info("Protocolos | %s — %d seção(ões) identificada(s)", filename, len(sections))

        for section in sections:
            section_title = str(section["title"])
            section_index = int(section["section_index"])
            section_body = str(section["content"])
            subchunks = _split_section_content(section_body, _PROTOCOL_SECTION_MAX_CHARS)

            for chunk_index, chunk_text in enumerate(subchunks):
                docs.append(
                    Document(
                        page_content=(
                            f"Protocol: {filename}\n"
                            f"Section: {section_title}\n"
                            f"Content:\n"
                            f"{chunk_text}"
                        ),
                        metadata={
                            "source": filename,
                            "source_type": "curated_protocol",
                            "category": category,
                            "categoria": category,
                            "path": rel_path,
                            "section_title": section_title,
                            "section_index": section_index,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    log.info(
        "Protocolos | %d seção(ões) parseada(s), %d chunk(s) indexado(s)",
        total_sections,
        len(docs),
    )
    return docs


def inicializar_rag(embeddings_model):
    """Inicializa RAG com medquad.csv e data/protocols/*.md. Persiste FAISS em disco."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "medquad.csv")
        protocols_dir = os.path.join(base_dir, "data", "protocols")
        faiss_dir = os.path.join(base_dir, "data", "faiss_index")
        hash_file = os.path.join(faiss_dir, "csv.hash")

        current_hash = _hash_index_sources(dataset_path, protocols_dir)
        cached_hash = open(hash_file).read().strip() if os.path.exists(hash_file) else ""

        if os.path.exists(faiss_dir) and current_hash == cached_hash:
            log.info("RAG | carregando índice FAISS do disco (%s)", faiss_dir)
            vectorstore = FAISS.load_local(
                faiss_dir, embeddings_model, allow_dangerous_deserialization=True
            )
        else:
            log.info("RAG | construindo índice FAISS (fontes novas ou modificadas)...")
            docs = _carregar_documentos(dataset_path)
            medquad_count = len(docs)
            protocol_docs = _carregar_protocolos(protocols_dir, base_dir)
            docs.extend(protocol_docs)
            log.info(
                "RAG | total de documentos a indexar: %d (MedQuAD=%d, protocolos=%d)",
                len(docs),
                medquad_count,
                len(protocol_docs),
            )
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
