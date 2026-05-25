"""
Importa MedQuAD, filtra categorias de saúde da mulher e salva em data/medquad.csv.
Mantém conteúdo em inglês (sem tradução).

Filtragem por conteúdo do elemento <Focus> do XML (não pelo nome do arquivo).

Execução:
    uv run python fine_tuning/import_medquad.py
    uv run python fine_tuning/import_medquad.py --max-per-category 100
"""

import argparse
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "medquad.csv"

GITHUB_API = "https://api.github.com/repos/abachaa/MedQuAD/contents"
GITHUB_RAW = "https://raw.githubusercontent.com/abachaa/MedQuAD/master"

# Pastas reais do repo → instrucao do projeto
# scan_limit=0 significa escanear todos os arquivos da pasta
CATEGORY_MAP = {
    "1_CancerGov_QA": "Prevenção Câncer",
    "4_MPlus_Health_Topics_QA": "Saúde da Mulher",
    "7_SeniorHealth_QA": "Menopausa",
}

# scan_limit por categoria (0 = todos)
CATEGORY_SCAN_LIMIT = {
    "1_CancerGov_QA": 0,        # 116 arquivos — escaneia todos
    "4_MPlus_Health_Topics_QA": 300,
    "7_SeniorHealth_QA": 0,
}

# Palavras-chave no conteúdo do <Focus> para filtrar documentos relevantes
FOCUS_KEYWORDS = [
    # Câncer ginecológico (exatos do CancerGov)
    "breast", "endometrial", "uterine", "ovarian", "fallopian",
    "cervical", "vaginal", "vulvar", "gestational trophoblastic",
    # Saúde reprodutiva
    "pregnancy", "prenatal", "maternal", "fetal", "labor", "childbirth",
    "miscarriage", "abortion", "placenta", "eclampsia", "breastfeed",
    "lactation", "menopause", "climacteric",
    # Ginecologia geral
    "gynecolog", "obstetric", "fibroid", "endometriosis", "pcos",
    "polycystic ovary", "infertil", "contraception",
    # IST
    "hiv", "chlamydia", "gonorrhea", "syphilis", "herpes", "hpv",
    # Geral mulher
    "women", "female health", "osteoporosis",
]


def is_relevant_focus(focus_text: str) -> bool:
    text = focus_text.lower()
    return any(kw in text for kw in FOCUS_KEYWORDS)


def list_github_files(folder: str) -> list[str]:
    url = f"{GITHUB_API}/{folder}"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        print(f"  Erro ao listar {folder}: HTTP {resp.status_code}")
        return []
    return [f["name"] for f in resp.json() if f["name"].endswith(".xml")]


def download_xml(folder: str, filename: str) -> str | None:
    url = f"{GITHUB_RAW}/{folder}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


def parse_xml(content: str) -> tuple[str, list[dict]]:
    """Retorna (focus, lista de pares QA)."""
    pairs = []
    focus = ""
    try:
        root = ET.fromstring(content)
        focus_el = root.find("Focus")
        if focus_el is not None:
            focus = (focus_el.text or "").strip()
        for qapair in root.findall(".//QAPair"):
            q_el = qapair.find("Question")
            a_el = qapair.find("Answer")
            if q_el is not None and a_el is not None:
                q = (q_el.text or "").strip()
                a = (a_el.text or "").strip()
                if q and a and len(a) > 30:
                    pairs.append({"question": q, "answer": a})
    except ET.ParseError:
        pass
    return focus, pairs


def process_pairs(pairs: list, source: str, focus_area: str, max_count: int, current: int) -> list:
    results = []
    for pair in pairs:
        if current + len(results) >= max_count:
            break
        results.append({"question": pair["question"], "answer": pair["answer"], "source": source, "focus_area": focus_area})
    return results


def collect_category(folder: str, instrucao: str, max_per_category: int, scan_limit: int) -> list:
    print(f"[{folder}] → '{instrucao}'")
    all_files = list_github_files(folder)
    cat_limit = CATEGORY_SCAN_LIMIT.get(folder, scan_limit)
    scanned = all_files if cat_limit == 0 else all_files[:cat_limit]
    print(f"  {len(all_files)} arquivos no repo | escaneando {len(scanned)}")

    examples = []
    relevant_docs = 0

    for filename in scanned:
        if len(examples) >= max_per_category:
            break
        xml_content = download_xml(folder, filename)
        if not xml_content:
            time.sleep(0.5)
            continue
        focus, pairs = parse_xml(xml_content)
        if not is_relevant_focus(focus):
            continue
        relevant_docs += 1
        examples.extend(process_pairs(pairs, folder, instrucao, max_per_category, len(examples)))
        time.sleep(0.1)

    print(f"  Docs relevantes: {relevant_docs} | Exemplos coletados: {len(examples)}\n")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Importar MedQuAD para ConsultasMedica")
    parser.add_argument("--max-per-category", type=int, default=80,
                        help="Máximo de exemplos por categoria (default: 80)")
    parser.add_argument("--scan-limit", type=int, default=300,
                        help="Máximo de arquivos XML a escanear por categoria (default: 300)")
    args = parser.parse_args()

    new_examples = []

    for folder, instrucao in CATEGORY_MAP.items():
        new_examples.extend(
            collect_category(folder, instrucao, args.max_per_category, args.scan_limit)
        )

    existing_rows = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    merged = existing_rows + new_examples
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source", "focus_area"])
        writer.writeheader()
        writer.writerows(merged)

    print("=" * 50)
    print(f"Existentes       : {len(existing_rows)} exemplos")
    print(f"MedQuAD adicionou: {len(new_examples)} exemplos")
    print(f"Total final      : {len(merged)} exemplos")
    print(f"Salvo em         : {OUTPUT_PATH}")
    print("\nPróximo passo: uv run python fine_tuning/prepare_data.py")


if __name__ == "__main__":
    main()
