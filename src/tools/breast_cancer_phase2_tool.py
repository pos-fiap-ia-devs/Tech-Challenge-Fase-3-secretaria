"""
Tool Breast Cancer Fase 2 — FemCare AI (Tech Challenge FIAP Fase 3)

Módulo acadêmico de apoio à triagem inspirado no trabalho da Fase 2:
  - Breast Cancer Wisconsin Dataset
  - Random Forest otimizado com Algoritmo Genético
  - Foco em recall e redução de falsos negativos

IMPORTANTE — Aviso clínico:
  Esta tool NÃO fornece diagnóstico definitivo de câncer.
  Classifica níveis de atenção clínica para apoiar fluxos de triagem.
  Todas as saídas devem ser revisadas por profissionais de saúde qualificados.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Métricas documentadas da Fase 2 (Random Forest + Algoritmo Genético)
# ---------------------------------------------------------------------------
PHASE2_MODEL_METRICS: dict[str, float] = {
    "recall": 0.952381,
    "specificity": 0.972222,
    "precision": 0.952381,
}

# Seis features usadas apenas no fallback baseado em regras
FEATURE_KEYS: tuple[str, ...] = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave_points_mean",
)

# Conjunto completo Wisconsin (exames sintéticos Fase 3, nomes com underscore)
PHASE3_FEATURE_KEYS: tuple[str, ...] = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
)

# Limiares derivados dos pontos médios benigno/maligno do Wisconsin (contexto Fase 2).
_WISCONSIN_THRESHOLDS: dict[str, float] = {
    "radius_mean": 14.81,
    "texture_mean": 19.73,
    "perimeter_mean": 96.73,
    "area_mean": 720.59,
    "concavity_mean": 0.094,
    "concave_points_mean": 0.031,
}

_DEFAULT_SOURCES: tuple[str, ...] = (
    "phase2_breast_cancer_model_card.md",
    "breast_cancer_screening.md",
)

_MODEL_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "models" / "breast_cancer_rf_ga_pipeline.joblib",
    Path(__file__).resolve().parents[2] / "models" / "breast_cancer_rf_ga_pipeline.pkl",
    Path(__file__).resolve().parent / "models" / "breast_cancer_rf_phase2.pkl",
    Path(__file__).resolve().parents[2] / "data" / "models" / "breast_cancer_rf_phase2.pkl",
)

_LIMITATIONS_TEXT = (
    "Este resultado é apoio à triagem acadêmica de um MVP. "
    "Não representa diagnóstico definitivo e não deve ser usado "
    "para prescrever tratamento ou substituir avaliação médica profissional."
)

_POSITIVE_CLASS_CANDIDATES: tuple[Any, ...] = ("M", 1, "malignant", "maligno")


def _normalize_key(key: str) -> str:
    """Normaliza chaves de dicionário para strings minúsculas no estilo snake_case."""
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_mapping(data: dict[str, Any] | None) -> dict[str, Any]:
    """Retorna cópia de *data* com chaves normalizadas; dict vazio se a entrada for None."""
    if not data:
        return {}
    return {_normalize_key(str(k)): v for k, v in data.items()}


def _resolve_patient_id(patient_data: dict[str, Any]) -> str | None:
    """Extrai identificador de paciente a partir de nomes comuns em dados sintéticos."""
    normalized = _normalize_mapping(patient_data)
    patient_id = normalized.get("patient_id") or normalized.get("id") or normalized.get("nome")
    return str(patient_id) if patient_id is not None else None


def _to_float(value: Any) -> float | None:
    """Converte um valor para float quando possível."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_feature_availability(features: dict[str, float]) -> tuple[list[str], list[str]]:
    """Disponibilidade para fallback baseado em regras (6 features)."""
    available = [key for key in FEATURE_KEYS if key in features]
    missing = [key for key in FEATURE_KEYS if key not in features]
    return available, missing


def _extract_exam_features(exam_data: dict[str, Any]) -> dict[str, float]:
    """
    Extrai features numéricas Wisconsin de exam_data (até 30 chaves da Fase 3).

    Aceita nomes com underscore e variantes com espaço (ex.: concave points_mean).
    Não inventa valores ausentes.
    """
    normalized = _normalize_mapping(exam_data)
    features: dict[str, float] = {}

    for feature in PHASE3_FEATURE_KEYS:
        raw = normalized.get(feature)
        if raw is None:
            continue
        parsed = _to_float(raw)
        if parsed is None:
            logger.warning("Feature '%s' presente mas não numérica — ignorada", feature)
            continue
        features[feature] = parsed

    return features


def _feature_value_for_model_name(model_name: str, features: dict[str, float]) -> float | None:
    """Resolve um nome de coluna do modelo para um valor no dicionário de features."""
    if model_name in features:
        return features[model_name]

    normalized_name = _normalize_key(model_name)
    if normalized_name in features:
        return features[normalized_name]

    return None


def _adapt_feature_names_for_model(features: dict[str, float], model: Any) -> dict[str, float]:
    """
    Mapeia dict de features com underscore para os nomes esperados em model.feature_names_in_.

    Trata concave_points_* vs concave points_* automaticamente via normalização.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return dict(features)

    adapted: dict[str, float] = {}
    for name in expected:
        name_str = str(name)
        value = _feature_value_for_model_name(name_str, features)
        if value is not None:
            adapted[name_str] = value

    return adapted


def _compute_model_feature_availability(
    features: dict[str, float],
    model: Any,
) -> tuple[list[str], list[str]]:
    """Features disponíveis/ausentes usando model.feature_names_in_ quando presente."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        available = [key for key in PHASE3_FEATURE_KEYS if key in features]
        missing = [key for key in PHASE3_FEATURE_KEYS if key not in features]
        return available, missing

    available: list[str] = []
    missing: list[str] = []
    for name in expected:
        name_str = str(name)
        if _feature_value_for_model_name(name_str, features) is not None:
            available.append(name_str)
        else:
            missing.append(name_str)
    return available, missing


def _build_model_dataframe(
    features: dict[str, float],
    model: Any,
) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """
    Monta DataFrame de uma linha na ordem exata esperada pelo modelo.

    Retorna (None, available, missing) quando colunas obrigatórias estão ausentes.
    """
    adapted = _adapt_feature_names_for_model(features, model)
    expected = getattr(model, "feature_names_in_", None)

    if expected is not None:
        columns = [str(col) for col in expected]
        available, missing = _compute_model_feature_availability(features, model)
        if missing:
            return None, available, missing
        row = {col: adapted[col] for col in columns}
        return pd.DataFrame([row], columns=columns), available, missing

    columns = list(PHASE3_FEATURE_KEYS)
    available, missing = _compute_model_feature_availability(features, model)
    if missing:
        return None, available, missing
    row = {col: features[col] for col in columns}
    return pd.DataFrame([row], columns=columns), available, missing


def _load_phase2_model() -> tuple[Any | None, bool, str | None]:
    """
    Carrega Random Forest da Fase 2 a partir de caminhos candidatos joblib/pkl.

    Retorna:
        (model, model_loaded, model_path)
    """
    for path in _MODEL_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        logger.info("Tentando carregar modelo da Fase 2 em: %s", path)
        try:
            model = joblib.load(path)
        except Exception:
            logger.exception("Falha ao carregar modelo de %s — tentando próximo candidato", path)
            continue

        if not hasattr(model, "predict_proba"):
            logger.error("Objeto carregado em %s não possui predict_proba — tentando próximo candidato", path)
            continue

        logger.info("Modelo da Fase 2 carregado com sucesso de %s", path)
        return model, True, str(path)

    logger.info(
        "Modelo da Fase 2 não encontrado — caminhos candidatos: %s",
        [str(p) for p in _MODEL_CANDIDATE_PATHS],
    )
    return None, False, None


def _load_pickle_model() -> tuple[Any | None, bool]:
    """Wrapper de compatibilidade em torno de _load_phase2_model."""
    model, loaded, _ = _load_phase2_model()
    return model, loaded


def _resolve_positive_class_index(
    model: Any,
    probabilities: Any,
) -> tuple[int, str]:
    """Escolhe índice/classe positiva (maligno) na saída de predict_proba."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        for label in _POSITIVE_CLASS_CANDIDATES:
            if label in class_list:
                idx = class_list.index(label)
                logger.info(
                    "Classe positiva resolvida — index=%d label=%s",
                    idx,
                    label,
                )
                return idx, str(label)
        idx = len(class_list) - 1
        label = str(class_list[idx])
        logger.info(
            "Classe positiva resolvida — index=%d label=%s (última classe)",
            idx,
            label,
        )
        return idx, label

    idx = len(probabilities) - 1
    logger.info("Classe positiva resolvida — index=%d label=last_column (sem classes_)", idx)
    return idx, "last_column"

# !!! Função de predição que aplica o modelo da Fase 2!!!


def _predict_with_model(
    model: Any,
    features: dict[str, float],
) -> tuple[float | None, dict[str, Any]]:
    """
    Executa predict_proba no modelo da Fase 2 carregado usando DataFrame alinhado às features.

    Retorna:
        (probability, metadata) — probability é None quando a inferência não pode ser executada.
    """
    frame, available, missing = _build_model_dataframe(features, model)
    expected = getattr(model, "feature_names_in_", None)
    model_feature_count = len(expected) if expected is not None else len(PHASE3_FEATURE_KEYS)

    metadata: dict[str, Any] = {
        "model_feature_count": model_feature_count,
        "model_available_features": available,
        "model_missing_features": missing,
        "positive_class": None,
        "positive_class_index": None,
    }

    if frame is None:
        logger.warning(
            "Inferência do modelo ignorada — features ausentes (%d): %s",
            len(missing),
            missing,
        )
        return None, metadata

    try:
        probabilities = model.predict_proba(frame)[0]
        positive_index, positive_label = _resolve_positive_class_index(model, probabilities)
        metadata["positive_class"] = positive_label
        metadata["positive_class_index"] = positive_index
        return float(probabilities[positive_index]), metadata
    except Exception:
        logger.exception("predict_proba falhou no modelo da Fase 2")
        return None, metadata


def _rule_based_probability(features: dict[str, float]) -> float:
    """
    Fallback simples baseado em regras quando não há inferência do modelo.

    Usa apenas FEATURE_KEYS e _WISCONSIN_THRESHOLDS.
    """
    fallback_features = {k: features[k] for k in FEATURE_KEYS if k in features}

    if not fallback_features:
        return 0.45

    above_threshold = 0
    evaluated = 0

    for feature, value in fallback_features.items():
        threshold = _WISCONSIN_THRESHOLDS.get(feature)
        if threshold is None:
            continue
        evaluated += 1
        if value > threshold:
            above_threshold += 1

    if evaluated == 0:
        return 0.45

    ratio = above_threshold / evaluated
    probability = 0.12 + ratio * 0.80
    return round(min(max(probability, 0.0), 1.0), 4)


def _map_probability_to_risk(probability: float) -> tuple[str, str, str]:
    """Mapeia probabilidade do modelo para rótulos cautelosos de atenção clínica em português."""
    if probability >= 0.70:
        return (
            "alto",
            "maior atenção clínica",
            "Encaminhar para avaliação profissional e exames confirmatórios.",
        )
    if probability >= 0.40:
        return (
            "moderado",
            "atenção clínica moderada",
            "Agendar avaliação médica para revisão dos achados e conduta preventiva.",
        )
    return (
        "baixo",
        "baixa atenção clínica pelo modelo",
        "Manter acompanhamento preventivo de rotina conforme orientação profissional.",
    )


def _patient_context_snippet(patient_data: dict[str, Any]) -> str:
    """Monta linha curta de contexto da paciente, sem linguagem diagnóstica."""
    normalized = _normalize_mapping(patient_data)
    parts: list[str] = []

    patient_id = normalized.get("patient_id") or normalized.get("id") or normalized.get("nome")
    if patient_id:
        parts.append(f"paciente sintética '{patient_id}'")

    age = normalized.get("idade") or normalized.get("age")
    if age is not None:
        parts.append(f"idade {age} anos")

    family_history = normalized.get("historico_familiar_cancer_mama") or normalized.get(
        "historico_familiar"
    )
    if family_history is not None:
        if isinstance(family_history, str):
            flag = family_history.strip().lower() in {"true", "sim", "yes", "1"}
        else:
            flag = bool(family_history)
        if flag:
            parts.append("histórico familiar de câncer de mama informado")

    if not parts:
        return "Caso sintético analisado"

    return "Caso de " + ", ".join(parts)


def _build_explanation(
    patient_data: dict[str, Any],
    features: dict[str, float],
    probability: float,
    inference_method: str,
) -> str:
    """Compõe explicação em linguagem natural cautelosa (português, orientada à triagem)."""
    context = _patient_context_snippet(patient_data)

    if inference_method == "phase2_joblib_model":
        method_note = "modelo Random Forest + GA da Fase 2 (joblib)"
        method_article = "O"
    else:
        method_note = "regras simplificadas inspiradas no Wisconsin Dataset (fallback acadêmico)"
        method_article = "As"

    feature_notes: list[str] = []
    for feature in FEATURE_KEYS:
        if feature not in features:
            continue
        value = features[feature]
        threshold = _WISCONSIN_THRESHOLDS[feature]
        status = "acima" if value > threshold else "dentro"
        feature_notes.append(f"{feature}={value:.3f} ({status} do limite de referência)")

    features_text = (
        "; ".join(feature_notes)
        if feature_notes
        else "dados de exame limitados ou incompletos para análise detalhada"
    )

    return (
        f"{context}. {method_article} {method_note} indicou probabilidade estimada de {probability:.2%} "
        f"para padrão que requer atenção clínica, com base em: {features_text}. "
        "Este resultado é apoio à triagem e não confirma a presença de câncer."
    )


# !!! Função que gera texto exibindo e interpretando score!!!


def analyze_breast_cancer_case(patient_data: dict, exam_data: dict) -> dict:
    """
    Analisa um caso sintético de triagem de câncer de mama (ponto de entrada Fase 3).

    Usa modelo joblib da Fase 2 quando disponível; caso contrário, pontuação baseada em regras.
    """
    logger.info("analyze_breast_cancer_case iniciada")

    patient_id = _resolve_patient_id(patient_data)
    if patient_id:
        logger.info("Contexto da paciente resolvido — patient_id=%s", patient_id)
    else:
        logger.warning("patient_id não encontrado em patient_data — continuando com caso anônimo")

    if not exam_data:
        logger.warning("exam_data vazio — inferência usará defaults do fallback")

    features = _extract_exam_features(exam_data)
    available_features, missing_features = _compute_feature_availability(features)

    logger.info("Features do exame extraídas (%d chaves): %s", len(features), list(features.keys()))
    if available_features:
        logger.info(
            "Disponibilidade de features do fallback (%d/%d): %s",
            len(available_features),
            len(FEATURE_KEYS),
            available_features,
        )
    if missing_features:
        logger.warning(
            "Features ausentes no fallback (%d/%d): %s",
            len(missing_features),
            len(FEATURE_KEYS),
            missing_features,
        )

    inference_method = "rule_based_fallback"
    probability: float | None = None
    model_path: str | None = None
    model_feature_count = 0
    model_available_features: list[str] = []
    model_missing_features: list[str] = []
    positive_class: str | None = None
    positive_class_index: int | None = None
    model_inference_failed = False

    model, model_loaded, model_path = _load_phase2_model()

    if model_loaded and model is not None:
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            logger.info("Model feature_names_in_: %s", list(expected))
            model_feature_count = len(expected)
        else:
            model_feature_count = len(PHASE3_FEATURE_KEYS)

        logger.info("Tentando inferência com modelo da Fase 2 em %s", model_path)
        probability, predict_meta = _predict_with_model(model, features)
        model_available_features = list(predict_meta.get("model_available_features") or [])
        model_missing_features = list(predict_meta.get("model_missing_features") or [])
        positive_class = predict_meta.get("positive_class")
        positive_class_index = predict_meta.get("positive_class_index")
        model_feature_count = int(predict_meta.get("model_feature_count") or model_feature_count)

        if probability is not None:
            inference_method = "phase2_joblib_model"
            logger.info(
                "Inferência bem-sucedida via phase2_joblib_model — probability=%.4f",
                probability,
            )
        else:
            model_inference_failed = True
            inference_method = "rule_based_fallback"
            logger.warning(
                "Inferência do modelo da Fase 2 falhou — alternando para rule_based_fallback "
                "(model_loaded=True, missing=%s)",
                model_missing_features,
            )
    else:
        model_loaded = False
        logger.info("Usando rule_based_fallback (nenhum modelo válido da Fase 2 disponível)")

    if probability is None:
        probability = _rule_based_probability(features)
        logger.info("Probabilidade do fallback baseado em regras calculada: %.4f", probability)

    probability = round(float(probability), 4)
    risk_level, prediction_label, recommended_action = _map_probability_to_risk(probability)

    normalized_patient = _normalize_mapping(patient_data)
    family_flag = normalized_patient.get("historico_familiar_cancer_mama") or normalized_patient.get(
        "historico_familiar"
    )
    if family_flag is not None:
        if isinstance(family_flag, str):
            has_family_history = family_flag.strip().lower() in {"true", "sim", "yes", "1"}
        else:
            has_family_history = bool(family_flag)

        if has_family_history and risk_level == "baixo":
            logger.info("Risco elevado para moderado devido a histórico familiar (heurística de triagem)")
            risk_level = "moderado"
            prediction_label = "atenção clínica moderada (histórico familiar)"
            recommended_action = (
                "Agendar avaliação médica considerando histórico familiar de câncer de mama."
            )

    explanation = _build_explanation(patient_data, features, probability, inference_method)

    result = {
        "flow": "breast_cancer",
        "risk_level": risk_level,
        "prediction_label": prediction_label,
        "probability": probability,
        "model_metrics": dict(PHASE2_MODEL_METRICS),
        "explanation": explanation,
        "limitations": _LIMITATIONS_TEXT,
        "recommended_action": recommended_action,
        "sources": list(_DEFAULT_SOURCES),
        "inference_method": inference_method,
        "model_loaded": model_loaded,
        "missing_features": missing_features,
        "available_features": available_features,
        "model_path": model_path,
        "model_feature_count": model_feature_count,
        "model_available_features": model_available_features,
        "model_missing_features": model_missing_features,
        "positive_class": positive_class,
        "positive_class_index": positive_class_index,
        "model_inference_failed": model_inference_failed,
    }

    logger.info(
        "analyze_breast_cancer_case concluída — patient_id=%s inference_method=%s "
        "probability=%.4f risk_level=%s model_loaded=%s model_path=%s",
        patient_id or "anonymous",
        inference_method,
        probability,
        risk_level,
        model_loaded,
        model_path,
    )

    return result


def _sample_exam_p002() -> dict[str, float]:
    """Exame sintético com 30 features alinhado a P002 (maior atenção)."""
    return {
        "radius_mean": 18.9,
        "texture_mean": 23.8,
        "perimeter_mean": 122.6,
        "area_mean": 1010.5,
        "smoothness_mean": 0.115,
        "compactness_mean": 0.235,
        "concavity_mean": 0.178,
        "concave_points_mean": 0.074,
        "symmetry_mean": 0.205,
        "fractal_dimension_mean": 0.064,
        "radius_se": 0.85,
        "texture_se": 1.55,
        "perimeter_se": 5.9,
        "area_se": 95.0,
        "smoothness_se": 0.0085,
        "compactness_se": 0.045,
        "concavity_se": 0.065,
        "concave_points_se": 0.018,
        "symmetry_se": 0.024,
        "fractal_dimension_se": 0.0045,
        "radius_worst": 24.9,
        "texture_worst": 32.6,
        "perimeter_worst": 164.0,
        "area_worst": 1860.0,
        "smoothness_worst": 0.156,
        "compactness_worst": 0.52,
        "concavity_worst": 0.62,
        "concave_points_worst": 0.21,
        "symmetry_worst": 0.36,
        "fractal_dimension_worst": 0.096,
    }


def _sample_exam_p006() -> dict[str, float]:
    """Exame sintético com 30 features alinhado a P006 (menor atenção)."""
    return {
        "radius_mean": 10.6,
        "texture_mean": 15.3,
        "perimeter_mean": 67.1,
        "area_mean": 378.6,
        "smoothness_mean": 0.047,
        "compactness_mean": 0.069,
        "concavity_mean": 0.017,
        "concave_points_mean": 0.007,
        "symmetry_mean": 0.165,
        "fractal_dimension_mean": 0.06,
        "radius_se": 0.18,
        "texture_se": 0.98,
        "perimeter_se": 1.25,
        "area_se": 12.0,
        "smoothness_se": 0.0048,
        "compactness_se": 0.015,
        "concavity_se": 0.01,
        "concave_points_se": 0.0045,
        "symmetry_se": 0.016,
        "fractal_dimension_se": 0.0028,
        "radius_worst": 11.7,
        "texture_worst": 20.2,
        "perimeter_worst": 75.0,
        "area_worst": 450.0,
        "smoothness_worst": 0.085,
        "compactness_worst": 0.125,
        "concavity_worst": 0.06,
        "concave_points_worst": 0.026,
        "symmetry_worst": 0.23,
        "fractal_dimension_worst": 0.068,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("=== FemCare AI — Tool Breast Cancer Fase 2 (teste local) ===\n")

    model_probe, model_probe_loaded, model_probe_path = _load_phase2_model()
    if model_probe_loaded and model_probe is not None:
        names = getattr(model_probe, "feature_names_in_", None)
        print(f"[MODELO] carregado de {model_probe_path}")
        print(f"[MODELO] feature_names_in_ ({len(names) if names is not None else 0}):")
        print(list(names) if names is not None else "N/A")
    else:
        print("[MODELO] não carregado — rule_based_fallback esperado")

    print("\n[P002 — maior atenção, 30 features]")
    result_p002 = analyze_breast_cancer_case(
        {"patient_id": "P002", "idade": 28, "historico_familiar_cancer_mama": True},
        _sample_exam_p002(),
    )
    print(
        f"inference_method={result_p002['inference_method']} "
        f"model_loaded={result_p002['model_loaded']} "
        f"model_path={result_p002.get('model_path')}"
    )
    print(f"model_missing_features={result_p002.get('model_missing_features')}")
    print(json.dumps(result_p002, indent=2, ensure_ascii=False))

    print("\n[P006 — menor atenção, 30 features]")
    result_p006 = analyze_breast_cancer_case(
        {"patient_id": "P006", "idade": 52, "historico_familiar_cancer_mama": False},
        _sample_exam_p006(),
    )
    print(
        f"inference_method={result_p006['inference_method']} "
        f"model_loaded={result_p006['model_loaded']} "
        f"model_path={result_p006.get('model_path')}"
    )
    print(f"model_missing_features={result_p006.get('model_missing_features')}")
    print(json.dumps(result_p006, indent=2, ensure_ascii=False))

    print("\n[EXAME PARCIAL — apenas 6 features]")
    result_partial = analyze_breast_cancer_case(
        {"patient_id": "P001", "idade": 45},
        {
            "radius_mean": 15.0,
            "texture_mean": 20.0,
            "perimeter_mean": 98.0,
            "area_mean": 800.0,
            "concavity_mean": 0.10,
            "concave_points_mean": 0.04,
        },
    )
    print(
        f"inference_method={result_partial['inference_method']} "
        f"model_missing_features={result_partial.get('model_missing_features')}"
    )
