from __future__ import annotations

"""Cliente OpenAI para el backend de Evalify AI."""

import logging

from langchain_openai import ChatOpenAI

from app.config import OPENAI_MODEL

logger = logging.getLogger(__name__)


def build_llm(model_name: str | None = None) -> ChatOpenAI:
    """Construye un cliente ChatOpenAI con parametros seguros por defecto."""
    selected_model = (model_name or OPENAI_MODEL).strip()
    kwargs = {"model": selected_model}

    # o-series models reject explicit temperature in current Chat Completions API.
    # For those models we rely on provider defaults.
    if not selected_model.lower().startswith("o"):
        kwargs["temperature"] = 0

    logger.debug("Inicializando cliente OpenAI model=%s explicit_temperature=%s", selected_model, "temperature" in kwargs)
    return ChatOpenAI(**kwargs)
