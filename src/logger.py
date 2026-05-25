"""
Logger centralizado do ConsultasMedica.
Grava em console e em logs/consultas_medica.log com rotação diária.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "consultas_medica.log")

os.makedirs(LOG_DIR, exist_ok=True)

_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console — INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_fmt, _datefmt))

    # Arquivo — DEBUG+, rotação diária, mantém 30 dias
    fh = TimedRotatingFileHandler(LOG_FILE, when="midnight", backupCount=30, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_fmt, _datefmt))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
