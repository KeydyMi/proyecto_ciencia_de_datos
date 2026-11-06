import logging
import sys

def get_logger(name: str, log_file: str = "job.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Formateador compartido para consola y archivo
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        
        # 1. Handler para la consola (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 2. Handler para el archivo de logs
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.setLevel(logging.INFO)
        
    return logger