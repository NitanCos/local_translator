import os
import sys
import logging
from logging.handlers import RotatingFileHandler


def setup_logging():
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    os.makedirs("Debug", exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # 統一輸出到 All.log
    all_handler = RotatingFileHandler(
        "Debug/All.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    all_handler.setFormatter(formatter)
    all_handler.setLevel(logging.DEBUG)

    main_logger = logging.getLogger("main")
    main_logger.addHandler(all_handler)
    main_logger.addHandler(console_handler)
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False

    ocr_logger = logging.getLogger("ocr_processor")
    ocr_logger.addHandler(all_handler)
    ocr_logger.addHandler(console_handler)
    ocr_logger.setLevel(logging.DEBUG)
    ocr_logger.propagate = False

    translator_logger = logging.getLogger("translater")
    translator_logger.addHandler(all_handler)
    translator_logger.addHandler(console_handler)
    translator_logger.setLevel(logging.DEBUG)
    translator_logger.propagate = False

    api_logger = logging.getLogger("API_mode")
    api_logger.addHandler(all_handler)
    api_logger.addHandler(console_handler)
    api_logger.setLevel(logging.DEBUG)

    region_logger = logging.getLogger("region_capture")
    region_logger.addHandler(all_handler)
    region_logger.addHandler(console_handler)
    region_logger.setLevel(logging.DEBUG)

