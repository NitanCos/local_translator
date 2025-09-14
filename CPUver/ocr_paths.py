import os

def _basename_or_default(path_str: str, default_name: str) -> str:
    """Return basename of a path string or a default name if empty/invalid."""
    try:
        b = os.path.basename(path_str.replace("\\", "/").rstrip("/"))
        return b if b else default_name
    except Exception:
        return default_name


def build_ocr_dirs_from_base(base_dir: str, cfg) -> dict:
    """
    Using cfg's *_model_dir and *_model_name, construct model directories under base_dir.
    Rule: if *_model_dir is empty, use "{*_model_name}_infer". Otherwise, take its basename and
    join under base_dir.
    """

    def infer_subdir(current_dir: str, default_from_name: str) -> str:
        base = _basename_or_default(current_dir, f"{default_from_name}_infer")
        return os.path.join(base_dir, base)

    return {
        "doc_unwarping_model_dir":            infer_subdir(cfg.doc_unwarping_model_dir,            cfg.doc_unwarping_model_name),
        "textline_orientation_model_dir":     infer_subdir(cfg.textline_orientation_model_dir,     cfg.textline_orientation_model_name),
        "doc_orientation_classify_model_dir": infer_subdir(cfg.doc_orientation_classify_model_dir, cfg.doc_orientation_classify_model_name),
        "text_detection_model_dir":           infer_subdir(cfg.text_detection_model_dir,           cfg.text_detection_model_name),
        "text_recognition_model_dir":         infer_subdir(cfg.text_recognition_model_dir,         cfg.text_recognition_model_name),
    }


def guess_ocr_base_dir_from_cfg(cfg) -> str | None:
    """Guess a common OCR base directory from current *_model_dir values."""
    candidates = [
        cfg.doc_unwarping_model_dir,
        cfg.textline_orientation_model_dir,
        cfg.doc_orientation_classify_model_dir,
        cfg.text_detection_model_dir,
        cfg.text_recognition_model_dir,
    ]
    for p in candidates:
        if p and isinstance(p, str) and os.path.isdir(p):
            return os.path.dirname(p)
    for p in candidates:
        if p and isinstance(p, str):
            return os.path.dirname(p)
    return None

