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


def _norm_parent(p: str) -> str | None:
    try:
        if not p or not isinstance(p, str):
            return None
        p = os.path.normpath(p)
        # If given a directory path, return its parent as the base
        return os.path.dirname(p)
    except Exception:
        return None


def guess_ocr_base_dir_from_cfg(cfg) -> str | None:
    """Guess a robust common base directory from current *_model_dir values.

    Strategy:
    - Collect parents of all provided model dirs (existing or not).
    - Prefer an existing directory parent shared by multiple entries (common path).
    - Fallback to the first non-empty parent.
    """
    raw_dirs = [
        getattr(cfg, "doc_unwarping_model_dir", None),
        getattr(cfg, "textline_orientation_model_dir", None),
        getattr(cfg, "doc_orientation_classify_model_dir", None),
        getattr(cfg, "text_detection_model_dir", None),
        getattr(cfg, "text_recognition_model_dir", None),
    ]
    parents = [d for d in (_norm_parent(p) for p in raw_dirs) if d]
    if not parents:
        return None

    # Prefer a common existing parent
    try:
        common = os.path.commonpath(parents)
        if os.path.isdir(common):
            return common
    except Exception:
        pass

    # Next, prefer any existing parent that contains at least one provided dir
    for d in parents:
        if os.path.isdir(d):
            return d

    # Fallback: first parent even if it does not exist yet
    return parents[0]

