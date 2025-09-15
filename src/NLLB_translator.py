import argparse
import logging
from dataclasses import dataclass, field
import os
from typing import Optional, Tuple, Union, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========== 日誌 ==========
def get_logger():
    logger = logging.getLogger("translater")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    logger.debug("Translater logger initialized")
    return logger


# ========== 組態 ==========
@dataclass
class NLLBConfig:
    """
    僅本地載入（不下載）：請把 local_dir 指到「模型完整資料夾」（含 config.json、tokenizer.json、
    generation_config.json、pytorch_model.bin 或 model.safetensors）
    """
    model_name: str = "facebook/nllb-200-distilled-1.3B"   # 僅做記錄/顯示用
    src_language: Optional[str] = None
    tgt_language: str = "eng_Latn"

    # 必填：本地模型資料夾
    local_dir: Optional[str] = None

    # 其它
    torch_dtype: Union[torch.dtype, str] = "float32"
    low_cpu_mem_usage: bool = True
    prefer_safetensors: bool = False  # NLLB 多數倉庫無 safetensors，預設關閉
    revision: str = "main"            # 僅記錄用途

    LANGUAGE_MAP = {
        "English": "eng_Latn",
        "Japanese": "jpn_Jpan",
        "Simplified Chinese": "zho_Hans",
        "Traditional Chinese": "zho_Hant",
    }


@dataclass
class TranslateConfig:
    max_new_tokens: int = 1024
    min_length: int = 0
    num_beams: int = field(default_factory=lambda: 15)
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.2
    do_sample: bool = True
    temperature: float = 1.5
    top_k: int = 100
    top_p: float = 0.9
    forced_bos_token_id: Optional[int] = field(init=False, default=None)

    def adjust_for_model(self, model_name: str):
        if "600M" in model_name:
            self.num_beams = min(self.num_beams, 10)
        elif "1.3B" in model_name:
            self.num_beams = min(self.num_beams, 15)
        elif "3.3B" in model_name:
            self.num_beams = min(self.num_beams, 20)


# ========== 工具 ==========
REQUIRED_ANY_TOKENIZER = ["tokenizer.json", "spm.model", "sentencepiece.bpe.model"]
REQUIRED_MIN = ["config.json", "generation_config.json"]
REQUIRED_ANY_WEIGHTS = ["pytorch_model.bin", "model.safetensors"]

def _normalize_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, torch.float32)
    return torch.float32

def _require_local_files(dirpath: str):
    """
    驗證 local_dir 內是否具備必要檔案；若缺少，拋出 ValueError（清楚列出缺少的檔案）
    """
    missing: List[str] = []
    for f in REQUIRED_MIN:
        if not os.path.isfile(os.path.join(dirpath, f)):
            missing.append(f)
    if not any(os.path.isfile(os.path.join(dirpath, f)) for f in REQUIRED_ANY_TOKENIZER):
        missing.append("tokenizer.(json/spm.model/sentencepiece.bpe.model 任一)")
    if not any(os.path.isfile(os.path.join(dirpath, f)) for f in REQUIRED_ANY_WEIGHTS):
        missing.append("權重 (pytorch_model.bin 或 model.safetensors)")
    if missing:
        raise ValueError(
            "本地 NLLB 模型缺少必要檔案：\n  - " + "\n  - ".join(missing) +
            f"\n請將完整模型檔放入：{dirpath}"
        )


# ========== 載入器（只讀本地、完全離線；缺檔就丟錯） ==========
def _load_nllb_model_and_tokenizer(cfg: NLLBConfig):
    logger = get_logger()
    if not cfg.local_dir:
        raise ValueError("local_dir 未設定。請先於 GUI 選擇本地 NLLB 模型資料夾。")
    local_dir = cfg.local_dir

    # 強制離線：避免任何網路請求
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    _require_local_files(local_dir)
    torch_dtype = _normalize_dtype(cfg.torch_dtype)

    common_kwargs = dict(
        revision="main",               # 僅作記錄；本地載入不會用到
        trust_remote_code=False,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        local_files_only=True,
    )

    logger.info(f"Loading tokenizer from local_dir: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True, **common_kwargs)

    # 權重：優先 safetensors（若你想），否則 .bin；但完全不嘗試下載
    if cfg.prefer_safetensors and os.path.isfile(os.path.join(local_dir, "model.safetensors")):
        logger.info("Loading model (safetensors) from local_dir")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            local_dir, torch_dtype=torch_dtype, use_safetensors=True, **common_kwargs
        )
    else:
        logger.info("Loading model (.bin) from local_dir")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            local_dir, torch_dtype=torch_dtype, use_safetensors=False, **common_kwargs
        )
    return tokenizer, model


# ========== 翻譯器 ==========
class NLLBTranslator:
    def __init__(self, cfg: NLLBConfig):
        self.cfg = cfg
        logger = get_logger()
        logger.info(
            "初始化 NLLBTranslator (本地離線)：model_name=%s, local_dir=%s, src_lang=%s, tgt_lang=%s",
            cfg.model_name, cfg.local_dir, cfg.src_language, cfg.tgt_language
        )

        self.tokenizer, self.model = _load_nllb_model_and_tokenizer(cfg)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 設定 forced_bos_token_id（目標語言）
        lang_codes = getattr(self.tokenizer, 'lang_code_to_id', None)
        if lang_codes and cfg.tgt_language in lang_codes:
            self.bos_token_id = lang_codes[cfg.tgt_language]
        else:
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(cfg.tgt_language)

    def translate(self, text: str, gen_cfg: TranslateConfig) -> str:
        logger = get_logger()
        gen_cfg.adjust_for_model(self.cfg.model_name)
        gen_cfg.forced_bos_token_id = self.bos_token_id

        encoded = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        out = self.model.generate(
            **encoded,
            max_new_tokens=gen_cfg.max_new_tokens,
            min_length=gen_cfg.min_length,
            num_beams=gen_cfg.num_beams,
            early_stopping=gen_cfg.early_stopping,
            length_penalty=gen_cfg.length_penalty,
            no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
            repetition_penalty=gen_cfg.repetition_penalty,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p,
            forced_bos_token_id=gen_cfg.forced_bos_token_id
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ========== CLI（本地測試） ==========
def main():
    ap = argparse.ArgumentParser(description="NLLB 本地離線翻譯（不下載、不連網）")
    ap.add_argument("--local_dir", required=True, help="本地模型資料夾")
    ap.add_argument("--model_name", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--tgt_lang", default="eng_Latn")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    cfg = NLLBConfig(
        model_name=args.model_name,
        src_language=None,
        tgt_language=args.tgt_lang,
        local_dir=args.local_dir,
        prefer_safetensors=False,
        low_cpu_mem_usage=True,
        torch_dtype="float32",
    )
    gcfg = TranslateConfig()
    tr = NLLBTranslator(cfg)
    print(tr.translate(args.text, gcfg))

if __name__ == "__main__":
    main()
