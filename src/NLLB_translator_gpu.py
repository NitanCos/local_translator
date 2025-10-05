
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

    # 加速/裝置
    device: str = "auto"              # {"auto","cuda","cpu"}
    load_in_8bit: bool = False        # 需 bitsandbytes
    load_in_4bit: bool = False        # 需 bitsandbytes
    use_tf32: bool = True             # Ampere(3080) 支援 TF32
    gpu_id: int = 0                   # 指定 GPU

    LANGUAGE_MAP = {
        "English": "eng_Latn",
        "Japanese": "jpn_Jpan",
        "Simplified Chinese": "zho_Hans",
        "Traditional Chinese": "zho_Hant",
    }


@dataclass
class TranslateConfig:
    max_new_tokens: int = 512
    min_length: int = 0
    num_beams: int = field(default_factory=lambda: 5)
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    forced_bos_token_id: Optional[int] = field(init=False, default=None)

    def adjust_for_model(self, model_name: str):
        if "600M" in model_name:
            self.num_beams = min(self.num_beams, 6)
        elif "1.3B" in model_name:
            self.num_beams = min(self.num_beams, 8)
        elif "3.3B" in model_name:
            self.num_beams = min(self.num_beams, 6)  # 3.3B 請適度降低 beam 以節省 VRAM


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


def _resolve_device(cfg: NLLBConfig) -> Tuple[str, Optional[torch.dtype], Optional[torch.dtype]]:
    """
    回傳: (device_str, param_dtype, autocast_dtype)
    """
    logger = get_logger()
    if cfg.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = cfg.device

    if dev == "cuda":
        # Ampere (RTX 3080) 推薦 fp16，自動混合精度計算
        param_dtype = _normalize_dtype(cfg.torch_dtype)
        if param_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            param_dtype = torch.float16
        autocast_dtype = torch.bfloat16 if (param_dtype == torch.bfloat16) else torch.float16
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if cfg.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        logger.info(f"Using CUDA device with dtype={param_dtype}")
        return "cuda", param_dtype, autocast_dtype
    else:
        logger.info("Using CPU")
        return "cpu", _normalize_dtype(cfg.torch_dtype), None


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
    device, param_dtype, _ = _resolve_device(cfg)

    common_kwargs = dict(
        revision="main",               # 僅作記錄；本地載入不會用到
        trust_remote_code=False,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        local_files_only=True,
    )

    logger.info(f"Loading tokenizer from local_dir: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True, **common_kwargs)

    load_kwargs = dict(common_kwargs)
    load_kwargs["torch_dtype"] = param_dtype

    # 可選量化（需 bitsandbytes）
    if cfg.load_in_8bit or cfg.load_in_4bit:
        try:
            if cfg.load_in_4bit:
                load_kwargs.update(dict(load_in_4bit=True, device_map="auto"))
                logger.info("Loading model in 4-bit with device_map=auto")
            else:
                load_kwargs.update(dict(load_in_8bit=True, device_map="auto"))
                logger.info("Loading model in 8-bit with device_map=auto")
        except Exception as e:
            logger.warning(f"8/4-bit 量化載入失敗，改用標準精度: {e}")

    # 權重：優先 safetensors（若你想），否則 .bin；但完全不嘗試下載
    use_safetensors = cfg.prefer_safetensors and os.path.isfile(os.path.join(local_dir, "model.safetensors"))
    if use_safetensors:
        logger.info("Loading model (safetensors) from local_dir")
        model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, use_safetensors=True, **load_kwargs)
    else:
        logger.info("Loading model (.bin) from local_dir")
        model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, use_safetensors=False, **load_kwargs)

    if device == "cuda" and not (cfg.load_in_8bit or cfg.load_in_4bit):
        model.to(device)

    return tokenizer, model, device, param_dtype


# ========== 翻譯器 ==========
class NLLBTranslator:
    def __init__(self, cfg: NLLBConfig):
        self.cfg = cfg
        logger = get_logger()
        logger.info(
            "初始化 NLLBTranslator (本地離線)：model_name=%s, local_dir=%s, src_lang=%s, tgt_lang=%s",
            cfg.model_name, cfg.local_dir, cfg.src_language, cfg.tgt_language
        )

        self.tokenizer, self.model, self.device, self.param_dtype = _load_nllb_model_and_tokenizer(cfg)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 設定 forced_bos_token_id（目標語言）
        lang_codes = getattr(self.tokenizer, 'lang_code_to_id', None)
        if lang_codes and cfg.tgt_language in lang_codes:
            self.bos_token_id = lang_codes[cfg.tgt_language]
        else:
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(cfg.tgt_language)

        # 決定 autocast dtype
        self.autocast_dtype = torch.bfloat16 if self.param_dtype == torch.bfloat16 else torch.float16

    @torch.inference_mode()
    def translate(self, text: str, gen_cfg: TranslateConfig) -> str:
        logger = get_logger()
        gen_cfg.adjust_for_model(self.cfg.model_name)
        gen_cfg.forced_bos_token_id = self.bos_token_id

        encoded = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        # 把張量搬到裝置
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # CUDA 下啟用混合精度
        if self.device == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype)
        else:
            # CPU 無需 autocast
            class DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            autocast_ctx = DummyCtx()

        with autocast_ctx:
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
    ap = argparse.ArgumentParser(description="NLLB 本地離線翻譯（CUDA/混合精度/可選量化；不下載、不連網）")
    ap.add_argument("--local_dir", required=True, help="本地模型資料夾")
    ap.add_argument("--model_name", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--tgt_lang", default="eng_Latn")
    ap.add_argument("--text", required=True)

    # 裝置 & 精度
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="預設 auto（有 GPU 則用 CUDA）")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"], help="權重精度（CUDA 推薦 float16 或 bfloat16）")
    ap.add_argument("--gpu_id", type=int, default=0, help="要使用的 GPU ID")
    ap.add_argument("--no-tf32", action="store_true", help="關閉 TF32")

    # 量化（需 bitsandbytes）
    ap.add_argument("--load_in_8bit", action="store_true", help="以 8-bit 量化載入（需 bitsandbytes）")
    ap.add_argument("--load_in_4bit", action="store_true", help="以 4-bit 量化載入（需 bitsandbytes）")

    # 生成選項（可選）
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)

    args = ap.parse_args()

    cfg = NLLBConfig(
        model_name=args.model_name,
        src_language=None,
        tgt_language=args.tgt_lang,
        local_dir=args.local_dir,
        prefer_safetensors=False,
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_tf32=not args.no_tf32,
        gpu_id=args.gpu_id,
    )
    gcfg = TranslateConfig(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    tr = NLLBTranslator(cfg)
    print(tr.translate(args.text, gcfg))

if __name__ == "__main__":
    main()
