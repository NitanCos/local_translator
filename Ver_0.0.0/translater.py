import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
import os
import requests  # For HTTPError

# 日誌記錄器 / Logger (使用動態名稱基於模型大小)
def get_logger(model_size):
    return logging.getLogger(f"translater_{model_size}")

@dataclass
class NLLBConfig:
    """
    模型與認證相關設定
    """
    model_name: str = "facebook/nllb-200-distilled-600M"
    src_language: str = "jpn_Jpan"
    tgt_language: str = "eng_Latn"
    torch_dtype: torch.dtype = torch.float32
    low_cpu_mem_usage: bool = True
    auth_token: str = None  # HuggingFace 訪問 token，如有需要

@dataclass
class TranslateConfig:
    """
    生成參數設定 (基於模型調整默認值)
    """
    max_new_tokens: int = 100
    min_length: int = 0
    num_beams: int = field(default_factory=lambda: 5)  # 默認5，可根據模型調整
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.2
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    forced_bos_token_id: int = field(init=False, default=None)

    def adjust_for_model(self, model_name: str):
        """根據模型名稱調整默認生成參數"""
        if "600M" in model_name:
            self.num_beams = 5
            self.early_stopping = False
            self.no_repeat_ngram_size = 0
            self.repetition_penalty = 1.2
        elif "1.3B" in model_name:
            self.num_beams = 5
            self.early_stopping = True
            self.no_repeat_ngram_size = 2
            self.repetition_penalty = 1.2
        elif "3.3B" in model_name:
            self.num_beams = 10
            self.early_stopping = True
            self.no_repeat_ngram_size = 0
            self.repetition_penalty = 1.0

class NLLBTranslator:
    def __init__(self, cfg: NLLBConfig):
        self.cfg = cfg
        model_size = self._get_model_size(cfg.model_name)
        logger = get_logger(model_size)
        logger.info("初始化 NLLBTranslator: %s", cfg)

        try:
            # 載入 tokenizer，並設定 src_lang
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                use_fast=True,
                src_lang=cfg.src_language,
                use_auth_token=cfg.auth_token
            )
            logger.info("Tokenizer loaded: vocab_size=%d", self.tokenizer.vocab_size)

            # 載入模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_name,
                torch_dtype=cfg.torch_dtype,
                low_cpu_mem_usage=cfg.low_cpu_mem_usage,
                use_auth_token=cfg.auth_token
            )
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            total_m = sum(p.numel() for p in self.model.parameters()) / 1e6
            logger.info("Model loaded: %.1fM params", total_m)

            # 計算 forced_bos_token_id: 使用 tokenizer.lang_code_to_id
            lang_codes = getattr(self.tokenizer, 'lang_code_to_id', None)
            if lang_codes and cfg.tgt_language in lang_codes:
                self.bos_token_id = lang_codes[cfg.tgt_language]
            else:
                # 對於 NLLB，使用 convert_tokens_to_ids(cfg.tgt_language)
                self.bos_token_id = self.tokenizer.convert_tokens_to_ids(cfg.tgt_language)
            logger.info("forced_bos_token_id set to %d", self.bos_token_id)

        except OSError as e:
            logger.error(f"OSError during model/tokenizer loading: {e}. Check model path or internet connection.")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTPError during model/tokenizer download: {e}. Check auth_token or network.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _get_model_size(self, model_name: str) -> str:
        """從模型名稱提取大小 (e.g., '600M', '1_3B', '3_3B')"""
        if "600M" in model_name:
            return "600M"
        elif "1.3B" in model_name:
            return "1_3B"
        elif "3.3B" in model_name:
            return "3_3B"
        return "unknown"

    def translate(self, text: str, gen_cfg: TranslateConfig) -> str:
        model_size = self._get_model_size(self.cfg.model_name)
        logger = get_logger(model_size)
        # 更新 forced_bos_token_id
        gen_cfg.forced_bos_token_id = self.bos_token_id
        logger.info("使用 TranslateConfig: %s", gen_cfg)

        # 直接使用文本，無前綴 (tokenizer 會處理 src_lang)
        inputs = text
        logger.debug("輸入: %s", inputs)

        # Tokenize
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        logger.debug("Encoded IDs: %s...", encoded["input_ids"][0][:5].tolist())

        # Generate
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
        logger.info("生成完成, token 長度=%d", out.shape[-1])

        # Decode with skip_special_tokens
        result = self.tokenizer.decode(out[0], skip_special_tokens=True)
        logger.debug("解碼結果: %s", result)
        return result

def main():
    parser = argparse.ArgumentParser(description="NLLB-200 翻譯 (CPU，config 化，支持多模型)")
    parser.add_argument("--model", type=str, default=NLLBConfig.model_name, help="模型名稱 (e.g., facebook/nllb-200-distilled-600M)")
    parser.add_argument("--src_lang", type=str, help="源語言代碼覆寫")
    parser.add_argument("--tgt_lang", type=str, help="目標語言代碼覆寫")
    parser.add_argument("--auth_token", type=str, help="HuggingFace 訪問 Token")
    parser.add_argument("--text", type=str, required=True, help="待翻譯文本")

    args = parser.parse_args()

    # 建立配置
    ncfg = NLLBConfig(
        model_name=args.model,
        src_language=args.src_lang or NLLBConfig.src_language,
        tgt_language=args.tgt_lang or NLLBConfig.tgt_language,
        auth_token=args.auth_token
    )
    gcfg = TranslateConfig()
    gcfg.adjust_for_model(ncfg.model_name)  # 調整默認值

    # 翻譯
    translator = NLLBTranslator(ncfg)
    output = translator.translate(args.text, gcfg)
    print(output)

if __name__ == "__main__":
    main()