import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
import os
import requests

# 日誌記錄器
def get_logger():
    logger = logging.getLogger("translater")
    logger.info("Translater logger initialized")  # 測試記錄
    return logger

@dataclass
class NLLBConfig:
    """
    模型與認證相關設定
    """
    model_name: str = "facebook/nllb-200-distilled-1.3B"
    src_language: str = None  # 預設 None，允許自動檢測
    tgt_language: str = "zho_Hant"  # 預設繁體中文
    torch_dtype: torch.dtype = torch.float32
    low_cpu_mem_usage: bool = True
    auth_token: str = None
    LANGUAGE_MAP = {
        "English": "eng_Latn",
        "Japanese": "jpn_Jpan",
        "Simplified Chinese": "zho_Hans",
        "Traditional Chinese": "zho_Hant",
    }

@dataclass
class TranslateConfig:
    """
    生成參數設定 (基於模型調整默認值)
    """
    max_new_tokens: int = 100
    min_length: int = 0
    num_beams: int = field(default_factory=lambda: 15)  
    early_stopping: bool = False
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.2
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    forced_bos_token_id: int = field(init=False, default=None)

    def adjust_for_model(self, model_name: str):
        """根據模型名稱調整默認生成參數"""
        if "600M" in model_name:
            self.num_beams = min(self.num_beams, 10)  # 600M 模型限制為 10
            #self.early_stopping = False
            # self.no_repeat_ngram_size = 0
            # self.repetition_penalty = 1.2
        elif "1.3B" in model_name:
            self.num_beams = min(self.num_beams, 15)  # 1.3B 模型限制為 15
            #self.early_stopping = False
            # self.no_repeat_ngram_size = 2
            # self.repetition_penalty = 1.2
        elif "3.3B" in model_name:
            self.num_beams = min(self.num_beams, 20)  # 3.3B 模型允許最大 20
            # self.early_stopping = False
            # self.no_repeat_ngram_size = 0
            # self.repetition_penalty = 1.0

class NLLBTranslator:
    def __init__(self, cfg: NLLBConfig):
        self.cfg = cfg
        model_size = self._get_model_size(cfg.model_name)
        logger = get_logger()
        logger.info("初始化 NLLBTranslator: model=%s, src_lang=%s, tgt_lang=%s", cfg.model_name, cfg.src_language, cfg.tgt_language)

        try:
            # 載入 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                use_fast=True,
                src_lang=cfg.src_language,
                use_auth_token=cfg.auth_token
            )
            logger.info("Tokenizer loaded: vocab_size=%d, model=%s", self.tokenizer.vocab_size, cfg.model_name)

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
            logger.info("Model loaded: %.1fM params, model=%s", total_m, cfg.model_name)

            # 計算 forced_bos_token_id
            lang_codes = getattr(self.tokenizer, 'lang_code_to_id', None)
            if lang_codes and cfg.tgt_language in lang_codes:
                self.bos_token_id = lang_codes[cfg.tgt_language]
            else:
                self.bos_token_id = self.tokenizer.convert_tokens_to_ids(cfg.tgt_language)
            logger.info("forced_bos_token_id set to %d, model=%s", self.bos_token_id, cfg.model_name)

        except OSError as e:
            logger.error("OSError during model/tokenizer loading: %s, model=%s", e, cfg.model_name)
            raise
        except requests.exceptions.HTTPError as e:
            logger.error("HTTPError during model/tokenizer download: %s, model=%s", e, cfg.model_name)
            raise
        except Exception as e:
            logger.error("Unexpected error during initialization: %s, model=%s", e, cfg.model_name)
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
        logger = get_logger()
        logger.info("Starting translate with text: %s", text[:50])
        gen_cfg.forced_bos_token_id = self.bos_token_id
        logger.info("Using TranslateConfig: num_beams=%d, model=%s", gen_cfg.num_beams, self.cfg.model_name)

        inputs = text
        logger.debug("input: %s, model=%s", inputs, self.cfg.model_name)

        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        logger.debug("Encoded IDs: %s..., model=%s", encoded["input_ids"][0][:5].tolist(), self.cfg.model_name)

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
        logger.info("生成完成, token 長度=%d, model=%s", out.shape[-1], self.cfg.model_name)

        result = self.tokenizer.decode(out[0], skip_special_tokens=True)
        logger.debug("解碼結果: %s, model=%s", result, self.cfg.model_name)
        #logger.info("Translate completed with result: %s", result[:50])
        return result

def main():
    parser = argparse.ArgumentParser(description="NLLB-200 翻譯 (CPU，config 化，支持多模型)")
    parser.add_argument("--model", type=str, default=NLLBConfig.model_name, help="模型名稱 (e.g., facebook/nllb-200-distilled-600M)")
    parser.add_argument("--src_lang", type=str, help="源語言代碼覆寫")
    parser.add_argument("--tgt_lang", type=str, help="目標語言代碼覆寫")
    parser.add_argument("--auth_token", type=str, help="HuggingFace 訪問 Token")
    parser.add_argument("--text", type=str, required=True, help="待翻譯文本")

    args = parser.parse_args()

    ncfg = NLLBConfig(
        model_name=args.model,
        src_language=args.src_lang or NLLBConfig.src_language,
        tgt_language=args.tgt_lang or NLLBConfig.tgt_language,
        auth_token=args.auth_token
    )
    gcfg = TranslateConfig()
    gcfg.adjust_for_model(ncfg.model_name)

    translator = NLLBTranslator(ncfg)
    output = translator.translate(args.text, gcfg)
    print(output)

if __name__ == "__main__":
    main()