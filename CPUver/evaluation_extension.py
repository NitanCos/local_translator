import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
import os
import requests
from evaluate import load  # Hugging Face evaluate library for metrics
import pandas as pd  # For loading test data

# Logger (dynamic name based on model size)
def get_logger(model_size):
    return logging.getLogger(f"translater_{model_size}")

@dataclass
class NLLBConfig:
    model_name: str = "facebook/nllb-200-distilled-600M"
    src_language: str = "jpn_Jpan"
    tgt_language: str = "eng_Latn"
    torch_dtype: torch.dtype = torch.float32
    low_cpu_mem_usage: bool = True
    auth_token: str = None

@dataclass
class TranslateConfig:
    max_new_tokens: int = 100
    min_length: int = 0
    num_beams: int = field(default_factory=lambda: 5)
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
        if "600M" in model_name:
            self.num_beams = 10
            self.early_stopping = False
            self.no_repeat_ngram_size = 0
            self.repetition_penalty = 1.2
        elif "1.3B" in model_name:
            self.num_beams = 10
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

def evaluate_models(test_data_path, models_to_eval=["facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B", "facebook/nllb-200-3.3B"]):
    # Load test data: Assume CSV with columns 'source' and 'reference'
    df = pd.read_csv(test_data_path)
    sources = df['source'].tolist()
    references = df['reference'].tolist()  # List of lists if multiple references per source

    metric = load("sacrebleu")  # Load sacreBLEU metric

    results = {}
    for model_name in models_to_eval:
        ncfg = NLLBConfig(model_name=model_name)
        gcfg = TranslateConfig()
        gcfg.adjust_for_model(model_name)
        translator = NLLBTranslator(ncfg)

        predictions = []
        for text in sources:
            pred = translator.translate(text, gcfg)
            predictions.append(pred)

        # Compute BLEU
        score = metric.compute(predictions=predictions, references=references)
        results[model_name] = score['score']
        logger = get_logger(translator._get_model_size(model_name))
        logger.info(f"BLEU score for {model_name}: {score['score']}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLLB-200 Translation Accuracy")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test CSV (columns: source, reference)")
    args = parser.parse_args()
    results = evaluate_models(args.test_data)
    print(results)