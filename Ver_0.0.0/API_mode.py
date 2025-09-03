import requests
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import os
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai
from ocr_processor import OCR_Processor, OCR_Processor_Config
import nltk
import re

# 日誌設定
logger = logging.getLogger("API_mode")
logger.setLevel(logging.DEBUG)

# 儲存 API 設定的檔案
CONFIG_FILE = "api_config.json"

# 下載 NLTK 數據
nltk.download('punkt', quiet=True)

class APIConfig:
    def __init__(self, api_type="Gemini", api_key="", model="", target_lang="Traditional Chinese"):
        self.api_type = api_type
        self.api_key = api_key
        self.model = model if api_type == "Gemini" else ""
        self.target_lang = target_lang

    def save(self):
        """儲存設定到 JSON 檔案"""
        data = {
            "api_type": self.api_type,
            "api_key": self.api_key,
            "model": self.model,
            "target_lang": self.target_lang
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("API config saved successfully")

    @classmethod
    def load(cls):
        """從 JSON 檔案載入設定"""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                api_type=data.get("api_type", "Gemini"),
                api_key=data.get("api_key", ""),
                model=data.get("model", ""),
                target_lang=data.get("target_lang", "Traditional Chinese")
            )
        return cls()  # 預設值

class APIMode:
    def __init__(self):
        self.config = APIConfig.load()
        self.ocr_processor = None  # 對於 DeepL 使用本地 OCR

    def init_ocr(self):
        """初始化 OCR Processor（用於 DeepL）"""
        if self.ocr_processor is None:
            ocr_config = OCR_Processor_Config()
            self.ocr_processor = OCR_Processor(ocr_config)
            logger.info("OCR Processor initialized for DeepL")

    def test_api(self):
        """測試 API 是否有效，回傳 (success, message)"""
        if self.config.api_type == "DeepL":
            return self._test_deepl()
        elif self.config.api_type == "Gemini":
            return self._test_gemini()
        else:
            return False, "未知的 API 類型 / Unknown API type"

    def _test_deepl(self):
        """測試 DeepL API"""
        url = "https://api-free.deepl.com/v2/translate"
        lang_code = {
            "Traditional Chinese": "ZH-HANT",
            "English": "EN",
            "Japanese": "JA"
        }.get(self.config.target_lang, "EN")
        params = {
            "auth_key": self.config.api_key,
            "text": "Hello, world!",
            "target_lang": lang_code
        }
        try:
            response = requests.post(url, data=params)
            response.raise_for_status()
            data = response.json()
            if "translations" in data:
                logger.info("DeepL test successful")
                return True, "測試成功 / Test successful"
            else:
                return False, "回應格式錯誤 / Invalid response format"
        except requests.exceptions.HTTPError as e:
            logger.error(f"DeepL test failed: {e}")
            return False, f"HTTP 錯誤 / HTTP error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"DeepL test failed: {e}")
            return False, f"測試失敗 / Test failed: {str(e)}"

    def _test_gemini(self):
        """測試 Gemini API"""
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")
            response = model.generate_content("Hello, world!")
            if response.text:
                logger.info("Gemini test successful")
                return True, "測試成功 / Test successful"
            else:
                return False, "無回應內容 / No response content"
        except Exception as e:
            logger.error(f"Gemini test failed: {e}")
            return False, f"測試失敗 / Test failed: {str(e)}"

    def translate_text(self, text):
        """翻譯純文字，回傳翻譯結果"""
        if self.config.api_type == "DeepL":
            return self._translate_deepl(text)
        elif self.config.api_type == "Gemini":
            return self._translate_gemini(text)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _translate_deepl(self, text):
        """使用 DeepL 翻譯文字"""
        url = "https://api-free.deepl.com/v2/translate"
        lang_code = {
            "Traditional Chinese": "ZH-HANT",
            "English": "EN",
            "Japanese": "JA"
        }.get(self.config.target_lang, "EN")
        params = {
            "auth_key": self.config.api_key,
            "text": text,
            "target_lang": lang_code
        }
        try:
            response = requests.post(url, data=params)
            response.raise_for_status()
            data = response.json()
            translated_text = data["translations"][0]["text"]
            logger.info("DeepL text translation successful")
            return translated_text
        except Exception as e:
            logger.error(f"DeepL text translation failed: {e}")
            return f"文字翻譯失敗 / Text translation failed: {str(e)}"

    def _translate_gemini(self, text):
        """使用 Gemini 翻譯文字"""
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

            # 提示詞，基於 translate_image.py 的結構，簡化為純文字翻譯
            prompt = """
            Role: Professional Text Translator

            Languages:
              - Source Text: Automatically detect (Japanese, English, or Chinese)
              - Translation: Translate to {target_lang}

            Instructions:
            1. Accurately detect the language of the input text.
            2. Preserve the original text format and structure:
               - Maintain bullet points, numbered lists, and other formatting elements.
               - Keep line breaks and paragraph structures intact.
               - Preserve any special characters or symbols used for formatting.
            3. Refine the translation:
               - Retain all meaningful punctuation.
               - Preserve the tone, style, and intent of the original text.
               - Adapt idiomatic expressions and cultural nuances appropriately.
            4. Output the result as the translated text only.
            """.format(target_lang=self.config.target_lang)

            response = model.generate_content(prompt + "\n\nInput text:\n" + text)
            translated_text = response.text.strip()
            logger.info("Gemini text translation successful")
            return translated_text
        except Exception as e:
            logger.error(f"Gemini text translation failed: {e}")
            return f"文字翻譯失敗 / Text translation failed: {str(e)}"

    def ocr_image(self, img_array):
        """僅進行 OCR 辨識文字，回傳未翻譯文字"""
        if self.config.api_type == "DeepL":
            return self._ocr_deepl(img_array)
        elif self.config.api_type == "Gemini":
            return self._ocr_gemini(img_array)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _ocr_deepl(self, img_array):
        """使用本地 OCR 辨識文字"""
        self.init_ocr()
        predict_res = self.ocr_processor.ocr_predict(img_array)
        all_text_list = self.ocr_processor.json_preview_and_get_all_text(predict_res)
        text = "\n".join(all_text_list)
        if not text:
            return "無偵測到文字 / No text detected"
        logger.info("DeepL OCR successful")
        return text

    def _ocr_gemini(self, img_array):
        """使用 Gemini 辨識圖片文字（未翻譯）"""
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

            # 將 NumPy 陣列轉為 PIL Image 並存為 bytes (JPEG)
            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # 提示詞參考 translate_image.py 的 get_prompt，但修改為純辨識未翻譯
            # 參考：https://github.com/google/generative-ai-python/issues/112 中的 OCR Prompt 示例，並調整為保留格式
            prompt = """
            Role: Professional Image Text Recognizer

            Languages:
              - Image Text: Automatically detect (Japanese or English or Chinese)

            Instructions:
            1. Accurately transcribe the text in the image, detecting the language.
            2. Preserve the original text format and structure:
               - Maintain bullet points, numbered lists, and other formatting elements.
               - Keep line breaks and paragraph structures intact.
               - Preserve any special characters or symbols used for formatting.
            3. Refine the transcription:
               - Retain all meaningful punctuation.
               - Accurately capture any emphasis (bold, italic, underline) if discernible.
            4. Do not translate the text; output only the original transcribed text.
            5. Ensure the transcription accurately reflects the original image text in content and format.
            6. Output the result as the transcribed text only, with sentences separated by newlines if appropriate.
            """

            # 使用 Gemini 生成內容
            contents = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}]}
            ]
            response = model.generate_content(contents)
            transcribed_text = response.text.strip()
            logger.info("Gemini OCR successful")
            return transcribed_text
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return f"OCR 失敗 / OCR failed: {str(e)}"

    def process_image(self, img_array):
        """舊邏輯：直接 OCR + 翻譯（如果需要，可保留或移除）"""
        # 此方法可視需求保留，但根據新邏輯，不再直接使用
        transcribed_text = self.ocr_image(img_array)
        return self.translate_text(transcribed_text)

    def update_config(self, api_type, api_key, model, target_lang):
        """更新並儲存設定"""
        self.config.api_type = api_type
        self.config.api_key = api_key
        self.config.model = model if api_type == "Gemini" else ""
        self.config.target_lang = target_lang
        self.config.save()
        logger.info("API config updated")