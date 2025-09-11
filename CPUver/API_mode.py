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
import re

# 日誌設定
logger = logging.getLogger("API_mode")
logger.setLevel(logging.DEBUG)

# 儲存 API 設定的檔案
CONFIG_FILE = "api_config.json"



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
        
    def perform_ocr(self, img_array, progress_callback, worker=None):
        """執行 OCR，支援取消檢查"""
        if worker and worker.is_canceled:
            logger.info("OCR canceled before starting")
            return None
        progress_callback(0)
        text = self.ocr_image(img_array)
        if worker and worker.is_canceled:
            logger.info("OCR canceled after processing")
            return None
        progress_callback(100)
        return text
    
    def format_ocr_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[\n。！？])', text)
        formatted = '\n'.join(s.strip() for s in sentences if s.strip())
        return formatted

    def translate_text(self, text, worker=None):
        """翻譯純文字，回傳翻譯結果"""
        if worker and worker.is_canceled:
            logger.info("Translation canceled before starting")
            return None
        if self.config.api_type == "DeepL":
            return self._translate_deepl(text, worker)
        elif self.config.api_type == "Gemini":
            return self._translate_gemini(text, worker)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _translate_deepl(self, text, worker=None):
        """使用 DeepL 翻译文字，支持多段落和格式保留"""
        if worker and worker.is_canceled:
            logger.info("DeepL translation canceled before starting")
            return None
        url = "https://api-free.deepl.com/v2/translate"
        lang_code = {
            "Traditional Chinese": "ZH-HANT",
            "English": "EN",
            "Japanese": "JA"
        }.get(self.config.target_lang, "EN")
        # 将文本按段落分割为列表，DeepL 支持多文本输入
        paragraphs = text.split('\n') if '\n' in text else [text]
        params = {
            "auth_key": self.config.api_key,
            "text": paragraphs,  # 传入文本列表
            "target_lang": lang_code,
            "preserve_formatting": "1"  # 保留格式（0 或 1）
        }
        try:
            response = requests.post(url, data=params)
            if worker and worker.is_canceled:
                logger.info("DeepL translation canceled after API call")
                return None
            response.raise_for_status()
            data = response.json()
            # 合并所有翻译结果，保留换行
            translated_paragraphs = [t["text"] for t in data["translations"]]
            translated_text = '\n'.join(translated_paragraphs)
            logger.info("DeepL text translation successful, paragraphs: %d", len(translated_paragraphs))
            logger.debug("Translated text: %s", translated_text[:50])
            return translated_text
        except Exception as e:
            logger.error(f"DeepL text translation failed: {e}")
            return f"文字翻译失败 / Text translation failed: {str(e)}"
        
    def _translate_gemini(self, text, worker=None):
        """使用 Gemini 翻譯文字"""
        if worker and worker.is_canceled:
            logger.info("Gemini translation canceled before starting")
            return None
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

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
            if worker and worker.is_canceled:
                logger.info("Gemini translation canceled after API call")
                return None
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

    def _ocr_deepl(self, img_array, worker=None):
        if worker and worker.is_canceled:
            logger.info("DeepL OCR canceled before starting")
            return None
        self.init_ocr()
        predict_res = self.ocr_processor.ocr_predict(img_array)
        if worker and worker.is_canceled:
            logger.info("DeepL OCR canceled after prediction")
            return None
        all_text_list = self.ocr_processor.json_preview_and_get_all_text(predict_res)
        text = "\n".join(all_text_list)
        if not text:
            return "無偵測到文字 / No text detected"
        formatted_text = self.format_ocr_text(text)
        logger.info("DeepL OCR successful")
        return formatted_text

    def _ocr_gemini(self, img_array, worker=None):
        """使用 Gemini 辨識圖片文字（未翻譯）"""
        if worker and worker.is_canceled:
            logger.info("Gemini OCR canceled before starting")
            return None
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

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

            contents = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}]}
            ]
            response = model.generate_content(contents)
            if worker and worker.is_canceled:
                logger.info("Gemini OCR canceled after API call")
                return None
            transcribed_text = response.text.strip()
            formatted_text = self.format_ocr_text(transcribed_text)
            logger.info("Gemini OCR successful")
            return formatted_text
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return f"OCR 失敗 / OCR failed: {str(e)}"

    def process_image(self, img_array, progress_callback, worker=None):
        """執行圖像處理（OCR + 翻譯），支援取消檢查"""
        if worker and worker.is_canceled:
            logger.info("Image processing canceled before starting")
            return None
        progress_callback(0)
        text = self.ocr_image(img_array)
        if worker and worker.is_canceled:
            logger.info("Image processing canceled after OCR")
            return None
        progress_callback(50)
        translated_text = self.translate_text(text)
        if worker and worker.is_canceled:
            logger.info("Image processing canceled after translation")
            return None
        progress_callback(100)
        return translated_text
    
    def update_config(self, api_type, api_key, model, target_lang):
        """更新並儲存設定"""
        self.config.api_type = api_type
        self.config.api_key = api_key
        self.config.model = model if api_type == "Gemini" else ""
        self.config.target_lang = target_lang
        self.config.save()
        logger.info("API config updated")