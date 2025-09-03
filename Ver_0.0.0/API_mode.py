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

# 日誌設定，與 main.py 一致
logger = logging.getLogger("API_mode")
logger.setLevel(logging.DEBUG)

# 儲存 API 設定的檔案
CONFIG_FILE = "api_config.json"

class APIConfig:
    def __init__(self, api_type="Gemini", api_key="", model="", target_lang="Traditional Chinese"):
        self.api_type = api_type
        self.api_key = api_key
        self.model = model if api_type == "Gemini" else ""  # 只對 Gemini 有 model
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
            "Traditional Chinese": "ZH",
            "English": "EN",
            "Japanese": "JA"  # 添加日文語言代碼
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

    def process_image(self, img_array):
        """處理圖片：OCR + 翻譯，回傳翻譯文字"""
        if self.config.api_type == "DeepL":
            return self._process_deepl(img_array)
        elif self.config.api_type == "Gemini":
            return self._process_gemini(img_array)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _process_deepl(self, img_array):
        """使用本地 OCR + DeepL 翻譯"""
        self.init_ocr()
        predict_res = self.ocr_processor.ocr_predict(img_array)
        all_text_list = self.ocr_processor.json_preview_and_get_all_text(predict_res)
        text = " ".join(all_text_list)  # 合併文字
        if not text:
            return "無偵測到文字 / No text detected"

        url = "https://api-free.deepl.com/v2/translate"
        lang_code = {
            "Traditional Chinese": "ZH",
            "English": "EN",
            "Japanese": "JA"  # 添加日文語言代碼
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
            logger.info("DeepL translation successful")
            return translated_text
        except Exception as e:
            logger.error(f"DeepL processing failed: {e}")
            return f"處理失敗 / Processing failed: {str(e)}"

    def _process_gemini(self, img_array):
        """使用 Gemini 直接處理圖片（OCR + 翻譯），提示詞參照 translate_image.py"""
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

            # 將 NumPy 陣列轉為 PIL Image 並存為 bytes (JPEG)
            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # 提示詞基於 translate_image.py 的 get_prompt()，調整為 OCR + 翻譯
            prompt = """
            Role: Professional Image Text Recognizer and Translator

            Languages:
              - Image Text: Automatically detect (Japanese or English or Chinese)
              - Translation: Translate to {target_lang}

            Instructions:
            1. Accurately transcribe the text in the image, detecting the language.
            2. Preserve the original text format and structure:
               - Maintain bullet points, numbered lists, and other formatting elements.
               - Keep line breaks and paragraph structures intact.
               - Preserve any special characters or symbols used for formatting.
            3. Refine the transcription:
               - Retain all meaningful punctuation.
               - Accurately capture any emphasis (bold, italic, underline) if discernible.
            4. Translate the transcribed text to {target_lang}.
            5. In the translation:
               - Maintain the original formatting, including lists and line breaks.
               - Preserve the tone, style, and intent of the original text.
               - Adapt idiomatic expressions and cultural nuances appropriately.
            6. Ensure both the transcription and translation accurately reflect the original image text in content and format.
            7. Output the result as the translated text only.
            """.format(target_lang=self.config.target_lang)

            # 使用 Gemini 生成內容
            contents = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}]}
            ]
            response = model.generate_content(contents)
            translated_text = response.text.strip()
            logger.info("Gemini processing successful")
            return translated_text
        except Exception as e:
            logger.error(f"Gemini processing failed: {e}")
            return f"處理失敗 / Processing failed: {str(e)}"

    def update_config(self, api_type, api_key, model, target_lang):
        """更新並儲存設定"""
        self.config.api_type = api_type
        self.config.api_key = api_key
        self.config.model = model if api_type == "Gemini" else ""
        self.config.target_lang = target_lang
        self.config.save()
        logger.info("API config updated")