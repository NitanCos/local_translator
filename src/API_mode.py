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
import multiprocessing as mp
import time
from PyQt6 import QtWidgets, QtCore  # 用於驗證中對話框


# 日誌設定
logger = logging.getLogger("API_mode")
logger.setLevel(logging.DEBUG)

# 儲存 API 設定的檔案
CONFIG_FILE = "api_config.json"

# ===== 子行程工具（api_test）=====

def _api_test_child(config_dict: dict, out_q: mp.Queue):
    """
    子行程：用傳入設定跑 APIMode.test_api()，把結果丟回主行程。
    回傳格式：{"ok": True/False, "kind": "ok/auth/network/timeout/error", "detail": str}
    """
    try:
        # 避免把不可序列化物件傳進來，這裡重新建立 APIMode 與設定
        from API_mode import APIMode, APIConfig  # 本檔名若不同，請改成實際檔名
        api = APIMode()
        cfg = APIConfig(
            api_type=config_dict.get("api_type", "Gemini"),
            api_key=config_dict.get("api_key", ""),
            model=config_dict.get("model", ""),
            target_lang=config_dict.get("target_lang", "Traditional Chinese"),
        )
        api.config = cfg
        ok, msg = api.test_api()
        out_q.put({"ok": bool(ok), "kind": "ok" if ok else "error", "detail": msg})
    except Exception as e:
        out_q.put({"ok": False, "kind": "error", "detail": repr(e)})

class VerifyingDialog(QtWidgets.QDialog):
    """
    簡潔『驗證中』視窗：不確定進度條＋可取消。提供 set_message() 更新文字。
    """
    canceled = QtCore.pyqtSignal()

    def __init__(self, parent=None, title="驗證中 / Verifying…"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(360)
        lay = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel("正在驗證連線與金鑰… / Checking connectivity & key…")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # 不確定模式
        lay.addWidget(self.progress)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.rejected.connect(self._on_cancel)
        lay.addWidget(btns)

        # 小動畫（... 效果）
        self._dots = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        dots = "." * self._dots
        base = "正在驗證連線與金鑰 / Verifying connectivity & key"
        self.label.setText(base + dots)

    def _on_cancel(self):
        self.canceled.emit()
        self.reject()

    def set_message(self, text: str):
        self.label.setText(text)


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

    def show_test_dialog(self, parent=None, timeout_sec: int = 30):
        """
        顯示『驗證中』視窗，並在子行程執行 test_api()。
        - Cancel = 立刻 kill 子行程
        - 逾時自動結束並提示
        """
        # 1) 準備快照（可序列化）
        snapshot = {
            "api_type": self.config.api_type,
            "api_key": self.config.api_key,
            "model": self.config.model,
            "target_lang": self.config.target_lang,
        }

        # 2) 啟動子行程
        q = mp.Queue()
        proc = mp.Process(target=_api_test_child, args=(snapshot, q), daemon=True)
        proc.start()

        # 3) 顯示『驗證中』對話框
        dlg = VerifyingDialog(parent, title="驗證中 / Verifying…")

        # 4) 輪詢 Queue 與逾時
        start_ts = time.time()
        hard_timeout = int(timeout_sec)

        def finish(res: dict):
            # 關閉輪詢與對話框
            timer.stop()
            if dlg.isVisible():
                dlg.close()
            # 終止子行程（若仍在）
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

            ok = bool(res.get("ok"))
            kind = res.get("kind") or "error"
            detail = res.get("detail") or ""

            # 視窗提示
            if ok:
                QtWidgets.QMessageBox.information(parent or dlg, "API 測試成功 / Test Succeeded", detail or "OK")
                logger.info("API test ok: %s", detail)
            else:
                QtWidgets.QMessageBox.warning(parent or dlg, "API 測試失敗 / Test Failed", f"{kind}: {detail}")
                logger.error("API test failed: %s - %s", kind, detail)

        def poll():
            # 讀取子行程結果
            try:
                while True:
                    res = q.get_nowait()
                    finish(res)
                    return
            except Exception:
                pass
            # 逾時
            if time.time() - start_ts > hard_timeout:
                finish({"ok": False, "kind": "timeout", "detail": f"Timeout > {hard_timeout}s"})

        def on_cancel():
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            timer.stop()
            logger.info("API verification canceled by user")

        dlg.canceled.connect(on_cancel)

        timer = QtCore.QTimer(dlg)
        timer.setInterval(120)  # 每 120ms 輪詢一次
        timer.timeout.connect(poll)
        timer.start()

        dlg.exec()
        

    # ---------------- Utils ----------------
    def _deepl_post_auto(self, params, timeout=60):
        """
        先打 Free 端點；若 403/404/405/410 等拒絕，改打 Pro 端點。
        這能自動因應使用者實際帳戶（Free/Pro）或地理限制導致的 403。
        """
        endpoints = [
            "https://api-free.deepl.com/v2/translate",
            "https://api.deepl.com/v2/translate",
        ]
        last_exc = None
        for idx, url in enumerate(endpoints):
            try:
                logger.debug(f"Trying DeepL endpoint {url}")
                resp = requests.post(url, data=params, timeout=timeout)
                # 某些狀況 403/404 = 端點不符或金鑰權限不符，改試另一個
                if resp.status_code >= 400:
                    logger.debug(f"DeepL endpoint {url} responded {resp.status_code}")
                    # 僅當第一個失敗才嘗試第二個；第二個失敗就拋出
                    if idx == 0 and resp.status_code in (401, 402, 403, 404, 405, 410):
                        continue
                    resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if idx == 0:
                    continue
                raise
        # 理論上走不到這；保底
        if last_exc:
            raise last_exc

    # ---------------- 測試 API ---------------- #
    def test_api(self):
        """測試 API 是否有效，回傳 (success, message)"""
        if self.config.api_type == "DeepL":
            return self._test_deepl()
        elif self.config.api_type == "Gemini":
            return self._test_gemini()
        else:
            return False, "未知的 API 類型 / Unknown API type"

    def _test_deepl(self):
        """測試 DeepL API（自動在 free/pro 端點間切換）"""
        url_lang = {
            "Traditional Chinese": "ZH-HANT",
            "English": "EN",
            "Japanese": "JA"
        }
        lang_code = url_lang.get(self.config.target_lang, "EN")
        params = {
            "auth_key": self.config.api_key,
            "text": "Hello, world!",
            "target_lang": lang_code
        }
        try:
            resp = self._deepl_post_auto(params)
            resp.raise_for_status()
            data = resp.json()
            if "translations" in data:
                # logger.info("DeepL test successful")
                return True, "測試成功 / Test successful"
            else:
                return False, "回應格式錯誤 / Invalid response format"
        except requests.exceptions.HTTPError as e:
            logger.error(f"DeepL test failed: {e}")
            return False, f"HTTP 錯誤 / HTTP error: {getattr(e.response,'status_code', 'unknown')}"
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
                # logger.info("Gemini test successful")
                return True, "測試成功 / Test successful"
            else:
                return False, "無回應內容 / No response content"
        except Exception as e:
            logger.error(f"Gemini test failed: {e}")
            return False, f"測試失敗 / Test failed: {str(e)}"

    # ---------------- 翻譯 ---------------- #
    def translate_text(self, text):
        """翻譯純文字，回傳翻譯結果；供應者由 config.api_type 決定"""
        if self.config.api_type == "DeepL":
            return self._translate_deepl(text)
        elif self.config.api_type == "Gemini":
            return self._translate_gemini(text)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _translate_deepl(self, text):
        """使用 DeepL 翻译文字（自動端點切換 + 多段落輸入）"""
        url_lang = {
            "Traditional Chinese": "ZH-HANT",
            "English": "EN",
            "Japanese": "JA"
        }
        lang_code = url_lang.get(self.config.target_lang, "EN")
        paragraphs = text.split('\n') if '\n' in text else [text]
        params = {
            "auth_key": self.config.api_key,
            "text": paragraphs,  # 允許列表
            "target_lang": lang_code,
            "preserve_formatting": "1"
        }
        try:
            resp = self._deepl_post_auto(params)
            resp.raise_for_status()
            data = resp.json()
            translated_paragraphs = [t["text"] for t in data.get("translations", [])]
            translated_text = '\n'.join(translated_paragraphs) if translated_paragraphs else ""
            logger.info("DeepL text translation successful, paragraphs: %d", len(translated_paragraphs))
            return translated_text or "（DeepL 無回傳內容）"
        except Exception as e:
            logger.error(f"DeepL text translation failed: {e}")
            return f"文字翻譯失敗 / Text translation failed: {str(e)}"

    def _translate_gemini(self, text):
        """使用 Gemini 翻譯文字（保留你的 Prompt）"""
        try:
            configure(api_key=self.config.api_key)
            model = GenerativeModel(self.config.model or "gemini-1.5-flash")

            prompt = f"""
            Role: Professional Text Translator

            Languages:
              - Source Text: Automatically detect (Japanese, English, or Chinese)
              - Translation: Translate to {self.config.target_lang}

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
            """

            response = model.generate_content(prompt + "\n\nInput text:\n" + text)
            translated_text = (response.text or "").strip()
            logger.info("Gemini text translation successful")
            return translated_text or "（Gemini 無回傳內容）"
        except Exception as e:
            logger.error(f"Gemini text translation failed: {e}")
            return f"文字翻譯失敗 / Text translation failed: {str(e)}"

    # ---------------- OCR ---------------- #
    def ocr_image(self, img_array):
        """僅進行 OCR 辨識文字，回傳未翻譯文字；供應者由 config.api_type 決定"""
        if self.config.api_type == "DeepL":
            return self._ocr_deepl(img_array)
        elif self.config.api_type == "Gemini":
            return self._ocr_gemini(img_array)
        else:
            raise ValueError("未知的 API 類型 / Unknown API type")

    def _ocr_deepl(self, img_array):
        """DeepL 模式下採本地 PaddleOCR"""
        self.init_ocr()
        predict_res = self.ocr_processor.ocr_predict(img_array)
        all_text_list = self.ocr_processor.json_preview_and_get_all_text(predict_res)
        text = "\n".join(all_text_list)
        if not text:
            return "無偵測到文字 / No text detected"
        formatted_text = self.format_ocr_text(text)
        logger.info("DeepL OCR (local) successful")
        return formatted_text

    def _ocr_gemini(self, img_array):
        """使用 Gemini 辨識圖片文字（未翻譯；保留你的 Prompt）"""
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
            transcribed_text = (response.text or "").strip()
            formatted_text = self.format_ocr_text(transcribed_text)
            logger.info("Gemini OCR successful")
            return formatted_text or "（Gemini OCR 無回傳內容）"
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return f"OCR 失敗 / OCR failed: {str(e)}"

    def process_image(self, img_array):
        """選配：OCR + 翻譯；如需可於主程式內自行組合此流程"""
        text = self.ocr_image(img_array)
        translated_text = self.translate_text(text)
        return translated_text

    def format_ocr_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[\n。！？])', text)
        formatted = '\n'.join(s.strip() for s in sentences if s.strip())
        return formatted

    def init_ocr(self):
        """初始化 OCR Processor（用於 DeepL）"""
        if self.ocr_processor is None:
            ocr_config = OCR_Processor_Config()
            self.ocr_processor = OCR_Processor(ocr_config)
            logger.info("OCR Processor initialized for DeepL (local mode OCR)")

    def update_config(self, api_type, api_key, model, target_lang):
        """更新並儲存設定"""
        self.config.api_type = api_type
        self.config.api_key = api_key
        self.config.model = model if api_type == "Gemini" else ""
        self.config.target_lang = target_lang
        self.config.save()
        # logger.info(f"API config updated: api_type={api_type}, target={target_lang}, model={self.config.model or '(default)'}")
        if api_type == "DeepL":
            self.ocr_processor = None  # 重設 OCR，待需要時再初始化