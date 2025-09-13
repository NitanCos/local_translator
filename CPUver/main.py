import sys
import os
import logging
import re
import timeit
from logging.handlers import RotatingFileHandler
import requests  # For HTTPError
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QThread, pyqtSignal, QRunnable, QObject, QThreadPool
from MainGUI import Ui_MainWindow
from ocr_processor import OCR_Processor, OCR_Processor_Config
from API_mode import APIMode
from region_capture import RegionSelector
from NLLB_translator import NLLBTranslator, NLLBConfig, TranslateConfig
from datetime import datetime
import json
from pathlib import Path
import torch

# 日誌設定 / Logging configuration
def setup_logging():
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    os.makedirs("Debug", exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # 原本的個別 handler
    # main_handler = RotatingFileHandler("Debug/Main.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    # main_handler.setFormatter(formatter)
    # main_handler.setLevel(logging.INFO)
    #
    # ocr_handler = RotatingFileHandler("Debug/OCR.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    # ocr_handler.setFormatter(formatter)
    # ocr_handler.setLevel(logging.DEBUG)
    #
    # translator_handler = RotatingFileHandler("Debug/Translater.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    # translator_handler.setFormatter(formatter)
    # translator_handler.setLevel(logging.DEBUG)

    # 新增：統一的 All.log handler（所有記錄整合到這裡）
    all_handler = RotatingFileHandler("Debug/All.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    all_handler.setFormatter(formatter)
    all_handler.setLevel(logging.DEBUG)  # 設定為 DEBUG 以捕捉所有級別的記錄

    # 主 logger
    main_logger = logging.getLogger("main")
    main_logger.addHandler(all_handler)  # 更改：使用 all_handler
    main_logger.addHandler(console_handler)
    main_logger.setLevel(logging.INFO)  # 新增：明確設定 logger 級別
    main_logger.propagate = False

    # OCR logger
    ocr_logger = logging.getLogger("ocr_processor")
    ocr_logger.addHandler(all_handler)  # 更改：使用 all_handler
    ocr_logger.addHandler(console_handler)
    ocr_logger.setLevel(logging.DEBUG)  # 新增：明確設定級別
    ocr_logger.propagate = False

    # Translator logger
    translator_logger = logging.getLogger("translater")
    translator_logger.addHandler(all_handler)  # 更改：使用 all_handler
    translator_logger.addHandler(console_handler)
    translator_logger.setLevel(logging.DEBUG)  # 新增：明確設定級別
    translator_logger.propagate = False

    # 新增：其他模組的 logger（如 API_mode、region_capture），確保它們也使用 all_handler
    # （因為 __name__ 會是模組名，我們可以為 root logger 添加 handler，讓子 logger 繼承，但由於 propagate=False，我們手動添加）
    api_logger = logging.getLogger("API_mode")
    api_logger.addHandler(all_handler)
    api_logger.addHandler(console_handler)
    api_logger.setLevel(logging.DEBUG)

    region_logger = logging.getLogger("region_capture")
    region_logger.addHandler(all_handler)
    region_logger.addHandler(console_handler)
    region_logger.setLevel(logging.DEBUG)



setup_logging()
logger = logging.getLogger("main")
logger.info("Logging system initialized")

class TranslationWorker(QObject):  # 改為 QObject
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, text, api_mode, is_api_mode, translate_config, translator):
        super().__init__()
        self.text = text
        self.api_mode = api_mode
        self.is_api_mode = is_api_mode
        self.translate_config = translate_config
        self.translator = translator
        self.is_canceled = False  # 新增取消旗標

    def run(self):
        try:
            paragraphs = self.preprocess_text(self.text)
            translated_paragraphs = []
            for paragraph in paragraphs:
                if self.is_canceled: return
                if self.is_api_mode:
                    translated_text = self.api_mode.translate_text(paragraph, self)
                else:
                    translated_text = self.translator.translate(paragraph, self.translate_config)
                if self.is_canceled: return
                translated_paragraphs.append(translated_text)
            if self.is_canceled: return
            translated_text = "\n\n".join(translated_paragraphs)
            self.result.emit(translated_text)
        except Exception as e:
            self.error.emit(str(e))

    def preprocess_text(self, text):
        """前处理文本以便翻译：按段落分割，保留换行 / Preprocess text for translation: split into paragraphs, preserve newlines"""
        logger.debug("Preprocessing text for translation: %s", text[:50])
        lines = text.splitlines()
        paragraphs = []
        current_para = []
        for line in lines:
            if line.strip():  # 非空行，添加到当前段落
                current_para.append(line)
            else:  # 空行，结束当前段落
                if current_para:  # 如果当前段落不为空，加入 paragraphs
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
        if current_para:  # 添加最后一个段落
            paragraphs.append('\n'.join(current_para))
        logger.debug("Preprocessed paragraphs: %d paragraphs", len(paragraphs))
        return paragraphs
    
class OCRWorker(QObject):  # 改為 QObject
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, img_array, api_mode, is_api_mode, ocr_processor):
        super().__init__()
        self.img_array = img_array
        self.api_mode = api_mode
        self.is_api_mode = is_api_mode
        self.ocr_processor = ocr_processor
        self.is_canceled = False  # 新增取消旗標

    def run(self):  # 改為 run() 方法（非 slot，但可連接）
        try:
            self.progress.emit(0)
            if self.is_canceled: return  # 立即檢查取消
            if self.is_api_mode:
                transcribed_text = self.api_mode.ocr_image(self.img_array)
            else:
                predict_res = self.ocr_processor.ocr_predict(self.img_array)
                if self.is_canceled: return  # 在預測後檢查（若預測是長任務，可在內部循環檢查）
                all_text_list = self.ocr_processor.json_preview_and_get_all_text(predict_res)
                transcribed_text = "\n".join(all_text_list)
            if self.is_canceled: return  # 最終檢查
            self.progress.emit(100)
            self.result.emit(transcribed_text)
        except Exception as e:
            self.error.emit(str(e))
            
class APIWorker(QObject):  # 改為 QObject
    progress = pyqtSignal(int)
    result = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, img_array, api_mode, task="ocr"):
        super().__init__()
        self.img_array = img_array
        self.api_mode = api_mode
        self.task = task
        self.is_canceled = False  # 新增取消旗標
        logger.debug(f"APIWorker initialized with task: {task}")

    def run(self):
        try:
            if self.is_canceled: return  # 检查取消
            if self.task == "ocr":
                text = self.api_mode.perform_ocr(self.img_array, lambda x: None, self)  # 传递空回调
            else:
                text = self.api_mode.process_image(self.img_array, lambda x: None, self)
            if self.is_canceled: return  # 再次检查取消
            if not text:
                self.error.emit("無有效結果 / No valid results")
                return
            self.result.emit(text)
        except Exception as e:
            logger.error(f"APIWorker error in task {self.task}: {str(e)}")
            self.error.emit(str(e))

class MainApplication(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Starting MainApplication initialization")
        self.setupUi(self)
        self.ocr_processor = None
        self.translator = None
        self.api_mode = APIMode()
        self.is_api_mode = False
        self.ocr_history = []
        self.translate_history = []
        self.max_history = 5
        self.init_components()
        self.setup_connections()
        self.load_history()
        logger.info("MainApplication initialization completed")

    def init_components(self):
        """初始化 OCR 和翻譯器 / Initialize OCR and translator"""
        self.init_ocr_processor()
        self.init_translator()
        self.translate_config = TranslateConfig()

    def init_ocr_processor(self):
        ocr_config = OCR_Processor_Config(device="cpu")
        self.ocr_processor = OCR_Processor(ocr_config)

    def init_translator(self):
        nllb_config = NLLBConfig()
        self.translator = NLLBTranslator(nllb_config)

    def setup_connections(self):
        """設定按鈕連接 / Set up button connections"""
        self.OCR_Detect.clicked.connect(self.execute_ocr)
        self.Translate_action.clicked.connect(self.execute_translate)
        self.OCR_Detect_Copy.clicked.connect(self.copy_ocr_text)
        self.OCR_Detect_Clear.clicked.connect(self.clear_ocr_text)
        self.Translated_Text_Copy.clicked.connect(self.copy_translated_text)
        self.Translated_Text_Clear.clicked.connect(self.clear_translated_text)
        self.actionSelect_Mode.triggered.connect(self.show_select_mode)
        self.actionOCR_Setting.triggered.connect(self.show_ocr_setting)
        self.actionTranslator_Setting.triggered.connect(self.show_translator_setting)
        self.actionShow_Detect_History.triggered.connect(self.show_ocr_history)
        self.actionShow_Translator_History.triggered.connect(self.show_translate_history)
        # self.actionMain_Log.triggered.connect(lambda: self.show_log("Debug/Main.log"))
        # self.actionOCR_Log.triggered.connect(lambda: self.show_log("Debug/OCR.log"))
        # self.actionTranslator_Log.triggered.connect(lambda: self.show_log("Debug/Translater.log")) 
        self.actionAll_Log.triggered.connect(lambda: self.show_log("Debug/All.log"))

    def load_history(self):
        """載入歷史記錄 / Load history"""
        logger.info("Loading history")
        try:
            if os.path.exists("ocr_history.json"):
                with open("ocr_history.json", "r", encoding="utf-8") as f:
                    self.ocr_history = json.load(f)[:self.max_history]
            if os.path.exists("translate_history.json"):
                with open("translate_history.json", "r", encoding="utf-8") as f:
                    self.translate_history = json.load(f)[:self.max_history]
            logger.info("History loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def save_history(self):
        """保存歷史記錄 / Save history"""
        logger.info("Saving history")
        try:
            with open("ocr_history.json", "w", encoding="utf-8") as f:
                json.dump(self.ocr_history, f, ensure_ascii=False, indent=2)
            with open("translate_history.json", "w", encoding="utf-8") as f:
                json.dump(self.translate_history, f, ensure_ascii=False, indent=2)
            logger.info("History saved successfully")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def preprocess_ocr_text(self, text_list):
        """前處理 OCR 文本，僅保留完整句子 / Preprocess OCR text to keep only complete sentences"""
        logger.info("Preprocessing OCR text")
        # 將文本列表合併為單一字符串
        full_text = " ".join(text_list)
        # 定義完整句子的正規表達式，匹配以全形或半形標點結尾的句子
        # 支持的標點：全形 (。！？) 和半形 (.!?)
        sentence_pattern = r'[^.!?。！？]+[.!?。！？]'
        sentences = re.findall(sentence_pattern, full_text)
        # 去除空白並過濾空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.info(f"Extracted {len(sentences)} complete sentences")
        return sentences

    def execute_ocr(self):
        """執行螢幕區域捕捉和 OCR / Execute screen capture and OCR"""
        logger.info("Executing OCR")
        try:
            selector = RegionSelector()
            if selector.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = f"screenshots/screenshot_{timestamp}.png"
                os.makedirs("screenshots", exist_ok=True)
                img_array = selector.capture_screenshot(save_path=save_path, save_image=self.checkBox.isChecked())
                if img_array is not None:
                    progress = QtWidgets.QProgressDialog("正在辨識文字... / Recognizing text...", "取消 / Cancel", 0, 0, self)
                    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.show()

                    if self.is_api_mode:
                        worker = APIWorker(img_array, self.api_mode, task="ocr")  # 使用 APIWorker
                    else:
                        worker = OCRWorker(img_array, self.api_mode, self.is_api_mode, self.ocr_processor)

                    thread = QThread()  # 新建 QThread
                    worker.moveToThread(thread)  # 移動 worker 到 thread
                    thread.started.connect(worker.run)  # 連接 run 方法

                    # 連接信號
                    worker.result.connect(self.on_ocr_finished)
                    worker.error.connect(self.on_ocr_error)
                    worker.result.connect(thread.quit)  # 完成後終止 thread
                    worker.error.connect(thread.quit)
                    worker.result.connect(lambda: progress.close())
                    worker.error.connect(lambda: progress.close())

                    # 取消邏輯
                    def cancel_task():
                        worker.is_canceled = True  # 設置取消旗標
                        thread.quit()  # 請求終止 thread
                        thread.wait()  # 等待 thread 結束
                        progress.close()
                        self.statusbar.showMessage("OCR 操作已取消 / OCR operation canceled", 5000)

                    progress.canceled.connect(cancel_task)

                    thread.start()  # 啟動 thread
                else:
                    logger.error("Failed to capture screenshot")
                    self.statusbar.showMessage("螢幕捕捉失敗 / Screenshot capture failed", 5000)
            else:
                logger.info("Region selection canceled")
        except Exception as e:
            logger.error(f"OCR execution failed: {e}")
            self.statusbar.showMessage(f"OCR 執行失敗 / OCR execution failed: {str(e)}", 5000)

    def format_ocr_text(self, text):
        """规则排列 OCR 文本：去除多余空格，按句子分割并添加换行"""
        import re
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        sentences = re.split(r'(?<=[\n。！？])', text)  # 按句尾标点分割
        formatted = '\n'.join(s.strip() for s in sentences if s.strip())
        return formatted
    
    def on_ocr_finished(self, transcribed_text):
        formatted_text = self.format_ocr_text(transcribed_text)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.OCR_Detect_Text.setPlainText(formatted_text)
        self.ocr_history.insert(0, {"timestamp": timestamp, "text": formatted_text, "image_path": None})
        self.ocr_history = self.ocr_history[:self.max_history]
        self.save_history()
        logger.info("OCR completed successfully")
        self.statusbar.showMessage("OCR 完成 / OCR completed", 5000)

    def on_ocr_error(self, error_msg):
        self.statusbar.showMessage(f"OCR 失敗: {error_msg} / OCR failed: {error_msg}", 5000)
        logger.error(f"OCR failed: {error_msg}")

    def execute_translate(self):
        logger.info("Executing translation")
        try:
            text = self.OCR_Detect_Text.toPlainText()
            if not text.strip():
                self.statusbar.showMessage("無文字可翻譯 / No text to translate", 5000)
                return

            progress = QtWidgets.QProgressDialog("正在翻譯... / Translating...", "取消 / Cancel", 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            worker = TranslationWorker(text, self.api_mode, self.is_api_mode, self.translate_config, self.translator)
            thread = QThread()  # 新建 QThread
            worker.moveToThread(thread)
            thread.started.connect(worker.run)

            # 連接信號
            worker.result.connect(self.on_translation_finished)
            worker.error.connect(self.on_translation_error)
            # worker.progress.connect(lambda current, total: progress.setValue(int(current / total * 100)))
            worker.result.connect(thread.quit)
            worker.error.connect(thread.quit)
            worker.result.connect(lambda: progress.close())
            worker.error.connect(lambda: progress.close())

            # 取消邏輯
            def cancel_task():
                worker.is_canceled = True
                thread.quit()
                thread.wait()
                progress.close()
                self.statusbar.showMessage("翻譯操作已取消 / Translation operation canceled", 5000)

            progress.canceled.connect(cancel_task)

            thread.start()
        except Exception as e:
            logger.error(f"Translation execution failed: {e}")
            self.statusbar.showMessage(f"翻譯執行失敗 / Translation execution failed: {str(e)}", 5000)
    
    def preprocess_text(self, text):
        """改進中日文斷句，支援全形標點（。！？）並保留格式"""
        logger.debug("Preprocessing text for translation: %s", text[:50])
        
        # 定義全形句末標點，排除逗號等非句末標點
        sentence_end_pattern = r'(?<=[\n。！？])(?![。！？])'
        
        # 先按換行分割，保留原始換行
        lines = text.splitlines()
        sentences = []
        
        for line in lines:
            # 檢查是否為項目符號行（例如 -, *, 1. 等）
            line = line.strip()
            if not line:
                continue  # 跳過空行
            is_bullet = bool(re.match(r'^[\-\*]\s+|^[\d]+\.\s+', line))
            
            # 按全形標點分割句子
            sub_sentences = re.split(sentence_end_pattern, line)
            # 清理並過濾空句子
            sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
            
            # 如果是項目符號行，保留整行作為單一句子
            if is_bullet:
                sentences.append(line)
            else:
                sentences.extend(sub_sentences)
        
        # 進一步清理：移除空句子並確保每句以標點或換行結尾
        refined_sentences = []
        for s in sentences:
            # 檢查句子是否以有效標點結尾，若無則嘗試修復
            if s and not re.search(r'[。！？\n]$', s):
                s += '。'  # 為無標點句子添加句號
            refined_sentences.append(s)
        
        logger.debug("Preprocessed sentences: %d sentences", len(refined_sentences))
        return refined_sentences

    def on_translation_finished(self, translated_text):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.Translated_Text.setPlainText(translated_text)
        self.translate_history.insert(0, {"timestamp": timestamp, "text": translated_text})
        self.translate_history = self.translate_history[:self.max_history]
        self.save_history()
        self.statusbar.showMessage("翻譯完成 / Translation completed", 5000)
        logger.info("Translation completed successfully")

    def on_translation_error(self, error_msg):
        self.statusbar.showMessage(f"翻譯失敗: {error_msg} / Translation failed: {error_msg}", 5000)
        logger.error(f"Translation failed: {error_msg}")

    def copy_ocr_text(self):
        """複製 OCR 文本 / Copy OCR text"""
        logger.info("Copying OCR text")
        text = self.OCR_Detect_Text.toPlainText()
        if text:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(text)
            self.statusbar.showMessage("OCR 文本已複製 / OCR text copied", 3000)

    def clear_ocr_text(self):
        """清除 OCR 文本 / Clear OCR text"""
        logger.info("Clearing OCR text")
        self.OCR_Detect_Text.clear()
        self.statusbar.showMessage("OCR 文本已清除 / OCR text cleared", 3000)

    def copy_translated_text(self):
        """複製翻譯文本 / Copy translated text"""
        logger.info("Copying translated text")
        text = self.Translated_Text.toPlainText()
        if text:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(text)
            self.statusbar.showMessage("翻譯文本已複製 / Translated text copied", 3000)

    def clear_translated_text(self):
        """清除翻譯文本 / Clear translated text"""
        logger.info("Clearing translated text")
        self.Translated_Text.clear()
        self.statusbar.showMessage("翻譯文本已清除 / Translated text cleared", 3000)

    def show_select_mode(self):
        """顯示選擇模式對話框：選擇 API 或本地模式 / Show select mode dialog: Choose API or Local mode"""
        logger.info("Showing select mode")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("選擇模式 / Choose Mode")
        layout = QtWidgets.QFormLayout()

        # 模式選擇
        mode_label = QtWidgets.QLabel("選擇模式 / Choose Mode:")
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItems(["Local Mode", "API Mode"])
        mode_combo.setCurrentText("API Mode" if self.is_api_mode else "Local Mode")
        layout.addRow(mode_label, mode_combo)

        # API 相關欄位
        api_label = QtWidgets.QLabel("選擇 API / Choose API:")
        api_combo = QtWidgets.QComboBox()
        api_combo.addItems(["DeepL", "Gemini"])
        api_combo.setCurrentText(self.api_mode.config.api_type)
        layout.addRow(api_label, api_combo)

        lang_label = QtWidgets.QLabel("翻譯語言 / Target Language:")
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItems(["Traditional Chinese", "English", "Japanese"])
        lang_combo.setCurrentText(self.api_mode.config.target_lang)
        layout.addRow(lang_label, lang_combo)

        key_label = QtWidgets.QLabel("API 金鑰 / API Key:")
        key_edit = QtWidgets.QLineEdit()
        key_edit.setText(self.api_mode.config.api_key)
        layout.addRow(key_label, key_edit)

        model_label = QtWidgets.QLabel("模型 / Model (僅 Gemini):")
        model_edit = QtWidgets.QLineEdit()
        model_edit.setText(self.api_mode.config.model)
        layout.addRow(model_label, model_edit)

        # 測試按鈕和結果顯示
        test_btn = QtWidgets.QPushButton("測試 / Test")
        result_label = QtWidgets.QLabel("")  # 用來顯示打勾/叉和訊息
        layout.addRow(test_btn)
        layout.addRow(result_label)

        # 動態隱藏/顯示 API 相關欄位
        def toggle_api_fields():
            is_api_mode = mode_combo.currentText() == "API Mode"
            api_label.setVisible(is_api_mode)
            api_combo.setVisible(is_api_mode)
            lang_label.setVisible(is_api_mode)
            lang_combo.setVisible(is_api_mode)
            key_label.setVisible(is_api_mode)
            key_edit.setVisible(is_api_mode)
            model_label.setVisible(is_api_mode and api_combo.currentText() == "Gemini")
            model_edit.setVisible(is_api_mode and api_combo.currentText() == "Gemini")
            test_btn.setVisible(is_api_mode)
            result_label.setVisible(is_api_mode)

        # 動態隱藏/顯示 model 輸入（僅 Gemini）
        def toggle_model_input():
            if mode_combo.currentText() == "API Mode" and api_combo.currentText() == "Gemini":
                model_label.show()
                model_edit.show()
            else:
                model_label.hide()
                model_edit.hide()

        mode_combo.currentTextChanged.connect(toggle_api_fields)
        api_combo.currentTextChanged.connect(toggle_model_input)
        toggle_api_fields()  # 初始化顯示

        def perform_test():
            api_type = api_combo.currentText()
            api_key = key_edit.text().strip()
            model = model_edit.text().strip() if api_type == "Gemini" else ""
            target_lang = lang_combo.currentText()
            self.api_mode.update_config(api_type, api_key, model, target_lang)
            success, message = self.api_mode.test_api()
            if success:
                result_label.setText("✅ " + message)
            else:
                result_label.setText("❌ " + message)

        test_btn.clicked.connect(perform_test)

        # 確認按鈕
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(lambda: self.switch_mode(
            dialog,
            mode_combo.currentText(),
            api_combo.currentText(),
            key_edit.text(),
            model_edit.text(),
            lang_combo.currentText()
        ))
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)
        dialog.exec()

    def enable_api_mode(self, dialog, api_type, api_key, model, target_lang):
        """啟用 API 模式並儲存設定"""
        self.api_mode.update_config(api_type, api_key, model, target_lang)
        self.is_api_mode = True
        self.statusbar.showMessage("已切換到 API 模式 / Switched to API mode", 5000)
        logger.info("API mode enabled")
        dialog.accept()
    
    def switch_mode(self, dialog, mode, api_type, api_key, model, target_lang):
        """切換模式並儲存設定 / Switch mode and save configuration"""
        logger.info(f"Switching to mode: {mode}")
        if mode == "API Mode":
            self.api_mode.update_config(api_type, api_key, model, target_lang)
            self.is_api_mode = True
            self.statusbar.showMessage("已切換到 API 模式 / Switched to API mode", 5000)
            logger.info("Switched to API mode")
        else:
            self.is_api_mode = False
            self.statusbar.showMessage("已切換到本地模式 / Switched to Local mode", 5000)
            logger.info("Switched to Local mode")
        dialog.accept()

    def show_ocr_setting(self):
        """顯示 OCR 設定對話框 / Show OCR settings dialog"""
        logger.info("Showing OCR settings")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("OCR 設定 / OCR Settings")
        layout = QtWidgets.QFormLayout()

        # 語言 / Language
        LANGUAGE_MAP = {
            "Traditional Chinese": "chinese_cht",
            "Simplified Chinese": "ch",
            "Japanese": "japan",
            "English": "en"
        }

        # 語言 / Language
        lang_label = QtWidgets.QLabel("語言 / Language:")
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItems(LANGUAGE_MAP.keys())
        # 設置當前選項，根據配置中的語言代碼映射回顯示名稱
        current_lang = next((key for key, value in LANGUAGE_MAP.items() if value == self.ocr_processor.config.lang))
        lang_combo.setCurrentText(current_lang)
        layout.addRow(lang_label, lang_combo)

        # 設備 / Device
        # device_label = QtWidgets.QLabel("設備 / Device:")
        # device_combo = QtWidgets.QComboBox()
        # device_combo.addItems(["cpu", "gpu"])
        # device_combo.setCurrentText(self.ocr_processor.config.device)
        # layout.addRow(device_label, device_combo)

        # CPU 線程數 / CPU Threads
        cpu_threads_label = QtWidgets.QLabel("CPU 線程數 / CPU Threads:")
        cpu_threads_spin = QtWidgets.QSpinBox()
        cpu_threads_spin.setRange(1, 32)
        cpu_threads_spin.setValue(self.ocr_processor.config.cpu_threads)
        layout.addRow(cpu_threads_label, cpu_threads_spin)

        # 高性能推理 / Enable HPI
        # enable_hpi_check = QtWidgets.QCheckBox("啟用高性能推理 / Enable HPI")
        # enable_hpi_check.setChecked(self.ocr_processor.config.enable_hpi)
        # layout.addRow(enable_hpi_check)

        # MKLDNN
        # enable_mkldnn_check = QtWidgets.QCheckBox("啟用 MKLDNN / Enable MKLDNN")
        # enable_mkldnn_check.setChecked(self.ocr_processor.config.enable_mkldnn)
        # layout.addRow(enable_mkldnn_check)

        # 文本圖像校正 / Use Doc Unwarping
        use_doc_unwarping_check = QtWidgets.QCheckBox("文本圖像校正 / Use Doc Unwarping")
        use_doc_unwarping_check.setChecked(self.ocr_processor.config.use_doc_unwarping)
        layout.addRow(use_doc_unwarping_check)

        # 文本行方向判斷 / Use Textline Orientation
        use_textline_orientation_check = QtWidgets.QCheckBox("文本行方向判斷 / Use Textline Orientation")
        use_textline_orientation_check.setChecked(self.ocr_processor.config.use_textline_orientation)
        layout.addRow(use_textline_orientation_check)

        # 文檔方向判斷 / Use Doc Orientation Classify
        use_doc_orientation_classify_check = QtWidgets.QCheckBox("文檔方向判斷 / Use Doc Orientation Classify")
        use_doc_orientation_classify_check.setChecked(self.ocr_processor.config.use_doc_orientation_classify)
        layout.addRow(use_doc_orientation_classify_check)

        # 文本檢測限制邊長 / Text Detection Limit Side Length
        text_det_limit_side_len_label = QtWidgets.QLabel("文本檢測限制邊長 / Text Detection Limit Side Length:")
        text_det_limit_side_len_spin = QtWidgets.QSpinBox()
        text_det_limit_side_len_spin.setRange(32, 960)
        text_det_limit_side_len_spin.setValue(self.ocr_processor.config.text_det_limit_side_len)
        layout.addRow(text_det_limit_side_len_label, text_det_limit_side_len_spin)

        # 文本檢測限制類型 / Text Detection Limit Type
        text_det_limit_type_label = QtWidgets.QLabel("文本檢測限制類型 / Text Detection Limit Type:")
        text_det_limit_type_combo = QtWidgets.QComboBox()
        text_det_limit_type_combo.addItems(["min", "max"])
        text_det_limit_type_combo.setCurrentText(self.ocr_processor.config.text_det_limit_type)
        layout.addRow(text_det_limit_type_label, text_det_limit_type_combo)

        # 文本檢測框閾值 / Text Detection Box Threshold
        text_det_box_thresh_label = QtWidgets.QLabel("文本檢測框閾值 / Text Detection Box Threshold:")
        text_det_box_thresh_spin = QtWidgets.QDoubleSpinBox()
        text_det_box_thresh_spin.setRange(0.0, 1.0)
        text_det_box_thresh_spin.setSingleStep(0.1)
        text_det_box_thresh_spin.setValue(self.ocr_processor.config.text_det_box_thresh)
        layout.addRow(text_det_box_thresh_label, text_det_box_thresh_spin)

        # 文本檢測像素閾值 / Text Detection Pixel Threshold
        text_det_thresh_label = QtWidgets.QLabel("文本檢測像素閾值 / Text Detection Pixel Threshold:")
        text_det_thresh_spin = QtWidgets.QDoubleSpinBox()
        text_det_thresh_spin.setRange(0.0, 1.0)
        text_det_thresh_spin.setSingleStep(0.1)
        text_det_thresh_spin.setValue(self.ocr_processor.config.text_det_thresh)
        layout.addRow(text_det_thresh_label, text_det_thresh_spin)

        # 文本檢測擴張係數 / Text Detection Unclip Ratio
        text_det_unclip_ratio_label = QtWidgets.QLabel("文本檢測擴張係數 / Text Detection Unclip Ratio:")
        text_det_unclip_ratio_spin = QtWidgets.QDoubleSpinBox()
        text_det_unclip_ratio_spin.setRange(0.0, 5.0)
        text_det_unclip_ratio_spin.setSingleStep(0.1)
        text_det_unclip_ratio_spin.setValue(self.ocr_processor.config.text_det_unclip_ratio)
        layout.addRow(text_det_unclip_ratio_label, text_det_unclip_ratio_spin)

        # 確認按鈕 / Confirm buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(lambda: self.update_ocr_config(
            LANGUAGE_MAP[lang_combo.currentText()],
            #device_combo.currentText(),
            'cpu', # 強制使用 CPU
            cpu_threads_spin.value(),
            # enable_hpi_check.isChecked(),
            # enable_mkldnn_check.isChecked(),
            use_doc_unwarping_check.isChecked(),
            use_textline_orientation_check.isChecked(),
            use_doc_orientation_classify_check.isChecked(),
            text_det_limit_side_len_spin.value(),
            text_det_limit_type_combo.currentText(),
            text_det_box_thresh_spin.value(),
            text_det_thresh_spin.value(),
            text_det_unclip_ratio_spin.value(),
            dialog
        ))
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        dialog.exec()

    def update_ocr_config(self, lang, device, cpu_threads, #enable_hpi, enable_mkldnn, 
                          use_doc_unwarping,use_textline_orientation, use_doc_orientation_classify, text_det_limit_side_len,
                         text_det_limit_type, text_det_box_thresh, text_det_thresh, text_det_unclip_ratio, dialog):
        """更新 OCR 配置 / Update OCR configuration"""
        logger.info(f"Updating OCR config: lang={lang}, device={device}, cpu_threads={cpu_threads}")
        try:
            self.ocr_processor.config = OCR_Processor_Config(
                lang=lang,
                device=device,
                cpu_threads=cpu_threads,
                # enable_hpi=enable_hpi,
                # enable_mkldnn=enable_mkldnn,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                use_doc_orientation_classify=use_doc_orientation_classify,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_box_thresh=text_det_box_thresh,
                text_det_thresh=text_det_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio
            )
            self.ocr_processor = OCR_Processor(self.ocr_processor.config)
            logger.info(f"OCR config updated successfully")
            self.statusbar.showMessage("OCR 設定已更新 / OCR Settings updated", 3000)
            dialog.accept()
        except Exception as e:
            logger.error(f"Failed to update OCR config: {e}")
            self.statusbar.showMessage(f"OCR 設定更新失敗 / Failed to update OCR settings: {str(e)}", 5000)

    def show_translator_setting(self):
        """顯示翻譯器設定對話框 / Show translator settings dialog"""
        logger.debug("Showing translator settings")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("翻譯器設定 / Translator Settings")
        layout = QtWidgets.QFormLayout()
        
        # 定義反向語言映射（從代碼到顯示名稱）
        INV_LANGUAGE_MAP = {v: k for k, v in NLLBConfig.LANGUAGE_MAP.items()}

        # 模型選擇
        model_label = QtWidgets.QLabel("模型 / Model:")
        model_combo = QtWidgets.QComboBox()
        model_combo.addItems(["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-3.3B"])
        model_combo.setCurrentText(self.translator.cfg.model_name)
        layout.addRow(model_label, model_combo)

        # 源語言選擇（從當前 cfg 載入）
        src_label = QtWidgets.QLabel("源語言 / Source Language:")
        src_combo = QtWidgets.QComboBox()
        src_combo.addItems(["Auto", "English", "Japanese", "Simplified Chinese", "Traditional Chinese"])
        if self.translator.cfg.src_language is None:
            src_combo.setCurrentText("Auto")  # 如果 None，顯示 Auto
        else:
            src_combo.setCurrentText(INV_LANGUAGE_MAP.get(self.translator.cfg.src_language, "Auto"))  # 否則載入對應顯示名稱
        layout.addRow(src_label, src_combo)

        # 目標語言選擇（從當前 cfg 載入）
        tgt_label = QtWidgets.QLabel("目標語言 / Target Language:")
        tgt_combo = QtWidgets.QComboBox()
        tgt_combo.addItems(["English", "Japanese", "Simplified Chinese", "Traditional Chinese"])
        tgt_combo.setCurrentText(INV_LANGUAGE_MAP.get(self.translator.cfg.tgt_language, "Traditional Chinese"))  # 載入對應顯示名稱
        layout.addRow(tgt_label, tgt_combo)

        # 最大生成 token 數
        max_new_tokens_label = QtWidgets.QLabel("最大生成 Token 數 / Max New Tokens:")
        max_new_tokens_spin = QtWidgets.QSpinBox()
        max_new_tokens_spin.setRange(1, 99999)
        max_new_tokens_spin.setSingleStep(100)
        max_new_tokens_spin.setValue(self.translate_config.max_new_tokens)
        layout.addRow(max_new_tokens_label, max_new_tokens_spin)

        # 最小長度
        min_length_label = QtWidgets.QLabel("最小長度 / Min Length:")
        min_length_spin = QtWidgets.QSpinBox()
        min_length_spin.setRange(0, 1024)
        min_length_spin.setValue(self.translate_config.min_length)
        layout.addRow(min_length_label, min_length_spin)

        # Beam 數
        num_beams_label = QtWidgets.QLabel("Beam 數 / Num Beams:")
        num_beams_spin = QtWidgets.QSpinBox()
        num_beams_spin.setRange(1, 20)  
        num_beams_spin.setValue(self.translate_config.num_beams)
        layout.addRow(num_beams_label, num_beams_spin)

        # 早期停止
        early_stopping_check = QtWidgets.QCheckBox("早期停止 / Early Stopping")
        early_stopping_check.setChecked(self.translate_config.early_stopping)
        layout.addRow(early_stopping_check)

        # 長度懲罰
        length_penalty_label = QtWidgets.QLabel("長度懲罰 / Length Penalty:")
        length_penalty_spin = QtWidgets.QDoubleSpinBox()
        length_penalty_spin.setRange(-2.0, 2.0)
        length_penalty_spin.setSingleStep(0.1)
        length_penalty_spin.setValue(self.translate_config.length_penalty)
        layout.addRow(length_penalty_label, length_penalty_spin)

        # 重複 N-gram 大小
        no_repeat_ngram_size_label = QtWidgets.QLabel("重複 N-gram 大小 / No Repeat Ngram Size:")
        no_repeat_ngram_size_spin = QtWidgets.QSpinBox()
        no_repeat_ngram_size_spin.setRange(0, 5)
        no_repeat_ngram_size_spin.setValue(self.translate_config.no_repeat_ngram_size)
        layout.addRow(no_repeat_ngram_size_label, no_repeat_ngram_size_spin)

        # 重複懲罰
        repetition_penalty_label = QtWidgets.QLabel("重複懲罰 / Repetition Penalty:")
        repetition_penalty_spin = QtWidgets.QDoubleSpinBox()
        repetition_penalty_spin.setRange(1.0, 10.0)
        repetition_penalty_spin.setSingleStep(0.1)
        repetition_penalty_spin.setValue(self.translate_config.repetition_penalty)
        layout.addRow(repetition_penalty_label, repetition_penalty_spin)

        # 隨機採樣
        do_sample_check = QtWidgets.QCheckBox("隨機採樣 / Do Sample")
        do_sample_check.setChecked(self.translate_config.do_sample)
        layout.addRow(do_sample_check)

        # 溫度
        temperature_label = QtWidgets.QLabel("溫度 / Temperature:")
        temperature_spin = QtWidgets.QDoubleSpinBox()
        temperature_spin.setRange(0.1, 5.0)
        temperature_spin.setSingleStep(0.1)
        temperature_spin.setValue(self.translate_config.temperature)
        layout.addRow(temperature_label, temperature_spin)

        # Top K
        top_k_label = QtWidgets.QLabel("Top K:")
        top_k_spin = QtWidgets.QSpinBox()
        top_k_spin.setRange(0, 100)
        top_k_spin.setValue(self.translate_config.top_k)
        layout.addRow(top_k_label, top_k_spin)

        # Top P
        top_p_label = QtWidgets.QLabel("Top P:")
        top_p_spin = QtWidgets.QDoubleSpinBox()
        top_p_spin.setRange(0.0, 1.0)
        top_p_spin.setSingleStep(0.1)
        top_p_spin.setValue(self.translate_config.top_p)
        layout.addRow(top_p_label, top_p_spin)

        # 確認按鈕
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(lambda: self.update_translator_config(
            model_combo.currentText(),
            NLLBConfig.LANGUAGE_MAP.get(src_combo.currentText(), None) if src_combo.currentText() != "Auto" else None,
            NLLBConfig.LANGUAGE_MAP.get(tgt_combo.currentText(), "zho_Hant"),
            max_new_tokens_spin.value(),
            min_length_spin.value(),
            num_beams_spin.value(),
            early_stopping_check.isChecked(),
            length_penalty_spin.value(),
            no_repeat_ngram_size_spin.value(),
            repetition_penalty_spin.value(),
            do_sample_check.isChecked(),
            temperature_spin.value(),
            top_k_spin.value(),
            top_p_spin.value(),
            dialog
        ))
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)
        dialog.exec()

    def update_translator_config(self, model_name, src_lang, tgt_lang, max_new_tokens, min_length, num_beams,
                            early_stopping, length_penalty, no_repeat_ngram_size, repetition_penalty,
                            do_sample, temperature, top_k, top_p, dialog):
        """更新翻譯器配置 / Update translator configuration"""
        logger.debug(f"Updating translator config: model={model_name}, src_lang={src_lang}, tgt_lang={tgt_lang}")
        try:
            self.translator.cfg = NLLBConfig(
                model_name=model_name,
                src_language=src_lang,  # 接受 None 以支援自動檢測
                tgt_language=tgt_lang,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                auth_token=None
            )
            self.translate_config = TranslateConfig(
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            self.translate_config.adjust_for_model(model_name)
            self.translator = NLLBTranslator(self.translator.cfg)
            logger.info("Translator config updated successfully")
            self.statusbar.showMessage("翻譯設定已更新 / Translator Settings updated", 3000)
            dialog.accept()
        except Exception as e:
            logger.error(f"Failed to update translator config: {e}")
            self.statusbar.showMessage(f"翻譯設定更新失敗 / Failed to update translator settings: {str(e)}", 5000)


    def show_ocr_history(self):
        """顯示 OCR 歷史記錄 / Show OCR history"""
        logger.debug("Showing OCR history")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("OCR 歷史記錄 / OCR History")
        layout = QtWidgets.QVBoxLayout()
        
        list_widget = QtWidgets.QListWidget()
        for item in self.ocr_history:
            list_widget.addItem(f"{item['timestamp']}: {item['text'][:50]}...")
        list_widget.itemClicked.connect(lambda item: self.recall_ocr_text(list_widget.currentRow()))
        layout.addWidget(list_widget)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        dialog.exec()

    def recall_ocr_text(self, index):
        """回呼 OCR 歷史文本 / Recall OCR history text"""
        logger.debug(f"Recalling OCR history item {index}")
        if 0 <= index < len(self.ocr_history):
            formatted_text = self.format_ocr_text(self.ocr_history[index]["text"])
            self.OCR_Detect_Text.setPlainText(formatted_text)
            self.statusbar.showMessage("已載入 OCR 歷史記錄 / OCR history loaded", 3000)
            logger.info(f"Recalled OCR history item {index}")

    def show_translate_history(self):
        """顯示翻譯歷史記錄 / Show translation history"""
        logger.debug("Showing translate history")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("翻譯歷史記錄 / Translation History")
        layout = QtWidgets.QVBoxLayout()
        
        list_widget = QtWidgets.QListWidget()
        for item in self.translate_history:
            list_widget.addItem(f"{item['timestamp']}: {item['text'][:50]}...")
        list_widget.itemClicked.connect(lambda item: self.recall_translated_text(list_widget.currentRow()))
        layout.addWidget(list_widget)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        dialog.setLayout(layout)
        dialog.exec()

    def recall_translated_text(self, index):
        """回呼翻譯歷史文本 / Recall translation history text"""
        logger.debug(f"Recalling translate history item {index}")
        if 0 <= index < len(self.translate_history):
            formatted_text = self.format_ocr_text(self.translate_history[index]["text"])  # 使用相同的格式化函數
            self.Translated_Text.setPlainText(formatted_text)
            self.statusbar.showMessage("已載入翻譯歷史記錄 / Translation history loaded", 3000)
            logger.info(f"Recalled translate history item {index}")

    def show_log(self, log_file):
        """顯示日誌文件內容 / Show log file content"""
        logger.debug(f"Showing log file: {log_file}")
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {log_file}, trying 'gbk'")
            try:
                with open(log_file, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to decode {log_file} with GBK: {e}")
                content = f"無法讀取日誌檔案 / Unable to read log file: {e}"
        except Exception as e:
            logger.error(f"Failed to show log {log_file}: {e}")
            content = f"無法顯示日誌 {log_file} / Failed to show log {log_file}: {e}"

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"日誌 / Log: {log_file}")
        layout = QtWidgets.QVBoxLayout()
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(content)
        layout.addWidget(text_edit)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        dialog.setLayout(layout)
        dialog.resize(600, 400)
        dialog.exec()

if __name__ == "__main__":
    logger.debug("Starting application")
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = MainApplication()
        window.show()
        logger.info("Application window shown")
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)