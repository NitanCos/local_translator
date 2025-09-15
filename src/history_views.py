import logging
from PyQt6 import QtWidgets
import os
import json
from datetime import datetime

logger = logging.getLogger("main")

class HistoryManager:
    """管理 OCR 和翻譯歷史記錄，包括載入、儲存、新增和顯示功能"""
    
    def __init__(self, max_history):
        """初始化歷史管理器
        
        Args:
            max_history (int): 最大歷史記錄數量
        """
        self.max_history = max_history
        self.ocr_history = []
        self.translate_history = []
    
    def load(self):
        """從 JSON 檔案載入 OCR 和翻譯歷史記錄"""
        logger.info("Loading history")
        try:
            if os.path.exists("ocr_history.json"):
                with open("ocr_history.json", "r", encoding="utf-8") as f:
                    self.ocr_history = json.load(f)[:self.max_history]
            if os.path.exists("translate_history.json"):
                with open("translate_history.json", "r", encoding="utf-8") as f:
                    self.translate_history = json.load(f)[:self.max_history]
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def save(self):
        """儲存 OCR 和翻譯歷史記錄到 JSON 檔案"""
        try:
            with open("ocr_history.json", "w", encoding="utf-8") as f:
                json.dump(self.ocr_history, f, ensure_ascii=False, indent=2)
            with open("translate_history.json", "w", encoding="utf-8") as f:
                json.dump(self.translate_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def add_ocr(self, text):
        """新增 OCR 歷史記錄並儲存
        
        Args:
            text (str): 要新增的 OCR 文字
        """
        if not text.strip():
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.ocr_history.insert(0, {"timestamp": timestamp, "text": text})
            self.ocr_history = self.ocr_history[:self.max_history]
            self.save()
            logger.debug(f"Added OCR history item: {timestamp}")
        except Exception as e:
            logger.error(f"Failed to add OCR history: {e}")

    def add_translate(self, text):
        """新增翻譯歷史記錄並儲存
        
        Args:
            text (str): 要新增的翻譯文字
        """
        if not text.strip():
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.translate_history.insert(0, {"timestamp": timestamp, "text": text})
            self.translate_history = self.translate_history[:self.max_history]
            self.save()
            logger.debug(f"Added translate history item: {timestamp}")
        except Exception as e:
            logger.error(f"Failed to add translate history: {e}")

    def show_ocr_history(self, app):
        """顯示 OCR 歷史記錄對話框
        
        Args:
            app: 主應用程式實例，提供 UI 元素和 format_ocr_text 方法
        """
        logger.debug("Showing OCR history")
        dialog = QtWidgets.QDialog(app)
        dialog.setWindowTitle("OCR 歷史記錄 / OCR History")
        layout = QtWidgets.QVBoxLayout()
        list_widget = QtWidgets.QListWidget()
        for item in self.ocr_history:
            try:
                preview = item.get("text", "")
                list_widget.addItem(f"{item.get('timestamp', '')}: {preview[:50]}...")
            except Exception:
                continue
        list_widget.itemClicked.connect(lambda item: self.recall_ocr(app, list_widget.currentRow()))
        layout.addWidget(list_widget)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        dialog.setLayout(layout)
        dialog.exec()

    def recall_ocr(self, app, index: int):
        """回調指定的 OCR 歷史記錄到 UI
        
        Args:
            app: 主應用程式實例，提供 UI 元素和 format_ocr_text 方法
            index (int): 要回調的歷史記錄索引
        """
        logger.debug(f"Recalling OCR history item {index}")
        if 0 <= index < len(self.ocr_history):
            try:
                formatted_text = app.format_ocr_text(self.ocr_history[index].get("text", ""))
                app.OCR_Detect_Text.setPlainText(formatted_text)
                if hasattr(app, "statusbar") and app.statusbar:
                    app.statusbar.showMessage("已載入 OCR 歷史記錄 / OCR history loaded", 3000)
                logger.info(f"Recalled OCR history item {index}")
            except Exception as e:
                logger.exception(f"Failed recalling OCR history: {e}")

    def show_translate_history(self, app):
        """顯示翻譯歷史記錄對話框
        
        Args:
            app: 主應用程式實例，提供 UI 元素和 format_ocr_text 方法
        """
        logger.debug("Showing translate history")
        dialog = QtWidgets.QDialog(app)
        dialog.setWindowTitle("翻譯歷史記錄 / Translation History")
        layout = QtWidgets.QVBoxLayout()
        list_widget = QtWidgets.QListWidget()
        for item in self.translate_history:
            try:
                preview = item.get("text", "")
                list_widget.addItem(f"{item.get('timestamp', '')}: {preview[:50]}...")
            except Exception:
                continue
        list_widget.itemClicked.connect(lambda item: self.recall_translate(app, list_widget.currentRow()))
        layout.addWidget(list_widget)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        dialog.setLayout(layout)
        dialog.exec()

    def recall_translate(self, app, index: int):
        """回調指定的翻譯歷史記錄到 UI
        
        Args:
            app: 主應用程式實例，提供 UI 元素和 format_ocr_text 方法
            index (int): 要回調的歷史記錄索引
        """
        logger.debug(f"Recalling translate history item {index}")
        if 0 <= index < len(self.translate_history):
            try:
                formatted_text = app.format_ocr_text(self.translate_history[index].get("text", ""))
                app.Translated_Text.setPlainText(formatted_text)
                if hasattr(app, "statusbar") and app.statusbar:
                    app.statusbar.showMessage("已載入翻譯歷史 / Translation history loaded", 3000)
                logger.info(f"Recalled translate history item {index}")
            except Exception as e:
                logger.exception(f"Failed recalling translation history: {e}")