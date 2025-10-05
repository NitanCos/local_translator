import sys
import os
import logging
import re
from logging.handlers import RotatingFileHandler
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QFileDialog
from MainGUI import Ui_MainWindow
from ocr_processor import OCR_Processor, OCR_Processor_Config
from API_mode import APIMode
from region_capture import RegionSelector
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
from workers import TranslationWorker, OCRWorker
import ocr_paths as ocr_paths_mod
from history_views import HistoryManager
import log_viewer as log_viewer_mod
from NLLB_translator_gpu import NLLBTranslator, NLLBConfig, TranslateConfig
from logging_utils import setup_logging
from PIL import Image
import tempfile

try:
    import torch
except ImportError:
    torch = None

# =========================
# 日誌系統初始化
# =========================
setup_logging()
logger = logging.getLogger("main")
logger.info("Logging system initialized")

# =========================
# MainApplication
# =========================

class MainApplication(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Starting MainApplication initialization")
        self.setupUi(self)

        # 狀態成員
        self.max_history = 10
        self.ocr_processor = None
        self.ocr_cfg = None
        self.translator = None
        self.api_mode = APIMode()
        self.is_api_mode = False  # 由 UI 切換
        self.history_manager = HistoryManager(self.max_history)
        self.history_manager.load()
        self.ocr_history = self.history_manager.ocr_history
        self.translate_history = self.history_manager.translate_history
        #

        self.nllb_cfg = NLLBConfig(
            model_name="facebook/nllb-200-distilled-1.3B",
            src_language=None,
            tgt_language="zho_Hant",
            local_dir=None,               
            prefer_safetensors=False,     
            low_cpu_mem_usage=True,
            torch_dtype="float32",
            device="auto",
            load_in_8bit=False,
            load_in_4bit=False,
            use_tf32=True,
            gpu_id=0,
        )
        # 生成參數
        self.translate_config = TranslateConfig()

        # 初始化引擎
        self.init_components()
        # 建立 UI 連接
        self.setup_connections()
        logger.info("MainApplication initialization completed")

        # 保存 worker 引用，避免 GC
        self.ocr_worker = None
        self.trans_worker = None

        try:
            from PyQt6 import QtCore
            QtCore.QTimer.singleShot(0, lambda: (
                self.show_translator_setting() if (not self.is_api_mode and not self.nllb_cfg.local_dir) else None
            ))
        except Exception:
            pass

    # ---------- 初始化 ----------
    def init_components(self):
        self.init_ocr_processor()
        #self.init_translator()

    def init_ocr_processor(self):
        # Prepare OCR config only; defer heavy model init to run time
        self.ocr_cfg = OCR_Processor_Config(device="cpu")
        self.ocr_processor = None

    def init_translator(self):
        self.translator = None

    # ---------- 連接 ----------
    def setup_connections(self):
        self.OCR_Detect.clicked.connect(self.execute_ocr)
        self.Translate_action.clicked.connect(self.execute_translate)
        self.OCR_Detect_Copy.clicked.connect(self.copy_ocr_text)
        self.OCR_Detect_Clear.clicked.connect(self.clear_ocr_text)
        self.Translated_Text_Copy.clicked.connect(self.copy_translated_text)
        self.Translated_Text_Clear.clicked.connect(self.clear_translated_text)
        self.actionSelect_Mode.triggered.connect(self.show_select_mode)
        self.actionOCR_Setting.triggered.connect(self.show_ocr_setting)
        self.actionTranslator_Setting.triggered.connect(self.show_translator_setting)
        self.actionShow_Detect_History.triggered.connect(lambda: self.history_manager.show_ocr_history(self))
        self.actionShow_Translator_History.triggered.connect(lambda: self.history_manager.show_translate_history(self))
        self.actionAll_Log.triggered.connect(lambda: log_viewer_mod.show_log(self, "Debug/All.log"))

    # ---------- 工具 ----------
    def format_ocr_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[\n。！？])', text)
        formatted = '\n'.join(s.strip() for s in sentences if s.strip())
        return formatted

    # =========================
    # OCR：擷取 + 子進程 OCR
    # =========================
    def execute_ocr(self):
        logger.info("Executing OCR")
        provider = self.api_mode.config.api_type if self.is_api_mode else "Local(PaddleOCR)"
        logger.info(f"[OCR] mode={'API' if self.is_api_mode else 'Local'} provider={provider}")
        self.statusbar.showMessage(f"OCR 模式：{'API' if self.is_api_mode else '本地'} / 供應者：{provider}", 3000)

        try:
            selector = RegionSelector()
            if selector.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = f"screenshots/screenshot_{timestamp}.png"
                os.makedirs("screenshots", exist_ok=True)
                img_array = selector.capture_screenshot(save_path=save_path, save_image=self.checkBox.isChecked())
                if img_array is None:
                    logger.error("Failed to capture screenshot")
                    self.statusbar.showMessage("螢幕捕捉失敗 / Screenshot capture failed", 5000)
                    return

                progress = QtWidgets.QProgressDialog("正在辨識文字... / Recognizing text...", "取消 / Cancel", 0, 0, self)
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QtWidgets.QApplication.processEvents()  # 先讓 UI 畫好，避免白框

                self.ocr_worker = OCRWorker(img_array, self.api_mode, self.is_api_mode, self.ocr_cfg)

                def finished_callback(result_text):
                    progress.close()
                    # API Mode 的 DeepL 會回傳格式化後文字；Local Mode 需我們自行格式化
                    formatted_text = self.format_ocr_text(result_text) if not self.is_api_mode else result_text
                    self.on_ocr_finished(formatted_text)

                def error_callback(err_msg):
                    progress.close()
                    self.on_ocr_error(err_msg)

                # 延遲數十毫秒啟動，避免白色空窗
                QtCore.QTimer.singleShot(100, lambda: self.ocr_worker.start(finished_callback, error_callback))

                # 取消 → 直接 terminate 子進程
                progress.canceled.connect(lambda: (self.ocr_worker.cancel(), progress.close()))
            else:
                logger.info("Region selection canceled")
        except Exception as e:
            logger.error(f"OCR execution failed: {e}")
            self.statusbar.showMessage(f"OCR 執行失敗 / OCR execution failed: {str(e)}", 5000)

    def on_ocr_finished(self, transcribed_text):
        formatted_text = self.format_ocr_text(transcribed_text)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.OCR_Detect_Text.setPlainText(formatted_text)
        self.ocr_history.insert(0, {"timestamp": timestamp, "text": formatted_text, "image_path": None})
        self.ocr_history = self.ocr_history[:self.max_history]
        self.history_manager.save()
        logger.info("OCR completed successfully")
        self.statusbar.showMessage("OCR 完成 / OCR completed", 5000)

    def on_ocr_error(self, error_msg):
        self.statusbar.showMessage(f"OCR 失敗: {error_msg} / OCR failed: {error_msg}", 5000)
        logger.error(f"OCR failed: {error_msg}")

    # =========================
    # 翻譯：子進程翻譯
    # =========================
    def execute_translate(self):
        logger.info("Executing translation")
        provider = self.api_mode.config.api_type if self.is_api_mode else "Local(NLLB)"
        logger.info(f"[Translate] mode={'API' if self.is_api_mode else 'Local'} provider={provider}")
        self.statusbar.showMessage(f"翻譯模式：{'API' if self.is_api_mode else '本地'} / 供應者：{provider}", 3000)

        if not self.is_api_mode:

            if not self.nllb_cfg.local_dir:
                QtWidgets.QMessageBox.information(self, "本地模型未設定", "請選擇本地 NLLB 模型資料夾後再執行。")
                self.show_translator_setting()
                if not self.nllb_cfg.local_dir:
                    return  # 使用者取消

        try:
            text = self.OCR_Detect_Text.toPlainText()
            if not text.strip():
                self.statusbar.showMessage("無文字可翻譯 / No text to translate", 5000)
                return

            progress = QtWidgets.QProgressDialog("正在翻譯... / Translating...", "取消 / Cancel", 0, 0, self)
            progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QtWidgets.QApplication.processEvents()  # 先讓 UI 畫好，避免白框

            self.trans_worker = TranslationWorker(
            text,
            self.api_mode,
            self.is_api_mode,
            self.translate_config,
            self.nllb_cfg,     # ★ 傳設定，不傳 translator 物件
            )

            def finished_callback(translated_text):
                progress.close()
                self.on_translation_finished(translated_text)

            def error_callback(err_msg):
                progress.close()
                self.on_translation_error(err_msg)

            QtCore.QTimer.singleShot(50, lambda: self.trans_worker.start(finished_callback, error_callback))
            progress.canceled.connect(lambda: (self.trans_worker.cancel(), progress.close()))
        except Exception as e:
            logger.error(f"Translation execution failed: {e}")
            self.statusbar.showMessage(f"翻譯執行失敗 / Translation execution failed: {str(e)}", 5000)

    def on_translation_finished(self, translated_text):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.Translated_Text.setPlainText(translated_text)
        self.translate_history.insert(0, {"timestamp": timestamp, "text": translated_text})
        self.translate_history = self.translate_history[:self.max_history]
        self.history_manager.save()
        self.statusbar.showMessage("翻譯完成 / Translation completed", 5000)
        logger.info("Translation completed successfully")

    def on_translation_error(self, error_msg):
        self.statusbar.showMessage(f"翻譯失敗: {error_msg} / Translation failed: {error_msg}", 5000)
        logger.error(f"Translation failed: {error_msg}")

    # =========================
    # 文字操作
    # =========================
    def copy_ocr_text(self):
        text = self.OCR_Detect_Text.toPlainText()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.statusbar.showMessage("OCR 文本已複製 / OCR text copied", 3000)

    def clear_ocr_text(self):
        self.OCR_Detect_Text.clear()
        self.statusbar.showMessage("OCR 文本已清除 / OCR text cleared", 3000)

    def copy_translated_text(self):
        text = self.Translated_Text.toPlainText()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.statusbar.showMessage("翻譯文本已複製 / Translated text copied", 3000)

    def clear_translated_text(self):
        self.Translated_Text.clear()
        self.statusbar.showMessage("翻譯文本已清除 / Translated text cleared", 3000)

    # =========================
    # API更新設定
    # =========================

    def switch_mode(self, dialog, mode, api_type, api_key, model, target_lang):
        """
        切換 Local / API 模式並更新設定。
        - API Mode：更新 self.api_mode.config，顯示狀態訊息
        - Local Mode：關閉 API 模式旗標
        """
        import logging
        logger = logging.getLogger("main")

        try:
            if mode == "API Mode":
                # 更新 API 設定（會自動保存至檔案）
                self.api_mode.update_config(
                    api_type=api_type,
                    api_key=api_key.strip(),
                    model=(model.strip() if api_type == "Gemini" else ""),
                    target_lang=target_lang,
                )
                self.is_api_mode = True
                # 若需要，這裡可關閉本地翻譯或做資源釋放
                self.translator = None  # 進入 API 模式時不保留本地模型
                if hasattr(self, "statusbar") and self.statusbar:
                    self.statusbar.showMessage("已切換到 API 模式 / Switched to API mode", 5000)
                logger.info("Switched to API mode: api=%s, target=%s, model=%s",api_type, target_lang, self.api_mode.config.model or "(default)")
            else:
                self.is_api_mode = False
                # 回到本地模式時，如需重新初始化本地翻譯/OCR，可在這裡處理
                self.init_translator()  # 即時重建本地 NLLB
                if hasattr(self, "statusbar") and self.statusbar:
                    self.statusbar.showMessage("已切換到本地模式 / Switched to Local mode", 5000)
                logger.info("Switched to Local mode")

            # 關閉選擇視窗
            if dialog is not None:
                dialog.accept()

        except Exception as e:
            logger.exception("Switch mode failed")
            # 保底提示
            try:
                from PyQt6 import QtWidgets
                QtWidgets.QMessageBox.critical(self, "切換失敗 / Switch Failed", str(e))
            except Exception:
                pass


    # =========================
    # 模式/設定對話框（沿用原本）
    # =========================
    def show_select_mode(self):
        from PyQt6 import QtWidgets
        logger.info("Showing select mode")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("選擇模式 / Choose Mode")
        layout = QtWidgets.QFormLayout()

        # --- 模式與 API 選擇 ---
        mode_label = QtWidgets.QLabel("選擇模式 / Choose Mode:")
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItems(["Local Mode", "API Mode"])
        mode_combo.setCurrentText("API Mode" if self.is_api_mode else "Local Mode")
        layout.addRow(mode_label, mode_combo)

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
        key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        layout.addRow(key_label, key_edit)

        model_label = QtWidgets.QLabel("模型 / Model (僅 Gemini):")
        model_edit = QtWidgets.QLineEdit()
        model_edit.setText(self.api_mode.config.model)
        layout.addRow(model_label, model_edit)

        # --- 測試按鈕：改成呼叫 API_mode 內建「驗證中」對話框 ---
        test_btn = QtWidgets.QPushButton("測試 / Test")
        result_label = QtWidgets.QLabel("")  # 保留占位；實際結果用訊息框顯示
        layout.addRow(test_btn)
        layout.addRow(result_label)

        # --- 顯示/隱藏欄位邏輯 ---
        def toggle_api_fields():
            is_api = mode_combo.currentText() == "API Mode"
            api_label.setVisible(is_api)
            api_combo.setVisible(is_api)
            lang_label.setVisible(is_api)
            lang_combo.setVisible(is_api)
            key_label.setVisible(is_api)
            key_edit.setVisible(is_api)
            model_label.setVisible(is_api and api_combo.currentText() == "Gemini")
            model_edit.setVisible(is_api and api_combo.currentText() == "Gemini")
            test_btn.setVisible(is_api)
            result_label.setVisible(is_api)

        def toggle_model_input():
            model_label.setVisible(mode_combo.currentText() == "API Mode" and api_combo.currentText() == "Gemini")
            model_edit.setVisible(mode_combo.currentText() == "API Mode" and api_combo.currentText() == "Gemini")

        mode_combo.currentTextChanged.connect(toggle_api_fields)
        api_combo.currentTextChanged.connect(toggle_model_input)
        toggle_api_fields()

        # --- 測試：更新 config → 呼叫 API_mode.show_test_dialog(self) ---
        def perform_test():
            api_type = api_combo.currentText()
            api_key = key_edit.text().strip()
            model = model_edit.text().strip() if api_type == "Gemini" else ""
            target_lang = lang_combo.currentText()
            # 先更新配置（API_mode 內部會存檔）
            self.api_mode.update_config(api_type, api_key, model, target_lang)
            # 開啟「驗證中」視窗並在子行程進行測試（可取消/逾時）
            try:
                self.api_mode.show_test_dialog(self, timeout_sec=30)
            except Exception as e:
                # 萬一 GUI/多程序有例外，回退成同步測試
                ok, msg = self.api_mode.test_api()
                QtWidgets.QMessageBox.warning(self, "API 測試結果 / Test Result", (("✅ " if ok else "❌ ") + msg))

        test_btn.clicked.connect(perform_test)

        # --- OK/Cancel ---
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        def on_accept():
            self.switch_mode(
                dialog,
                mode_combo.currentText(),
                api_combo.currentText(),
                key_edit.text(),
                model_edit.text(),
                lang_combo.currentText()
            )
        btn_box.accepted.connect(on_accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)
        dialog.exec()
    
    def show_log(self, log_file):
        return log_viewer_mod.show_log(self, log_file)
    # ========= OCR 設定視窗（含即時檢查、紅框紅字、自動補齊）=========
    def show_ocr_setting(self):
        from PyQt6 import QtWidgets
        from PyQt6.QtWidgets import QFileDialog

        cfg = self.ocr_cfg

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("OCR 設定 / OCR Settings")
        form = QtWidgets.QFormLayout(dlg)

        # ===== 基底模型路徑 =====
        base_label = QtWidgets.QLabel("基底模型路徑 / Base Model Directory")
        base_edit  = QtWidgets.QLineEdit(ocr_paths_mod.guess_ocr_base_dir_from_cfg(cfg) or r"D:\models\paddleocr")
        browse_btn = QtWidgets.QPushButton("瀏覽… / Browse…")
        h_base = QtWidgets.QHBoxLayout(); h_base.addWidget(base_edit); h_base.addWidget(browse_btn)
        form.addRow(base_label, h_base)

        def on_browse():
            d = QFileDialog.getExistingDirectory(self, "選擇 PaddleOCR 模型基底 / Select base model dir")
            if d:
                base_edit.setText(d)
                autofill_paths()
        browse_btn.clicked.connect(on_browse)

        # ===== 功能開關 =====
        chk_unwarp = QtWidgets.QCheckBox("啟用文檔校正 / Enable Doc Unwarping")
        chk_unwarp.setChecked(cfg.use_doc_unwarping)
        form.addRow(chk_unwarp)

        chk_textline = QtWidgets.QCheckBox("啟用文本行方向 / Enable Textline Orientation")
        chk_textline.setChecked(cfg.use_textline_orientation)
        form.addRow(chk_textline)

        chk_docori = QtWidgets.QCheckBox("啟用文檔方向分類 / Enable Doc Orientation Classifier")
        chk_docori.setChecked(cfg.use_doc_orientation_classify)
        form.addRow(chk_docori)

        # ===== 模型名稱 =====
        unwarp_name   = QtWidgets.QLineEdit(cfg.doc_unwarping_model_name)
        textline_name = QtWidgets.QLineEdit(cfg.textline_orientation_model_name)
        docori_name   = QtWidgets.QLineEdit(cfg.doc_orientation_classify_model_name)
        det_name      = QtWidgets.QLineEdit(cfg.text_detection_model_name)
        rec_name      = QtWidgets.QLineEdit(cfg.text_recognition_model_name)

        form.addRow("校正模型名 / Unwarping Name", unwarp_name)
        form.addRow("文本行方向 名稱 / Textline Ori Name", textline_name)
        form.addRow("文檔方向 名稱 / Doc Ori Name", docori_name)
        form.addRow("檢測模型名 / Detection Name", det_name)
        form.addRow("識別模型名 / Recognition Name", rec_name)

        # ===== 模型路徑（可自動帶入，也可手動改）=====
        unwarp_dir   = QtWidgets.QLineEdit(cfg.doc_unwarping_model_dir)
        textline_dir = QtWidgets.QLineEdit(cfg.textline_orientation_model_dir)
        docori_dir   = QtWidgets.QLineEdit(cfg.doc_orientation_classify_model_dir)
        det_dir      = QtWidgets.QLineEdit(cfg.text_detection_model_dir)
        rec_dir      = QtWidgets.QLineEdit(cfg.text_recognition_model_dir)

        form.addRow("校正模型路徑 / Unwarping Dir", unwarp_dir)
        form.addRow("文本行方向 路徑 / Textline Ori Dir", textline_dir)
        form.addRow("文檔方向 路徑 / Doc Ori Dir", docori_dir)
        form.addRow("檢測模型路徑 / Detection Dir", det_dir)
        form.addRow("識別模型路徑 / Recognition Dir", rec_dir)

        auto_btn = QtWidgets.QPushButton("依基底自動推導 / Autofill from Base")
        form.addRow(auto_btn)

        def autofill_paths():
            base = base_edit.text().strip()
            if not base:
                return
            paths = ocr_paths_mod.build_ocr_dirs_from_base(base, cfg)
            # Always fill based on base directory selection
            unwarp_dir.setText(paths["doc_unwarping_model_dir"])
            textline_dir.setText(paths["textline_orientation_model_dir"])
            docori_dir.setText(paths["doc_orientation_classify_model_dir"])
            det_dir.setText(paths["text_detection_model_dir"])
            rec_dir.setText(paths["text_recognition_model_dir"])

        auto_btn.clicked.connect(autofill_paths)

        # ===== 原本執行參數 =====
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItems(["en", "ch", "chinese_cht", "japan"])
        if lang_combo.findText(cfg.lang) >= 0:
            lang_combo.setCurrentText(cfg.lang)
        form.addRow("語言 / Language", lang_combo)

        device_combo = QtWidgets.QComboBox()
        device_combo.addItems(["cpu", "gpu"])
        device_combo.setCurrentText(cfg.device)
        form.addRow("設備 / Device", device_combo)

        cpu_spin = QtWidgets.QSpinBox(); cpu_spin.setRange(1, 256); cpu_spin.setValue(cfg.cpu_threads)
        form.addRow("CPU 線程數 / CPU Threads", cpu_spin)

        hpi_chk = QtWidgets.QCheckBox("啟用 HPI（僅 Linux） / Enable HPI (Linux only)")
        hpi_chk.setChecked(cfg.enable_hpi)
        form.addRow(hpi_chk)

        mkldnn_chk = QtWidgets.QCheckBox("啟用 MKLDNN（Intel CPU） / Enable MKLDNN (Intel)")
        mkldnn_chk.setChecked(cfg.enable_mkldnn)
        form.addRow(mkldnn_chk)

        det_side_len = QtWidgets.QSpinBox(); det_side_len.setRange(1, 4096); det_side_len.setValue(cfg.text_det_limit_side_len)
        form.addRow("檢測限制邊長 / Det Limit Side Len", det_side_len)

        det_limit_type = QtWidgets.QComboBox()
        det_limit_type.addItems(["min", "max"])
        det_limit_type.setCurrentText(cfg.text_det_limit_type)
        form.addRow("檢測限制類型 / Det Limit Type", det_limit_type)

        det_box_thresh = QtWidgets.QDoubleSpinBox(); det_box_thresh.setRange(0.0, 1.0); det_box_thresh.setDecimals(3); det_box_thresh.setSingleStep(0.01)
        det_box_thresh.setValue(cfg.text_det_box_thresh)
        form.addRow("檢測框閾值 / Box Thresh", det_box_thresh)

        det_thresh = QtWidgets.QDoubleSpinBox(); det_thresh.setRange(0.0, 1.0); det_thresh.setDecimals(3); det_thresh.setSingleStep(0.01)
        det_thresh.setValue(cfg.text_det_thresh)
        form.addRow("像素閾值 / Pixel Thresh", det_thresh)

        unclip_ratio = QtWidgets.QDoubleSpinBox(); unclip_ratio.setRange(0.1, 10.0); unclip_ratio.setDecimals(2); unclip_ratio.setSingleStep(0.1)
        unclip_ratio.setValue(cfg.text_det_unclip_ratio)
        form.addRow("擴張係數 / Unclip Ratio", unclip_ratio)

        # ===== 先建立按鈕 =====
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(btns)

        # ===== 驗證工具 & 即時驗證 =====
        def _set_path_ok(widget: QtWidgets.QLineEdit, ok: bool, reason: str = ""):
            widget.setStyleSheet("QLineEdit { border: 1px solid %s; padding: 2px; }" %
                                ("#28a745" if ok else "#dc3545"))
            widget.setToolTip("" if ok else reason)

        def _dir_exists(p: str) -> bool:
            return bool(p and Path(p).is_dir())

        def _find_model_in_dir(root: str) -> tuple[bool, str]:
            """
            合格條件（任一即通過）：
            A) 任意子層有 *.onnx
            B) 任意子層有 *.nb
            C) 任意子層、同一目錄、同一 stem 的三件套：<name>.json + <name>.yml + <name>.pdiparams
                （例如 inference.json / inference.yml / inference.pdiparams）
            D) 任意子層、同一目錄、同一 stem 的 Paddle 原生配對：<name>.pdmodel + <name>.pdiparams
            回傳：(ok, reason)
            """
            from pathlib import Path
            from collections import defaultdict

            if not root:
                return False, "未設定路徑 / path empty"
            p = Path(root)
            if not p.is_dir():
                return False, "目錄不存在 / directory not found"

            # A) onnx
            if any(p.rglob("*.onnx")):
                return True, ""

            # B) nb
            if any(p.rglob("*.nb")):
                return True, ""

            # 收集三種副檔名與 paddle 原生副檔的 stem 分佈
            dir2json = defaultdict(set)
            dir2yml  = defaultdict(set)
            dir2pdip = defaultdict(set)
            dir2pdm  = defaultdict(set)

            for fp in p.rglob("*.json"):
                dir2json[fp.parent].add(fp.stem)
            for fp in p.rglob("*.yml"):
                dir2yml[fp.parent].add(fp.stem)
            for fp in p.rglob("*.pdiparams"):
                dir2pdip[fp.parent].add(fp.stem)
            for fp in p.rglob("*.pdmodel"):
                dir2pdm[fp.parent].add(fp.stem)

            # C) 三件套：<name>.json + <name>.yml + <name>.pdiparams（同目錄同名）
            for d in set(list(dir2json.keys()) + list(dir2yml.keys()) + list(dir2pdip.keys())):
                if dir2json[d] & dir2yml[d] & dir2pdip[d]:
                    return True, ""

            # D) Paddle 原生：<name>.pdmodel + <name>.pdiparams（同目錄同名）
            for d in set(list(dir2pdm.keys()) + list(dir2pdip.keys())):
                if dir2pdm[d] & dir2pdip[d]:
                    return True, ""

            # 進一步回傳更精準的錯誤原因（方便排查）
            # 只找到部分檔案時的提示
            has_json = any(dir2json.values())
            has_yml  = any(dir2yml.values())
            has_pdip = any(dir2pdip.values())
            has_pdm  = any(dir2pdm.values())

            if has_json or has_yml or has_pdip or has_pdm:
                # 檢查三件套缺哪個
                if (has_json and has_yml and not has_pdip):
                    return False, "找到 .json + .yml，但缺 .pdiparams / missing .pdiparams"
                if (has_json and has_pdip and not has_yml):
                    return False, "找到 .json + .pdiparams，但缺 .yml / missing .yml"
                if (has_yml and has_pdip and not has_json):
                    return False, "找到 .yml + .pdiparams，但缺 .json / missing .json"
                # 檢查 paddle 原生缺哪個
                if (has_pdm and not has_pdip):
                    return False, "找到 .pdmodel，但缺 .pdiparams / missing .pdiparams"
                if (has_pdip and not has_pdm):
                    return False, "找到 .pdiparams，但缺 .pdmodel / missing .pdmodel"

            return False, "找不到 *.onnx / *.nb，或 (.json+.yml+.pdiparams) / (.pdmodel+.pdiparams) 配對"




        def _mk_status_label():
            lab = QtWidgets.QLabel("")
            lab.setStyleSheet("QLabel { color: #6c757d; font-size: 11px; }")
            return lab

        lab_base   = _mk_status_label()
        lab_unwarp = _mk_status_label()
        lab_tline  = _mk_status_label()
        lab_docori = _mk_status_label()
        lab_det    = _mk_status_label()
        lab_rec    = _mk_status_label()
        form.addRow("", lab_base)
        form.addRow("", lab_unwarp)
        form.addRow("", lab_tline)
        form.addRow("", lab_docori)
        form.addRow("", lab_det)
        form.addRow("", lab_rec)

        summary = QtWidgets.QLabel("")
        summary.setWordWrap(True)
        summary.setStyleSheet("QLabel { color:#6c757d; }")
        form.addRow("狀態 / Status", summary)

        def validate_all():
            missing = []

            base_ok = _dir_exists(base_edit.text().strip())
            _set_path_ok(base_edit, base_ok, "目錄不存在 / directory not found")
            lab_base.setText(f"基底資料夾 / Base dir：{'OK' if base_ok else '缺檔'}")
            if not base_ok: missing.append("Base")

            if chk_unwarp.isChecked():
                ok, reason = _find_model_in_dir(unwarp_dir.text().strip())
                _set_path_ok(unwarp_dir, ok, reason)
                lab_unwarp.setText(f"校正模型 / Unwarping：{'OK' if ok else '缺檔'}")
                if not ok: missing.append("Unwarping")
            else:
                _set_path_ok(unwarp_dir, True, ""); lab_unwarp.setText("校正模型 / Unwarping：未啟用 / disabled")

            if chk_textline.isChecked():
                ok, reason = _find_model_in_dir(textline_dir.text().strip())
                _set_path_ok(textline_dir, ok, reason)
                lab_tline.setText(f"文本行方向 / Textline Ori：{'OK' if ok else '缺檔'}")
                if not ok: missing.append("TextlineOri")
            else:
                _set_path_ok(textline_dir, True, ""); lab_tline.setText("文本行方向 / Textline Ori：未啟用 / disabled")

            if chk_docori.isChecked():
                ok, reason = _find_model_in_dir(docori_dir.text().strip())
                _set_path_ok(docori_dir, ok, reason)
                lab_docori.setText(f"文檔方向分類 / Doc Ori：{'OK' if ok else '缺檔'}")
                if not ok: missing.append("DocOri")
            else:
                _set_path_ok(docori_dir, True, ""); lab_docori.setText("文檔方向分類 / Doc Ori：未啟用 / disabled")

            ok, reason = _find_model_in_dir(det_dir.text().strip())
            _set_path_ok(det_dir, ok, reason)
            lab_det.setText(f"檢測模型 / Detection：{'OK' if ok else '缺檔'}")
            if not ok: missing.append("Detection")

            ok, reason = _find_model_in_dir(rec_dir.text().strip())
            _set_path_ok(rec_dir, ok, reason)
            lab_rec.setText(f"識別模型 / Recognition：{'OK' if ok else '缺檔'}")
            if not ok: missing.append("Recognition")

            if missing:
                summary.setStyleSheet("QLabel { color:#dc3545; }")
                summary.setText("缺少項目 / Missing: " + ", ".join(missing))
            else:
                summary.setStyleSheet("QLabel { color:#28a745; }")
                summary.setText("所有路徑就緒 / All paths look good")

            btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(len(missing) == 0)

        # 綁定事件：基底變更→自動補齊＋驗證；子欄位/勾選變更→驗證
        def _on_base_changed():
            autofill_paths()
            validate_all()
        base_edit.textChanged.connect(_on_base_changed)

        for w in [unwarp_dir, textline_dir, docori_dir, det_dir, rec_dir]:
            w.textChanged.connect(validate_all)
        for w in [chk_unwarp, chk_textline, chk_docori]:
            w.toggled.connect(validate_all)

        # 初次自動帶入與驗證
        if not all([unwarp_dir.text(), textline_dir.text(), docori_dir.text(), det_dir.text(), rec_dir.text()]):
            autofill_paths()
        validate_all()

        # ===== OK / Cancel =====
        def _hard_validate_or_msgbox() -> bool:
            errs = []
            def need_ok(path: str, name: str, enabled: bool = True):
                if not enabled: return
                ok, reason = _find_model_in_dir(path)
                if not ok: errs.append(f"{name} 缺檔 / missing: {path}（{reason}）")

            if not _dir_exists(base_edit.text().strip()):
                errs.append("Base dir 不存在 / Base directory not found")
            need_ok(unwarp_dir.text().strip(), "Unwarping", chk_unwarp.isChecked())
            need_ok(textline_dir.text().strip(), "TextlineOri", chk_textline.isChecked())
            need_ok(docori_dir.text().strip(), "DocOri", chk_docori.isChecked())
            need_ok(det_dir.text().strip(), "Detection", True)
            need_ok(rec_dir.text().strip(), "Recognition", True)

            if errs:
                QtWidgets.QMessageBox.warning(self, "路徑驗證失敗 / Path validation failed", "\n".join(errs))
                return False
            return True

        def on_accept():
            if not _hard_validate_or_msgbox():
                return
            self.update_ocr_config(
                base_dir=base_edit.text().strip(),
                use_doc_unwarping=chk_unwarp.isChecked(),
                doc_unwarping_model_name=unwarp_name.text().strip(),
                doc_unwarping_model_dir=unwarp_dir.text().strip(),
                use_textline_orientation=chk_textline.isChecked(),
                textline_orientation_model_name=textline_name.text().strip(),
                textline_orientation_model_dir=textline_dir.text().strip(),
                textline_orientation_batch_size=cfg.textline_orientation_batch_size,
                use_doc_orientation_classify=chk_docori.isChecked(),
                doc_orientation_classify_model_dir=docori_dir.text().strip(),
                doc_orientation_classify_model_name=docori_name.text().strip(),
                text_detection_model_dir=det_dir.text().strip(),
                text_detection_model_name=det_name.text().strip(),
                text_recognition_model_dir=rec_dir.text().strip(),
                text_recognition_model_name=rec_name.text().strip(),
                lang=lang_combo.currentText(),
                device=device_combo.currentText(),
                cpu_threads=cpu_spin.value(),
                enable_hpi=hpi_chk.isChecked(),
                enable_mkldnn=mkldnn_chk.isChecked(),
                text_det_limit_side_len=det_side_len.value(),
                text_det_limit_type=det_limit_type.currentText(),
                text_det_box_thresh=det_box_thresh.value(),
                text_det_thresh=det_thresh.value(),
                text_det_unclip_ratio=unclip_ratio.value(),
            )
            dlg.accept()

        btns.accepted.connect(on_accept)
        btns.rejected.connect(dlg.reject)
        dlg.exec()




    # ========= 更新 OCR 設定並重建 Processor（一致的嚴格驗證）=========
    def update_ocr_config(self,
                        base_dir: str,
                        use_doc_unwarping: bool,
                        doc_unwarping_model_name: str,
                        doc_unwarping_model_dir: str,
                        use_textline_orientation: bool,
                        textline_orientation_model_name: str,
                        textline_orientation_model_dir: str,
                        textline_orientation_batch_size: int,
                        use_doc_orientation_classify: bool,
                        doc_orientation_classify_model_dir: str,
                        doc_orientation_classify_model_name: str,
                        text_detection_model_dir: str,
                        text_detection_model_name: str,
                        text_recognition_model_dir: str,
                        text_recognition_model_name: str,
                        lang: str,
                        device: str,
                        cpu_threads: int,
                        enable_hpi: bool,
                        enable_mkldnn: bool,
                        text_det_limit_side_len: int,
                        text_det_limit_type: str,
                        text_det_box_thresh: float,
                        text_det_thresh: float,
                        text_det_unclip_ratio: float):
        """
        依視窗設定更新 OCR_Processor_Config，並立即重建 self.ocr_processor。
        若 base_dir 有填，會用 base_dir 自動補齊空白的模型路徑。
        內含嚴格路徑驗證（遞迴掃描 .onnx / .nb / .pdmodel+.pdiparams），缺檔直接拋出例外。
        """
        from pathlib import Path
        import logging

        # 路徑自動補齊（僅填空）
        temp_cfg = OCR_Processor_Config(
            ocr_version=self.ocr_cfg.ocr_version,

            use_doc_unwarping=use_doc_unwarping,
            doc_unwarping_model_name=doc_unwarping_model_name,
            doc_unwarping_model_dir=doc_unwarping_model_dir,

            use_textline_orientation=use_textline_orientation,
            textline_orientation_model_name=textline_orientation_model_name,
            textline_orientation_model_dir=textline_orientation_model_dir,
            textline_orientation_batch_size=textline_orientation_batch_size,

            use_doc_orientation_classify=use_doc_orientation_classify,
            doc_orientation_classify_model_dir=doc_orientation_classify_model_dir,
            doc_orientation_classify_model_name=doc_orientation_classify_model_name,

            text_detection_model_dir=text_detection_model_dir,
            text_detection_model_name=text_detection_model_name,

            text_recognition_model_dir=text_recognition_model_dir,
            text_recognition_model_name=text_recognition_model_name,

            lang=lang,
            device=device,
            cpu_threads=int(cpu_threads),
            enable_hpi=bool(enable_hpi),
            enable_mkldnn=bool(enable_mkldnn),
            text_det_limit_side_len=int(text_det_limit_side_len),
            text_det_limit_type=text_det_limit_type,
            text_det_box_thresh=float(text_det_box_thresh),
            text_det_thresh=float(text_det_thresh),
            text_det_unclip_ratio=float(text_det_unclip_ratio),
        )

        if base_dir:
            auto_paths = ocr_paths_mod.build_ocr_dirs_from_base(base_dir, temp_cfg)
            if not temp_cfg.doc_unwarping_model_dir:
                temp_cfg.doc_unwarping_model_dir = auto_paths["doc_unwarping_model_dir"]
            if not temp_cfg.textline_orientation_model_dir:
                temp_cfg.textline_orientation_model_dir = auto_paths["textline_orientation_model_dir"]
            if not temp_cfg.doc_orientation_classify_model_dir:
                temp_cfg.doc_orientation_classify_model_dir = auto_paths["doc_orientation_classify_model_dir"]
            if not temp_cfg.text_detection_model_dir:
                temp_cfg.text_detection_model_dir = auto_paths["text_detection_model_dir"]
            if not temp_cfg.text_recognition_model_dir:
                temp_cfg.text_recognition_model_dir = auto_paths["text_recognition_model_dir"]

        # —— 嚴格驗證：與 GUI 相同邏輯（遞迴掃描）——
        def _dir_exists(p: str) -> bool:
            return bool(p and Path(p).is_dir())

        def _find_model_in_dir(root: str) -> tuple[bool, str]:
            """
            合格條件（任一即通過）：
            A) 任意子層有 *.onnx
            B) 任意子層有 *.nb
            C) 任意子層、同一目錄、同一 stem 的三件套：<name>.json + <name>.yml + <name>.pdiparams
                （例如 inference.json / inference.yml / inference.pdiparams）
            D) 任意子層、同一目錄、同一 stem 的 Paddle 原生配對：<name>.pdmodel + <name>.pdiparams
            回傳：(ok, reason)
            """
            from pathlib import Path
            from collections import defaultdict

            if not root:
                return False, "未設定路徑 / path empty"
            p = Path(root)
            if not p.is_dir():
                return False, "目錄不存在 / directory not found"

            # A) onnx
            if any(p.rglob("*.onnx")):
                return True, ""

            # B) nb
            if any(p.rglob("*.nb")):
                return True, ""

            # 收集三種副檔名與 paddle 原生副檔的 stem 分佈
            dir2json = defaultdict(set)
            dir2yml  = defaultdict(set)
            dir2pdip = defaultdict(set)
            dir2pdm  = defaultdict(set)

            for fp in p.rglob("*.json"):
                dir2json[fp.parent].add(fp.stem)
            for fp in p.rglob("*.yml"):
                dir2yml[fp.parent].add(fp.stem)
            for fp in p.rglob("*.pdiparams"):
                dir2pdip[fp.parent].add(fp.stem)
            for fp in p.rglob("*.pdmodel"):
                dir2pdm[fp.parent].add(fp.stem)

            # C) 三件套：<name>.json + <name>.yml + <name>.pdiparams（同目錄同名）
            for d in set(list(dir2json.keys()) + list(dir2yml.keys()) + list(dir2pdip.keys())):
                if dir2json[d] & dir2yml[d] & dir2pdip[d]:
                    return True, ""

            # D) Paddle 原生：<name>.pdmodel + <name>.pdiparams（同目錄同名）
            for d in set(list(dir2pdm.keys()) + list(dir2pdip.keys())):
                if dir2pdm[d] & dir2pdip[d]:
                    return True, ""

            # 進一步回傳更精準的錯誤原因（方便排查）
            # 只找到部分檔案時的提示
            has_json = any(dir2json.values())
            has_yml  = any(dir2yml.values())
            has_pdip = any(dir2pdip.values())
            has_pdm  = any(dir2pdm.values())

            if has_json or has_yml or has_pdip or has_pdm:
                # 檢查三件套缺哪個
                if (has_json and has_yml and not has_pdip):
                    return False, "找到 .json + .yml，但缺 .pdiparams / missing .pdiparams"
                if (has_json and has_pdip and not has_yml):
                    return False, "找到 .json + .pdiparams，但缺 .yml / missing .yml"
                if (has_yml and has_pdip and not has_json):
                    return False, "找到 .yml + .pdiparams，但缺 .json / missing .json"
                # 檢查 paddle 原生缺哪個
                if (has_pdm and not has_pdip):
                    return False, "找到 .pdmodel，但缺 .pdiparams / missing .pdiparams"
                if (has_pdip and not has_pdm):
                    return False, "找到 .pdiparams，但缺 .pdmodel / missing .pdmodel"

            return False, "找不到 *.onnx / *.nb，或 (.json+.yml+.pdiparams) / (.pdmodel+.pdiparams) 配對"


        errs = []
        def must_ok(path: str, name: str, enabled: bool = True):
            if not enabled: return
            ok, reason = _find_model_in_dir(path)
            if not ok: errs.append(f"{name} 缺檔 / missing: {path}（{reason}）")

        must_ok(temp_cfg.text_detection_model_dir, "Detection", True)
        must_ok(temp_cfg.text_recognition_model_dir, "Recognition", True)
        must_ok(temp_cfg.doc_unwarping_model_dir, "Unwarping", temp_cfg.use_doc_unwarping)
        must_ok(temp_cfg.textline_orientation_model_dir, "TextlineOri", temp_cfg.use_textline_orientation)
        must_ok(temp_cfg.doc_orientation_classify_model_dir, "DocOri", temp_cfg.use_doc_orientation_classify)

        if errs:
            raise ValueError("；".join(errs))

        # 重建 Processor
        try:
            # Lazy store config; defer model initialization to OCR run
            self.ocr_cfg = temp_cfg
            self.ocr_processor = None
            if hasattr(self, "statusbar") and self.statusbar:
                self.statusbar.showMessage("OCR 設定已更新 / OCR settings updated", 3000)
            logging.getLogger("main").info("OCR settings updated successfully")
        except Exception as e:
            logging.getLogger("main").exception("Failed to update OCR config")
            if hasattr(self, "statusbar") and self.statusbar:
                self.statusbar.showMessage(f"OCR 設定更新失敗 / Failed to update OCR settings: {e}", 5000)
            raise




    def show_translator_setting(self):

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("本地 NLLB 設定 / Local NLLB Settings")
        form = QtWidgets.QFormLayout(dlg)

        # 可視名稱 <-> 代碼 對應
        NAME_TO_CODE = NLLBConfig.LANGUAGE_MAP
        CODE_TO_NAME = {v: k for k, v in NAME_TO_CODE.items()}

        # ===== 模型名稱（僅作標示；實際載入看 local_dir）=====
        model_combo = QtWidgets.QComboBox()
        model_combo.addItems([
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-1.3B",
            "facebook/nllb-200-3.3B"
        ])
        model_combo.setCurrentText(self.nllb_cfg.model_name)
        form.addRow("模型名稱 / Model", model_combo)

        # ===== 源語言 / Source Language（含 Auto）=====
        src_combo = QtWidgets.QComboBox()
        src_items = ["Auto", "English", "Japanese", "Simplified Chinese", "Traditional Chinese"]
        src_combo.addItems(src_items)
        if self.nllb_cfg.src_language is None:
            src_combo.setCurrentText("Auto")
        else:
            src_combo.setCurrentText(CODE_TO_NAME.get(self.nllb_cfg.src_language, "Auto"))
        form.addRow("源語言 / Source Language", src_combo)

        # ===== 目標語言 / Target Language =====
        tgt_combo = QtWidgets.QComboBox()
        tgt_items = ["English", "Japanese", "Simplified Chinese", "Traditional Chinese"]
        tgt_combo.addItems(tgt_items)
        tgt_combo.setCurrentText(CODE_TO_NAME.get(self.nllb_cfg.tgt_language, "Traditional Chinese"))
        form.addRow("目標語言 / Target Language", tgt_combo)

        # ===== 本地模型資料夾 =====
        local_edit = QtWidgets.QLineEdit(self.nllb_cfg.local_dir or "")
        browse = QtWidgets.QPushButton("選擇資料夾 / Browse…")
        h = QtWidgets.QHBoxLayout()
        h.addWidget(local_edit); h.addWidget(browse)
        form.addRow("本地模型資料夾 / Local Model Folder", h)

        def on_browse():
            d = QFileDialog.getExistingDirectory(
                self, "選擇 NLLB 本地模型資料夾（需包含 config/tokenizer/權重） / Select local NLLB folder"
            )
            if d:
                local_edit.setText(d)
        browse.clicked.connect(on_browse)

        # ===== safetensors 偏好 =====
        st_chk = QtWidgets.QCheckBox("優先使用 safetensors（若存在） / Prefer safetensors if available")
        st_chk.setChecked(self.nllb_cfg.prefer_safetensors)
        form.addRow(st_chk)

        gpu_chk = QtWidgets.QCheckBox("使用 GPU 加速 (需要 CUDA) / Use GPU acceleration")
        gpu_available = bool(torch and torch.cuda.is_available())
        if not gpu_available:
            gpu_chk.setEnabled(False)
            gpu_chk.setToolTip("未偵測到可用的 GPU / No compatible GPU detected")
        current_device = getattr(self.nllb_cfg, "device", "auto")
        gpu_chk.setChecked(gpu_available and current_device != "cpu")
        form.addRow(gpu_chk)


        # ===== 生成參數（完整保留）=====
        # === SpinBox 設定區 ===
        max_tok = QtWidgets.QSpinBox(); max_tok.setRange(1, 99999)
        max_tok.setValue(self.translate_config.max_new_tokens)
        max_tok.setAccelerated(True)

        # 自適應步幅
        def _adapt_max_tok_step(val: int):
            if val < 256:
                step = 16
            elif val < 1024:
                step = 32
            elif val < 4096:
                step = 64
            else:
                step = 128
            max_tok.setSingleStep(step)
        _adapt_max_tok_step(max_tok.value())
        max_tok.valueChanged.connect(_adapt_max_tok_step)

        min_len = QtWidgets.QSpinBox(); min_len.setRange(0, 99999)
        min_len.setValue(self.translate_config.min_length)
        min_len.setSingleStep(16); min_len.setAccelerated(True)

        num_beams = QtWidgets.QSpinBox(); num_beams.setRange(1, 20)
        num_beams.setValue(self.translate_config.num_beams)
        num_beams.setSingleStep(1); num_beams.setAccelerated(True)

        early_chk = QtWidgets.QCheckBox("早期停止 / Early Stopping")
        early_chk.setChecked(self.translate_config.early_stopping)

        len_penalty = QtWidgets.QDoubleSpinBox(); len_penalty.setRange(0.0, 10.0)
        len_penalty.setDecimals(2); len_penalty.setValue(self.translate_config.length_penalty)
        len_penalty.setSingleStep(0.10); len_penalty.setAccelerated(True)

        ngram_spin = QtWidgets.QSpinBox(); ngram_spin.setRange(0, 20)
        ngram_spin.setValue(self.translate_config.no_repeat_ngram_size)
        ngram_spin.setSingleStep(1); ngram_spin.setAccelerated(True)

        rep_penalty = QtWidgets.QDoubleSpinBox(); rep_penalty.setRange(0.0, 10.0)
        rep_penalty.setDecimals(2); rep_penalty.setValue(self.translate_config.repetition_penalty)
        rep_penalty.setSingleStep(0.10)

        do_sample_chk = QtWidgets.QCheckBox("隨機採樣 / Do Sampling")
        do_sample_chk.setChecked(self.translate_config.do_sample)

        temp_spin = QtWidgets.QDoubleSpinBox(); temp_spin.setRange(0.1, 5.0)
        temp_spin.setDecimals(2); temp_spin.setValue(self.translate_config.temperature)
        temp_spin.setSingleStep(0.05)

        topk_spin = QtWidgets.QSpinBox(); topk_spin.setRange(1, 500)
        topk_spin.setValue(self.translate_config.top_k)
        topk_spin.setSingleStep(5); topk_spin.setAccelerated(True)

        topp_spin = QtWidgets.QDoubleSpinBox(); topp_spin.setRange(0.0, 1.0)
        topp_spin.setDecimals(2); topp_spin.setValue(self.translate_config.top_p)
        topp_spin.setSingleStep(0.01)

        # === 隨機採樣控制 ===
        def _toggle_sampling_controls(checked: bool):
            temp_spin.setEnabled(checked)
            topk_spin.setEnabled(checked)
            topp_spin.setEnabled(checked)

        _toggle_sampling_controls(do_sample_chk.isChecked())
        do_sample_chk.toggled.connect(_toggle_sampling_controls)


        form.addRow("最大生成 Token / Max New Tokens", max_tok)
        form.addRow("最小長度 / Min Length", min_len)
        form.addRow("Beam 數 / Num Beams", num_beams)
        form.addRow(early_chk)
        form.addRow("長度懲罰 / Length Penalty", len_penalty)
        form.addRow("禁止重複 N-gram / No-Repeat N-gram", ngram_spin)
        form.addRow("重複懲罰 / Repetition Penalty", rep_penalty)
        form.addRow(do_sample_chk)
        form.addRow("溫度 / Temperature", temp_spin)
        form.addRow("Top-K", topk_spin)
        form.addRow("Top-P", topp_spin)

        # ===== 按鈕 =====
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(btns)

        def accept():
            # 轉換源語言：Auto -> None；其餘 -> 代碼
            src_name = src_combo.currentText()
            src_code = None if src_name == "Auto" else NAME_TO_CODE.get(src_name)

            gpu_enabled = gpu_chk.isChecked() and gpu_available
            device = "cuda" if gpu_enabled else "cpu"
            torch_dtype = "float16" if gpu_enabled else "float32"


            self.update_translator_config(
                model_name=model_combo.currentText(),
                src_lang_name=src_name,              # 傳可視名稱，函式內會轉換
                tgt_lang_name=tgt_combo.currentText(),
                local_dir=local_edit.text(),
                prefer_safetensors=st_chk.isChecked(),
                max_new_tokens=max_tok.value(),
                min_length=min_len.value(),
                num_beams=num_beams.value(),
                early_stopping=early_chk.isChecked(),
                length_penalty=len_penalty.value(),
                no_repeat_ngram_size=ngram_spin.value(),
                repetition_penalty=rep_penalty.value(),
                do_sample=do_sample_chk.isChecked(),
                temperature=temp_spin.value(),
                top_k=topk_spin.value(),
                top_p=topp_spin.value(),
                device=device,
                torch_dtype=torch_dtype,
            )
            dlg.accept()

        btns.accepted.connect(accept)
        btns.rejected.connect(dlg.reject)
        dlg.exec()




    def update_translator_config(
        self,
        *,
        model_name: str,
        src_lang_name: str,            # "Auto" / "English" / "Japanese" / "Simplified Chinese" / "Traditional Chinese"
        tgt_lang_name: str,
        local_dir: str | None,
        prefer_safetensors: bool,

        # 生成參數
        max_new_tokens: int,
        min_length: int,
        num_beams: int,
        early_stopping: bool,
        length_penalty: float,
        no_repeat_ngram_size: int,
        repetition_penalty: float,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        device: str,
        torch_dtype: str,
    ):
        """
        寫回本地 NLLB 與生成參數設定（不在此階段建模；真正載入在子進程）。
        """
        try:
            # --- 語言代碼轉換 ---
            NAME_TO_CODE = NLLBConfig.LANGUAGE_MAP
            # 源語言：Auto -> None
            src_lang = None if src_lang_name == "Auto" else NAME_TO_CODE.get(src_lang_name, None)
            tgt_lang = NAME_TO_CODE.get(tgt_lang_name, "zho_Hant")

            # --- 正規化路徑 ---
            local_dir = (local_dir or "").strip() or None

            # --- 回寫 NLLB 設定（純資料） ---
            self.nllb_cfg.model_name = model_name
            self.nllb_cfg.src_language = src_lang
            self.nllb_cfg.tgt_language = tgt_lang
            self.nllb_cfg.local_dir = local_dir
            self.nllb_cfg.prefer_safetensors = prefer_safetensors
            self.nllb_cfg.device = device
            self.nllb_cfg.torch_dtype = torch_dtype
            self.nllb_cfg.use_tf32 = bool(device == "cuda")

            # --- 回寫生成參數 ---
            tc = self.translate_config
            tc.max_new_tokens = int(max_new_tokens)
            tc.min_length = int(min_length)
            tc.num_beams = int(num_beams)
            tc.early_stopping = bool(early_stopping)
            tc.length_penalty = float(length_penalty)
            tc.no_repeat_ngram_size = int(no_repeat_ngram_size)
            tc.repetition_penalty = float(repetition_penalty)
            tc.do_sample = bool(do_sample)
            tc.temperature = float(temperature)
            tc.top_k = int(top_k)
            tc.top_p = float(top_p)

            tc.adjust_for_model(self.nllb_cfg.model_name)

            # --- 提示（雙語） ---
            if hasattr(self, "statusbar") and self.statusbar:
                msg = "翻譯設定已更新；本地模型將於執行時載入 / Settings updated; local model loads at run time"
                self.statusbar.showMessage(msg, 3000)

            logger.info(
                ("Translator config updated: model=%s, src=%s, tgt=%s, local_dir=%s, prefer_safetensors=%s, "
                "device=%s, torch_dtype=%s, max_new_tokens=%d, min_length=%d, beams=%d, early=%s, "
                "len_penalty=%.2f, ngram=%d, rep_penalty=%.2f, sample=%s, temp=%.2f, top_k=%d, top_p=%.2f"),
                self.nllb_cfg.model_name, self.nllb_cfg.src_language, self.nllb_cfg.tgt_language,
                self.nllb_cfg.local_dir, self.nllb_cfg.prefer_safetensors,
                self.nllb_cfg.device, getattr(self.nllb_cfg.torch_dtype, "name", self.nllb_cfg.torch_dtype),
                tc.max_new_tokens, tc.min_length, tc.num_beams, tc.early_stopping, tc.length_penalty,
                tc.no_repeat_ngram_size, tc.repetition_penalty, tc.do_sample, tc.temperature, tc.top_k, tc.top_p
            )

        except Exception as e:
            logger.exception("Failed to update translator config")
            if hasattr(self, "statusbar") and self.statusbar:
                self.statusbar.showMessage(f"翻譯設定更新失敗 / Failed to update settings：{e}", 5000)

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
