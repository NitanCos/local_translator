import sys
import os
import logging
import re
from logging.handlers import RotatingFileHandler
import requests  # For HTTPError
from PyQt6 import QtWidgets, QtCore, QtGui
from MainGUI import Ui_MainWindow
from ocr_processor import OCR_Processor, OCR_Processor_Config
from region_capture import RegionSelector
from NLLB_translator import NLLBTranslator, NLLBConfig, TranslateConfig
from datetime import datetime
import json
from pathlib import Path
import torch

# 日誌設定 / Logging configuration
def setup_logging():
    """設置日誌配置，為每個模組創建獨立處理器並支援旋轉和控制台輸出 / Set up logging configuration with separate handlers for each module, supporting rotation and console output"""
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    os.makedirs("Debug", exist_ok=True)

    # 共用格式器 / Shared formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # 控制台處理器 / Console handler for all loggers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # 控制台顯示 DEBUG 級別以上 / Console shows DEBUG and above

    # 為 main 創建旋轉檔案處理器 / Rotating file handler for main
    main_handler = RotatingFileHandler("Debug/Main.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    main_handler.setFormatter(formatter)
    main_handler.setLevel(logging.INFO)

    # 為 ocr_processor 創建旋轉檔案處理器 / Rotating file handler for ocr_processor
    ocr_handler = RotatingFileHandler("Debug/OCR.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    ocr_handler.setFormatter(formatter)
    ocr_handler.setLevel(logging.DEBUG)

    # 為 translater 創建單一旋轉檔案處理器（統一所有模型） / Rotating file handler for translater (unified for all models)
    translator_handler = RotatingFileHandler("Debug/Translater.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    translator_handler.setFormatter(formatter)
    translator_handler.setLevel(logging.DEBUG)

    # 為每個模組的記錄器添加處理器（檔案 + 控制台） / Add handlers (file + console) to module loggers
    main_logger = logging.getLogger("main")
    main_logger.addHandler(main_handler)
    main_logger.addHandler(console_handler)
    main_logger.propagate = False

    ocr_logger = logging.getLogger("ocr_processor")
    ocr_logger.addHandler(ocr_handler)
    ocr_logger.addHandler(console_handler)
    ocr_logger.propagate = False

    translator_logger = logging.getLogger("translater")
    translator_logger.addHandler(translator_handler)
    translator_logger.addHandler(console_handler)
    translator_logger.propagate = False

# 初始化日誌 / Initialize logging
setup_logging()
logger = logging.getLogger("main")

class MainApplication(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Starting MainApplication initialization")
        self.setupUi(self)
        self.ocr_processor = None
        self.translator = None
        self.ocr_history = []
        self.translate_history = []
        self.max_history = 5  # 限制歷史記錄最多5筆 / Limit history to 5 entries
        self.init_components()
        self.setup_connections()
        self.load_history()
        logger.info("MainApplication initialization completed")

    def init_components(self):
        """初始化 OCR 和翻譯器 / Initialize OCR and Translator"""
        logger.info("Initializing components")
        try:
            ocr_config = OCR_Processor_Config()
            self.ocr_processor = OCR_Processor(ocr_config)
            logger.info("OCR Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR Processor: {e}")
            self.statusbar.showMessage("OCR 初始化失敗 / OCR initialization failed", 5000)

        try:
            nllb_config = NLLBConfig()
            translate_config = TranslateConfig()
            self.translator = NLLBTranslator(nllb_config)
            self.translate_config = translate_config
            self.translate_config.adjust_for_model(self.translator.cfg.model_name)  # 調整默認基於模型 / Adjust defaults based on model
            logger.info("Translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Translator: {e}")
            self.statusbar.showMessage("翻譯器初始化失敗 / Translator initialization failed", 5000)

    def setup_connections(self):
        """設置按鈕和選單的連接 / Set up button and menu connections"""
        logger.info("Setting up connections")
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
        self.actionMain_Log.triggered.connect(lambda: self.show_log("Debug/Main.log"))
        self.actionOCR_Log.triggered.connect(lambda: self.show_log("Debug/OCR.log"))
        self.actionTranslator_Log.triggered.connect(lambda: self.show_log("Debug/Translater.log"))  # 統一為單一 log / Unified to single log

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
                    predict_res = self.ocr_processor.ocr_predict(img_array)
                    all_text = self.ocr_processor.json_preview_and_get_all_text(predict_res)
                    # 前處理 OCR 文本，僅保留完整句子
                    processed_sentences = self.preprocess_ocr_text(all_text)
                    text = "\n".join(processed_sentences)
                    self.OCR_Detect_Text.setPlainText(text)
                    self.ocr_history.insert(0, {"timestamp": timestamp, "text": text, "image_path": save_path if self.checkBox.isChecked() else None})
                    self.ocr_history = self.ocr_history[:self.max_history]
                    self.save_history()
                    logger.info("OCR completed successfully with %d sentences", len(processed_sentences))
                    self.statusbar.showMessage("OCR 完成 / OCR completed", 5000)
                else:
                    logger.error("Failed to capture screenshot")
                    self.statusbar.showMessage("螢幕捕捉失敗 / Screenshot capture failed", 5000)
            else:
                logger.info("Region selection cancelled")
        except Exception as e:
            logger.error(f"Error in execute_ocr: {e}")
            self.statusbar.showMessage("OCR 處理失敗 / OCR processing failed", 5000)

    def execute_translate(self):
        """執行翻譯 / Execute translation"""
        logger.info("Executing translation")
        try:
            text = self.OCR_Detect_Text.toPlainText()
            if not text:
                logger.warning("No text to translate")
                self.statusbar.showMessage("無文本可翻譯 / No text to translate", 5000)
                return
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            translated_text = self.translator.translate(text, self.translate_config)
            self.Translated_Text.setPlainText(translated_text)
            self.translate_history.insert(0, {"timestamp": timestamp, "text": translated_text})
            self.translate_history = self.translate_history[:self.max_history]
            self.save_history()
            logger.info("Translation completed successfully")
            self.statusbar.showMessage("翻譯完成 / Translation completed", 5000)
        except Exception as e:
            logger.error(f"Error in execute_translate: {e}")
            self.statusbar.showMessage("翻譯處理失敗 / Translation processing failed", 5000)

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
        """顯示選擇模式對話框（未來功能） / Show select mode dialog (future feature)"""
        logger.info("Showing select mode")
        QtWidgets.QMessageBox.information(self, "Select Mode", "此功能即將推出，敬請期待！\nThis feature is coming soon, stay tuned!")
        logger.info("Select Mode accessed (placeholder)")

    def show_ocr_setting(self):
        """顯示 OCR 設定對話框 / Show OCR settings dialog"""
        logger.info("Showing OCR settings")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("OCR 設定 / OCR Settings")
        layout = QtWidgets.QFormLayout()

        # 語言 / Language
        lang_label = QtWidgets.QLabel("語言 / Language:")
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItems(["japan", "ch", "en"])
        lang_combo.setCurrentText(self.ocr_processor.config.lang)
        layout.addRow(lang_label, lang_combo)

        # 設備 / Device
        device_label = QtWidgets.QLabel("設備 / Device:")
        device_combo = QtWidgets.QComboBox()
        device_combo.addItems(["cpu", "gpu"])
        device_combo.setCurrentText(self.ocr_processor.config.device)
        layout.addRow(device_label, device_combo)

        # CPU 線程數 / CPU Threads
        cpu_threads_label = QtWidgets.QLabel("CPU 線程數 / CPU Threads:")
        cpu_threads_spin = QtWidgets.QSpinBox()
        cpu_threads_spin.setRange(1, 32)
        cpu_threads_spin.setValue(self.ocr_processor.config.cpu_threads)
        layout.addRow(cpu_threads_label, cpu_threads_spin)

        # 高性能推理 / Enable HPI
        enable_hpi_check = QtWidgets.QCheckBox("啟用高性能推理 / Enable HPI")
        enable_hpi_check.setChecked(self.ocr_processor.config.enable_hpi)
        layout.addRow(enable_hpi_check)

        # MKLDNN
        enable_mkldnn_check = QtWidgets.QCheckBox("啟用 MKLDNN / Enable MKLDNN")
        enable_mkldnn_check.setChecked(self.ocr_processor.config.enable_mkldnn)
        layout.addRow(enable_mkldnn_check)

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
            lang_combo.currentText(),
            device_combo.currentText(),
            cpu_threads_spin.value(),
            enable_hpi_check.isChecked(),
            enable_mkldnn_check.isChecked(),
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

    def update_ocr_config(self, lang, device, cpu_threads, enable_hpi, enable_mkldnn, use_doc_unwarping,
                         use_textline_orientation, use_doc_orientation_classify, text_det_limit_side_len,
                         text_det_limit_type, text_det_box_thresh, text_det_thresh, text_det_unclip_ratio, dialog):
        """更新 OCR 配置 / Update OCR configuration"""
        logger.info(f"Updating OCR config: lang={lang}, device={device}, cpu_threads={cpu_threads}")
        try:
            self.ocr_processor.config = OCR_Processor_Config(
                lang=lang,
                device=device,
                cpu_threads=cpu_threads,
                enable_hpi=enable_hpi,
                enable_mkldnn=enable_mkldnn,
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
        logger.info("Showing translator settings")
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("翻譯設定 / Translator Settings")
        layout = QtWidgets.QFormLayout()

        # 模型選擇 / Model selection
        model_label = QtWidgets.QLabel("模型 / Model:")
        model_combo = QtWidgets.QComboBox()
        model_combo.addItems([
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-1.3B",
            "facebook/nllb-200-3.3B"
        ])
        model_combo.setCurrentText(self.translator.cfg.model_name)
        layout.addRow(model_label, model_combo)

        # 源語言 / Source Language
        src_label = QtWidgets.QLabel("源語言 / Source Language:")
        src_combo = QtWidgets.QComboBox()
        src_combo.addItems(["jpn_Jpan", "zho_Hans", "eng_Latn"])
        src_combo.setCurrentText(self.translator.cfg.src_language)
        layout.addRow(src_label, src_combo)

        # 目標語言 / Target Language
        tgt_label = QtWidgets.QLabel("目標語言 / Target Language:")
        tgt_combo = QtWidgets.QComboBox()
        tgt_combo.addItems(["eng_Latn", "zho_Hans", "jpn_Jpan"])
        tgt_combo.setCurrentText(self.translator.cfg.tgt_language)
        layout.addRow(tgt_label, tgt_combo)

        # 最大生成 token 數 / Max New Tokens
        max_new_tokens_label = QtWidgets.QLabel("最大生成 token 數 / Max New Tokens:")
        max_new_tokens_spin = QtWidgets.QSpinBox()
        max_new_tokens_spin.setRange(1, 1000)
        max_new_tokens_spin.setValue(self.translate_config.max_new_tokens)
        layout.addRow(max_new_tokens_label, max_new_tokens_spin)

        # 最小長度 / Min Length
        min_length_label = QtWidgets.QLabel("最小長度 / Min Length:")
        min_length_spin = QtWidgets.QSpinBox()
        min_length_spin.setRange(0, 1000)
        min_length_spin.setValue(self.translate_config.min_length)
        layout.addRow(min_length_label, min_length_spin)

        # Beam 數 / Num Beams
        num_beams_label = QtWidgets.QLabel("Beam 數 / Num Beams:")
        num_beams_spin = QtWidgets.QSpinBox()
        num_beams_spin.setRange(1, 20)
        num_beams_spin.setValue(self.translate_config.num_beams)
        layout.addRow(num_beams_label, num_beams_spin)

        # 提前停止 / Early Stopping
        early_stopping_check = QtWidgets.QCheckBox("提前停止 / Early Stopping")
        early_stopping_check.setChecked(self.translate_config.early_stopping)
        layout.addRow(early_stopping_check)

        # 長度懲罰 / Length Penalty
        length_penalty_label = QtWidgets.QLabel("長度懲罰 / Length Penalty:")
        length_penalty_spin = QtWidgets.QDoubleSpinBox()
        length_penalty_spin.setRange(-5.0, 5.0)
        length_penalty_spin.setSingleStep(0.1)
        length_penalty_spin.setValue(self.translate_config.length_penalty)
        layout.addRow(length_penalty_label, length_penalty_spin)

        # N-gram 重複限制 / No Repeat N-gram Size
        no_repeat_ngram_size_label = QtWidgets.QLabel("N-gram 重複限制 / No Repeat N-gram Size:")
        no_repeat_ngram_size_spin = QtWidgets.QSpinBox()
        no_repeat_ngram_size_spin.setRange(0, 10)
        no_repeat_ngram_size_spin.setValue(self.translate_config.no_repeat_ngram_size)
        layout.addRow(no_repeat_ngram_size_label, no_repeat_ngram_size_spin)

        # 重複懲罰 / Repetition Penalty
        repetition_penalty_label = QtWidgets.QLabel("重複懲罰 / Repetition Penalty:")
        repetition_penalty_spin = QtWidgets.QDoubleSpinBox()
        repetition_penalty_spin.setRange(1.0, 10.0)
        repetition_penalty_spin.setSingleStep(0.1)
        repetition_penalty_spin.setValue(self.translate_config.repetition_penalty)
        layout.addRow(repetition_penalty_label, repetition_penalty_spin)

        # 隨機採樣 / Do Sample
        do_sample_check = QtWidgets.QCheckBox("隨機採樣 / Do Sample")
        do_sample_check.setChecked(self.translate_config.do_sample)
        layout.addRow(do_sample_check)

        # 溫度 / Temperature
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

        # 確認按鈕 / Confirm buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(lambda: self.update_translator_config(
            model_combo.currentText(),
            src_combo.currentText(),
            tgt_combo.currentText(),
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
                src_language=src_lang,
                tgt_language=tgt_lang,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
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
            self.translate_config.adjust_for_model(model_name)  # 調整默認值 / Adjust defaults
            self.translator = NLLBTranslator(self.translator.cfg)
            logger.info(f"Translator config updated successfully")
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
            self.OCR_Detect_Text.setPlainText(self.ocr_history[index]["text"])
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
            self.Translated_Text.setPlainText(self.translate_history[index]["text"])
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