from API_mode import APIMode
from ocr_processor import OCR_Processor, OCR_Processor_Config
from multiprocessing import Process, Queue
from PyQt6 import QtCore
from PIL import Image
import numpy as np
import os
import tempfile


# ------- Child process target: Translation -------

def run_translation(queue, text: str, is_api_mode: bool,
                     api_snapshot: dict | None,
                     nllb_cfg_dict: dict | None,
                     translate_cfg_dict: dict | None):
    try:
        if is_api_mode:
            api = APIMode()
            api.config.api_type   = api_snapshot["api_type"]
            api.config.api_key    = api_snapshot["api_key"]
            api.config.model      = api_snapshot["model"]
            api.config.target_lang= api_snapshot["target_lang"]
            paras = text.split("\n\n")
            out = [api.translate_text(p) for p in paras]
            queue.put(("result", "\n\n".join(out)))
        else:
            from NLLB_translator import NLLBTranslator, NLLBConfig, TranslateConfig
            cfg = NLLBConfig(
                model_name=nllb_cfg_dict["model_name"],
                src_language=nllb_cfg_dict.get("src_language"),
                tgt_language=nllb_cfg_dict["tgt_language"],
                local_dir=nllb_cfg_dict["local_dir"],          # ★ 必填：由 GUI 選擇
                prefer_safetensors=nllb_cfg_dict.get("prefer_safetensors", False),
                low_cpu_mem_usage=True,
                torch_dtype=nllb_cfg_dict.get("torch_dtype", "float32"),
                revision="main",
            )
            gcfg = TranslateConfig(
                max_new_tokens=translate_cfg_dict["max_new_tokens"],
                min_length=translate_cfg_dict["min_length"],
                num_beams=translate_cfg_dict["num_beams"],
                early_stopping=translate_cfg_dict["early_stopping"],
                length_penalty=translate_cfg_dict["length_penalty"],
                no_repeat_ngram_size=translate_cfg_dict["no_repeat_ngram_size"],
                repetition_penalty=translate_cfg_dict["repetition_penalty"],
                do_sample=translate_cfg_dict["do_sample"],
                temperature=translate_cfg_dict["temperature"],
                top_k=translate_cfg_dict["top_k"],
                top_p=translate_cfg_dict["top_p"],
            )
            translator = NLLBTranslator(cfg)  # ★ 這裡才真正載入（本地、離線、無下載）
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            out = [translator.translate(p, gcfg) for p in paragraphs]
            queue.put(("result", "\n\n".join(out)))
    except Exception as e:
        queue.put(("error", str(e)))


# ------- Child process target: OCR --------

def run_ocr(queue, img_source, is_api_mode: bool, api_snapshot: dict | None, ocr_cfg_dict: dict | None):
    try:
        if is_api_mode:
            api = APIMode()
            api.config.api_type    = api_snapshot["api_type"]
            api.config.api_key     = api_snapshot["api_key"]
            api.config.model       = api_snapshot["model"]
            api.config.target_lang = api_snapshot["target_lang"]
            # Accept either a file path or an ndarray; open path here to avoid large pickling
            if isinstance(img_source, str):
                img = Image.open(img_source).convert("RGB")
                img_array = np.array(img)
            else:
                img_array = img_source
            text = api.ocr_image(img_array)
            queue.put(("result", text))
        else:
            cfg = OCR_Processor_Config(
                ocr_version=ocr_cfg_dict["ocr_version"],

                use_doc_unwarping=ocr_cfg_dict["use_doc_unwarping"],
                doc_unwarping_model_name=ocr_cfg_dict["doc_unwarping_model_name"],
                doc_unwarping_model_dir=ocr_cfg_dict["doc_unwarping_model_dir"],

                use_textline_orientation=ocr_cfg_dict["use_textline_orientation"],
                textline_orientation_model_name=ocr_cfg_dict["textline_orientation_model_name"],
                textline_orientation_model_dir=ocr_cfg_dict["textline_orientation_model_dir"],
                textline_orientation_batch_size=ocr_cfg_dict["textline_orientation_batch_size"],

                use_doc_orientation_classify=ocr_cfg_dict["use_doc_orientation_classify"],
                doc_orientation_classify_model_dir=ocr_cfg_dict["doc_orientation_classify_model_dir"],
                doc_orientation_classify_model_name=ocr_cfg_dict["doc_orientation_classify_model_name"],

                text_detection_model_dir=ocr_cfg_dict["text_detection_model_dir"],
                text_detection_model_name=ocr_cfg_dict["text_detection_model_name"],

                text_recognition_model_dir=ocr_cfg_dict["text_recognition_model_dir"],
                text_recognition_model_name=ocr_cfg_dict["text_recognition_model_name"],

                lang=ocr_cfg_dict["lang"],
                device=ocr_cfg_dict["device"],
                cpu_threads=ocr_cfg_dict["cpu_threads"],
                enable_hpi=ocr_cfg_dict["enable_hpi"],
                enable_mkldnn=ocr_cfg_dict["enable_mkldnn"],
                text_det_limit_side_len=ocr_cfg_dict["text_det_limit_side_len"],
                text_det_limit_type=ocr_cfg_dict["text_det_limit_type"],
                text_det_box_thresh=ocr_cfg_dict["text_det_box_thresh"],
                text_det_thresh=ocr_cfg_dict["text_det_thresh"],
                text_det_unclip_ratio=ocr_cfg_dict["text_det_unclip_ratio"],
            )
            ocr = OCR_Processor(cfg)
            predict_res = ocr.ocr_predict(img_source)
            all_text_list = ocr.json_preview_and_get_all_text(predict_res)
            queue.put(("result", "\n".join(all_text_list)))
    except Exception as e:
        queue.put(("error", str(e)))


# ------- Qt-side wrapper workers (parent process) -------

class TranslationWorker:
    def __init__(self, text, api_mode, is_api_mode, translate_config, nllb_cfg):
        self.text = text
        self.is_api_mode = is_api_mode
        self.api_snapshot = {
            "api_type": api_mode.config.api_type,
            "api_key":  api_mode.config.api_key,
            "model":    api_mode.config.model,
            "target_lang": api_mode.config.target_lang,
        }
        self.nllb_cfg_dict = {
            "model_name": nllb_cfg.model_name,
            "src_language": nllb_cfg.src_language,
            "tgt_language": nllb_cfg.tgt_language,
            "local_dir": nllb_cfg.local_dir,
            "prefer_safetensors": nllb_cfg.prefer_safetensors,
            "torch_dtype": getattr(nllb_cfg.torch_dtype, "name", nllb_cfg.torch_dtype),
        }
        self.translate_cfg_dict = {
            "max_new_tokens": translate_config.max_new_tokens,
            "min_length": translate_config.min_length,
            "num_beams": translate_config.num_beams,
            "early_stopping": translate_config.early_stopping,
            "length_penalty": translate_config.length_penalty,
            "no_repeat_ngram_size": translate_config.no_repeat_ngram_size,
            "repetition_penalty": translate_config.repetition_penalty,
            "do_sample": translate_config.do_sample,
            "temperature": translate_config.temperature,
            "top_k": translate_config.top_k,
            "top_p": translate_config.top_p,
        }

        self.queue = Queue()
        self.process = None
        self.timer = None

    def start(self, callback_result, callback_error):
        self.process = Process(
            target=run_translation,
            args=(self.queue, self.text, self.is_api_mode,
                  self.api_snapshot, self.nllb_cfg_dict, self.translate_cfg_dict)
        )
        self.process.start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self._check_queue(callback_result, callback_error))
        self.timer.start(200)

    def _check_queue(self, callback_result, callback_error):
        try:
            msg_type, data = self.queue.get_nowait()
        except Exception:
            return
        self.timer.stop()
        if msg_type == "result":
            callback_result(data)
        else:
            callback_error(data)

    def cancel(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()


class OCRWorker:
    def __init__(self, img_array, api_mode, is_api_mode, ocr_cfg):
        # Accept ndarray or file path; if ndarray, we'll persist to temp file before starting process
        self.img_source = img_array
        self.is_api_mode = is_api_mode

        self.api_snapshot = {
            "api_type": api_mode.config.api_type,
            "api_key":  api_mode.config.api_key,
            "model":    api_mode.config.model,
            "target_lang": api_mode.config.target_lang,
        }

        cfg = ocr_cfg
        self.ocr_cfg_dict = {
            "ocr_version": cfg.ocr_version,

            "use_doc_unwarping": cfg.use_doc_unwarping,
            "doc_unwarping_model_name": cfg.doc_unwarping_model_name,
            "doc_unwarping_model_dir": cfg.doc_unwarping_model_dir,

            "use_textline_orientation": cfg.use_textline_orientation,
            "textline_orientation_model_name": cfg.textline_orientation_model_name,
            "textline_orientation_model_dir": cfg.textline_orientation_model_dir,
            "textline_orientation_batch_size": cfg.textline_orientation_batch_size,

            "use_doc_orientation_classify": cfg.use_doc_orientation_classify,
            "doc_orientation_classify_model_dir": cfg.doc_orientation_classify_model_dir,
            "doc_orientation_classify_model_name": cfg.doc_orientation_classify_model_name,

            "text_detection_model_dir": cfg.text_detection_model_dir,
            "text_detection_model_name": cfg.text_detection_model_name,

            "text_recognition_model_dir": cfg.text_recognition_model_dir,
            "text_recognition_model_name": cfg.text_recognition_model_name,

            "lang": cfg.lang,
            "device": cfg.device,
            "cpu_threads": cfg.cpu_threads,
            "enable_hpi": cfg.enable_hpi,
            "enable_mkldnn": cfg.enable_mkldnn,
            "text_det_limit_side_len": cfg.text_det_limit_side_len,
            "text_det_limit_type": cfg.text_det_limit_type,
            "text_det_box_thresh": cfg.text_det_box_thresh,
            "text_det_thresh": cfg.text_det_thresh,
            "text_det_unclip_ratio": cfg.text_det_unclip_ratio,
        }

        self.queue = Queue()
        self.process = None
        self.timer = None
        self._temp_path = None

    def start(self, callback_result, callback_error):
        img_arg = self.img_source
        # If ndarray, persist to a temp file to avoid pickling large arrays on Windows spawn
        try:
            if isinstance(img_arg, np.ndarray):
                os.makedirs("screenshots", exist_ok=True)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="screenshot_", dir="screenshots")
                Image.fromarray(img_arg).save(tmp.name)
                self._temp_path = tmp.name
                img_arg = self._temp_path
        except Exception:
            # Fall back to original ndarray if saving fails
            pass

        self.process = Process(
            target=run_ocr,
            args=(self.queue, img_arg, self.is_api_mode,
                  self.api_snapshot, self.ocr_cfg_dict)
        )
        self.process.start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self._check_queue(callback_result, callback_error))
        self.timer.start(200)

    def _check_queue(self, callback_result, callback_error):
        try:
            msg_type, data = self.queue.get_nowait()
        except Exception:
            return
        self.timer.stop()
        if msg_type == "result":
            callback_result(data)
        else:
            callback_error(data)
        self._cleanup_temp()

    def cancel(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self._cleanup_temp()

    def _cleanup_temp(self):
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except Exception:
                pass
            finally:
                self._temp_path = None
