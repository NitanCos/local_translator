from paddleocr import PaddleOCR
import logging
import numpy as np
import os
import mss
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from pandas.core.internals.blocks import libinternals
import requests
from io import BytesIO
import glob
import json
from pathlib import Path
import re


# 日誌設定
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
os.makedirs("Debug", exist_ok=True)  # 創建 Debug 目錄
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename="Debug/OCR.log", encoding="utf-8")
logger = logging.getLogger(__name__)

@dataclass
class OCR_Processor_Config:
    lang: str = 'en' #語言
    device: str = 'cpu' #設備
    cpu_threads: int = 12 #CPU 線程數
    enable_hpi: bool = False #是否啟用高性能推理
    enable_mkldnn: bool = False #是否啟用 MKLDNN
    use_doc_unwarping: bool = True  #文本圖像校正
    use_textline_orientation: bool = False  #文本行方向判斷
    use_doc_orientation_classify: bool = False  #文檔方向判斷
    text_det_limit_side_len: int = 16  #文本檢測限制邊長
    text_det_limit_type: str = "min"  #文本檢測限制類型
    text_det_box_thresh: float = 0.3  #文本檢測框閾值
    text_det_thresh: float = 0.3  #文本檢測像素閾值
    text_det_unclip_ratio: float = 1.6  #文本檢測擴張係數



class OCR_Processor:
    def __init__(self, config: OCR_Processor_Config):
        self.config = config
        #logger.info("Initializing OCR_Processor with config: %s", config)
        #logger.info(f"語言: {config.lang}")
        #logger.info(f"使用文本圖像校正: {config.use_doc_unwarping}")
        #logger.info(f"使用文本行方向判斷: {config.use_textline_orientation}")
        #logger.info(f"使用文檔方向判斷: {config.use_doc_orientation_classify}")
        #logger.info(f"文本檢測限制邊長: {config.text_det_limit_side_len}")
        #logger.info(f"文本檢測限制類型: {config.text_det_limit_type}")
        #logger.info(f"文本檢測框閾值: {config.text_det_box_thresh}")
        #logger.info(f"文本檢測像素閾值: {config.text_det_thresh}")
        #logger.info(f"文本檢測擴張係數: {config.text_det_unclip_ratio}")
        try:
            self.ocr = PaddleOCR( 
                device=config.device,
                cpu_threads=config.cpu_threads,
                enable_hpi=config.enable_hpi,
                enable_mkldnn=config.enable_mkldnn,
                lang=config.lang,
                ocr_version='PP-OCRv5',
                use_doc_unwarping=config.use_doc_unwarping,
                use_textline_orientation=config.use_textline_orientation,
                use_doc_orientation_classify=config.use_doc_orientation_classify,
                text_det_limit_side_len=config.text_det_limit_side_len,
                text_det_limit_type=config.text_det_limit_type,
                text_det_box_thresh=config.text_det_box_thresh,
                text_det_thresh=config.text_det_thresh,
                text_det_unclip_ratio=config.text_det_unclip_ratio,
            )
            logger.info(f"PaddleOCR initialized successfully.，語言: {config.lang}")
        except Exception as e:
            logger.error(f"PaddleOCR 初始化失敗: {e}")
            raise e

    def is_url(self, s: str) -> bool:
        url_pattern = re.compile(
            r'^(https?://)'  # 協議 (http 或 https)
            r'([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'  # 域名
            r'(/.*)?$'  # 可選路徑
        )
        return bool(url_pattern.match(s.strip()))

    def _load_url_image(self, url: str) -> np.ndarray:
        logger.info(f"Downloading image from URL: {url}")
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        image_array = np.array(img)
        logger.info(f"Downloaded image size: {image_array.shape}")
        return image_array
    
    def _expand_input(self, input) : #將 inputs 扁平化並展開目錄、網址，回傳 list[Union[str, np.ndarray]]
        #logger.info(f"Expanding input: {input}")
        if not isinstance(input, list): 
            #logger.info(f"Input is not a list, converting to list: {input}")
            input = [input]
        output = []
        for item in input:
            if isinstance(item, np.ndarray):
                logger.debug("Array image, shape: %s", item.shape)
                output.append(item)
            elif isinstance(item, str):
                if self.is_url(item):
                    logger.debug("URL, load image: %s", item)
                    output.append(item)
                elif os.path.isdir(item):
                    # 只展開常見圖檔 
                    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
                        files = glob.glob(os.path.join(item, ext))
                        logger.debug("Found %d %s files", len(files), ext)
                        output.extend(files)
                else:
                    # 單一檔案（Image / PDF）
                    logger.debug("Single file, adding to output: %s", item)
                    output.append(item)
            else:
                logger.error("Unsupported input type: %s", type(item))
                raise TypeError(f"Unsupported input type: {type(item)}")
        logger.debug("Expanded list contains %d items", len(output))
        return output
    
    def ocr_predict(self, predict_input):
        
        # Predict inputs 支援：
        # - numpy.ndarray
        # - str: 本地檔案路徑（圖片、PDF）、目錄、HTTP/HTTPS URL
        # - list: 以上任意型別組合
        # Predict return value: list of OCR 結果（依輸入順序扁平化）
        
        expanded_inputs = self._expand_input(predict_input)
        #logger.info(f"Expanded inputs: {expanded_inputs}")
        predict_res = []

        for item in expanded_inputs:
            try:
                if isinstance(item, np.ndarray):
                    predict_res = self.ocr.predict(item)
                elif isinstance(item, str):
                    if self.is_url(item):
                        img = self._load_url_image(item)
                        predict_res = self.ocr.predict(img)
                    else:
                        predict_res = self.ocr.predict(item)
                else:
                    logger.error(f"Unsupported input type: {type(item)}")
                    continue
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                continue
            #logger.info(f"Prediction result: {predict_res}")
        return predict_res
    
    def ocr_print(self, predict_res, format_json = bool, indent = int, ensure_ascii = bool):
        for res in predict_res:
            res.print(format_json, indent, ensure_ascii)
    
    def ocr_save_to_img(self, predict_res, save_path = str): 
        for res in predict_res:
            res.save_to_img(save_path)
    
    def ocr_save_to_json(self, predict_res, save_path = str, indent = int, ensure_ascii = bool):
        for res in predict_res:
            res.save_to_json(save_path, indent, ensure_ascii)
    
    def json_preview_and_get_all_text(self, predict_res): #json 預覽 type: dict
        for res in predict_res:
            json_result = res.json
        all_text_list = json_result['res']['rec_texts']
        return all_text_list
    


if __name__ == "__main__":
    ocr = OCR_Processor(OCR_Processor_Config())
    ocr_predict = ocr.ocr_predict("./sample.png")
    ocr.ocr_save_to_img(ocr_predict, save_path="./result/")
    ocr.ocr_save_to_json(ocr_predict, save_path="./result/", indent=4, ensure_ascii=False)
    all_text_list = ocr.json_preview_and_get_all_text(ocr_predict)
    #for text in all_text_list:
    #    print(text)
    #    print(type(text))