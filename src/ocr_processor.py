from paddleocr import PaddleOCR
import logging
import numpy as np
import os
from dataclasses import dataclass
from PIL import Image
import requests
from io import BytesIO
import glob
import json
from pathlib import Path
import re



# 日誌設定
# LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# os.makedirs("Debug", exist_ok=True)  # 創建 Debug 目錄
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filename="Debug/OCR.log", encoding="utf-8")
# logger = logging.getLogger(__name__)

@dataclass
class OCR_Processor_Config:

    ocr_version: str = 'PP-OCRv5' #OCR 版本

    use_doc_unwarping: bool = True  
    doc_unwarping_model_name: str = "UVDoc"  
    doc_unwarping_model_dir: str = r"D:\models\paddleocr\UVDoc_infer"

    use_textline_orientation: bool = True  
    textline_orientation_model_name	: str = "PP-LCNet_x0_25_textline_ori"  
    textline_orientation_model_dir: str = r"D:\models\paddleocr\PP-LCNet_x0_25_textline_ori_infer"
    textline_orientation_batch_size: int = 1  

    use_doc_orientation_classify: bool = True 
    doc_orientation_classify_model_dir: str = r"D:\models\paddleocr\PP-LCNet_x1_0_doc_ori_infer"
    doc_orientation_classify_model_name: str = "PP-LCNet_x1_0_doc_ori"

    text_detection_model_dir: str = r"D:\models\paddleocr\PP-OCRv5_mobile_det_infer"
    text_detection_model_name: str = "PP-OCRv5_mobile_det"

    text_recognition_model_dir: str = r"D:\models\paddleocr\PP-OCRv5_mobile_rec_infer"
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec"

    lang: str = 'en' #語言
    device: str = 'cpu' #設備
    cpu_threads: int = 12 #CPU 線程數
    enable_hpi: bool = False #是否啟用高性能推理 (only on Linux)
    enable_mkldnn: bool = False #是否啟用 MKLDNN (only intel cpu)
    text_det_limit_side_len: int = 64 
    text_det_limit_type: str = "min"  
    text_det_box_thresh: float = 0.3  
    text_det_thresh: float = 0.3  
    text_det_unclip_ratio: float = 1.6  

class OCR_Processor:
    def __init__(self, config: OCR_Processor_Config):
        self.config = config
        try:
            self.ocr = PaddleOCR( 

                ocr_version=config.ocr_version,
                use_doc_unwarping=config.use_doc_unwarping,
                doc_unwarping_model_name=config.doc_unwarping_model_name,
                doc_unwarping_model_dir=config.doc_unwarping_model_dir,

                use_textline_orientation=config.use_textline_orientation,  
                textline_orientation_model_name=config.textline_orientation_model_name,
                textline_orientation_model_dir=config.textline_orientation_model_dir,
                textline_orientation_batch_size=config.textline_orientation_batch_size,

                use_doc_orientation_classify=config.use_doc_orientation_classify,
                doc_orientation_classify_model_dir = config.doc_orientation_classify_model_dir,
                doc_orientation_classify_model_name = config.doc_orientation_classify_model_name,
                
                text_detection_model_dir=config.text_detection_model_dir,
                text_detection_model_name=config.text_detection_model_name,

                text_recognition_model_dir=config.text_recognition_model_dir,
                text_recognition_model_name=config.text_recognition_model_name,

                lang=config.lang,
                device=config.device,
                cpu_threads=config.cpu_threads,
                enable_hpi=config.enable_hpi, #是否啟用高性能推理 (only on Linux)
                enable_mkldnn=config.enable_mkldnn, #是否啟用 MKLDNN (only intel cpu)
                text_det_limit_side_len=config.text_det_limit_side_len,
                text_det_limit_type=config.text_det_limit_type,
                text_det_box_thresh= config.text_det_box_thresh,
                text_det_thresh= config.text_det_thresh, 
                text_det_unclip_ratio= config.text_det_unclip_ratio
            )
        except Exception as e:
            raise e

    def is_url(self, s: str) -> bool:
        url_pattern = re.compile(
            r'^(https?://)'  # 協議 (http 或 https)
            r'([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'  # 域名
            r'(/.*)?$'  # 可選路徑
        )
        return bool(url_pattern.match(s.strip()))

    def _load_url_image(self, url: str) -> np.ndarray:

        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        image_array = np.array(img)

        return image_array
    
    def _expand_input(self, input) : #將 inputs 扁平化並展開目錄、網址，回傳 list[Union[str, np.ndarray]]
        if not isinstance(input, list): 
            input = [input]
        output = []
        for item in input:
            if isinstance(item, np.ndarray):
                output.append(item)
            elif isinstance(item, str):
                if self.is_url(item):
                    output.append(item)
                elif os.path.isdir(item):
                    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
                        files = glob.glob(os.path.join(item, ext))
                        output.extend(files)
                else:
                    # 單一檔案（Image / PDF）
                    output.append(item)
            else:
                raise TypeError(f"Unsupported input type: {type(item)}")
        return output
    
    def ocr_predict(self, predict_input):
        
        # Predict inputs 支援：
        # - numpy.ndarray
        # - str: 本地檔案路徑（圖片、PDF）、目錄、HTTP/HTTPS URL
        # - list: 以上任意型別組合
        # Predict return value: list of OCR 結果（依輸入順序扁平化）
        
        expanded_inputs = self._expand_input(predict_input)
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
                    continue
            except Exception as e:
                continue
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