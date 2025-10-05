# Local and API translator

  <a href="https://www.python.org/downloads/">
    <img alt="PyPI - Python Version" src="https://img.shields.io/badge/pyversion-3.12%2B-blue?style=flat&label=python">
  </a>  

## Project Preview

This is a desktop application for Optical Character Recognition (OCR) and text translation, supporting multiple languages and modes.  

It uses PaddleOCR for text detection and recognition, and integrates translation via local NLLB models or external APIs like Gemini and DeepL.  

The tool features a PyQt6-based GUI for easy screenshot capture, OCR processing, and translation.
Ideal for users needing quick text extraction and translation from images, such as screenshots or documents.

## Feature

- Screenshot Capture: Select and capture screen regions for OCR.
- OCR Processing: Detect and recognize text in images using PPOCR v5, with configurable parameters like language and thresholds.
- Translation:
  - Local mode: Uses Facebook's NLLB-200 models (distilled variants: 600M, 1.3B, 3.3B) for offline translation.(and you can freely define and choose the local model you want to use.)
  - API mode: Integrates with Gemini (for OCR + translation) or DeepL (for translation after local OCR).
- Supports source language auto-detection and target languages like English, Japanese, Simplified/Traditional Chinese.
- GUI Interface: User-friendly interface with text display, copy/clear buttons, history logs, and settings menus.
- History and Logging: Maintains OCR and translation history; detailed logging for debugging.
- Configurable Settings: Adjust OCR parameters (e.g., CPU threads, box thresholds) and translation configs (e.g., beams, temperature).
- Save Options: Optionally save captured images with timestamps.

## Installation

### Prerequisites

Python 3.12+

### Step

1. Clone the repository:

   ```bash
   https://github.com/NitanCos/local_translator.git
   ```

2. Navigate to the project directory:

   ```bash
   cd local_translator/src
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   Base on your current device (OCR)
   ```bash
   #cpu
   python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   #gpu,cuda=12.9
   python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
   ```

4. For NLLB models:  
   - [600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
   - [1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)
   - [3.3B](https://huggingface.co/facebook/nllb-200-3.3B)

   (HTTP)

   ```bash
   # Select the model you want install
   # Make sure git-lfs is installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/facebook/nllb-200-distilled-600M
   git clone https://huggingface.co/facebook/nllb-200-distilled-1.3B
   git clone https://huggingface.co/facebook/nllb-200-3.3B
   ```

   (CLI)

   ```bash
   # Select the model you want install
   # Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
   hf download facebook/nllb-200-distilled-600M
   hf download facebook/nllb-200-distilled-1.3B
   hf download facebook/nllb-200-3.3B
   ```

5. For PaddleOCR:  
   *You need to download the required model through the following link and use a model that matches your device’s performance.*
   - [PP-LCNet_x1_0_doc_ori](https://www.paddleocr.ai/latest/version3.x/module_usage/doc_img_orientation_classification.html)
   - [UVaDoc](https://www.paddleocr.ai/latest/version3.x/module_usage/text_image_unwarping.html)
   - [PP-LCNet_x1_0_textline_ori or PP-LCNet_x0_25_textline_ori](https://www.paddleocr.ai/latest/version3.x/module_usage/textline_orientation_classification.html)
   - [PP-OCRv5_det](https://www.paddleocr.ai/latest/version3.x/module_usage/text_detection.html#_2)
   - [PP-OCRv5_rec](https://www.paddleocr.ai/latest/version3.x/module_usage/text_recognition.html)

### Run

```bash
python main.py
```

## Usage Step

   1. Setup the Local Translator model location **(if you just want to use the API mode, select Cancel)**

      The meaning of parameters can find in **[reference.md](./CPUver/reference.md)**
      <p align="left"><img src="./img/img1.png" alt="Translater setting" width="500"></p>
   2. Select Mode (and enter the API Key)
      <p align="left"><img src="./img/img2.png" alt="API mode setting" width="300"></p>
      <p align="left"><img src="./img/img3.png" alt="Local mode setting" width="300"></p>
   3. Setup the OCR model and parameters (except use Gemini API. Gemini API will use own OCR to recognize text.)
      <p align="left"><img src="./img/img4.png" alt="Local mode setting" width="300"></p>
      please create and unzip the OCR model to the directory:  

      ```text
      ./models/paddleocr
      ```

      The meaning of parameters can find in **[reference.md](./CPUver/reference.md)**
   4. Start Translate (GUI view)
      <p align="left"><img src="./img/img5.png" alt="Local mode setting" width="300"></p>

## Credits

- PaddleOCR - <https://github.com/PaddlePaddle/PaddleOCR>
- Hugging Face Transformers for NLLB models.(nllb-200-distilled-600M) - <https://huggingface.co/facebook/nllb-200-distilled-600M>
- Hugging Face Transformers for NLLB models.(nllb-200-distilled-1.3B) - <https://huggingface.co/facebook/nllb-200-distilled-1.3B>
- Hugging Face Transformers for NLLB models.(nllb-200-3.3B) - <https://huggingface.co/facebook/nllb-200-3.3B>

## License

   This project is released under the [Apache 2.0 license](./LICENSE.md)
   > **[!NOTICE]**
      > This project uses the NLLB-200 model from Meta AI (license under CC-BY-NC-4.0).  
      > As a result, while the source code is Apchace-2.0, the overall project is restricted to NON-COMMERIAL-USE.
