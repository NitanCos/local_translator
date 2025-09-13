# Local and API translator
## Project Preview
This is a desktop application for Optical Character Recognition (OCR) and text translation, supporting multiple languages and modes. 
It uses PaddleOCR for text detection and recognition, and integrates translation via local NLLB models or external APIs like Gemini and DeepL. 
The tool features a PyQt6-based GUI for easy screenshot capture, OCR processing, and translation. 
Ideal for users needing quick text extraction and translation from images, such as screenshots or documents.<br/>

Key technologies: Python, PyQt6, PaddleOCR, Transformers (for NLLB), Requests (for APIs).<br/>

## Feature
 - Screenshot Capture: Select and capture screen regions for OCR.
 - OCR Processing: Detect and recognize text in images using PaddleOCR v5, with configurable parameters like language and thresholds.
 - Translation:
    + Local mode: Uses Facebook's NLLB-200 models (distilled variants: 600M, 1.3B, 3.3B) for offline translation.
    + API mode: Integrates with Gemini (for OCR + translation) or DeepL (for translation after local OCR).
 - Supports source language auto-detection and target languages like English, Japanese, Simplified/Traditional Chinese.
 - GUI Interface: User-friendly interface with text display, copy/clear buttons, history logs, and settings menus.
 - History and Logging: Maintains OCR and translation history; detailed logging for debugging.
 - Configurable Settings: Adjust OCR parameters (e.g., CPU threads, box thresholds) and translation configs (e.g., beams, temperature).
 - Save Options: Optionally save captured images with timestamps.

## Installation
### Prerequisites
Python 3.12+
### Step
1. Clone the repository : <br/>
   ```
   https://github.com/NitanCos/local_translator
   ```
3. Navigate to the project directory :<br/>
   ```
   cd local_translator/CPUver
   ```
5. Install dependencies :<br/>
   ```
   pip install -r requirements.txt
   ```
7. For NLLB models: Automatically Download from Hugging Face in first execution.
8. For APIs: Obtain keys for Gemini or DeepL and configure in settings menu.

### Run
```
python main.py
```

## Acknowledgments
 - PaddleOCR for OCR engine.
 - Hugging Face Transformers for NLLB models.
 - Google Gemini and DeepL for API integrations.
 - PyQt6 for GUI framework.
