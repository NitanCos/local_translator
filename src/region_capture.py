from PyQt6 import QtWidgets, QtCore, QtGui
import mss
import logging
from PIL import Image
import numpy as np
import sys
from mss.exception import ScreenShotError
from datetime import datetime  # 用於生成時間戳記

# 設置日誌格式，調整 datefmt 以移除毫秒，便於查看
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class RegionSelector(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("選擇捕捉區域")
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.drawing = False
        self.selected_region = None
        
        # 使用 Qt 原生 screens() 計算所有螢幕的聯合邊界（支持負坐標的多螢幕）
        try:
            screens = QtGui.QGuiApplication.screens()
            if screens:
                min_left = min(s.geometry().x() for s in screens)
                min_top = min(s.geometry().y() for s in screens)
                max_right = max(s.geometry().x() + s.geometry().width() for s in screens)
                max_bottom = max(s.geometry().y() + s.geometry().height() for s in screens)
                width = max_right - min_left
                height = max_bottom - min_top
                self.setGeometry(min_left, min_top, width, height)
                # 保存計算邊界以確保絕對座標正確（避免系統調整負坐標）
                self.min_left = min_left
                self.min_top = min_top
                logging.info(f"Geometry set based on Qt screens: left={min_left}, top={min_top}, width={width}, height={height}")
                logging.debug(f"Actual window geometry after set: left={self.geometry().x()}, top={self.geometry().y()}")
            else:
                # 若無螢幕偵測，預設回退
                self.setGeometry(0, 0, 1920, 1080)
                self.min_left = 0
                self.min_top = 0
                logging.warning("No screens detected, using default geometry.")
        except Exception as e:
            logging.exception(f"Error setting geometry with Qt screens: {e}")
            self.setGeometry(0, 0, 1920, 1080)
            self.min_left = 0
            self.min_top = 0  # 強制預設
        
        logging.info("Initialized RegionSelector")

    def paintEvent(self, event):
        try:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 100))
            if self.drawing:
                rect = QtCore.QRect(self.start_x, self.start_y, self.end_x - self.start_x, self.end_y - self.start_y)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
                painter.drawRect(rect)
        except Exception as e:
            logging.exception(f"Error in paintEvent: {e}")

    def mousePressEvent(self, event):
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.start_x = event.pos().x()
                self.start_y = event.pos().y()
                self.end_x = self.start_x
                self.end_y = self.start_y
                self.drawing = True
                self.update()
                logging.debug(f"Mouse pressed at ({self.start_x}, {self.start_y})")
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                # 右鍵按下取消選擇
                self.reject()
                logging.info("Region selection cancelled by right mouse click")
        except Exception as e:
            logging.exception(f"Error in mousePressEvent: {e}")

    def mouseMoveEvent(self, event):
        try:
            if self.drawing:
                self.end_x = event.pos().x()
                self.end_y = event.pos().y()
                self.update()
        except Exception as e:
            logging.exception(f"Error in mouseMoveEvent: {e}")

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.drawing = False
                rel_top = min(self.start_y, self.end_y)
                rel_left = min(self.start_x, self.end_x)
                width = abs(self.end_x - self.start_x)
                height = abs(self.end_y - self.start_y)
                if width <= 0 or height <= 0:
                    logging.warning("Invalid region selected: width or height <= 0")
                    self.reject()
                    return
                # 轉換為絕對座標，使用保存的 min_top/min_left 以避免負坐標系統調整問題
                abs_top = self.min_top + rel_top
                abs_left = self.min_left + rel_left
                self.selected_region = (abs_top, abs_left, width, height)
                logging.info(f"Selected region: top={abs_top}, left={abs_left}, width={width}, height={height}")
                self.accept()
        except Exception as e:
            logging.exception(f"Error in mouseReleaseEvent: {e}")

    def keyPressEvent(self, event):
        try:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.reject()
                logging.info("Region selection cancelled by user")
        except Exception as e:
            logging.exception(f"Error in keyPressEvent: {e}")

    def capture_screenshot(self, save_path=None, save_image: bool = True):
        if self.selected_region is None:
            logging.warning("No region selected for screenshot")
            return None
        top, left, width, height = self.selected_region
        try:
            with mss.mss() as sct:
                monitor = {"top": top, "left": left, "width": width, "height": height}
                sct_img = sct.grab(monitor)
            logging.debug("Screenshot grabbed successfully.")
        except ScreenShotError as e:
            logging.error(f"MSS ScreenShotError: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in screenshot grab: {e}")
            return None
        
        try:
            # 轉換為 PIL Image
            img = Image.frombytes("RGB", (width, height), sct_img.bgra, "raw", "BGRX")
            logging.debug("Image converted from bytes successfully.")
            if save_image:
                # 如果未提供 save_path，生成帶時間戳記的檔案名稱
                if save_path is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path = f"screenshot_{timestamp}.png"
                img.save(save_path)
                logging.info(f"Screenshot saved to {save_path}")
            else:
                logging.info("Screenshot captured but not saved (save_image=False)")
            # 返回 NumPy 陣列，以便傳遞到 OCR
            return np.array(img)
        except (ValueError, IOError) as e:
            logging.error(f"PIL Image creation or save error: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in image processing: {e}")
            return None

if __name__ == "__main__":
    try:
        app = QtWidgets.QApplication(sys.argv)
        logger.debug("QApplication initialized.")
        selector = RegionSelector()
        logger.debug("RegionSelector instantiated.")
        if selector.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            img_array = selector.capture_screenshot()  # 未提供 save_path，讓它自動生成帶時間的檔案；預設 save_image=True
            if img_array is not None:
                print("Screenshot captured and saved successfully.")
                logger.info("Screenshot process completed successfully.")
            else:
                logger.error("Failed to capture screenshot.")
        else:
            logger.info("Region selection was cancelled or failed.")
        app.quit()  # 明確退出應用程式
        logger.debug("QApplication quit called.")
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
    finally:
        sys.exit(0)  # 確保程式結束