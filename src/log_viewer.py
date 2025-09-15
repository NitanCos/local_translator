import logging
from PyQt6 import QtWidgets


logger = logging.getLogger("main")


def show_log(app, log_file: str):
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

    dialog = QtWidgets.QDialog(app)
    dialog.resize(900, 600)
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
    dialog.exec()
