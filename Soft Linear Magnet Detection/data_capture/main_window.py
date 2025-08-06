from PyQt5 import QtWidgets
from window import Ui_MainWindow

import sys
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader_online import DataLoaderOnline
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化串口資料讀取器
        self.data_loader = DataLoaderOnline()
        self.collected_data = []

        # 綁定按鈕功能
        self.ui.btn_open.clicked.connect(self.open_serial)
        self.ui.btn_start.clicked.connect(self.start_acquire)

    def open_serial(self):
        success = self.data_loader.open("COM6")
        if success:
            QtWidgets.QMessageBox.information(self, "Info", "串口已開啟 COM6")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "❌ 串口打開失敗！")

    def start_acquire(self):
        if not self.data_loader.m_serial:
            QtWidgets.QMessageBox.warning(self, "Warning", "請先打開串口")
            return
            # 每次采集前清空舊數據
        save_path = os.path.join("dataset", "captured_data.npy")
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"⚠️ 已清除舊數據: {save_path}")
        self.collected_data = []
        # 啟動讀取執行緒
        self.data_loader.start_reading(self.handle_data)
        QtWidgets.QMessageBox.information(self, "Info", "開始讀取數據...")

    def handle_data(self, data):
        # 假設data為numpy array，處理或儲存
        print("接收到數據:", data)
        self.collected_data.append(data)

    def closeEvent(self, a0):
        # 關閉視窗時關閉串口與儲存數據
        if self.data_loader:
            self.data_loader.close(None, None)

        if self.collected_data:
            save_path = os.path.join("dataset", "captured_data.npy")
            os.makedirs("dataset", exist_ok=True)
            np.save(save_path, np.array(self.collected_data))
            print(f"✅ 數據已儲存到 {save_path}")
        if a0 is not None:
            a0.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.setWindowTitle("Magnetic Sensor GUI")
    main_win.show()
    sys.exit(app.exec_())
