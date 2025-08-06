from PyQt5 import QtWidgets
from window import Ui_MainWindow

import sys
import numpy as np
import os
from data_capture_cmd import DataLoaderOnline
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

    def handle_data(self, data_tuple):
        geo_cnt, data = data_tuple
        if geo_cnt == 0:
            # 這是已校正數據
            print("✅ 校正後數據:", data.shape)
            self.collected_data.append(data)
        else:
            print(f"⏳ 地磁校正中 ({geo_cnt}/50)...")

    def closeEvent(self, a0):
        # 關閉視窗時關閉串口與儲存數據
        if self.data_loader:
            self.data_loader.close(None, None)

        if self.collected_data:
            all_data = np.array(self.collected_data)  # shape: (N, 85, 3)
            all_data = all_data.reshape(all_data.shape[0], -1)  # (N, 255)

            # 加入位姿（選擇性）
            magnet_pos = np.array([30, 30, 65]) * 1e-3
            magnet_ori = np.array([0, 0, 1])
            Bt = 7.68750001e-08
            pose_info = np.concatenate((magnet_pos, magnet_ori, [Bt]))  # shape: (7,)
            pose_info = np.tile(pose_info, (all_data.shape[0], 1))  # (N, 7)
            
            
            save_path = os.path.join("dataset", "captured_data.npy")
            os.makedirs("dataset", exist_ok=True)
            full_data = np.concatenate((all_data, pose_info), axis=1)
            np.save(save_path, full_data)
            print(f"✅ 數據已儲存到 {save_path}")
        if a0 is not None:
            a0.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.setWindowTitle("Magnetic Sensor GUI")
    main_win.show()
    sys.exit(app.exec_())
