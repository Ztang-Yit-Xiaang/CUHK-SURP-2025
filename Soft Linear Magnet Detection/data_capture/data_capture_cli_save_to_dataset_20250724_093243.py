
import os
import numpy as np
import time
from data_capture_cmd_copy import DataLoaderOnline

SAVE_DIR = os.path.join(os.path.dirname(__file__), '../dataset')
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, 'captured_' + time.strftime('%Y%m%d_%H%M%S') + '.npy')

def main():
    loader = DataLoaderOnline()
    if not loader.open('COM6'):
        return

    loader.start_read()
    collected_data = []

    print("📐 開始地磁校正...")
    while loader.geo_cnt < 50:
        try:
            flag, data = loader.get_data()
            print("⏳ 標定中 {}/50".format(flag))
        except Exception as e:
            print("⚠️ 等待資料超時，請檢查串口或感測器")
            time.sleep(0.1)

    print("✅ 地磁校正完成，開始記錄資料")

    try:
        while True:
            flag, data = loader.get_data()
            collected_data.append(data)
            print("✅ 已接收第 {} 幀資料".format(len(collected_data)))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("🛑 偵測到中斷，正在關閉...")
        loader.close()

        if collected_data:
            print("💾 儲存 {} 筆資料至 {}".format(len(collected_data), SAVE_PATH))
            collected_array = np.array(collected_data).reshape(len(collected_data), -1)
            np.save(SAVE_PATH, collected_array)
        else:
            print("⚠️ 沒有資料被儲存")

if __name__ == "__main__":
    main()
