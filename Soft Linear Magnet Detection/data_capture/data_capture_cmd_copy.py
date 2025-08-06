import serial
import serial.tools.list_ports
import time
import threading
import signal
from queue import Queue
import numpy as np

SENSOR_NUM = 17 * 5
SENSOR_BYTES = 3 * 2
PACKET_SIZE = SENSOR_NUM * SENSOR_BYTES + 4


class DataLoaderOnline:
    def __init__(self):
        super(DataLoaderOnline, self).__init__()

        ports_list = list(serial.tools.list_ports.comports())
        if not ports_list:
            print("❌ 沒有可用串口")
        else:
            print("✅ 可用串口:")
            for port in ports_list:
                print(f"  {port.device}: {port.description}")

        #signal.signal(signal.SIGINT, self.close)

        self.queue = Queue(maxsize=5)
        self.t1 = threading.Thread(target=self.read_thread)
        self.flag_running = False

        self.geo_cnt = 0
        self.geo_data = np.zeros((SENSOR_NUM, 3), dtype=np.float64)
        self.callback = None
        self.m_serial = None

    def open(self, port_name='COM6'):
        try:
            self.m_serial = serial.Serial(port_name, 115200)
            if self.m_serial.is_open:
                self.send_data('Run\r\n')
                print(f"✅ 串口 {port_name} 已開啟")
                return True
        except serial.SerialException as e:
            print(f"❌ 無法開啟串口 {port_name}: {e}")
        return False

    def close(self, signal=None, frame=None):
        self.flag_running = False
        if self.t1.is_alive():
            self.t1.join()
        if self.m_serial and self.m_serial.is_open:
            self.send_data('Stop\r\n')
            self.m_serial.close()
        print("🔌 串口已關閉")

    def send_data(self, value):
        if not self.m_serial or not self.m_serial.is_open:
            print("⚠️ 串口未開啟，無法發送指令")
            return
        self.m_serial.write(value.encode('utf-8'))

    def raw_to_sensor(self, raw_data):
        def decode(low, high):
            return np.array((high << 8) + low, dtype=np.uint16).astype(np.int16)

        sensor_data = np.zeros((SENSOR_NUM, 3), dtype=np.float32)
        for i in range(SENSOR_NUM):
            x = decode(raw_data[i*6+0], raw_data[i*6+1]) * 100 / 6842
            y = decode(raw_data[i*6+2], raw_data[i*6+3]) * 100 / 6842
            z = decode(raw_data[i*6+4], raw_data[i*6+5]) * 100 / 6842
            sensor_data[i] = [x, y, z]
        return sensor_data

    # ✅ 用於 CLI 模式：不使用 callback
    def start_read(self):
        self.flag_running = True
        self.geo_cnt = 0
        self.geo_data = np.zeros((SENSOR_NUM, 3), dtype=np.float64)
        self.t1 = threading.Thread(target=self.read_thread)
        self.t1.daemon = True
        self.t1.start()

    # ✅ 用於 GUI 模式：使用 callback 處理資料
    def start_reading(self, callback):
        self.callback = callback
        self.start_read()

    def read_thread(self):
        print("🧵 讀取執行緒啟動")
        while self.flag_running:
            if self.m_serial and self.m_serial.is_open:
                data = self.m_serial.read(PACKET_SIZE)

                print(f"📦 串口讀到 {len(data)} bytes")
                print("🔍 結尾 bytes:", list(data[-4:]))

                if len(data) >= PACKET_SIZE:
                    raw = data[:SENSOR_NUM * SENSOR_BYTES]
                    try:
                        sensor_data = self.raw_to_sensor(raw)
                        if self.queue.full():
                            self.queue.get()
                        self.queue.put(sensor_data)
                        print("⚠️ 忽略尾碼校驗，成功加入 queue")
                    except Exception as e:
                        print("❌ raw_to_sensor() 解析失敗:", e)


    # ✅ CLI 模式下主動拉資料（例如用在 while-loop）
    def get_data(self):
        data = self.queue.get()

        if self.geo_cnt < 50:
            self.geo_data += data
            self.geo_cnt += 1
            if self.geo_cnt == 50:
                self.geo_data /= 50
            return (self.geo_cnt, data)
        else:
            return (0, data - self.geo_data)


# 🧪 測試主程式：CLI 模式下使用 get_data() 讀取校正後磁場
if __name__ == "__main__":
    import time

    data_loader = DataLoaderOnline()
    if not data_loader.open('COM6'):
        exit()

    data_loader.start_read()  # ✅ 啟用 CLI 模式

    try:
        while True:
            flag, data = data_loader.get_data()
            if flag == 0:
                print("已校正:", data[-5])
            else:
                print(f"標定中 {flag}/50")
            time.sleep(0.05)
    except KeyboardInterrupt:
        data_loader.close()
