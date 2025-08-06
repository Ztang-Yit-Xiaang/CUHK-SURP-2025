import serial
import threading
import time
import numpy as np

SENSOR_NUM = 85
SENSOR_BYTES = 6  # 每個 sensor：3 軸 × 2 bytes

class DataLoaderOnline:
    def __init__(self):
        self.m_serial = None
        self.reading = False
        self.read_thread_handle = None
        self.data_callback = None

    def open(self, port_name='COM6'):
        try:
            self.m_serial = serial.Serial(port_name, 115200)
            if self.m_serial.isOpen():
                send_value = 'Run\r\n'
                self.send_command(send_value)
                print(f"✅ 串口 {port_name} 已打開")
                return True
        except Exception as e:
            print(f"❌ 無法打開串口 {port_name}：{e}")
        return False

    def send_command(self, data_str):
        if self.m_serial and self.m_serial.isOpen():
            self.m_serial.write(data_str.encode())

    def start_read(self):
        self.reading = True
        self.read_thread_handle = threading.Thread(target=self.read_thread)
        self.read_thread_handle.start()

    def start_reading(self, callback):
        self.data_callback = callback
        self.start_read()

    def parse_raw_sensor_data(self, raw_sensor_data):
        sensor_data = np.zeros((SENSOR_NUM, 3))
        for i in range(SENSOR_NUM):
            x_raw = (raw_sensor_data[i*SENSOR_BYTES+1] << 8) + raw_sensor_data[i*SENSOR_BYTES+0]
            y_raw = (raw_sensor_data[i*SENSOR_BYTES+3] << 8) + raw_sensor_data[i*SENSOR_BYTES+2]
            z_raw = (raw_sensor_data[i*SENSOR_BYTES+5] << 8) + raw_sensor_data[i*SENSOR_BYTES+4]

            x = 100 * np.int16(x_raw) / 6842
            y = 100 * np.int16(y_raw) / 6842
            z = 100 * np.int16(z_raw) / 6842

            sensor_data[i] = [x, y, z]
        return sensor_data

    def read_thread(self):
        while self.reading and self.m_serial and self.m_serial.isOpen():
            try:
                raw = self.m_serial.read(SENSOR_NUM * SENSOR_BYTES)
                if len(raw) != SENSOR_NUM * SENSOR_BYTES:
                    print("⚠️ 資料長度不足，略過")
                    continue
                sensor_data = self.parse_raw_sensor_data(raw)
                print("📥 Data:", sensor_data.shape)
                if self.data_callback:
                    self.data_callback(sensor_data)
            except Exception as e:
                print("❗Binary Read Error:", e)
            time.sleep(0.01)

    def close(self, sig=None, frame=None):
        self.reading = False
        if self.m_serial and self.m_serial.isOpen():
            self.m_serial.close()
        print("🔌 串口關閉")
