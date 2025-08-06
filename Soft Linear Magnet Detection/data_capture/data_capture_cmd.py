import serial
import serial.tools.list_ports
import time, threading
import signal
from queue import Queue
import numpy as np
import keyboard

SENSOR_NUM = 17*5
SENSOR_BYTES = 3*2
PACKET_SIZE = SENSOR_NUM*SENSOR_BYTES + 4

class DataLoaderOnline():
    def __init__(self):
        super(DataLoaderOnline, self).__init__()

        ports_list = list(serial.tools.list_ports.comports())
        if len(ports_list) <= 0:
            print("Don't have available ports")
        else:
            print("The available ports are as follows: ")
            for comport in ports_list:
                print(comport.device, comport.description)
        
        signal.signal(signal.SIGINT, self.close)
        time.sleep(2)
        self.queue = Queue(maxsize=5)
        self.t1 = threading.Thread(target=self.read_thread)


    def open(self, port_name='COM6'):
        # self.m_serial = serial.Serial(port_name, 115200)
        # if self.m_serial.is_open:
        #     send_value = 'Run\r\n'      # Run Hex
        #     self.send_data(send_value)
        #     return True
        # else:
        #     return False
        try:
            self.m_serial = serial.Serial(port_name, 115200)
            if self.m_serial.is_open:
                send_value = 'Run\r\n'
                self.send_data(send_value)
                print(f"✅ 串口 {port_name} 已打開")
                return True
        except serial.SerialException as e:
            print(f"❌ 無法打開串口 {port_name}: {e}")
            self.m_serial = None
            return False

        if self.m_serial is not None and self.m_serial.is_open:
            try:
                self.send_data("Run\r\n")
            except Exception as e:
                print(f"⚠️ 發送 Run 指令失敗：{e}")
            return True
        return False



    def close(self, signal=None, frame=None):
        self.flag_running = False
        # ✅ 安全 join：只有執行緒存在且已啟動才 join
        if hasattr(self, "t1") and self.t1.is_alive():
            self.t1.join()

        if self.m_serial is not None and self.m_serial.is_open:
            send_value = 'Stop\r\n'
            self.m_serial.write(send_value.encode('utf-8'))
            self.m_serial.close()
        print('Close Port.\r\n')
        exit()


    def send_data(self, send_value):
        if self.m_serial is None or not self.m_serial.is_open:
            print('Send data failed as serial has not opened.\r\n')
            return
        send_value = send_value.encode('utf-8')
        self.m_serial.write(send_value)

    # def start_read(self):
    #     self.flag_running = True
    #     self.t1.start()
    #     self.geo_cnt = 0
    #     self.geo_data = np.zeros((SENSOR_NUM, 3), dtype=np.double)
    def start_reading(self, callback):
        self.flag_running = True
        self.geo_cnt = 0
        self.geo_data = np.zeros((SENSOR_NUM, 3), dtype=np.double)
        self.callback = callback
        self.t1.start()


    def raw_to_sensor(self, raw_sensor_data):
        '''
        将传感器原始数据转换成读数(uT)
        '''
        sensor_data = np.zeros((0, 3))
        def decode_value(low, high):
            return np.array((high << 8) + low, dtype=np.uint16).astype(np.int16)
        for i in range(SENSOR_NUM):
            x_raw = decode_value(raw_sensor_data[i*SENSOR_BYTES+0], raw_sensor_data[i*SENSOR_BYTES+1])
            y_raw = decode_value(raw_sensor_data[i*SENSOR_BYTES+2], raw_sensor_data[i*SENSOR_BYTES+3])
            z_raw = decode_value(raw_sensor_data[i*SENSOR_BYTES+4], raw_sensor_data[i*SENSOR_BYTES+5])

            x = 100 * x_raw / 6842
            y = 100 * y_raw / 6842
            z = 100 * z_raw / 6842

            temp = np.array([x, y, z]).reshape(1, 3)
            sensor_data = np.append(sensor_data, temp, axis=0)
        return sensor_data


    def read_thread(self):
        print("🧵 進入讀取執行緒...")
        while self.flag_running:
            if self.m_serial is not None and self.m_serial.is_open:
                data = self.m_serial.read(PACKET_SIZE)
                # print(len(data))      # 514
                if (len(data) > 4):
                    if (data[-4] == 222) and (data[-3] == 173) and \
                       (data[-2] == 190) and (data[-1] == 239):
                        raw_sensor_data = data[0:-4]

                        sensor_data = self.raw_to_sensor(raw_sensor_data)

                        if self.queue.full():       # 如果队列满了，先删再入
                            self.queue.get()
                        self.queue.put(sensor_data)
                        if self.geo_cnt < 50:
                            self.geo_data += sensor_data
                            self.geo_cnt += 1
                            if self.geo_cnt == 50:
                                self.geo_data = self.geo_data / 50
                        
                        if hasattr(self, 'callback') and callable(self.callback):
                            # 处理地磁校正
                            self.callback((self.geo_cnt if self.geo_cnt < 50 else 0,
                                            sensor_data - self.geo_data if self.geo_cnt >= 50 else sensor_data))
            else:
                time.sleep(0.1)



    def get_data(self):
        data = self.queue.get()

        if self.geo_cnt < 50:    # 进入去地磁操作
            self.geo_data = self.geo_data + data
            self.geo_cnt = self.geo_cnt + 1
            if self.geo_cnt == 50:
                self.geo_data = self.geo_data / 50
            
            return (self.geo_cnt, data)
        else:
            return (0, data-self.geo_data)



flag_save = False
def save_data():
    global flag_save

    print("Save")
    flag_save = True

if __name__ == "__main__":
    data_loader = DataLoaderOnline()
    if not data_loader.open('COM6'):
        print('Open Serial failed.\r\n')
        exit()

    data_loader.start_reading(lambda x: None)

    keyboard.add_hotkey("ctrl+s", save_data) # lambda : flag_save = True
    try:
        while True:
            flag, data = data_loader.get_data()
            if flag == 0:
                print(data[-5])
            else:
                print(data)

            if flag_save:
                # 变更传感器数据的格式: 一行对应一帧
                print("Save Data")
                data = data.reshape(1, -1)
                if True:        # 同时也保存下位姿
                    magnet_pos = np.array([30, 30, 65]) * 1e-3      # unit: m
                    magnet_ori = np.array([0, 0, 1])
                    Bt = 7.68750001e-08
                    data = np.append(data, np.concatenate((magnet_pos, magnet_ori)))
                    data = np.append(data, Bt)
                
                np.save("./dataset/real_30_30.npy", data)
                flag_save = False

    except KeyboardInterrupt:
        data_loader.close(None, None)
        print('Exit.\r\n')
