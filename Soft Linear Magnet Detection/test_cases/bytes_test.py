
import serial
import numpy as np

SENSOR_NUM = 17*5
SENSOR_BYTES = 3*2
PACKET_SIZE = SENSOR_NUM*SENSOR_BYTES + 4

m_serial = serial.Serial("COM4", '115200', timeout=0.1)   # 1 seconds
if m_serial.isOpen():
    print("opened success")
    send_value = 'Run\r\n'      # Run Hex
    send_value = send_value.encode('utf-8')
    m_serial.write(send_value)

cnt = 0

while True:
    data = m_serial.read(PACKET_SIZE)
    print(len(data))
    if (len(data) > 4):
        print(data[-1])
        print(data[-2])
        print(data[-3])
        print(data[-4])
        if (data[-4] == 222) and (data[-3] == 173) and \
           (data[-2] == 190) and (data[-1] == 239):
            print("OK")
            raw_sensor_data = data[0:-4]
            break

print(raw_sensor_data)
# 解析
sensor_data = np.zeros((0, 3))
for i in range(SENSOR_NUM):
    x = np.int16((raw_sensor_data[i*SENSOR_BYTES+0] << 8) + raw_sensor_data[i*SENSOR_BYTES+1])/1711
    y = np.int16((raw_sensor_data[i*SENSOR_BYTES+2] << 8) + raw_sensor_data[i*SENSOR_BYTES+3])/1711
    z = np.int16((raw_sensor_data[i*SENSOR_BYTES+4] << 8) + raw_sensor_data[i*SENSOR_BYTES+5])/1711
    # print("%dS: %f, %f, %f" % (i, x, y, z))

    temp = np.array([x, y, z]).reshape(1, 3)
    sensor_data = np.append(sensor_data, temp, axis=0)

print(sensor_data)
print(sensor_data.shape)
