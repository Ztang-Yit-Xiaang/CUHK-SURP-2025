import numpy as np
data = np.load("dataset/captured_20250731_161948.npy")
data = data.reshape(-1, 85, 3)  # 确保数据是三维的
print(data.shape)  # 如果是 (0, 255) 或 (0,) 就是空的
# data1 = data[100, :]  # 取第100帧的数据
# print("第100帧的数据:", data1)
# data2 = data[228, :] 
# print("第228帧的数据:", data2)
# data3 = data[400, :]  
# print("第400帧的数据:", data3)
B_frame = data[200, :, :]  # 取第50帧的数据
print("B_frame shape:", B_frame)
print("B min:", np.min(B_frame))
print("B max:", np.max(B_frame))
print("B mean vector:", np.mean(B_frame, axis=0))