import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('./')

# df = pd.read_csv("./results/volunteer3/fix_pose_sequ_120919_nogeo.txt",delimiter=",")
# df.to_csv("./results/volunteer3/fix_pose_sequ_120919_nogeo.csv", encoding='utf-8', index=False)

df = pd.read_csv("./results/volunteer3/fix_pose_sequ_120919_nogeo.csv")

fig = plt.figure(1)

ax1 = fig.add_subplot(111, projection='3d')
ax1.view_init(60,35)
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


index = int(len(df)/10)
x = list(df['x'])
y = list(df['y'])
z = list(df['z'])
ax1.set_zlim(min(z),max(z))

plt.ion()
plt.xlim(min(x), max(x))
plt.ylim(min(y),max(y))
#plt.zlim(0.05,0.23)
for k in range(int(0.3 * index), int(0.7 * index)):
    if k % 5 == 0:
        ax1 = plt.gca()
        ax1.scatter(x[k], y[k], z[k], c='royalblue')
        plt.draw()
        plt.show()
        plt.pause(0.0001)
plt.ioff()