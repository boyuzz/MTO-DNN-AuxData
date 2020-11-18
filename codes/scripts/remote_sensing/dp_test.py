from rdp import rdp
import numpy as np
import matplotlib.pyplot as plt

# 圆的基本信息
# 1.圆半径
r = 2.0
# 2.圆心坐标
a, b = (0., 0.)
theta = np.arange(0, 2*np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)

round = list(zip(x,y))
dp_round = rdp(round)
dp_x = np.array([dp[0] for dp in dp_round])
dp_y = np.array([dp[1] for dp in dp_round])

# print(rdp(round))
fig = plt.figure(figsize=(12,6))
axes = fig.add_subplot(121)
axes.plot(x, y) # 上半部
# axes.plot(x, -y) # 下半部

axes = fig.add_subplot(122)
axes.plot(dp_x, dp_y) # 上半部
# axes.plot(dp_x, -dp_y) # 下半部

plt.axis('equal')

plt.show()