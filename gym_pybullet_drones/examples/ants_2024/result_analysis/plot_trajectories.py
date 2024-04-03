import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

experiment_date = "21_03_2024"
experiment_time = "15_04_05"

filename_base = "_" + experiment_date + "|" + experiment_time + ".npy"
filename_dir = "../self_logs/"
pos_xs = np.load(filename_dir + "log_pos_xs" + filename_base, allow_pickle=True).T
pos_ys = np.load(filename_dir + "log_pos_ys" + filename_base, allow_pickle=True).T
pos_zs = np.load(filename_dir + "log_pos_zs" + filename_base, allow_pickle=True).T
pos_hxs = np.load(filename_dir + "log_pos_hxs" + filename_base, allow_pickle=True).T
pos_hys = np.load(filename_dir + "log_pos_hys" + filename_base, allow_pickle=True).T
pos_hzs = np.load(filename_dir + "log_pos_hzs" + filename_base, allow_pickle=True).T
exp_length = pos_xs.shape[0]
num_agents = pos_xs.shape[1]
sensing_range = 3.0

min_d_ij = np.zeros(exp_length)



print("a")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
for i in range(5):
    ax.plot(pos_xs[:, i], pos_ys[:, i], pos_zs[:, i], label='Trajectory')

# Label axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Add a title and a legend
ax.set_title('3D Trajectory Plot')
ax.legend()

# Show the plot
plt.show()

sum_xc = np.zeros(exp_length)
sum_yc = np.zeros(exp_length)
sum_zc = np.zeros(exp_length)
order = np.zeros(exp_length)

for t in range(pos_hxs.shape[0]):
    _pos_xs = pos_xs[t, :]
    _pos_ys = pos_ys[t, :]
    _pos_zs = pos_zs[t, :]

    d_ij = np.hypot(np.hypot(_pos_xs[:, None] - _pos_xs, _pos_ys[:, None] - _pos_ys), _pos_zs[:, None] - _pos_zs)
    d_ij[(d_ij > sensing_range) | (d_ij == 0)] = np.inf
    min_d_ij[t] = np.min(d_ij)

    for agent in range(num_agents):
        sum_xc[t] += np.cos(pos_hxs[t, agent])
        sum_yc[t] += np.cos(pos_hys[t, agent])
        sum_zc[t] += np.cos(pos_hzs[t, agent])
    order[t] = np.sqrt(np.power(sum_xc[t], 2) + np.power(sum_yc[t], 2) + np.power(sum_zc[t], 2)) / num_agents

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(order)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(min_d_ij)
plt.show()





