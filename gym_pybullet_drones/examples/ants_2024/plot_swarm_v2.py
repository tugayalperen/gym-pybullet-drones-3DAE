import matplotlib.pyplot as plt
import numpy as np

class SwarmPlotter:
    def __init__(self, n_agents, x_lim, y_lim, z_lim):
        self.n_agents = n_agents
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        # Initialize scatter plot
        self.scat = self.ax.scatter([], [], [], color="black", s=15)
        # Initialize quiver plot
        self.quiv = self.ax.quiver([], [], [], [], [], [], color="black")
        self.ax.set_xlim(0, x_lim)
        self.ax.set_ylim(0, y_lim)
        self.ax.set_zlim(0, z_lim)

    def update_plot(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs):
        # Update scatter plot data
        self.scat._offsets3d = (pos_xs, pos_ys, pos_zs)
        # Update quiver plot data
        self.quiv.remove()  # Currently necessary due to Matplotlib's limitations with 3D quiver updates
        self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="black", length=0.2)
        plt.pause(0.0000001)  # Adjust this value as needed for your visualization needs
