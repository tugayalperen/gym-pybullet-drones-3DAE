import numpy as np
import plot_swarm_v2


class FlockingUtils:
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        self.n_agents = n_agents
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.spacing = spacing

        self.boun_thresh = 0.5
        self.boun_x = 20
        self.boun_y = 20
        self.boun_z = 20
        self.sensing_range = 3.0
        self.sigma = 0.5
        # self.sigma = 0.3
        self.sigmas = self.sigma * np.ones(n_agents)
        self.sigmas_b = 0.05 * np.ones(n_agents)
        self.epsilon = 12.0
        self.alpha = 0.05
        self.beta = 2.0
        self.k1 = 0.5
        # self.k1 = 0.3
        self.k2 = 0.1
        self.umax_const = 0.3
        # self.umax_const = 0.1
        self.wmax = 1.5708*2
        self.h_alignment = True
        self.dt = 0.042
        self.noise_pos = 0.05
        self.noise_h = np.pi / 72
        self.rng = np.random.default_rng(1234)
        self.mean_noise = 0.5


        self.d_ij = np.zeros([n_agents, n_agents])
        self.pos_h_xc = np.zeros(n_agents)
        self.pos_h_yc = np.zeros(n_agents)
        self.pos_h_zc = np.zeros(n_agents)
        self.ij_ang_x = np.zeros([n_agents, n_agents])
        self.ij_ang_y = np.zeros([n_agents, n_agents])
        self.ij_ang_z = np.zeros([n_agents, n_agents])
        self.f_x = np.zeros(n_agents)
        self.f_y = np.zeros(n_agents)
        self.f_z = np.zeros(n_agents)
        self.fa_x = np.zeros(n_agents)
        self.fa_y = np.zeros(n_agents)
        self.fa_z = np.zeros(n_agents)
        self.u = np.zeros(n_agents)
        self.w = np.zeros(n_agents)

        map_x, map_y, map_z = [np.linspace(-1, 1, 150) for _ in range(3)]
        X, Y, Z = np.meshgrid(map_x, map_y, map_z)
        self.map_3d = 255 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * self.sigma ** 2))
        self.grad_const_x, self.grad_const_y, self.grad_const_z = [150 / boun for boun in (self.boun_x, self.boun_y, self.boun_z)]

        self.plotter = plot_swarm_v2.SwarmPlotter(self.n_agents, self.boun_x, self.boun_y, self.boun_z)

    # def initialize_positions(self, random_seed):
    #     rng = np.random.default_rng(random_seed)
    #
    #     # Approximate equal distribution in 3D space
    #     def approximate_distribution(n):
    #         for z in range(int(n ** (1 / 3)), 0, -1):
    #             if n % z == 0:
    #                 rest = n // z
    #                 for y in range(int(rest ** 0.5), 0, -1):
    #                     if rest % y == 0:
    #                         x = rest // y
    #                         return x, y, z
    #         return 1, 1, n  # Fallback for n = 1
    #
    #     n_points_x, n_points_y, n_points_z = approximate_distribution(self.n_agents)
    #
    #     spacing = self.spacing
    #     init_x = self.center_x
    #     init_y = self.center_y
    #     init_z = self.center_z
    #
    #     x_min, x_max = init_x, init_x + n_points_x * spacing - spacing
    #     y_min, y_max = init_y, init_y + n_points_y * spacing - spacing
    #     z_min, z_max = init_z, init_z + n_points_z * spacing - spacing
    #
    #     x_values = np.linspace(x_min, x_max, n_points_x)
    #     y_values = np.linspace(y_min, y_max, n_points_y)
    #     z_values = np.linspace(z_min, z_max, n_points_z)
    #     xx, yy, zz = np.meshgrid(x_values, y_values, z_values)
    #
    #     total_points = n_points_x * n_points_y * n_points_z
    #     pos_xs = xx.ravel() + (rng.random(total_points) * spacing * 0.5) - spacing * 0.25
    #     pos_ys = yy.ravel() + (rng.random(total_points) * spacing * 0.5) - spacing * 0.25
    #     pos_zs = zz.ravel() + (rng.random(total_points) * spacing * 0.5) - spacing * 0.25
    #
    #     theta = np.random.uniform(0, 2 * np.pi, self.n_agents)
    #     phi = np.random.uniform(0, np.pi, self.n_agents)
    #
    #     self.pos_h_xc = np.sin(phi) * np.cos(theta)
    #     self.pos_h_yc = np.sin(phi) * np.sin(theta)
    #     self.pos_h_zc = np.cos(phi)
    #
    #     pos_h_m = np.sqrt(np.square(self.pos_h_xc) + np.square(self.pos_h_yc) + np.square(self.pos_h_zc))
    #
    #     return pos_xs, pos_ys, pos_zs, self.pos_h_xc, self.pos_h_yc, self.pos_h_zc

    def initialize_positions(self):
        """
        Place agents in a 3D grid around an initialization point with specified spacing and noise.

        Parameters:
            num_agents (int): Number of agents to place.
            init_pos (tuple): Initialization point (x, y, z).
            spacing (float): Spacing between agents.
            mean_noise (float): Mean value of noise to apply to positions.

        Returns:
            np.array: 3D positions of agents.
        """
        # Approximate cube root to start searching for dimensions
        num_agents = self.n_agents
        spacing = self.spacing
        init_pos = (self.center_x, self.center_y, self.center_z)
        mean_noise = self.mean_noise

        cube_root = round(num_agents ** (1 / 3))

        # Find dimensions that fill a space as equally as possible, even if some agents are left out
        best_diff = float('inf')
        for x in range(cube_root, 0, -1):
            for y in range(x, 0, -1):
                z = int(np.ceil(num_agents / (x * y)))
                total_agents = x * y * z
                diff = max(abs(x - y), abs(y - z), abs(x - z))
                if diff < best_diff and total_agents >= num_agents:
                    best_diff = diff
                    dimensions = (x, y, z)

        # Generate grid positions
        grid_positions = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]].reshape(3, -1).T
        grid_positions = grid_positions * spacing

        # Center the grid around the init_pos
        offset = np.array(init_pos) - (np.array(dimensions) * spacing / 2)
        grid_positions += offset

        # Apply noise
        noise = np.random.normal(loc=mean_noise, scale=mean_noise / 3, size=grid_positions.shape)
        grid_positions += noise

        theta = np.random.uniform(0, 2 * np.pi, self.n_agents)
        phi = np.random.uniform(0, np.pi, self.n_agents)

        self.pos_h_xc = np.sin(phi) * np.cos(theta)
        self.pos_h_yc = np.sin(phi) * np.sin(theta)
        self.pos_h_zc = np.cos(phi)

        return (grid_positions[:num_agents, 0], grid_positions[:num_agents, 1], grid_positions[:num_agents, 2],
                self.pos_h_xc, self.pos_h_yc, self.pos_h_zc)

    def calculate_rotated_vector_batch(self, X1, Y1, Z1, X2, Y2, Z2, wdt):
        # Convert inputs to NumPy arrays if they aren't already
        # X1, Y1, Z1, X2, Y2, Z2, wdt = [np.asarray(a) for a in [X1, Y1, Z1, X2, Y2, Z2, wdt]]

        # Stack the original and target vectors for batch processing
        vector1 = np.stack([X1, Y1, Z1], axis=-1)
        vector2 = np.stack([X2, Y2, Z2], axis=-1)

        # Calculate magnitudes for normalization
        original_magnitude = np.linalg.norm(vector1, axis=1, keepdims=True)
        vector2_magnitude = np.linalg.norm(vector2, axis=1, keepdims=True)

        # Normalize and scale vector2
        vector2_normalized_scaled = vector2 * (original_magnitude / vector2_magnitude)

        # Calculate the normal vector for each pair
        normal_vector = np.cross(vector1, vector2_normalized_scaled)
        normal_magnitude = np.linalg.norm(normal_vector, axis=1, keepdims=True)

        # Avoid division by zero by ensuring non-zero magnitude
        normal_vector /= np.where(normal_magnitude > 0, normal_magnitude, 1)

        # Rodrigues' rotation formula for batch
        k_cross_vector1 = np.cross(normal_vector, vector1)
        cos_theta = np.cos(wdt)[:, np.newaxis]
        sin_theta = np.sin(wdt)[:, np.newaxis]
        one_minus_cos_theta = (1 - cos_theta)

        dot_product = np.sum(normal_vector * vector1, axis=1, keepdims=True)
        v_rot = vector1 * cos_theta + k_cross_vector1 * sin_theta + normal_vector * dot_product * one_minus_cos_theta

        return v_rot.T

    def calculate_av_heading(self, x_components, y_components, z_components):
        # Normalize each vector and sum them to get an average direction
        normalized_vectors = []
        for x, y, z in zip(x_components, y_components, z_components):
            vec = np.array([x, y, z])
            norm = np.linalg.norm(vec)
            if norm != 0:  # Avoid division by zero
                normalized_vectors.append(vec / norm)
            else:
                normalized_vectors.append(vec)  # Keep zero vectors as is

        # Calculate the average vector (sum of normalized vectors)
        sum_of_normalized_vectors = np.sum(normalized_vectors, axis=0)

        # Normalize the sum to get the unit vector with the average direction
        unit_vector_average_direction = sum_of_normalized_vectors / np.linalg.norm(sum_of_normalized_vectors)

        return unit_vector_average_direction

    def detect_bounds(self, pos_x, pos_y, pos_z):
        result_x = np.zeros_like(pos_x)
        result_y = np.zeros_like(pos_y)
        result_z = np.zeros_like(pos_z)

        result_x[pos_x < self.boun_thresh] = 1
        result_x[pos_x > self.boun_x - self.boun_thresh] = -1

        result_y[pos_y < self.boun_thresh] = 1
        result_y[pos_y > self.boun_y - self.boun_thresh] = -1

        result_z[pos_z < self.boun_thresh] = 1
        result_z[pos_z > self.boun_z - self.boun_thresh] = -1

        return result_x, result_y, result_z

    def calc_dij(self, pos_xs, pos_ys, pos_zs):
        self.d_ij = np.hypot(np.hypot(pos_xs[:, None] - pos_xs, pos_ys[:, None] - pos_ys), pos_zs[:, None] - pos_zs)
        self.d_ij[(self.d_ij > self.sensing_range) | (self.d_ij == 0)] = np.inf
        self.d_ij_noise = self.d_ij + self.rng.uniform(-self.noise_pos, self.noise_pos, (self.n_agents, self.n_agents)) * self.dt

    def calc_ang_ij(self, pos_xs, pos_ys, pos_zs):
        print((pos_xs - pos_xs[:, None]) / self.d_ij)
        self.ij_ang_x = np.arccos((pos_xs - pos_xs[:, None]) / self.d_ij)
        self.ij_ang_y = np.arccos((pos_ys - pos_ys[:, None]) / self.d_ij)
        self.ij_ang_z = np.arccos((pos_zs - pos_zs[:, None]) / self.d_ij)

    def calc_grad_vals(self, pos_xs, pos_ys, pos_zs):
        grad_x = np.clip(np.ceil(pos_xs * self.grad_const_x).astype(int), 0, 149)
        grad_y = np.clip(np.ceil(pos_ys * self.grad_const_y).astype(int), 0, 149)
        grad_z = np.clip(np.ceil(pos_zs * self.grad_const_z).astype(int), 0, 149)

        grad_vals = self.map_3d[grad_x, grad_y, grad_z]
        grad_vals = np.clip(grad_vals, 0, 255)
        # self.sigmas = self.sigma + (grad_vals / 255.0) * 0.5

    def calc_p_forces(self):
        forces = -self.epsilon * (2 * (self.sigmas[:, np.newaxis] ** 4 / self.d_ij_noise ** 5) -
                                  (self.sigmas[:, np.newaxis] ** 2 / self.d_ij_noise ** 3))

        cos_ij_ang_x = np.cos(self.ij_ang_x)
        cos_ij_ang_y = np.cos(self.ij_ang_y)
        cos_ij_ang_z = np.cos(self.ij_ang_z)

        self.f_x = self.alpha * np.sum(forces * cos_ij_ang_x, axis=1)
        self.f_x = np.where(self.f_x == 0, 0.00001, self.f_x)

        self.f_y = self.alpha * np.sum(forces * cos_ij_ang_y, axis=1)
        self.f_y = np.where(self.f_y == 0, 0.00001, self.f_y)

        self.f_z = self.alpha * np.sum(forces * cos_ij_ang_z, axis=1)
        self.f_z = np.where(self.f_z == 0, 0.00001, self.f_z)

    def calc_alignment_forces(self):
        av_heading = self.calculate_av_heading(self.pos_h_xc, self.pos_h_yc, self.pos_h_zc)

        self.fa_x = int(self.h_alignment) * self.beta * av_heading[0]
        self.fa_y = int(self.h_alignment) * self.beta * av_heading[1]
        self.fa_z = int(self.h_alignment) * self.beta * av_heading[2]

    def calc_boun_rep(self, pos_xs, pos_ys, pos_zs):
        d_bxi = np.minimum(np.abs(self.boun_x - pos_xs), pos_xs)
        d_byi = np.minimum(np.abs(self.boun_y - pos_ys), pos_ys)
        d_bzi = np.minimum(np.abs(self.boun_z - pos_zs), pos_zs)

        close_to_bound_x = np.logical_or(pos_xs < self.boun_thresh, pos_xs > (self.boun_x - self.boun_thresh))
        close_to_bound_y = np.logical_or(pos_ys < self.boun_thresh, pos_ys > (self.boun_y - self.boun_thresh))
        close_to_bound_z = np.logical_or(pos_zs < self.boun_thresh, pos_zs > (self.boun_z - self.boun_thresh))

        if np.any(close_to_bound_x) or np.any(close_to_bound_y) or np.any(close_to_bound_z):
            db_bxi, db_byi, db_bzi = self.detect_bounds(pos_xs, pos_ys, pos_zs)

            boundary_effect_x = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_bxi ** 5) -
                                                (self.sigmas_b ** 2 / d_bxi ** 3))
            boundary_effect_y = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_byi ** 5) -
                                                (self.sigmas_b ** 2 / d_byi ** 3))
            boundary_effect_z = -self.epsilon * 5 * (2 * (self.sigmas_b ** 4 / d_bzi ** 5) -
                                                (self.sigmas_b ** 2 / d_bzi ** 3))

            boundary_effect_x[boundary_effect_x < 0] = 0.0
            boundary_effect_y[boundary_effect_y < 0] = 0.0
            boundary_effect_z[boundary_effect_z < 0] = 0.0

            self.f_x += self.fa_x + boundary_effect_x * db_bxi
            self.f_y += self.fa_y + boundary_effect_y * db_byi
            self.f_z += self.fa_z + boundary_effect_z * db_bzi
        else:
            self.f_x += self.fa_x
            self.f_y += self.fa_y
            self.f_z += self.fa_z

    def calc_u_w(self):
        f_mag = np.sqrt(np.square(self.f_x) + np.square(self.f_y) + np.square(self.f_z))
        f_mag = np.where(f_mag == 0, 0.00001, f_mag)
        dot_f_h = self.f_x * self.pos_h_xc + self.f_y * self.pos_h_yc + self.f_z * self.pos_h_zc

        cos_dot_f_h = np.clip(dot_f_h / f_mag, -1.0, 1.0)

        ang_f_h = np.arccos(cos_dot_f_h)
        # ang_f_h += self.rng.uniform(-self.noise_h, self.noise_h, self.n_agents) * self.dt

        self.u = self.k1 * f_mag * np.cos(ang_f_h) + 0.05
        self.w = self.k2 * f_mag * np.sin(ang_f_h)

        self.u = np.clip(self.u, 0, self.umax_const)
        self.w = np.clip(self.w, -self.wmax, self.wmax)

        return self.u

    def get_heading(self):
        pos_h_m = np.sqrt(np.square(self.pos_h_xc) + np.square(self.pos_h_yc) + np.square(self.pos_h_zc))
        pos_hxs = np.arccos(self.pos_h_xc / pos_h_m)
        pos_hys = np.arccos(self.pos_h_yc / pos_h_m)
        pos_hzs = np.arccos(self.pos_h_zc / pos_h_m)

        return pos_hxs, pos_hys, pos_hzs

    def update_heading(self):
        v_rot = self.calculate_rotated_vector_batch(
            self.pos_h_xc, self.pos_h_yc, self.pos_h_zc, self.f_x, self.f_y, self.f_z, self.w * self.dt)

        self.pos_h_xc = v_rot[0, :]
        self.pos_h_yc = v_rot[1, :]
        self.pos_h_zc = v_rot[2, :]

    def plot_swarm(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs):
        self.plotter.update_plot(pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs)


class FlockingUtilsVec:
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        self.n_agents = n_agents
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.spacing = spacing

        self.boun_thresh = 0.5
        self.boun_x = 20
        self.boun_y = 20
        self.boun_z = 20
        self.center_pos = np.array((self.boun_x, self.boun_y, self.boun_z)).reshape((3, 1)) / 2
        self.sensing_range = 3.0
        self.sigma = 0.5
        # self.sigma = 0.3
        self.sigmas = self.sigma * np.ones(n_agents)
        self.sigmas2 = self.sigmas**2
        self.sigmas4 = self.sigmas**2
        self.sigmas_b = 0.05 * np.ones(n_agents)
        self.sigmas_b2 = self.sigmas_b**2
        self.sigmas_b4 = self.sigmas_b**4
        self.epsilon = 12.0
        self.alpha = 0.05
        self.beta = 2.0
        self.k1 = 0.5
        # self.k1 = 0.3
        self.k2 = 0.1
        self.umax_const = 0.3
        # self.umax_const = 0.1
        self.wmax = 1.5708*2
        self.h_alignment = True
        self.dt = 0.042
        self.noise_pos = 0.05
        self.noise_h = np.pi / 72
        self.rng = np.random.default_rng(1234)
        self.mean_noise = 0.5


        self.headings = None
        self.positions = None
        self.fa_vec = np.zeros((3, n_agents))
        self.force_vecs = np.zeros((3, n_agents))
        self.dist_ratio = None
        self.dist_norm = None
        self.forces = None
        self.u = np.zeros(n_agents)
        self.w = np.zeros(n_agents)

        map_x, map_y, map_z = [np.linspace(-1, 1, 150) for _ in range(3)]
        X, Y, Z = np.meshgrid(map_x, map_y, map_z)
        self.map_3d = 255 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * self.sigma ** 2))
        self.grad_const_x, self.grad_const_y, self.grad_const_z = [150 / boun for boun in (self.boun_x, self.boun_y, self.boun_z)]

        self.plotter = plot_swarm_v2.SwarmPlotter(self.n_agents, self.boun_x, self.boun_y, self.boun_z)

    def initialize_positions(self):
        """
        Place agents in a 3D grid around an initialization point with specified spacing and noise.

        Parameters:
            num_agents (int): Number of agents to place.
            init_pos (tuple): Initialization point (x, y, z).
            spacing (float): Spacing between agents.
            mean_noise (float): Mean value of noise to apply to positions.

        Returns:
            np.array: 3D positions of agents.
        """
        # Approximate cube root to start searching for dimensions
        num_agents = self.n_agents
        spacing = self.spacing
        init_pos = (self.center_x, self.center_y, self.center_z)
        mean_noise = self.mean_noise

        cube_root = round(num_agents ** (1 / 3))

        # Find dimensions that fill a space as equally as possible, even if some agents are left out
        best_diff = float('inf')
        for x in range(cube_root, 0, -1):
            for y in range(x, 0, -1):
                z = int(np.ceil(num_agents / (x * y)))
                total_agents = x * y * z
                diff = max(abs(x - y), abs(y - z), abs(x - z))
                if diff < best_diff and total_agents >= num_agents:
                    best_diff = diff
                    dimensions = (x, y, z)

        # Generate grid positions
        grid_positions = np.mgrid[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]].reshape(3, -1).T
        grid_positions = grid_positions * spacing

        # Center the grid around the init_pos
        offset = np.array(init_pos) - (np.array(dimensions) * spacing / 2)
        grid_positions += offset

        # Apply noise
        noise = np.random.normal(loc=mean_noise, scale=mean_noise / 3, size=grid_positions.shape)
        grid_positions += noise

        theta = np.random.uniform(0, 2 * np.pi, self.n_agents)
        phi = np.random.uniform(0, np.pi, self.n_agents)

        pos_h_xc = np.sin(phi) * np.cos(theta)
        pos_h_yc = np.sin(phi) * np.sin(theta)
        pos_h_zc = np.cos(phi)

        self.positions = grid_positions.T
        self.headings = np.stack((pos_h_xc, pos_h_yc, pos_h_zc))

        return (self.positions, self.headings)

    def calculate_rotated_vector_batch(self, V1, V2, wdt):
        e3 = np.cross(V1.T, V2.T)

        # Rodrigues' rotation formula for batch
        e2 = np.cross(e3, V1.T).T
        e2_norm = np.linalg.norm(e2, axis=0)
        e2_norm[e2_norm <= 0] = 1
        e2 /= e2_norm

        v_rot = V1 * np.cos(wdt) + e2 * np.sin(wdt) * np.linalg.norm(V1, axis=0)
        return v_rot

    def calculate_av_heading(self, heading_vecs):
        # Normalize each vector and sum them to get an average direction
        heading_vecs_normalised = heading_vecs / np.linalg.norm(heading_vecs, axis=0)

        # Calculate the average vector (sum of normalized vectors)
        sum_of_heading_vecs_normalised = np.sum(heading_vecs_normalised, axis=1)

        # Normalize the sum to get the unit vector with the average direction
        unit_vector_average_direction = sum_of_heading_vecs_normalised / np.linalg.norm(sum_of_heading_vecs_normalised)
        return unit_vector_average_direction.reshape(3, 1)

    def calc_dij(self, positions):
        delta_x = positions[0] - positions[0][:, np.newaxis]
        delta_y = positions[1] - positions[1][:, np.newaxis]
        delta_z = positions[2] - positions[2][:, np.newaxis]
        dist = np.stack((delta_x, delta_y, delta_z))
        self.dist_norm = np.linalg.norm(dist, axis=0)
        self.dist_norm[(self.dist_norm > sensing_range) | (self.dist_norm == 0)] = np.inf
        self.dist_ratio = dist / self.dist_norm

    def calc_p_forces(self):
        self.forces = -self.epsilon * (2 * (self.sigmas4 / self.dist_norm ** 5) - (self.sigmas2 / self.dist_norm ** 3))
        self.force_vecs = self.alpha * np.sum(self.forces * self.dist_ratio, axis=2)

    def calc_alignment_forces(self):
        av_heading = self.calculate_av_heading(self.headings)
        self.fa_vec = int(self.h_alignment) * self.beta * av_heading

    def calc_boun_rep(self, positions):
        d_bounds = self.center_pos - np.abs(positions - self.center_pos)
        in_bounds = d_bounds < self.boun_thresh
        if np.any(in_bounds):
            boundary_effect = np.maximum(-self.epsilon * 5 * (2 * (self.sigmas_b4 / d_bounds ** 5) - (self.sigmas_b2 / d_bounds ** 3)), 0)

            f_b =  boundary_effect * in_bounds * -np.sign(positions - self.center_pos)
            self.force_vecs += self.fa_vec + f_b
        else:
            self.force_vecs += self.fa_vec

    def calc_u_w(self):
        f_mag = np.linalg.norm(self.force_vecs, axis=0)
        f_mag = np.maximum(f_mag, 0.00001)

        dot_f_h = np.sum(self.force_vecs*self.headings, axis=0)

        cos_dot_f_h = dot_f_h / (f_mag * np.linalg.norm(headings, axis=0))
        ang_f_h = np.arccos(cos_dot_f_h)

        self.u = self.k1 * f_mag * np.cos(ang_f_h) + 0.05
        self.w = self.k2 * f_mag * np.sin(ang_f_h)

        self.u = np.clip(u, 0, self.umax_const)
        self.w = np.clip(w, -self.wmax, self.wmax)
        return self.u

    def get_headings(self):
        return self.headings

    def update_heading(self):
        self.headings = self.calculate_rotated_vector_batch(self.headings, self.force_vecs, self.w * self.dt)

    def plot_swarm(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs):
        self.plotter.update_plot(pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs)
