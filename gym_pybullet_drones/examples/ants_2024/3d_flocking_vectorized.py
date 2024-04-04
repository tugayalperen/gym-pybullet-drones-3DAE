"""CrazyFlie software-in-the-loop control example.

Setup
-----
Step 1: Clone pycffirmware from https://github.com/utiasDSL/pycffirmware
Step 2: Follow the install instructions for pycffirmware in its README

Example
-------
In terminal, run:
python gym_pybullet_drones/examples/cf.py

"""

###
# Logging is added!
###

import time
import argparse
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool
from flocking_utils import FlockingUtilsVec
import pybullet as p
from datetime import datetime


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results'
DURATION_SEC = 200

NUM_DRONES = 10
if_cube = True
init_center_x = 4
init_center_y = 2
init_center_z = 1
spacing = 0.8
self_log = True
save_dir = "./self_logs/"

f_util = FlockingUtilsVec(NUM_DRONES, init_center_x, init_center_y, init_center_z, spacing)
positions, headings = f_util.initialize_positions()

INIT_XYZ = positions.T
INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])


def run(
        drone=DEFAULT_DRONES,
        num_drones=NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        duration_sec=DURATION_SEC
):
    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=ARGS.num_drones,
                     initial_xyzs=INIT_XYZ,
                     initial_rpys=INIT_RPY,
                     physics=ARGS.physics,
                     # physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     pyb_freq=ARGS.simulation_freq_hz,
                     ctrl_freq=ARGS.control_freq_hz,
                     gui=ARGS.gui,
                     user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    if self_log:
        log_pos_xs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_ys = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_zs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hxs = np.zeros([NUM_DRONES, DURATION_SEC*DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hys = np.zeros([NUM_DRONES, DURATION_SEC * DEFAULT_CONTROL_FREQ_HZ])
        log_pos_hzs = np.zeros([NUM_DRONES, DURATION_SEC * DEFAULT_CONTROL_FREQ_HZ])

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    pos_x = np.zeros(NUM_DRONES)
    pos_y = np.zeros(NUM_DRONES)
    pos_z = np.zeros(NUM_DRONES)

    for i in range(0, int(ARGS.duration_sec * env.CTRL_FREQ)):
        obs, reward, done, info, _ = env.step(action)
        positions = env.pos().T

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-40,
                                     cameraTargetPosition=[pos_x[1], pos_y[1], pos_z[1] - 1])

        f_util.calc_dij(positions)
        f_util.calc_ang_ij(positions)
        f_util.calc_grad_vals(positions)
        f_util.calc_p_forces()
        f_util.calc_alignment_forces()
        f_util.calc_boun_rep(positions)
        u = f_util.calc_u_w()
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()

        if i % (env.CTRL_FREQ*1) == 0:
            f_util.plot_swarm(pos_x, pos_y, pos_z, pos_hxs, pos_hys, pos_hzs)

        vel_cmd = u*headings
        for j in range(NUM_DRONES):
            # pos_cmd = vel_cmd * (1 / env.CTRL_FREQ) + np.array([pos_x[j], pos_y[j], pos_z[j]])
            pos_cmd = positions.T
            action[j], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                              state=obs[j],
                                                              target_pos=pos_cmd,
                                                              target_vel=vel_cmd,
                                                              target_rpy=np.array([0, 0, 0])
                                                              )

        #### Log the simulation ####################################
        if self_log:
            log_pos_xs[:, i] = positions[0]
            log_pos_ys[:, i] = positions[1]
            log_pos_zs[:, i] = positions[2]
            log_pos_hxs[:, i] = headings[0]
            log_pos_hys[:, i] = headings[1]
            log_pos_hzs[:, i] = headings[2]

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    if self_log:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y|%H_%M_%S")
        filename_posx = save_dir + "log_pos_xs_" + dt_string + ".npy"
        filename_posy = save_dir + "log_pos_ys_" + dt_string + ".npy"
        filename_posz = save_dir + "log_pos_zs_" + dt_string + ".npy"
        filename_pos_hxs = save_dir + "log_pos_hxs_" + dt_string + ".npy"
        filename_pos_hys = save_dir + "log_pos_hys_" + dt_string + ".npy"
        filename_pos_hzs = save_dir + "log_pos_hzs_" + dt_string + ".npy"
        np.save(filename_posx, log_pos_xs)
        np.save(filename_posy, log_pos_ys)
        np.save(filename_posz, log_pos_zs)
        np.save(filename_pos_hxs, log_pos_hxs)
        np.save(filename_pos_hys, log_pos_hys)
        np.save(filename_pos_hzs, log_pos_hzs)


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: BETA)',
                        metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=NUM_DRONES, type=int, help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 500)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 25)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--duration_sec',default=DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
