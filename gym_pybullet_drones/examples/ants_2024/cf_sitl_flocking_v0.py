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
import time
import argparse
import numpy as np
import csv

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from flocking_utils import FlockingUtils
import pybullet as p


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results'
DURATION_SEC = 10

NUM_DRONES = 1
init_center_x = 0
init_center_y = 0
init_center_z = 1.0
spacing = 0.8

f_util = FlockingUtils(NUM_DRONES, init_center_x, init_center_y, init_center_z, spacing)
pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions(5563)

INIT_XYZ = np.zeros([NUM_DRONES, 3])
INIT_XYZ[:, 0] = pos_xs
INIT_XYZ[:, 1] = pos_ys
INIT_XYZ[:, 2] = pos_zs
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
                     initial_rpys=INIT_XYZ,
                     physics=ARGS.physics,
                     neighbourhood_radius=10,
                     pyb_freq=ARGS.simulation_freq_hz,
                     ctrl_freq=ARGS.control_freq_hz,
                     gui=ARGS.gui,
                     user_debug_gui=ARGS.user_debug_gui
                         )

    # ctrl = CTBRControl(drone_model=drone)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=NUM_DRONES,
                    output_folder=output_folder,
                    )

    # #### Run the simulation ####################################
    # delta = 75  # 3s @ 25hz control loop
    # trajectory = [[0, 0, 0] for i in range(delta)] + \
    #              [[0, 0, i / delta] for i in range(delta)] + \
    #              [[i / delta, 0, 1] for i in range(delta)] + \
    #              [[1, i / delta, 1] for i in range(delta)] + \
    #              [[1 - i / delta, 1, 1] for i in range(delta)] + \
    #              [[0, 1 - i / delta, 1] for i in range(delta)] + \
    #              [[0, 0, 1 - i / delta] for i in range(delta)]

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    for i in range(0, int(ARGS.duration_sec * env.CTRL_FREQ)):
        obs, reward, done, info, _ = env.step(action)
        states = env._getDroneStateVector(0)
        pos_x = states[0]
        pos_y = states[1]
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-30, cameraPitch=-70,
                                     cameraTargetPosition=[pos_xs[0], pos_ys[0], 0])
        # vel_cmd = np.array([0.3 * np.sin(2 * np.pi * i / (2 * env.CTRL_FREQ)),
        #                     0.3 * np.sin(2 * np.pi * i / (2 * env.CTRL_FREQ)),
        #                     0.0])
        if i < int(ARGS.duration_sec * env.CTRL_FREQ) / 4:
            vel_cmd = np.array([0.15, 0.0, 0.0])
        elif int(ARGS.duration_sec * env.CTRL_FREQ) / 4 < i < 2 * int(ARGS.duration_sec * env.CTRL_FREQ) / 4:
            vel_cmd = np.array([0.0, 0.15, 0.0])
        elif 2 * int(ARGS.duration_sec * env.CTRL_FREQ) / 4 < i < 3 * int(ARGS.duration_sec * env.CTRL_FREQ) / 4:
            vel_cmd = np.array([-0.15, 0.0, 0.0])
        else:
            vel_cmd = np.array([0.0, -0.15, 0.0])

        # pos_cmd = np.array([pos_x + 0.3 * np.sin(2 * np.pi * i / (2 * env.CTRL_FREQ)),
        #                     pos_y + 0.3 * np.sin(2 * np.pi * i / (2 * env.CTRL_FREQ)),
        #                     1.0])
        pos_cmd = vel_cmd*(1/env.CTRL_FREQ) + np.array([pos_x, pos_y, 1.0])
        if i == 0:
            vel_cmd_old = vel_cmd

        action[0], _, _ = ctrl[0].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                           state=obs[0],
                                                           target_pos=pos_cmd,
                                                           target_vel=vel_cmd,
                                                           target_rpy=np.array([0, 0, 0])
                                                               )

        #### Log the simulation ####################################
        for j in range(NUM_DRONES):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=obs[j]
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("beta")  # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


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
