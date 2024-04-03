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
# Only flocking, no logging
###

import time
import argparse
import numpy as np
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
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results'
DURATION_SEC = 10

NUM_DRONES = 5
if_cube = True
init_center_x = 1
init_center_y = 1
init_center_z = 1
spacing = 0.8

f_util = FlockingUtils(NUM_DRONES, init_center_x, init_center_y, init_center_z, spacing)
pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions(123)

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

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=NUM_DRONES,
                    output_folder=output_folder,
                    )

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    pos_x = np.zeros(NUM_DRONES)
    pos_y = np.zeros(NUM_DRONES)
    pos_z = np.zeros(NUM_DRONES)

    for i in range(0, int(ARGS.duration_sec * env.CTRL_FREQ)):
        obs, reward, done, info, _ = env.step(action)
        for j in range(NUM_DRONES):
            states = env._getDroneStateVector(j)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = states[2]

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-40,
                                         cameraTargetPosition=[pos_x[3], pos_y[3], pos_z[3] - 1])

        f_util.calc_dij(pos_x, pos_y, pos_z)
        f_util.calc_ang_ij(pos_x, pos_y, pos_z)
        f_util.calc_grad_vals(pos_x, pos_y, pos_z)
        f_util.calc_p_forces()
        f_util.calc_alignment_forces()
        f_util.calc_boun_rep(pos_x, pos_y, pos_z)
        u = f_util.calc_u_w()
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()

        if i % (env.CTRL_FREQ*1) == 0:
            f_util.plot_swarm(pos_x, pos_y, pos_z, pos_hxs, pos_hys, pos_hzs)

        for j in range(NUM_DRONES):
            vel_cmd = np.array([u[j]*np.cos(pos_hxs[j]), u[j]*np.cos(pos_hys[j]), u[j]*np.cos(pos_hzs[j])])
            # pos_cmd = vel_cmd * (1 / env.CTRL_FREQ) + np.array([pos_x[j], pos_y[j], pos_z[j]])
            pos_cmd = np.array([pos_x[j], pos_y[j], pos_z[j]])
            action[j], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                              state=obs[j],
                                                              target_pos=pos_cmd,
                                                              target_vel=vel_cmd,
                                                              target_rpy=np.array([0, 0, 0])
                                                              )

        #### Log the simulation ####################################
        for j in range(NUM_DRONES):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=np.array([pos_x[j], pos_y[j], pos_z[j]])
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
    # logger.save_as_csv("beta")  # Optional CSV save

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
