# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import math

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0))
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    num_robots = 16
    # create a grid pattern for initial positions for all robots
    # determine grid dimensions
    cols = 4
    rows = 4
    spacing = 0.8  # distance between robots

    # build a centered grid of (x, y, z=1.0) positions
    init_positions = []
    for i in range(num_robots):
        r = i // cols
        c = i % cols
        x = (c - (cols - 1) / 2) * spacing
        y = (r - (rows - 1) / 2) * spacing
        init_positions.append((x, y, 1.0))
    print(init_positions)
    all_robot_cfgs = []
    for i in range(num_robots):
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path=f"/World/Crazyflie_{i}")
        robot_cfg.spawn.func(f"/World/Crazyflie_{i}", robot_cfg.spawn, translation=init_positions[i])
        robot_cfg.init_state.pos = init_positions[i]
        all_robot_cfgs.append(robot_cfg)

    # create handles for the robots
    all_robots = []
    for robot_cfg in all_robot_cfgs:
        robot = Articulation(robot_cfg)
        all_robots.append(robot)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            
            # reset dof state
            for robot in all_robots:
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot (make the robot float in place)
        for i, robot in enumerate(all_robots):
            prop_body_ids = robot.find_bodies("m.*_prop")[0]
            robot_mass = robot.root_physx_view.get_masses().sum()
            gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
            forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
            torques = torch.zeros_like(forces)
            forces[..., 2] = robot_mass * gravity / 4.0
            robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in all_robots:
            robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
