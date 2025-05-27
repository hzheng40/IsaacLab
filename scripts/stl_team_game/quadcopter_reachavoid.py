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
parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate a quadcopter."
)
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
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip

import numpy as np
from typing import Dict
import yaml


class Tracker:
    def __init__(self, config: Dict, sim: SimulationContext):
        # TODO: load trajectories
        # shape (timesteps, num_trajectories, num_robots, 6)
        self.trajectories = np.load(config["saved_traj"])

        # load params
        self.Kp_pos = config["Kp_pos"]
        self.Kd_pos = config["Kd_pos"]
        self.Kp_att = config["Kp_att"]
        self.Kd_att = config["Kd_att"]

        # load ref traj dt
        self.dt = config["dt"]

        self.sim = sim

    @staticmethod
    def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of quaternions (x, y, z, w) to rotation matrices.
        Input:  quat shape (N,4)
        Output: R    shape (N,3,3)
        """
        x, y, z, w = quat.unbind(dim=1)
        # precompute products
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=1),
                torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=1),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=1),
            ],
            dim=1,
        )  # shape (N,3,3)
        return R

    @staticmethod
    def so3_to_vec(so3mat: torch.Tensor) -> torch.Tensor:
        """
        vee-map for a batch of skew-symmetric matrices:
        so3mat shape (N,3,3) -> returns (N,3) vector [m32-m23, m13-m31, m21-m12]
        """
        return torch.stack(
            [
                so3mat[:, 2, 1] - so3mat[:, 1, 2],
                so3mat[:, 0, 2] - so3mat[:, 2, 0],
                so3mat[:, 1, 0] - so3mat[:, 0, 1],
            ],
            dim=1,
        )

    def plan(self, robot: Articulation, robot_idx: int, traj_idx: int, sim_time: float):
        device = robot.device
        ref_traj = self.trajectories[:, traj_idx, robot_idx, :]
        # calculate current time index
        t_ind = int(sim_time / self.dt)
        if t_ind >= ref_traj.shape[0]:
            t_ind = ref_traj.shape[0] - 1

        # reference position and velocity
        pos_d = torch.tensor(ref_traj[t_ind, :3], device=device)
        vel_d = torch.tensor(ref_traj[t_ind, 3:6], device=device)

        # current position and velocity
        pos = robot.data.root_pos_w
        quat = robot.data.root_quat_w
        vel = robot.data.root_lin_vel_w
        ang_vel = robot.data.root_ang_vel_w

        # mass
        mass = robot.data.default_mass.sum(dim=1, keepdim=True)

        # 1. Outer loop PD -> desired world frame acceleration
        a_cmd = self.Kp_pos * (pos_d - pos) + self.Kd_pos * (vel_d - vel)
        # convert to total thrust in world frame
        g = torch.tensor(self.sim.cfg.gravity, device=device).view(1, 3)
        F_des = mass * (a_cmd + g)  # (N, 3)?

        # desired body-z and zero yaw ref
        z_b_des = F_des / F_des.norm(dim=1, keepdim=True)  # (N, 3)
        yaw_ref = torch.zeros(robot.num_instances, device=device)  # (N,)
        x_c = torch.stack(
            [torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref)], dim=1
        )  # (N, 3)
        y_b_des = torch.cross(z_b_des, x_c, dim=1)  # (N, 3)
        y_b_des = y_b_des / y_b_des.norm(dim=1, keepdim=True)
        x_b_des = torch.cross(y_b_des, z_b_des, dim=1)

        # desired and current rotations
        R_d = torch.stack([x_b_des, y_b_des, z_b_des], dim=2)  # (N, 3, 3)
        R = self.quaternion_to_rotation_matrix(quat)

        # attitude error in SO(3)
        err_mat = R_d.transpose(1, 2) @ R - R.transpose(1, 2) @ R_d  # (N, 3, 3)
        e_R = 0.5 * self.so3_to_vec(err_mat)  # (N, 3)
        # 2. Inner loop PD for attitude control
        tau = -self.Kp_att * e_R - self.Kd_att * ang_vel  # (N, 3)

        return F_des, tau


def main():
    """Main function."""
    # load planning config
    with open("/IsaacLab/scripts/stl_team_game/config/traj_track.yaml", "r") as f:
        planning_cfg = yaml.safe_load(f)

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(100, -20, 50), target=(0.0, 40.0, 0.0))

    # tracker = Tracker(planning_cfg, sim)

    # -------------------------------------------------------------------
    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0), size=(300.0, 300.0))
    cfg.func("/World/Ground", cfg)
    # Lights
    # cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # obstacles
    obs_cfg = sim_utils.CuboidCfg(
        size=(12.0, 12.0, 30.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    obs_cfg.func("/World/Obstacle", obs_cfg, translation=(-4.0, 16.0, 15.0))

    # goal markers
    num_goals = 8
    goal_translations = np.array(
        [
            [
                17.0,
                17.0,
                10.0,
            ],
            [
                32.0,
                17.0,
                10.0,
            ],
            [
                47.0,
                17.0,
                10.0,
            ],
            [
                17.0,
                32.0,
                10.0,
            ],
            [
                32.0,
                32.0,
                10.0,
            ],
            [
                47.0,
                32.0,
                10.0,
            ],
            [
                17.0,
                47.0,
                10.0,
            ],
            [
                32.0,
                47.0,
                10.0,
            ],
        ]
    )
    cubes_cfg = sim_utils.CuboidCfg(
        size=(4.0, 4.0, 4.0),
        visual_material=sim_utils.GlassMdlCfg(
            glass_color=(0.0, 1.0, 0.0),
            frosting_roughness=1.0,
            glass_ior=1.0,
        ),
    )
    for i in range(num_goals):
        sim_utils.spawn_cuboid(
            f"/World/Goal_{i}",
            cubes_cfg,
            translation=goal_translations[i] + 2.0,
        )

    # start markers
    ego_start_cfg = sim_utils.CuboidCfg(
        size=(11.0, 11.0, 11.0),
        visual_material=sim_utils.GlassMdlCfg(
            glass_color=(0.0, 0.0, 1.0),
            frosting_roughness=1.0,
            glass_ior=1.0,
        ),
    )
    opp_start_cfg = sim_utils.CuboidCfg(
        size=(11.0, 11.0, 11.0),
        visual_material=sim_utils.GlassMdlCfg(
            glass_color=(1.0, 0.0, 0.0),
            frosting_roughness=1.0,
            glass_ior=1.0,
        ),
    )
    sim_utils.spawn_cuboid(
        "/World/Ego_start",
        ego_start_cfg,
        translation=(-26.5, -7.5, 6.5),
    )
    sim_utils.spawn_cuboid(
        "/World/Opp_start",
        opp_start_cfg,
        translation=(-26.5, 45.5, 8.5),
    )

    # Robots
    num_ego = 8
    num_opp = 4

    ego_starts = np.array(
        [
            [
                -32.0 + 10 * (i % 4),
                -14.0 + 10 * (i // 4),
                5.0,
            ]
            for i in range(num_ego)
        ]
    )
    opp_starts = np.array(
        [
            [
                -32.0 + 10 * (i % 2),
                40.0 + 10 * (i // 2),
                3.0,
            ]
            for i in range(num_opp)
        ]
    )

    ego_cfgs = []
    egos = []
    opp_cfgs = []
    opps = []

    scaled_crazyflie_cfg = CRAZYFLIE_CFG.copy()
    scaled_crazyflie_cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=CRAZYFLIE_CFG.spawn.usd_path, scale=(15.0, 15.0, 15.0)
    )

    for i in range(num_ego):
        ego_cfgs.append(
            scaled_crazyflie_cfg.copy().replace(
                prim_path=f"/World/Ego_{i}",
                init_state=CRAZYFLIE_CFG.init_state.replace(pos=ego_starts[i]),
            )
        )
        egos.append(Articulation(ego_cfgs[i]))

    for i in range(num_opp):
        opp_cfgs.append(
            scaled_crazyflie_cfg.copy().replace(
                prim_path=f"/World/Opp_{i}",
                init_state=CRAZYFLIE_CFG.init_state.replace(pos=opp_starts[i]),
            )
        )
        opps.append(Articulation(opp_cfgs[i]))
    # -------------------------------------------------------------------

    # num_robots = 16
    # # create a grid pattern for initial positions for all robots
    # # determine grid dimensions
    # cols = 4
    # rows = 4
    # spacing = 0.8  # distance between robots

    # # build a centered grid of (x, y, z=1.0) positions
    # init_positions = []
    # for i in range(num_robots):
    #     r = i // cols
    #     c = i % cols
    #     x = (c - (cols - 1) / 2) * spacing
    #     y = (r - (rows - 1) / 2) * spacing
    #     init_positions.append((x, y, 1.0))
    # print(init_positions)
    # all_robot_cfgs = []
    # for i in range(num_robots):
    #     robot_cfg = CRAZYFLIE_CFG.replace(prim_path=f"/World/Crazyflie_{i}")
    #     robot_cfg.spawn.func(
    #         f"/World/Crazyflie_{i}", robot_cfg.spawn, translation=init_positions[i]
    #     )
    #     robot_cfg.init_state.pos = init_positions[i]
    #     all_robot_cfgs.append(robot_cfg)

    # # create handles for the robots
    # all_robots = []
    # for robot_cfg in all_robot_cfgs:
    #     robot = Articulation(robot_cfg)
    #     all_robots.append(robot)

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
            for robot in egos:
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel,
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                robot.reset()

            for robot in opps:
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel,
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot (make the robot float in place)
        for i, robot in enumerate(egos):
            prop_body_ids = robot.find_bodies("m.*_prop")[0]
            robot_mass = robot.root_physx_view.get_masses().sum()
            gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
            forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
            torques = torch.zeros_like(forces)
            forces[..., 2] = robot_mass * gravity / 4.0
            robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
            robot.write_data_to_sim()

        for i, robot in enumerate(opps):
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
        for robot in egos:
            robot.update(sim_dt)
        for robot in opps:
            robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
