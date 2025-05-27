# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaacsim.core.utils.prims as prim_utils

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterRAEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterRAEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)  # type: ignore
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


def create_ego_robot_cfg():
    # agents
    num_ego = 8
    # random grid pattern for egos based on num_ego
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
    ego_cfg = []
    # create xform prims for all robots
    for i in range(num_ego):
        # prim_utils.create_prim(f"/World/Ego_{i}", "Xform", translation=ego_starts[i])
        # ego_cfg.append(
        #     AssetBaseCfg(
        #         prim_path=f"/World/Ego_{i}",
        #         spawn=CRAZYFLIE_CFG,  # type: ignore
        #         init_state=AssetBaseCfg.InitialStateCfg(
        #             pos=ego_starts[i],
        #         ),
        #     )
        # )

        ego_cfg.append(
            CRAZYFLIE_CFG.replace(
                prim_path=f"/World/Ego_{i}",
                init_state=CRAZYFLIE_CFG.init_state.replace(
                    pos=ego_starts[i],
                ),
            )
        )

    return ego_cfg


def create_opp_robot_cfg():
    # agents
    num_opp = 4
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
    # create xform prims for all robots
    opp_cfg = []
    for j in range(num_opp):
        # prim_utils.create_prim(f"/World/Opp_{j}", "Xform", translation=opp_starts[j])
        # opp_cfg.append(
        #     AssetBaseCfg(
        #         prim_path=f"/World/Opp_{j}",
        #         spawn=CRAZYFLIE_CFG,  # type: ignore
        #         init_state=AssetBaseCfg.InitialStateCfg(
        #             pos=opp_starts[j],
        #         ),
        #     )
        # )
        opp_cfg.append(
            CRAZYFLIE_CFG.replace(
                prim_path=f"/World/Opp_{j}",
                init_state=CRAZYFLIE_CFG.init_state.replace(
                    pos=opp_starts[j],
                ),
            )
        )

    return opp_cfg


def create_scene(
    num_envs: int = 4096, env_spacing: float = 2.5, replicate_physics: bool = True
) -> InteractiveSceneCfg:
    # obstacles (collision)
    obs = AssetBaseCfg(
        prim_path="/World/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 12.0, 30.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-10.0, 10.0, 0.0)),
    )

    # goals (no collision)
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

    goal_markers_cfg = []
    for i in range(num_goals):
        #     prim_utils.create_prim(
        #         f"/World/Goal_{i}", "Xform", translation=goal_translations[i]
        #     )
        goal_markers_cfg.append(
            AssetBaseCfg(
                prim_path=f"/World/Goal_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(4.0, 4.0, 4.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.6, 0.0)
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=goal_translations[i],
                ),
            )
        )
    # goal_markers = VisualizationMarkers(goal_markers_cfg)
    # goal_markers.visualize(translations=goal_translations)

    # altitude zones (no collision)
    # altitude_zone_markers_cfg = VisualizationMarkersCfg(
    #     prim_path="/World/AltitudeZones",
    #     markers={
    #         "cuboid": sim_utils.CuboidCfg(
    #             size=(12.0, 12.0, 4.0),
    #             visual_material=sim_utils.PreviewSurfaceCfg(
    #                 diffuse_color=(0.0, 0.0, 1.0)
    #             ),
    #         ),
    #     },
    # )
    # altitude_zone_markers = VisualizationMarkers(altitude_zone_markers_cfg)
    # altitude_zone_markers.visualize(translations=np.array([[-10.0, 10.0, 2.0]]))

    # starting zones (no collision)
    ego_start_markers_cfg = VisualizationMarkersCfg(
        prim_path="/World/Ego_start",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(
                    11.0,
                    11.0,
                    11.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.3)
                ),
            ),
        },
    )
    ego_start_markers = VisualizationMarkers(ego_start_markers_cfg)
    ego_start_markers.visualize(
        translations=np.array(
            [
                [
                    -32.0,
                    -14.0,
                    1.0,
                ]
            ]
        )
    )

    opp_start_markers_cfg = VisualizationMarkersCfg(
        prim_path="/World/Opp_start",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(4.0, 4.0, 4.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.6, 0.0)
                ),
            ),
        },
    )
    opp_start_markers = VisualizationMarkers(opp_start_markers_cfg)
    opp_start_markers.visualize(translations=np.array([[-32.0, 40.0, 3.0]]))

    # ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    @configclass
    class DefaultReachAvoidScene(InteractiveSceneCfg):
        """Configuration for the Reach Avoid scene. The recommended order of specification is terrain,
        physics-related assets (articulations and rigid bodies), sensors and non-physics-related assets (lights).
        """

        # ground plane
        ground = terrain

        # obstacle collision
        obstacles = obs

        # robot
        ego_robot = create_ego_robot_cfg()
        opp_robot = create_opp_robot_cfg()

        ego_start = ego_start_markers_cfg
        opp_start = opp_start_markers_cfg
        goals = goal_markers_cfg

        # lighting
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))

    return DefaultReachAvoidScene(
        num_envs=num_envs, env_spacing=env_spacing, replicate_physics=replicate_physics
    )


@configclass
class QuadcopterRAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2

    # agents
    possible_agents = [
        "ego_0",
        "ego_1",
        "ego_2",
        "ego_3",
        "ego_4",
        "ego_5",
        "ego_6",
        "ego_7",
        "opp_0",
        "opp_1",
        "opp_2",
        "opp_3",
    ]
    action_spaces = {
        "ego_0": 4,
        "ego_1": 4,
        "ego_2": 4,
        "ego_3": 4,
        "ego_4": 4,
        "ego_5": 4,
        "ego_6": 4,
        "ego_7": 4,
        "opp_0": 4,
        "opp_1": 4,
        "opp_2": 4,
        "opp_3": 4,
    }
    # observation space
    # (position (3), velocity (3), rotation_matrix (3x3), angular_velocity (3))
    observation_spaces = {
        "ego_0": 18,
        "ego_1": 18,
        "ego_2": 18,
        "ego_3": 18,
        "ego_4": 18,
        "ego_5": 18,
        "ego_6": 18,
        "ego_7": 18,
        "opp_0": 18,
        "opp_1": 18,
        "opp_2": 18,
        "opp_3": 18,
    }

    state_space = 18 * len(possible_agents)
    debug_vis = True

    ui_window_class_type = QuadcopterRAEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = create_scene()

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterRAEnv(DirectMARLEnv):
    cfg: QuadcopterRAEnvCfg

    def __init__(
        self, cfg: QuadcopterRAEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Logging
        # TODO: redo key in the logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # instantiate articulations
        self._ego_robots = Articulation(self.cfg.ego_robot)
        self._opp_robots = Articulation(self.cfg.opp_robot)
        self.scene.articulations["ego_robots"] = self._ego_robots
        self.scene.articulations["opp_robots"] = self._opp_robots

        # instantiate terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Scene should already been instantiated
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight
            * self._robot_weight
            * (self._actions[:, 0] + 1.0)
            / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped
            * self.cfg.distance_to_goal_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES  # type: ignore

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(
            self._desired_pos_w[env_ids, :2]
        ).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # type: ignore
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # type: ignore

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()  # type: ignore
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
