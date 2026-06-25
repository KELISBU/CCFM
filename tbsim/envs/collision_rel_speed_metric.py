"""
Collision Relative Speed Metric for SAFE-SIM Paper Replication

This metric calculates the relative speed at collision time between vehicles.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tbsim.envs.env_metrics import EnvMetrics, split_agents_by_scene, agent_index_by_scene
from tbsim.utils.geometry_utils import detect_collision


class CollisionRelativeSpeedMetric(EnvMetrics):
    """
    Compute relative speed when collision occurs.

    For paper replication - measures the relative velocity between
    colliding vehicles at the moment of collision.

    Episode-level aggregation records only the first collision in each scene.
    """

    def __init__(self):
        super(CollisionRelativeSpeedMetric, self).__init__()
        self._df = pd.DataFrame(columns=['scene_index', 'track_id', 'ts', 'rel_speed', 'collision'])
        self._scene_ts = defaultdict(lambda: 0)

    def reset(self):
        self._df = pd.DataFrame(columns=['scene_index', 'track_id', 'ts', 'rel_speed', 'collision'])
        self._scene_ts = defaultdict(lambda: 0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        """
        Compute relative speed at collision for each agent.

        Args:
            state_info: Dictionary containing agent state information
                - centroid: [N, 2] positions
                - yaw: [N] headings
                - extent: [N, 3] vehicle dimensions
                - curr_speed: [N] current speeds (if available)
            all_scene_index: Array of scene indices
            agent_info: Optional dictionary for other agents (when computing ego-agent collisions)

        Returns:
            rel_speeds: Relative speed per row in state_info (0 if no collision).
        """
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        # Get velocities if available, otherwise estimate from positions
        if "curr_speed" in state_info:
            velocity_per_scene = split_agents_by_scene(state_info["curr_speed"], agent_scene_index, all_scene_index)
        else:
            # Estimate speed from yaw and assume constant velocity
            velocity_per_scene = [np.ones(len(pos)) * 5.0 for pos in pos_per_scene]  # Default 5 m/s

        # Handle agent_info for ego-agent collision computation
        if agent_info is not None:
            other_agent_scene_index = agent_info["scene_index"]
            other_pos_per_scene = split_agents_by_scene(agent_info["centroid"], other_agent_scene_index, all_scene_index)
            other_yaw_per_scene = split_agents_by_scene(agent_info["yaw"], other_agent_scene_index, all_scene_index)
            other_extent_per_scene = split_agents_by_scene(agent_info["extent"][..., :2], other_agent_scene_index, all_scene_index)

            if "curr_speed" in agent_info:
                other_velocity_per_scene = split_agents_by_scene(agent_info["curr_speed"], other_agent_scene_index, all_scene_index)
            else:
                other_velocity_per_scene = [np.ones(len(pos)) * 5.0 for pos in other_pos_per_scene]

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        rel_speeds = np.zeros(num_agents)
        has_collision = np.zeros(num_agents, dtype=bool)

        # For each scene, compute collision and relative speed
        for i in range(num_scenes):
            if agent_info is None:
                # All-agent collision detection (original logic)
                num_agents_in_scene = pos_per_scene[i].shape[0]

                for j in range(num_agents_in_scene):
                    other_agent_mask = np.arange(num_agents_in_scene) != j

                    # Check for collision
                    coll = detect_collision(
                        ego_pos=pos_per_scene[i][j],
                        ego_yaw=yaw_per_scene[i][j],
                        ego_extent=extent_per_scene[i][j],
                        other_pos=pos_per_scene[i][other_agent_mask],
                        other_yaw=yaw_per_scene[i][other_agent_mask],
                        other_extent=extent_per_scene[i][other_agent_mask]
                    )

                    if coll is not None:
                        # Collision detected - compute relative speed against the collided agent only
                        collided_local_idx = np.where(other_agent_mask)[0][coll[1]]
                        ego_speed = velocity_per_scene[i][j]
                        other_speed = velocity_per_scene[i][collided_local_idx]

                        # Compute velocity vectors
                        ego_vel = ego_speed * np.array([np.cos(yaw_per_scene[i][j]),
                                                         np.sin(yaw_per_scene[i][j])])
                        other_vel = other_speed * np.array([
                            np.cos(yaw_per_scene[i][collided_local_idx]),
                            np.sin(yaw_per_scene[i][collided_local_idx]),
                        ])

                        rel_speed = float(np.linalg.norm(other_vel - ego_vel))

                        agent_idx = agent_index_per_scene[i][j]
                        rel_speeds[agent_idx] = rel_speed
                        has_collision[agent_idx] = True
            else:
                # Ego-agent collision detection (ego is first agent in state_info)
                coll = detect_collision(
                    ego_pos=pos_per_scene[i][0],
                    ego_yaw=yaw_per_scene[i][0],
                    ego_extent=extent_per_scene[i][0],
                    other_pos=other_pos_per_scene[i],
                    other_yaw=other_yaw_per_scene[i],
                    other_extent=other_extent_per_scene[i]
                )

                if coll is not None:
                    # Collision detected - compute relative speed against the collided agent only
                    ego_speed = velocity_per_scene[i][0]
                    other_speed = other_velocity_per_scene[i][coll[1]]

                    # Compute velocity vectors
                    ego_vel = ego_speed * np.array([np.cos(yaw_per_scene[i][0]),
                                                     np.sin(yaw_per_scene[i][0])])
                    other_vel = other_speed * np.array([
                        np.cos(other_yaw_per_scene[i][coll[1]]),
                        np.sin(other_yaw_per_scene[i][coll[1]]),
                    ])

                    rel_speed = float(np.linalg.norm(other_vel - ego_vel))

                    agent_idx = agent_index_per_scene[i][0]
                    rel_speeds[agent_idx] = rel_speed
                    has_collision[agent_idx] = True

        return rel_speeds, has_collision

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        rel_speeds, has_collision = self.compute_per_step(state_info, all_scene_index, agent_info)

        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])

        step_df = pd.DataFrame({
            'scene_index': state_info["scene_index"],
            'track_id': state_info["track_id"],
            'ts': ts,
            'rel_speed': rel_speeds,
            'collision': has_collision
        })

        self._df = pd.concat([self._df, step_df], ignore_index=True)

        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid] += 1

    def get_episode_metrics(self):
        """
        Aggregate metrics per scene.

        Returns:
            Dictionary with:
            - collision_rel_speed: Relative speed at the first collision in each scene
                                   (0 if no collision occurred)
        """
        all_scenes = self._df["scene_index"].unique()
        result = np.zeros(len(all_scenes), dtype=np.float32)

        collision_df = self._df[self._df["collision"] == True]
        if len(collision_df) > 0:
            first_collision_by_scene = (
                collision_df.sort_values(["scene_index", "ts"])
                .groupby("scene_index")["rel_speed"]
                .first()
            )

            for idx, scene in enumerate(all_scenes):
                if scene in first_collision_by_scene.index:
                    result[idx] = first_collision_by_scene[scene]

        return {
            'collision_rel_speed': result
        }
