"""
Collision Relative Heading Metric for SAFE-SIM.

This metric calculates the relative heading difference (in radians)
at collision time between vehicles.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from tbsim.envs.env_metrics import EnvMetrics, split_agents_by_scene, agent_index_by_scene
from tbsim.utils.geometry_utils import detect_collision, angular_distance


class CollisionRelativeHeadingMetric(EnvMetrics):
    """
    Compute relative heading difference when collision occurs.

    The per-scene output `collision_rel_heading` is the absolute angular
    difference (in radians, range [0, pi]) at the first collision in that scene.
    """

    def __init__(self):
        super(CollisionRelativeHeadingMetric, self).__init__()
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts", "rel_heading", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)

    def reset(self):
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts", "rel_heading", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        """
        Compute relative heading at collision for each tracked row in state_info.

        Returns:
            rel_headings: [num_agents_in_state_info], radians in [0, pi] for collision rows, else 0.
            has_collision: [num_agents_in_state_info], bool mask.
        """
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        if agent_info is not None:
            other_agent_scene_index = agent_info["scene_index"]
            other_pos_per_scene = split_agents_by_scene(
                agent_info["centroid"], other_agent_scene_index, all_scene_index
            )
            other_yaw_per_scene = split_agents_by_scene(
                agent_info["yaw"], other_agent_scene_index, all_scene_index
            )
            other_extent_per_scene = split_agents_by_scene(
                agent_info["extent"][..., :2], other_agent_scene_index, all_scene_index
            )

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)
        rel_headings = np.zeros(num_agents, dtype=np.float32)
        has_collision = np.zeros(num_agents, dtype=bool)

        for i in range(num_scenes):
            if agent_info is None:
                num_agents_in_scene = pos_per_scene[i].shape[0]
                for j in range(num_agents_in_scene):
                    other_agent_mask = np.arange(num_agents_in_scene) != j
                    if not np.any(other_agent_mask):
                        continue

                    coll = detect_collision(
                        ego_pos=pos_per_scene[i][j],
                        ego_yaw=yaw_per_scene[i][j],
                        ego_extent=extent_per_scene[i][j],
                        other_pos=pos_per_scene[i][other_agent_mask],
                        other_yaw=yaw_per_scene[i][other_agent_mask],
                        other_extent=extent_per_scene[i][other_agent_mask],
                    )

                    if coll is None:
                        continue

                    collided_local_idx = np.where(other_agent_mask)[0][coll[1]]
                    rel_heading = float(
                        np.abs(
                            angular_distance(yaw_per_scene[i][j], yaw_per_scene[i][collided_local_idx])
                        )
                    )
                    agent_idx = agent_index_per_scene[i][j]
                    rel_headings[agent_idx] = rel_heading
                    has_collision[agent_idx] = True
            else:
                # Ego-agent mode: state_info contains ego rows; agent_info contains non-ego rows.
                if pos_per_scene[i].shape[0] == 0 or other_pos_per_scene[i].shape[0] == 0:
                    continue

                coll = detect_collision(
                    ego_pos=pos_per_scene[i][0],
                    ego_yaw=yaw_per_scene[i][0],
                    ego_extent=extent_per_scene[i][0],
                    other_pos=other_pos_per_scene[i],
                    other_yaw=other_yaw_per_scene[i],
                    other_extent=other_extent_per_scene[i],
                )

                if coll is None:
                    continue

                rel_heading = float(
                    np.abs(
                        angular_distance(yaw_per_scene[i][0], other_yaw_per_scene[i][coll[1]])
                    )
                )
                agent_idx = agent_index_per_scene[i][0]
                rel_headings[agent_idx] = rel_heading
                has_collision[agent_idx] = True

        return rel_headings, has_collision

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        rel_headings, has_collision = self.compute_per_step(state_info, all_scene_index, agent_info)

        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = pd.DataFrame(
            {
                "scene_index": state_info["scene_index"],
                "track_id": state_info["track_id"],
                "ts": ts,
                "rel_heading": rel_headings,
                "collision": has_collision,
            }
        )

        self._df = pd.concat([self._df, step_df], ignore_index=True)
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid] += 1

    def get_episode_metrics(self):
        """
        Returns:
            collision_rel_heading: per-scene first-collision abs heading diff (radians), 0 if no collision.
        """
        if len(self._df) == 0:
            return {
                "collision_rel_heading": np.array([], dtype=np.float32),
            }

        collision_df = self._df[self._df["collision"] == True]
        all_scenes = self._df["scene_index"].unique()

        rel_heading_first = np.zeros(len(all_scenes), dtype=np.float32)

        if len(collision_df) > 0:
            first_by_scene = (
                collision_df.sort_values(["scene_index", "ts"])
                .groupby("scene_index")["rel_heading"]
                .first()
            )

            for idx, scene in enumerate(all_scenes):
                if scene in first_by_scene.index:
                    rel_heading_first[idx] = first_by_scene[scene]

        return {
            "collision_rel_heading": rel_heading_first,
        }
