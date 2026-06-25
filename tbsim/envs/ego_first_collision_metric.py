"""
EgoFirstCollisionMetric: record the collision position type (FRONT/REAR/SIDE)
at the FIRST timestep ego collides, then report proportions that sum to 100%.
"""

import numpy as np
from collections import defaultdict

from tbsim.envs.env_metrics import EnvMetrics, split_agents_by_scene, agent_index_by_scene
from tbsim.utils.geometry_utils import CollisionType, detect_collision


class EgoFirstCollisionMetric(EnvMetrics):
    """Record ego's first collision position (FRONT/REAR/SIDE) per scene.

    Output metrics:
        ego_first_coll_front_ratio: proportion of collided scenes where first hit is FRONT
        ego_first_coll_rear_ratio:  proportion of collided scenes where first hit is REAR
        ego_first_coll_side_ratio:  proportion of collided scenes where first hit is SIDE
        ego_first_coll_total:       total number of scenes where ego collided
    """

    def __init__(self):
        super().__init__()
        # scene_id -> CollisionType of first collision (None if not yet collided)
        self._first_coll_type = {}
        self._scene_ts = defaultdict(lambda: 0)

    def reset(self):
        self._first_coll_type = {}
        self._scene_ts = defaultdict(lambda: 0)

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        if agent_info is None:
            for sid in np.unique(state_info["scene_index"]):
                self._scene_ts[sid] += 1
            return

        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)

        other_scene_index = agent_info["scene_index"]
        other_pos_per_scene = split_agents_by_scene(agent_info["centroid"], other_scene_index, all_scene_index)
        other_yaw_per_scene = split_agents_by_scene(agent_info["yaw"], other_scene_index, all_scene_index)
        other_extent_per_scene = split_agents_by_scene(agent_info["extent"][..., :2], other_scene_index, all_scene_index)

        for i in range(len(all_scene_index)):
            scene_id = all_scene_index[i]

            # Skip if already recorded first collision for this scene
            if scene_id in self._first_coll_type:
                continue

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
            if coll is not None:
                self._first_coll_type[scene_id] = coll[0]  # CollisionType enum

        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid] += 1

    def get_episode_metrics(self):
        types = list(self._first_coll_type.values())
        total = len(types)

        if total == 0:
            return {
                "ego_first_coll_front_ratio": np.array([0.0]),
                "ego_first_coll_rear_ratio": np.array([0.0]),
                "ego_first_coll_side_ratio": np.array([0.0]),
                "ego_first_coll_total": np.array([0.0]),
            }

        n_front = sum(1 for t in types if t == CollisionType.FRONT)
        n_rear = sum(1 for t in types if t == CollisionType.REAR)
        n_side = sum(1 for t in types if t == CollisionType.SIDE)

        return {
            "ego_first_coll_front_ratio": np.array([n_front / total]),
            "ego_first_coll_rear_ratio": np.array([n_rear / total]),
            "ego_first_coll_side_ratio": np.array([n_side / total]),
            "ego_first_coll_total": np.array([float(total)]),
        }
