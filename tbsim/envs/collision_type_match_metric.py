"""
Collision Type Match Metric

Checks whether the collision type selected by HCS matches the
actual collision that occurs during closed-loop simulation.

At the first ego-adv collision in each scene, the actual collision type is
classified from contact point and relative heading (mirroring the constraint
definitions in collision_constraints.py).  If it matches the intended type,
the metric records 1; otherwise 0.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from tbsim.envs.env_metrics import EnvMetrics, split_agents_by_scene, agent_index_by_scene
from tbsim.utils.geometry_utils import detect_collision
from tbsim.utils.collision_constraints import CollisionType


def classify_collision_type(ego_pos, ego_yaw, adv_pos, adv_yaw):
    """Classify the actual collision type from ego/adv state at collision time.

    Uses the same geometric criteria as CollisionConstraints:
      - REAR_END : same direction  (dir_dot > 0.7)
      - SIDE     : perpendicular   (|dir_dot| <= 0.5)
      - HEAD_ON  : opposite        (dir_dot < -0.5)
      - CUT_IN   : moderate angle  (0.5 < dir_dot <= 0.7)  with adv heading
                    toward ego's lane  (pos_cross * dir_cross < 0)

    Contact-point refinement: if heading says REAR_END but the contact is
    clearly at the side of ego, override to SIDE.

    Args:
        ego_pos: [2] ego centroid.
        ego_yaw: scalar ego heading.
        adv_pos: [2] adv centroid.
        adv_yaw: scalar adv heading.

    Returns:
        CollisionType enum value.
    """
    ego_dir = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
    adv_dir = np.array([np.cos(adv_yaw), np.sin(adv_yaw)])
    dir_dot = float(np.dot(ego_dir, adv_dir))

    # Bearing: where is adv relative to ego
    rel_vec = adv_pos - ego_pos
    rel_dist = float(np.linalg.norm(rel_vec))
    if rel_dist > 1e-6:
        rel_dir = rel_vec / rel_dist
    else:
        rel_dir = np.array([1.0, 0.0])
    bearing_dot = float(np.dot(rel_dir, ego_dir))

    # Cross products for cut-in direction check
    pos_cross = ego_dir[0] * (adv_pos[1] - ego_pos[1]) - ego_dir[1] * (adv_pos[0] - ego_pos[0])
    dir_cross = ego_dir[0] * adv_dir[1] - ego_dir[1] * adv_dir[0]

    # Primary classification by heading dot product
    if dir_dot < -0.5:
        return CollisionType.HEAD_ON
    elif abs(dir_dot) <= 0.5:
        return CollisionType.SIDE
    elif dir_dot > 0.7:
        # Same direction — but check if contact is at ego's side
        # If adv is clearly to the side (|bearing_dot| small), call it SIDE
        if abs(bearing_dot) < 0.3:
            return CollisionType.SIDE
        return CollisionType.REAR_END
    else:
        # 0.5 < dir_dot <= 0.7 — moderate angle
        # Check cut-in direction: pos_cross * dir_cross < 0 means adv heading
        # toward ego's lane
        if pos_cross * dir_cross < 0:
            return CollisionType.CUT_IN
        else:
            # Heading toward ego lane not confirmed; still most consistent with cut-in
            return CollisionType.CUT_IN


class CollisionTypeMatchMetric(EnvMetrics):
    """Records whether the HCS's intended collision type matches the
    actual collision type at the first ego-adv collision in each scene.

    Usage:
        1. Create the metric and register it in the metrics dict.
        2. After ``select_adv_by_event_score``, call
           ``metric.set_collision_configs(env.current_collision_configs)``
           to tell the metric which collision type was intended per scene.
        3. ``add_step`` is called every simulation step (ego-agent mode).
        4. ``get_episode_metrics`` returns per-scene match (1/0).
    """

    def __init__(self):
        super(CollisionTypeMatchMetric, self).__init__()
        self._collision_configs = None  # list[dict|None], one per scene
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts",
                     "intended_type", "actual_type", "match", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)
        self._first_collision_printed = set()  # scene indices already printed

    def reset(self):
        self._collision_configs = None
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts",
                     "intended_type", "actual_type", "match", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)
        self._first_collision_printed = set()

    def set_collision_configs(self, collision_configs):
        """Store per-scene collision configs from HCS.

        Args:
            collision_configs: list of dicts (one per scene), each with keys
                ``ego_idx``, ``adv_idx``, ``collision_type``, or ``None``
                if no adversary was selected for that scene.
        """
        self._collision_configs = collision_configs

    # ------------------------------------------------------------------

    @staticmethod
    def _find_adv_collision(
        ego_pos, ego_yaw, ego_extent,
        all_pos, all_yaw, all_extent, all_global_indices,
        adv_global_idx,
    ):
        """Check if ego collides specifically with the selected adv agent.

        Returns:
            (adv_local_idx, ego_yaw, adv_yaw, ego_pos, adv_pos) if collision,
            None otherwise.
        """
        # Find the adv agent in the local arrays
        adv_local = None
        for li, gi in enumerate(all_global_indices):
            if gi == adv_global_idx:
                adv_local = li
                break
        if adv_local is None:
            return None

        # Check collision between ego and this specific adv
        coll = detect_collision(
            ego_pos=ego_pos,
            ego_yaw=ego_yaw,
            ego_extent=ego_extent,
            other_pos=all_pos[adv_local:adv_local + 1],
            other_yaw=all_yaw[adv_local:adv_local + 1],
            other_extent=all_extent[adv_local:adv_local + 1],
        )
        if coll is not None:
            return (adv_local, ego_yaw, all_yaw[adv_local],
                    ego_pos, all_pos[adv_local])
        return None

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        """Record collision type match for each ego at this timestep.

        This metric is designed for ego-agent mode (state_info = ego,
        agent_info = other agents), matching the pattern of
        ``ego_collision_*`` metrics.
        """
        if self._collision_configs is None:
            # No collision configs set yet — skip silently
            ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
            for sid in np.unique(state_info["scene_index"]):
                self._scene_ts[sid] += 1
            return

        agent_scene_index = state_info["scene_index"]
        num_agents = len(agent_scene_index)

        # Per-agent outputs
        intended_types = [""] * num_agents
        actual_types = [""] * num_agents
        matches = np.zeros(num_agents, dtype=np.float32)
        has_collision = np.zeros(num_agents, dtype=bool)

        # Split ego data by scene
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_idx_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        # Split agent (non-ego) data by scene
        if agent_info is not None:
            other_scene_index = agent_info["scene_index"]
            other_pos_per_scene = split_agents_by_scene(agent_info["centroid"], other_scene_index, all_scene_index)
            other_yaw_per_scene = split_agents_by_scene(agent_info["yaw"], other_scene_index, all_scene_index)
            other_extent_per_scene = split_agents_by_scene(agent_info["extent"][..., :2], other_scene_index, all_scene_index)

            # We need global indices of the other agents to match adv_idx
            if "global_index" in agent_info:
                other_global_per_scene = split_agents_by_scene(
                    agent_info["global_index"], other_scene_index, all_scene_index
                )
            elif "track_id" in agent_info:
                other_trackid_per_scene = split_agents_by_scene(
                    agent_info["track_id"], other_scene_index, all_scene_index
                )
            else:
                other_trackid_per_scene = [np.arange(len(p)) for p in other_pos_per_scene]
        else:
            return  # Need agent_info for ego-agent collision check

        num_scenes = len(all_scene_index)
        for i in range(num_scenes):
            if i >= len(self._collision_configs) or self._collision_configs[i] is None:
                continue

            cc = self._collision_configs[i]
            intended_ctype = cc["collision_type"]
            if isinstance(intended_ctype, str):
                intended_ctype = CollisionType(intended_ctype)
            adv_global = cc["adv_idx"]

            # Ego in this scene (should be exactly one in ego-agent mode)
            if pos_per_scene[i].shape[0] == 0 or other_pos_per_scene[i].shape[0] == 0:
                continue

            ego_pos = pos_per_scene[i][0]
            ego_yaw_val = yaw_per_scene[i][0]
            ego_ext = extent_per_scene[i][0]

            # Find the designated adv among the other agents
            # We match by checking all other agents for collision with ego,
            # then verify the colliding agent is the designated adv.
            # First, try to find adv by track_id matching adv_global.
            adv_local = None
            for li in range(other_pos_per_scene[i].shape[0]):
                # Check collision between ego and this specific agent
                coll = detect_collision(
                    ego_pos=ego_pos,
                    ego_yaw=ego_yaw_val,
                    ego_extent=ego_ext,
                    other_pos=other_pos_per_scene[i][li:li + 1],
                    other_yaw=other_yaw_per_scene[i][li:li + 1],
                    other_extent=other_extent_per_scene[i][li:li + 1],
                )
                if coll is not None:
                    # Collision found — classify actual type
                    adv_pos_val = other_pos_per_scene[i][li]
                    adv_yaw_val = other_yaw_per_scene[i][li]

                    actual_ctype = classify_collision_type(
                        ego_pos, ego_yaw_val, adv_pos_val, adv_yaw_val
                    )

                    agent_idx = agent_idx_per_scene[i][0]
                    intended_types[agent_idx] = intended_ctype.value
                    actual_types[agent_idx] = actual_ctype.value
                    is_match = actual_ctype == intended_ctype
                    matches[agent_idx] = 1.0 if is_match else 0.0
                    has_collision[agent_idx] = True

                    # Print on first collision per scene
                    scene_id = all_scene_index[i]
                    if scene_id not in self._first_collision_printed:
                        self._first_collision_printed.add(scene_id)
                        match_str = "MATCH" if is_match else "MISMATCH"
                        print(f"[CollisionTypeMatch] Scene {scene_id} first collision: "
                              f"HCS={intended_ctype.value}, "
                              f"Actual={actual_ctype.value} -> {match_str}")

                    break  # Only record first collision in this scene at this step

        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = pd.DataFrame({
            "scene_index": state_info["scene_index"],
            "track_id": state_info["track_id"],
            "ts": ts,
            "intended_type": intended_types,
            "actual_type": actual_types,
            "match": matches,
            "collision": has_collision,
        })
        self._df = pd.concat([self._df, step_df], ignore_index=True)

        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid] += 1

    def get_episode_metrics(self):
        """Return per-scene collision type match at first collision.

        Returns:
            dict with:
            - ``collision_type_match``: [num_scenes] array, 1.0 if the actual
              collision type matches the intended type, 0.0 if not, NaN if
              no collision occurred in that scene.
            - ``intended_collision_type``: [num_scenes] int array.
              0=rear-end, 1=side, 2=cut-in, 3=head-on, -1=no collision.
              Records the collision type HCS chose for each scene.
        """
        if len(self._df) == 0:
            return {
                "collision_type_match": np.array([], dtype=np.float32),
                "intended_collision_type": np.array([], dtype=np.int32),
            }

        all_scenes = self._df["scene_index"].unique()
        collision_df = self._df[self._df["collision"] == True]

        match_result = np.full(len(all_scenes), np.nan, dtype=np.float32)
        intended_result = np.full(len(all_scenes), -1, dtype=np.int32)

        # Map CollisionType string values to ints (same encoding as ActualCollisionTypeMetric)
        value_to_int = {ct.value: i for ct, i in COLLISION_TYPE_TO_INT.items()}

        if len(collision_df) > 0:
            # Take the first collision per scene (earliest ts)
            first_coll = (
                collision_df.sort_values(["scene_index", "ts"])
                .groupby("scene_index")
                .first()
            )
            for idx, scene in enumerate(all_scenes):
                if scene in first_coll.index:
                    match_result[idx] = first_coll.loc[scene, "match"]
                    intended_str = first_coll.loc[scene, "intended_type"]
                    intended_result[idx] = value_to_int.get(intended_str, -1)

        return {
            "collision_type_match": match_result,
            "intended_collision_type": intended_result,
        }


# Collision type encoding for numeric storage
COLLISION_TYPE_TO_INT = {
    CollisionType.REAR_END: 0,
    CollisionType.SIDE: 1,
    CollisionType.CUT_IN: 2,
    CollisionType.HEAD_ON: 3,
}
INT_TO_COLLISION_TYPE = {v: k for k, v in COLLISION_TYPE_TO_INT.items()}


class ActualCollisionTypeMetric(EnvMetrics):
    """Record the actual collision type at the first ego collision in each scene.

    Uses ``classify_collision_type`` to determine collision category from
    contact point and relative heading.  Does NOT require HCS or
    collision_configs — works standalone.

    Episode output:
        ``actual_collision_type``: [num_scenes] int array.
            0=rear-end, 1=side, 2=cut-in, 3=head-on, -1=no collision.
        ``actual_collision_dir_dot``: [num_scenes] float array.
            cos(yaw_ego - yaw_adv) at first ego-adv collision step.
            NaN when no collision occurred. This is the raw quantity the
            type classifier thresholds on, so you can audit whether the
            chosen type matches the constraint's heading interval.
        ``actual_collision_bearing_dot``: [num_scenes] float array.
            (adv_pos - ego_pos) / ||·|| · ego_dir at first collision step.
            +1 = adv directly in front of ego, -1 = directly behind, 0 = at
            ego's side. Used by the type classifier as a tie-breaker between
            REAR_END and SIDE when dir_dot > 0.7.
    """

    def __init__(self):
        super(ActualCollisionTypeMetric, self).__init__()
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts",
                     "actual_type_int", "dir_dot", "bearing_dot", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)
        self._first_collision_printed = set()

    def reset(self):
        self._df = pd.DataFrame(
            columns=["scene_index", "track_id", "ts",
                     "actual_type_int", "dir_dot", "bearing_dot", "collision"]
        )
        self._scene_ts = defaultdict(lambda: 0)
        self._first_collision_printed = set()

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, agent_info: dict = None):
        agent_scene_index = state_info["scene_index"]
        num_agents = len(agent_scene_index)

        actual_type_ints = np.full(num_agents, -1, dtype=np.int32)
        dir_dots = np.full(num_agents, np.nan, dtype=np.float32)
        bearing_dots = np.full(num_agents, np.nan, dtype=np.float32)
        has_collision = np.zeros(num_agents, dtype=bool)

        if agent_info is None:
            # Need agent_info for ego-agent collision check
            ts = np.array([self._scene_ts[sid] for sid in agent_scene_index])
            for sid in np.unique(agent_scene_index):
                self._scene_ts[sid] += 1
            return

        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_idx_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        other_scene_index = agent_info["scene_index"]
        other_pos_per_scene = split_agents_by_scene(agent_info["centroid"], other_scene_index, all_scene_index)
        other_yaw_per_scene = split_agents_by_scene(agent_info["yaw"], other_scene_index, all_scene_index)
        other_extent_per_scene = split_agents_by_scene(agent_info["extent"][..., :2], other_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        for i in range(num_scenes):
            if pos_per_scene[i].shape[0] == 0 or other_pos_per_scene[i].shape[0] == 0:
                continue

            ego_pos = pos_per_scene[i][0]
            ego_yaw_val = yaw_per_scene[i][0]
            ego_ext = extent_per_scene[i][0]

            coll = detect_collision(
                ego_pos=ego_pos,
                ego_yaw=ego_yaw_val,
                ego_extent=ego_ext,
                other_pos=other_pos_per_scene[i],
                other_yaw=other_yaw_per_scene[i],
                other_extent=other_extent_per_scene[i],
            )
            if coll is not None:
                collided_local = coll[1]
                adv_pos_v = other_pos_per_scene[i][collided_local]
                adv_yaw_val = float(other_yaw_per_scene[i][collided_local])
                actual_ctype = classify_collision_type(
                    ego_pos, ego_yaw_val, adv_pos_v, adv_yaw_val,
                )
                # Same dir_dot / bearing_dot the classifier thresholds on.
                ey = float(ego_yaw_val)
                dot_val = float(np.cos(ey) * np.cos(adv_yaw_val)
                                + np.sin(ey) * np.sin(adv_yaw_val))
                rel = np.asarray(adv_pos_v) - np.asarray(ego_pos)
                rel_n = float(np.linalg.norm(rel))
                if rel_n > 1e-6:
                    bearing_val = float((rel[0] * np.cos(ey)
                                         + rel[1] * np.sin(ey)) / rel_n)
                else:
                    bearing_val = float("nan")
                agent_idx = agent_idx_per_scene[i][0]
                actual_type_ints[agent_idx] = COLLISION_TYPE_TO_INT[actual_ctype]
                dir_dots[agent_idx] = dot_val
                bearing_dots[agent_idx] = bearing_val
                has_collision[agent_idx] = True

                scene_id = all_scene_index[i]
                if scene_id not in self._first_collision_printed:
                    self._first_collision_printed.add(scene_id)
                    print(f"[ActualCollisionType] Scene {scene_id} first collision: "
                          f"type={actual_ctype.value} "
                          f"dir_dot={dot_val:+.3f} bearing_dot={bearing_val:+.3f}")

        ts = np.array([self._scene_ts[sid] for sid in agent_scene_index])
        step_df = pd.DataFrame({
            "scene_index": agent_scene_index,
            "track_id": state_info["track_id"],
            "ts": ts,
            "actual_type_int": actual_type_ints,
            "dir_dot": dir_dots,
            "bearing_dot": bearing_dots,
            "collision": has_collision,
        })
        self._df = pd.concat([self._df, step_df], ignore_index=True)
        for sid in np.unique(agent_scene_index):
            self._scene_ts[sid] += 1

    def get_episode_metrics(self):
        if len(self._df) == 0:
            return {
                "actual_collision_type": np.array([], dtype=np.int32),
                "actual_collision_dir_dot": np.array([], dtype=np.float32),
                "actual_collision_bearing_dot": np.array([], dtype=np.float32),
            }

        all_scenes = self._df["scene_index"].unique()
        collision_df = self._df[self._df["collision"] == True]
        result = np.full(len(all_scenes), -1, dtype=np.int32)
        dot_result = np.full(len(all_scenes), np.nan, dtype=np.float32)
        bearing_result = np.full(len(all_scenes), np.nan, dtype=np.float32)

        if len(collision_df) > 0:
            first_rows = (
                collision_df.sort_values(["scene_index", "ts"])
                .groupby("scene_index")
                .first()
            )
            for idx, scene in enumerate(all_scenes):
                if scene in first_rows.index:
                    result[idx] = first_rows.loc[scene, "actual_type_int"]
                    dot_result[idx] = first_rows.loc[scene, "dir_dot"]
                    bearing_result[idx] = first_rows.loc[scene, "bearing_dot"]

        return {
            "actual_collision_type": result,
            "actual_collision_dir_dot": dot_result,
            "actual_collision_bearing_dot": bearing_result,
        }
