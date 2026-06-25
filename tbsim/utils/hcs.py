"""
Event Selector for Safety-Critical Scenario Generation

Scores all (candidate, collision_type) pairs and selects the best one:
  S(i,c) = w_reach * S_reach(i)
          + w_geometry * S_geometry(i,c)
          + w_control * S_control(i,c)
          + w_legality * S_legality(i,c)

Legality scoring uses lane centerline relationships via
determine_centerline_relationship() and find_conflict_point() from
tbsim.utils.agent_rel_classify.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tbsim.utils.collision_constraints import CollisionType


class HCS:
    """Select the best (vehicle, collision_type) pair for adversarial generation."""

    def __init__(
        self,
        w_reach: float = 0.30,
        w_geometry: float = 0.30,
        w_control: float = 0.0,
        w_legality: float = 0.30,
        verbose: bool = False,
    ):
        total = w_reach + w_geometry + w_control + w_legality
        self.w_reach = w_reach / total
        self.w_geometry = w_geometry / total
        self.w_control = w_control / total
        self.w_legality = w_legality / total
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def select_event(
        self,
        ego_state: Dict[str, torch.Tensor],
        candidate_states: List[Dict[str, torch.Tensor]],
        lane_relationships: Optional[Dict] = None,
        lane_interactions: Optional[Dict] = None,
    ) -> Tuple[int, CollisionType, float, Dict]:
        """Select the optimal (candidate, collision_type) pair.

        Args:
            ego_state: dict with position[2], heading, velocity, length, width.
            candidate_states: list of dicts (same keys as ego_state).
            lane_relationships: output of determine_centerline_relationship()
                                keyed by (i,j) -> relationship string.
            lane_interactions: output of determine_centerline_relationship()
                               keyed by (i,j) -> (interaction_type, s1, s2, conflict_point).

        Returns:
            best_idx, best_type, best_score, details
        """
        n_cands = len(candidate_states)
        collision_types = list(CollisionType)

        all_scores = torch.zeros(n_cands, len(collision_types))
        all_details = []

        for i, cand in enumerate(candidate_states):
            for j, ctype in enumerate(collision_types):
                score, details = self._compute_score(
                    ego_state, cand, ctype,
                    candidate_idx=i,
                    lane_relationships=lane_relationships,
                    lane_interactions=lane_interactions,
                )
                all_scores[i, j] = score
                all_details.append({"candidate_idx": i, "collision_type": ctype, **details})

        flat_idx = torch.argmax(all_scores).item()
        best_i = flat_idx // len(collision_types)
        best_j = flat_idx % len(collision_types)
        return best_i, collision_types[best_j], all_scores[best_i, best_j].item(), all_details[flat_idx]

    def select_from_observation(
        self,
        obs_dict: Dict,
        ego_idx: int = 0,
    ) -> Tuple[int, CollisionType, float, Dict]:
        """High-level API: extract states from trajdata observation and run selection.

        Args:
            obs_dict: observation dictionary with centroid, yaw, curr_speed, extent,
                      and extras (centerline_world_xy, has_lane, ref_polyline_ids).
            ego_idx: index of ego in the batch dimension.

        Returns:
            best_candidate_batch_idx, best_type, best_score, details
        """
        centroids = obs_dict["centroid"]  # [B, 2]
        yaws = obs_dict["yaw"]  # [B]
        speeds = obs_dict["curr_speed"]  # [B]
        extents = obs_dict["extent"]  # [B, 2] or [B, 3]
        B = centroids.shape[0]

        ego_state = self._obs_to_state(centroids, yaws, speeds, extents, ego_idx)
        candidate_indices = [i for i in range(B) if i != ego_idx]
        candidate_states = [
            self._obs_to_state(centroids, yaws, speeds, extents, i)
            for i in candidate_indices
        ]

        # Compute lane relationships if centerline data available.
        lane_rels, lane_inters = None, None
        extras = obs_dict.get("extras", {}) or {}
        available_keys = list(extras.keys()) if extras else []
        if "centerline_world_xy" in extras:
            try:
                from tbsim.utils.agent_rel_classify import determine_centerline_relationship
                lane_rels, lane_inters = determine_centerline_relationship(obs_dict)
            except Exception as e:
                if self.verbose:
                    print(f"    [HCS] Lane rel failed: {e}")
                    print(f"    [HCS] extras keys: {available_keys}")
        elif self.verbose:
            print(f"    [HCS] No centerline_world_xy in extras. Available keys: {available_keys}")

        best_local, best_type, best_score, details = self.select_event(
            ego_state, candidate_states,
            lane_relationships=lane_rels,
            lane_interactions=lane_inters,
        )

        # Print detailed scoring for all candidates
        if self.verbose:
            collision_types = list(CollisionType)
            ctype_names = [ct.name for ct in collision_types]
            print(f"    [HCS] ego_idx={ego_idx}, {len(candidate_states)} candidates, lane_rels={'available' if lane_rels else 'None'}")
            for i, cand_batch_idx in enumerate(candidate_indices):
                dist = torch.norm(ego_state["position"] - candidate_states[i]["position"]).item()
                scores_per_type = []
                for ctype in collision_types:
                    _, d = self._compute_score(
                        ego_state, candidate_states[i], ctype,
                        candidate_idx=i,
                        lane_relationships=lane_rels,
                        lane_interactions=lane_inters,
                    )
                    scores_per_type.append(d)
                # Find best collision type for this candidate
                best_ct_idx = max(range(len(collision_types)), key=lambda j: scores_per_type[j]["total_score"])
                best_d = scores_per_type[best_ct_idx]
                lane_rel = lane_rels.get((min(0, i+1), max(0, i+1)), "N/A") if lane_rels else "N/A"
                marker = " <<<" if i == best_local else ""
                print(f"      cand {cand_batch_idx}: dist={dist:.1f}m, lane_rel={lane_rel}, "
                      f"best_type={ctype_names[best_ct_idx]}, "
                      f"reach={best_d['s_reach']:.3f}, geo={best_d['s_geometry']:.3f}, "
                      f"ctrl={best_d['s_control']:.3f}, legal={best_d['s_legality']:.3f}, "
                      f"total={best_d['total_score']:.4f}{marker}")

        # Map local candidate index back to batch index.
        best_batch_idx = candidate_indices[best_local]

        # Compute conflict point for the selected pair.
        conflict_point = self._get_conflict_point(obs_dict, ego_idx, best_batch_idx, lane_inters)
        details["conflict_point"] = conflict_point
        details["batch_adv_idx"] = best_batch_idx

        # Determine if adv is in front of ego (for rear-end sub-type)
        rel_vec = centroids[best_batch_idx] - centroids[ego_idx]
        rel_dir = rel_vec / rel_vec.norm().clamp(min=1e-6)
        ego_heading = ego_state["heading"]
        ego_dir = torch.stack([torch.cos(ego_heading), torch.sin(ego_heading)])
        bearing_dot = torch.dot(rel_dir, ego_dir).item()
        details["adv_in_front"] = bearing_dot > 0

        # Estimate T_collision (for logging / manual-mode use only; REAR_END
        # / CUT_IN / HEAD_ON are overridden to a fixed value in
        # ccfm_guidance._update_T_collision).
        T_collision = self.compute_T_collision(
            ego_state, candidate_states[best_local], best_type, conflict_point
        )
        details["T_collision"] = T_collision

        return best_batch_idx, best_type, best_score, details

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def _compute_score(self, ego, cand, ctype, candidate_idx=0,
                       lane_relationships=None, lane_interactions=None):
        s_reach = self._score_reach(ego, cand)
        s_geo = self._score_geometry(ego, cand, ctype)
        s_ctrl = self._score_control(ego, cand, ctype)
        s_legal = self._score_legality(ego, cand, ctype, candidate_idx,
                                       lane_relationships, lane_interactions)
        total = (self.w_reach * s_reach + self.w_geometry * s_geo
                 + self.w_control * s_ctrl + self.w_legality * s_legal)
        details = {
            "s_reach": s_reach.item(),
            "s_geometry": s_geo.item(),
            "s_control": s_ctrl.item(),
            "s_legality": s_legal.item(),
            "total_score": total.item(),
        }
        return total, details

    def _score_reach(self, ego, cand) -> torch.Tensor:
        """Gaussian distance score: closer => higher."""
        dist = torch.norm(ego["position"] - cand["position"])
        sigma = 50.0
        return torch.exp(-(dist ** 2) / (sigma ** 2))

    def _score_geometry(self, ego, cand, ctype) -> torch.Tensor:
        """Evaluate how well the current pose fits the collision type.

        Uses dot products instead of angle arithmetic:
            ego_dir · cand_dir  = cos(heading_diff):  +1 same, 0 perp, -1 opposite
            approach = -(rel_dir · cand_dir):          +1 toward ego, -1 away from ego
        """
        # Unit direction vectors
        ego_dir = torch.stack([torch.cos(ego["heading"]), torch.sin(ego["heading"])])
        cand_dir = torch.stack([torch.cos(cand["heading"]), torch.sin(cand["heading"])])

        rel_vec = cand["position"] - ego["position"]
        rel_dist = torch.norm(rel_vec).clamp(min=1e-6)
        rel_dir = rel_vec / rel_dist

        # dot(ego_dir, cand_dir): +1=same dir, -1=opposite, 0=perpendicular
        dir_dot = torch.dot(ego_dir, cand_dir)
        # approach: +1=candidate heading toward ego, -1=heading away
        approach = -torch.dot(rel_dir, cand_dir)
        toward = torch.exp(-((approach - 1) ** 2) / 0.5)           # peaks at approach=+1 (toward ego)

        if ctype == CollisionType.REAR_END:
            # Same direction + candidate heading toward ego
            same_dir = (dir_dot + 1) / 2                           # 1 when same dir, 0 when opposite
            return 0.6 * same_dir + 0.4 * toward

        elif ctype == CollisionType.SIDE:
            # Perpendicular headings + candidate heading toward ego
            perp = torch.exp(-(dir_dot ** 2) / 0.3)                # peaks at dir_dot=0 (perpendicular)
            return 0.5 * perp + 0.5 * toward

        elif ctype == CollisionType.CUT_IN:
            # Heading diff ~30° + candidate heading toward ego
            mid_dot = 0.837  # cos(30°)
            fit = torch.exp(-((dir_dot - mid_dot) ** 2) / 0.05)    # peaks at ~30° heading diff
            return 0.6 * fit + 0.4 * toward

        elif ctype == CollisionType.HEAD_ON:
            # Opposite directions + candidate heading toward ego
            opposite = torch.exp(-((dir_dot + 1) ** 2) / 0.3)      # peaks at dir_dot=-1 (opposite)
            return 0.6 * opposite + 0.4 * toward

        return torch.tensor(0.0)

    def _score_control(self, ego, cand, ctype) -> torch.Tensor:
        """TTC-based controllability score using CPA (Closest Point of Approach).

        Uses the same formula as the TTC guidance in guidance_utils.py:
            t_cpa = -(Δp · Δv) / ||Δv||²
            d_cpa = |Δv_y·Δp_x - Δv_x·Δp_y| / ||Δv||

        Optimal t_cpa ~5s, smaller d_cpa is better.
        """
        # Δp = p_adv - p_ego,  Δv = v_adv - v_ego
        delta_p = cand["position"] - ego["position"]
        ego_vel = ego["velocity"] * torch.stack([torch.cos(ego["heading"]), torch.sin(ego["heading"])])
        cand_vel = cand["velocity"] * torch.stack([torch.cos(cand["heading"]), torch.sin(cand["heading"])])
        delta_v = cand_vel - ego_vel

        speed_diff_sq = torch.sum(delta_v ** 2).clamp(min=1e-6)

        # t_cpa: time of closest point of approach
        t_cpa = -torch.dot(delta_p, delta_v) / speed_diff_sq
        t_cpa = torch.relu(t_cpa)  # only future matters

        # d_cpa: distance at closest point of approach
        d_cpa = torch.abs(delta_v[1] * delta_p[0] - delta_v[0] * delta_p[1]) / torch.sqrt(speed_diff_sq)

        # Score: prefer t_cpa ~ 5s and small d_cpa
        t_bw = 5  # time bandwidth
        d_bw = 2.5  # distance bandwidth
        score = torch.exp(-(t_cpa - 5.0) ** 2 / (t_bw ** 2) - d_cpa ** 2 / (d_bw ** 2))
        return score

    def _score_legality(self, ego, cand, ctype, candidate_idx,
                        lane_relationships, lane_interactions) -> torch.Tensor:
        """Lane-relationship-aware legality score.

        Uses determine_centerline_relationship() results to check:
        - intersection → side / head-on
        - merging → cut-in / side
        - same lane → rear-end
        - nearby lanes → rear-end / cut-in
        """
        if lane_relationships is None:
            return torch.tensor(0.01)

        # Find the relationship for the (ego_idx=0, candidate) pair.
        # Keys in lane_relationships are (i, j) with i < j.
        ego_idx_in_obs = 0  # ego is always first in the score loop context
        pair = (min(ego_idx_in_obs, candidate_idx + 1), max(ego_idx_in_obs, candidate_idx + 1))
        rel = lane_relationships.get(pair, "no relationship")

        # Map relationship to collision type compatibility.
        compat = {
            "intersection": {CollisionType.SIDE: 0.95, CollisionType.HEAD_ON: 0.7,
                             CollisionType.CUT_IN: 0.3, CollisionType.REAR_END: 0.3},
            "merging":      {CollisionType.CUT_IN: 0.95, CollisionType.SIDE: 0.6,
                             CollisionType.REAR_END: 0.4, CollisionType.HEAD_ON: 0.1},
            "same lane":    {CollisionType.REAR_END: 0.95, CollisionType.CUT_IN: 0.05,
                             CollisionType.SIDE: 0.05, CollisionType.HEAD_ON: 0.05},
            "nearby lanes": {CollisionType.CUT_IN: 0.7, CollisionType.REAR_END: 0.6,
                             CollisionType.SIDE: 0.4, CollisionType.HEAD_ON: 0.2},
        }

        scores = compat.get(rel, {})
        base = scores.get(ctype, 0.01)

        # Bonus if there is actual interaction (not "no_interaction").
        if lane_interactions is not None:
            inter = lane_interactions.get(pair)
            if inter is not None and inter[0] not in ("no_interaction", "already_passed_conflict"):
                base = min(base + 0.1, 1.0)

        return torch.tensor(base, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Conflict point and T_collision helpers
    # ------------------------------------------------------------------

    def _get_conflict_point(self, obs_dict, ego_idx, adv_idx, lane_interactions):
        """Extract the pre-computed conflict point from lane interactions, or compute one."""
        if lane_interactions is not None:
            pair = (min(ego_idx, adv_idx), max(ego_idx, adv_idx))
            inter = lane_interactions.get(pair)
            if inter is not None and len(inter) >= 4 and inter[3] is not None:
                cp = inter[3]
                if isinstance(cp, torch.Tensor):
                    return cp
                if hasattr(cp, '__len__') and len(cp) == 2:
                    return torch.tensor([cp[0], cp[1]], dtype=torch.float32)

        # Fallback: compute from centerlines if available.
        if "extras" in obs_dict and "centerline_world_xy" in obs_dict["extras"]:
            try:
                from tbsim.utils.agent_rel_classify import find_conflict_point
                ego_cl = obs_dict["extras"]["centerline_world_xy"][ego_idx]
                adv_cl = obs_dict["extras"]["centerline_world_xy"][adv_idx]
                cp, _, _ = find_conflict_point(ego_cl, adv_cl)
                return cp
            except Exception:
                pass

        # Last resort: midpoint between the two vehicles.
        return (obs_dict["centroid"][ego_idx] + obs_dict["centroid"][adv_idx]) / 2

    @staticmethod
    def compute_T_collision(
        ego_state: Dict, cand_state: Dict, ctype: CollisionType,
        conflict_point=None, dt: float = 0.1, horizon: int = 32,
    ) -> int:
        """Estimate the collision timestep index based on distance-to-conflict and speed.

        Returns an integer in [1, horizon-1].
        """
        if conflict_point is not None:
            ego_dist = torch.norm(ego_state["position"] - conflict_point.to(ego_state["position"].device))
            adv_dist = torch.norm(cand_state["position"] - conflict_point.to(cand_state["position"].device))

            ego_speed = ego_state["velocity"].clamp(min=1.0)
            adv_speed = cand_state["velocity"].clamp(min=1.0)

            # Use the slower time-to-arrive as the collision time.
            ego_time = ego_dist / ego_speed
            adv_time = adv_dist / adv_speed
            t_seconds = torch.max(ego_time, adv_time).item()
        else:
            # Fallback: use distance between vehicles and closing speed.
            dist = torch.norm(ego_state["position"] - cand_state["position"])
            closing = (ego_state["velocity"] + cand_state["velocity"]).clamp(min=2.0)
            t_seconds = (dist / closing).item()

        T = max(1, min(int(t_seconds / dt), horizon - 1))
        return T

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_to_state(centroids, yaws, speeds, extents, idx):
        return {
            "position": centroids[idx].detach().float(),
            "heading": yaws[idx].detach().float() if yaws[idx].dim() == 0 else yaws[idx, 0].detach().float(),
            "velocity": speeds[idx].detach().float() if speeds[idx].dim() == 0 else speeds[idx, 0].detach().float(),
            "length": extents[idx, 0].detach().float(),
            "width": extents[idx, 1].detach().float(),
        }
