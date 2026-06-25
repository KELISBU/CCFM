"""
Collision Constraints for Safety-Critical Scenario Generation

Defines hard constraints for 4 collision types:
1. Direction Constraint
2. Contact Point Constraint
3. Relative Heading Constraint
4. Svt Level Constraint (relative velocity)

Constraints operate on trajectories with state = (x, y, theta).
Velocity is computed numerically from position differences.
"""

import torch
from typing import Dict, Tuple
from enum import Enum


class CollisionType(Enum):
    """Collision type enumeration — canonical definition for the project."""
    REAR_END = "rear-end"
    SIDE = "side"
    CUT_IN = "cut-in"
    HEAD_ON = "head-on"


# Unified heading-alignment interval per collision type (paper Eq. 28-29).
# d̂_dot = n̂_{adv} · n̂_{ego}; constraint enforces d̂_dot ∈ [d_low, d_high].
# Note: cos(15°) ≈ 0.966.
HEADING_INTERVALS = {
    CollisionType.REAR_END: (0.966, 1.00),   # adv ≈ same direction as ego (±15°)
    CollisionType.SIDE:     (-0.17, 0.17),   # adv ⟂ ego (80°-100° or -100°--80°)
    CollisionType.CUT_IN:   (0.707, 0.966),  # adv at 15°-45° to ego
    CollisionType.HEAD_ON:  (-1.00, -0.966), # adv ≈ opposite of ego (±15° around 180°)
}


class CollisionConstraints:
    """Evaluate structured hard constraints for each collision type."""

    def __init__(self, device="cuda", dt=0.1):
        self.device = device
        self.dt = dt

        self.constraint_weights = {
            "contact_point": 1.5,#1.3,
            "relative_heading": 10,#10.0,#10.0, #1.0,
            "svt_level": 1.0,#1.0,#0.5,
        }

        # Dead-zone tolerances: constraint residual is 0 when within tolerance.
        # Only values exceeding the tolerance produce a non-zero residual.
        self.dead_zone = {
            "contact_dist": 0.1,                                      # ±1.0m around target distance
            "heading_angle": torch.deg2rad(torch.tensor(10.0)),       # ±10° around target heading
            "svt_speed": 1.0,                                      # ±1 m/s around target speed diff
        }

        # Distance gating for heading constraints:
        # gate = sigmoid((threshold - dist) / scale)
        # Far away → gate≈0 (heading inactive), close → gate≈1 (heading active)
        self.heading_dist_gate_threshold = 10.0  # meters
        self.heading_dist_gate_scale = 3.0       # transition sharpness

        # SIDE relative-heading sign cache for cross≈0 hysteresis.
        self._side_target_sign_cache = None
        self._side_cache_key = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_constraints(
        self,
        trajectories: torch.Tensor,
        collision_config: Dict,
    ) -> torch.Tensor:
        """Compute constraint residuals for the given collision type.

        Args:
            trajectories: [B, N_agents, T, state_dim] where state = (x, y, theta).
                          Velocity is derived numerically.
            collision_config: dict with keys ego_idx, adv_idx, collision_type,
                              T_collision, ego_length, ego_width, adv_length, adv_width.

        Returns:
            h: [B, 4] constraint residuals (h=0 means satisfied).
        """
        collision_type = collision_config["collision_type"]
        if isinstance(collision_type, str):
            collision_type = CollisionType(collision_type)

        dispatch = {
            CollisionType.REAR_END: self._rear_end_constraints,
            CollisionType.SIDE: self._side_collision_constraints,
            CollisionType.CUT_IN: self._cut_in_constraints,
            CollisionType.HEAD_ON: self._head_on_constraints,
        }
        fn = dispatch.get(collision_type)
        if fn is None:
            raise ValueError(f"Unknown collision type: {collision_type}")
        return fn(trajectories, collision_config)

    def check_constraint_satisfaction(self, h: torch.Tensor) -> Tuple[bool, Dict]:
        """Check whether all constraints are within tolerance.

        Constraints use dead-zone formulation: h=0 means within tolerance.
        So satisfied iff all h values are near zero.

        Args:
            h: [B, 3] constraint residuals (contact, heading, svt).

        Returns:
            satisfied: bool
            violations: dict of per-constraint mean absolute violation.
        """
        names = ["contact_point", "relative_heading", "svt_level"]
        violations = {}
        for i, name in enumerate(names):
            violations[name] = torch.abs(h[:, i]).mean().item()

        # h already incorporates dead-zone: h=0 means satisfied.
        # A small epsilon accounts for numerical noise.
        eps = 1e-3
        satisfied = all(v < eps for v in violations.values())
        return satisfied, violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_velocity(self, positions: torch.Tensor) -> torch.Tensor:
        """Estimate scalar speed from (x, y) positions via finite differences.

        Args:
            positions: [B, T, 2]

        Returns:
            speed at each timestep: [B, T]  (last step = second-to-last speed).
        """
        dp = positions[:, 1:] - positions[:, :-1]  # [B, T-1, 2]
        speed = torch.norm(dp, dim=-1) / self.dt  # [B, T-1]
        # Pad the last timestep with the previous speed.
        speed = torch.cat([speed, speed[:, -1:]], dim=1)  # [B, T]
        return speed

    def _extract_states(self, trajectories, config):
        """Extract ego and adv state at T_collision, including numerically-derived speed."""
        ego_idx = config["ego_idx"]
        adv_idx = config["adv_idx"]
        T_total = trajectories.shape[2]
        # Clamp T_collision to last valid index to avoid out-of-bounds
        T = min(config["T_collision"], T_total - 1)

        ego_traj = trajectories[:, ego_idx]   # [B, T_total, 3]
        adv_traj = trajectories[:, adv_idx]   # [B, T_total, 3]

        ego_speed = self._estimate_velocity(ego_traj[..., :2])  # [B, T_total]
        adv_speed = self._estimate_velocity(adv_traj[..., :2])  # [B, T_total]

        ego_x = ego_traj[:, T, 0]
        ego_y = ego_traj[:, T, 1]
        ego_theta = ego_traj[:, T, 2]
        ego_v = ego_speed[:, T]

        adv_x = adv_traj[:, T, 0]
        adv_y = adv_traj[:, T, 1]
        adv_theta = adv_traj[:, T, 2]
        adv_v = adv_speed[:, T]

        return (ego_x, ego_y, ego_theta, ego_v), (adv_x, adv_y, adv_theta, adv_v)

    @staticmethod
    def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _distance_gate(self, ego_x, ego_y, adv_x, adv_y) -> torch.Tensor:
        """Smooth gate that suppresses heading constraint when vehicles are far apart.

        Returns a value in (0, 1):
          - dist >> threshold  →  gate ≈ 0  (heading inactive)
          - dist << threshold  →  gate ≈ 1  (heading fully active)
        """
        dist = torch.sqrt((adv_x - ego_x) ** 2 + (adv_y - ego_y) ** 2)
        return torch.sigmoid(
            (self.heading_dist_gate_threshold - dist) / self.heading_dist_gate_scale
        )

    def _unified_heading_residual(
        self,
        ego_theta: torch.Tensor,
        adv_theta: torch.Tensor,
        ego_x: torch.Tensor,
        ego_y: torch.Tensor,
        adv_x: torch.Tensor,
        adv_y: torch.Tensor,
        ctype: "CollisionType",
    ) -> torch.Tensor:
        """Unified heading residual (paper Eq. 28-29).

        h_hdg = w · g_t · ReLU(max(d_low - d̂_dot, d̂_dot - d_high))²

        where d̂_dot = cos(adv_θ - ego_θ) = n̂_adv · n̂_ego,
              [d_low, d_high] = HEADING_INTERVALS[ctype].
        """
        ego_dir = torch.stack([torch.cos(ego_theta), torch.sin(ego_theta)], dim=-1)
        adv_dir = torch.stack([torch.cos(adv_theta), torch.sin(adv_theta)], dim=-1)
        dot = (ego_dir * adv_dir).sum(dim=-1)

        d_low, d_high = HEADING_INTERVALS[ctype]
        excess = torch.relu(torch.maximum(
            torch.tensor(d_low, device=dot.device) - dot,
            dot - torch.tensor(d_high, device=dot.device),
        ))
        gate = self._distance_gate(ego_x, ego_y, adv_x, adv_y)
        return excess.pow(2) * self.constraint_weights["relative_heading"] * gate

    def _select_side_target_heading(
        self,
        ego_x: torch.Tensor,
        ego_y: torch.Tensor,
        ego_theta: torch.Tensor,
        adv_x: torch.Tensor,
        adv_y: torch.Tensor,
        config: Dict,
    ) -> torch.Tensor:
        """Choose SIDE heading target (±pi/2) that points toward ego.

        Near cross=0, keep previous sign to avoid left/right oscillation.
        """
        dx = adv_x - ego_x
        dy = adv_y - ego_y
        cross = torch.cos(ego_theta) * dy - torch.sin(ego_theta) * dx

        # Compare which perpendicular direction (+/- pi/2) is more aligned with adv->ego.
        to_ego = torch.stack([ego_x - adv_x, ego_y - adv_y], dim=-1)
        to_ego = to_ego / torch.norm(to_ego, dim=-1, keepdim=True).clamp(min=1e-6)

        plus_dir = torch.stack(
            [torch.cos(ego_theta + torch.pi / 2), torch.sin(ego_theta + torch.pi / 2)],
            dim=-1,
        )
        minus_dir = torch.stack(
            [torch.cos(ego_theta - torch.pi / 2), torch.sin(ego_theta - torch.pi / 2)],
            dim=-1,
        )
        plus_score = (plus_dir * to_ego).sum(dim=-1)
        minus_score = (minus_dir * to_ego).sum(dim=-1)

        toward_sign = torch.where(
            plus_score >= minus_score,
            torch.ones_like(cross),
            -torch.ones_like(cross),
        )

        cross_stable_eps = float(config.get("side_cross_stable_eps", 0.5))
        cache_key = (int(config.get("ego_idx", -1)), int(config.get("adv_idx", -1)))

        need_reset = (
            self._side_target_sign_cache is None
            or self._side_cache_key != cache_key
            or self._side_target_sign_cache.shape != toward_sign.shape
        )
        if need_reset:
            cache = toward_sign.detach().clone()
        else:
            cache = self._side_target_sign_cache.to(toward_sign.device)

        near_zero = torch.abs(cross) <= cross_stable_eps
        stable_sign = torch.where(near_zero, cache, toward_sign)
        updated_cache = torch.where(near_zero, cache, stable_sign).detach().clone()

        self._side_target_sign_cache = updated_cache
        self._side_cache_key = cache_key

        return stable_sign * (torch.pi / 2)

    # ------------------------------------------------------------------
    # Per-type constraints
    # ------------------------------------------------------------------

    def _rear_end_constraints(self, trajectories, config):
        ego, adv = self._extract_states(trajectories, config)
        ego_x, ego_y, ego_theta, ego_v = ego
        adv_x, adv_y, adv_theta, adv_v = adv
        ego_length = config.get("ego_length", 4.5)
        adv_length = config.get("adv_length", 4.5)
        w = self.constraint_weights
        dz = self.dead_zone

        ego_dir = torch.stack([torch.cos(ego_theta), torch.sin(ego_theta)], dim=-1)
        adv_dir = torch.stack([torch.cos(adv_theta), torch.sin(adv_theta)], dim=-1)

        # Determine if adv is in front of ego using INITIAL state (t=0),
        # not the projected T_collision state.  This prevents bearing_dot
        # from flipping during GN iterations.
        ego_idx = config["ego_idx"]
        adv_idx = config["adv_idx"]
        ego_t0 = trajectories[:, ego_idx, 0]   # [B, 3] at t=0
        adv_t0 = trajectories[:, adv_idx, 0]   # [B, 3] at t=0
        ego_dir_t0 = torch.stack([torch.cos(ego_t0[:, 2]), torch.sin(ego_t0[:, 2])], dim=-1)
        rel_vec_t0 = torch.stack([adv_t0[:, 0] - ego_t0[:, 0], adv_t0[:, 1] - ego_t0[:, 1]], dim=-1)
        rel_dist_t0 = torch.norm(rel_vec_t0, dim=-1).clamp(min=1e-6)
        bearing_dot_t0 = ((rel_vec_t0 / rel_dist_t0.unsqueeze(-1)) * ego_dir_t0).sum(dim=-1)
        adv_in_front = (bearing_dot_t0 > 0).float().detach()  # fixed across GN iters

        # 1. Contact (squared distance, target = 0, no relu):
        #    adv behind: (adv_front - ego_rear)² + (adv_front - ego_rear)²
        #    adv front:  (ego_front - adv_rear)² + (ego_front - adv_rear)²
        # h = ‖A - B‖² × w ; minimum at A == B (bumpers perfectly aligned).
        ego_rear_x = ego_x - (ego_length / 2) * torch.cos(ego_theta)
        ego_rear_y = ego_y - (ego_length / 2) * torch.sin(ego_theta)
        adv_front_x = adv_x + (adv_length / 2) * torch.cos(adv_theta)
        adv_front_y = adv_y + (adv_length / 2) * torch.sin(adv_theta)

        ego_front_x = ego_x + (ego_length / 2) * torch.cos(ego_theta)
        ego_front_y = ego_y + (ego_length / 2) * torch.sin(ego_theta)
        adv_rear_x = adv_x - (adv_length / 2) * torch.cos(adv_theta)
        adv_rear_y = adv_y - (adv_length / 2) * torch.sin(adv_theta)

        dist_behind = torch.sqrt(
            (adv_front_x - ego_rear_x) ** 2 + (adv_front_y - ego_rear_y) ** 2 + 1e-6
        )
        dist_front = torch.sqrt(
            (ego_front_x - adv_rear_x) ** 2 + (ego_front_y - adv_rear_y) ** 2 + 1e-6
        )
        dist = adv_in_front * dist_front + (1 - adv_in_front) * dist_behind
        h_contact = dist * w["contact_point"]

        # 2. Heading (unified, paper Eq. 28-29): dot ∈ [0.95, 1.00] (same direction)
        h_heading = self._unified_heading_residual(
            ego_theta, adv_theta, ego_x, ego_y, adv_x, adv_y, CollisionType.REAR_END
        )

        # 3. Svt: branch on adv position
        #    adv behind: adv faster than ego (adv_v - ego_v >= 3)
        #    adv in front: ego faster than adv (ego_v - adv_v >= 3), i.e. adv braked
        svt_behind = 3.0 - (adv_v - ego_v)
        svt_front = 3.0 - (ego_v - adv_v)
        speed_deficit = adv_in_front * svt_front + (1 - adv_in_front) * svt_behind
        h_svt = torch.relu(speed_deficit - dz["svt_speed"]) * w["svt_level"]

        return torch.stack([h_contact, h_heading, h_svt], dim=-1)

    def _side_collision_constraints(self, trajectories, config):
        ego, adv = self._extract_states(trajectories, config)
        ego = self._override_ego_with_target(ego, config)
        ego_x, ego_y, ego_theta, ego_v = ego
        adv_x, adv_y, adv_theta, adv_v = adv
        ego_width = config.get("ego_width", 1.8)
        w = self.constraint_weights
        dz = self.dead_zone

        # 1. Contact (squared distance, target = 0, no relu):
        #    contact pair = (adv_center, ego_side_point)
        #    ego_side_point = ego_center + ego_width/2 × ego_right × side_sign
        #    side_sign = +1 if adv on ego's right (frozen at t=0), else -1.
        #    h = 0 when adv_center meets ego_side_point (adv pressed on ego's side).
        ego_idx = config["ego_idx"]
        adv_idx = config["adv_idx"]
        ego_t0 = trajectories[:, ego_idx, 0]
        adv_t0 = trajectories[:, adv_idx, 0]
        right_x_t0 = torch.sin(ego_t0[:, 2])
        right_y_t0 = -torch.cos(ego_t0[:, 2])
        lateral_t0 = (adv_t0[:, 0] - ego_t0[:, 0]) * right_x_t0 + \
                     (adv_t0[:, 1] - ego_t0[:, 1]) * right_y_t0
        adv_on_right = (lateral_t0 > 0).float().detach()
        side_sign = 2.0 * adv_on_right - 1.0   # +1 right, -1 left

        right_x = torch.sin(ego_theta)
        right_y = -torch.cos(ego_theta)
        ego_side_x = ego_x + (ego_width / 2) * side_sign * right_x
        ego_side_y = ego_y + (ego_width / 2) * side_sign * right_y

        dx = adv_x - ego_side_x
        dy = adv_y - ego_side_y
        h_contact = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6) * w["contact_point"]

        # 2. Heading (unified, paper Eq. 28-29): dot ∈ [-0.17, 0.17] (perpendicular)
        h_heading = self._unified_heading_residual(
            ego_theta, adv_theta, ego_x, ego_y, adv_x, adv_y, CollisionType.SIDE
        )

        # 3. Lateral relative velocity >= 2 m/s, dead-zone ±1 m/s
        ego_vel = torch.stack([ego_v * torch.cos(ego_theta), ego_v * torch.sin(ego_theta)], dim=-1)
        adv_vel = torch.stack([adv_v * torch.cos(adv_theta), adv_v * torch.sin(adv_theta)], dim=-1)
        rel_vel = adv_vel - ego_vel
        perp = torch.stack([-torch.sin(ego_theta), torch.cos(ego_theta)], dim=-1)
        lateral_vel = torch.abs((rel_vel * perp).sum(dim=-1))
        speed_deficit = 2.0 - lateral_vel
        h_svt = torch.relu(speed_deficit - dz["svt_speed"]) * w["svt_level"]

        return torch.stack([h_contact, h_heading, h_svt], dim=-1)

    @staticmethod
    def _override_ego_with_target(ego, config):
        """Replace ego state at T_collision with pre-computed target (CUT_IN/HEAD_ON).

        When ccfm_guidance has stored ``ego_target_pos`` / ``ego_target_yaw``
        / ``ego_target_speed`` in the config (set by the CUT_IN/HEAD_ON
        branch of ``_update_T_collision``), use those as the ego reference
        the constraint compares adv against.  This decouples:
          - adv's optimization horizon (fixed T_collision = 15)
          - the collision waypoint on ego's future path (picked via
            ego_plan closest to adv, same as SIDE).
        """
        pos = config.get("ego_target_pos")
        if pos is None:
            return ego
        ego_x, _ego_y, ego_theta, ego_v = ego
        device = ego_x.device
        ones = torch.ones_like(ego_x)
        new_x = ones * pos[0].to(device)
        new_y = ones * pos[1].to(device)
        yaw = config.get("ego_target_yaw")
        spd = config.get("ego_target_speed")
        new_theta = ones * yaw.to(device) if yaw is not None else ego_theta
        new_v = ones * spd.to(device) if spd is not None else ego_v
        return (new_x, new_y, new_theta, new_v)

    def _cut_in_constraints(self, trajectories, config):
        ego, adv = self._extract_states(trajectories, config)
        ego = self._override_ego_with_target(ego, config)
        ego_x, ego_y, ego_theta, ego_v = ego
        adv_x, adv_y, adv_theta, adv_v = adv
        w = self.constraint_weights
        dz = self.dead_zone

        # 1. Contact (squared distance, target = 0, no relu):
        #    contact pair = (adv_center, ego_front) — ego's front bumper
        #    should meet adv's body centre at collision.
        ego_length = config.get("ego_length", 4.5)
        ego_front_x = ego_x + (ego_length / 2) * torch.cos(ego_theta)
        ego_front_y = ego_y + (ego_length / 2) * torch.sin(ego_theta)
        dx = adv_x - ego_front_x
        dy = adv_y - ego_front_y
        h_contact = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6) * w["contact_point"]

        # 2. Heading (unified, paper Eq. 28-29): dot ∈ [0.707, 0.966] (15°-45° offset)
        h_heading = self._unified_heading_residual(
            ego_theta, adv_theta, ego_x, ego_y, adv_x, adv_y, CollisionType.CUT_IN
        )

        # 3. Relative velocity >= 5 m/s, dead-zone ±1 m/s
        rel_v = torch.sqrt(
            ((adv_v * torch.cos(adv_theta)) - (ego_v * torch.cos(ego_theta))) ** 2
            + ((adv_v * torch.sin(adv_theta)) - (ego_v * torch.sin(ego_theta))) ** 2
        )
        speed_deficit = 5.0 - rel_v
        h_svt = torch.relu(speed_deficit - dz["svt_speed"]) * w["svt_level"]

        return torch.stack([h_contact, h_heading, h_svt], dim=-1)

    def _head_on_constraints(self, trajectories, config):
        ego, adv = self._extract_states(trajectories, config)
        ego = self._override_ego_with_target(ego, config)
        ego_x, ego_y, ego_theta, ego_v = ego
        adv_x, adv_y, adv_theta, adv_v = adv
        ego_length = config.get("ego_length", 4.5)
        adv_length = config.get("adv_length", 4.5)
        w = self.constraint_weights
        dz = self.dead_zone

        # 1. Contact (squared distance, target = 0, no relu):
        #    contact pair = (adv_front, ego_front) — nose-to-nose at collision.
        ego_front_x = ego_x + (ego_length / 2) * torch.cos(ego_theta)
        ego_front_y = ego_y + (ego_length / 2) * torch.sin(ego_theta)
        adv_front_x = adv_x + (adv_length / 2) * torch.cos(adv_theta)
        adv_front_y = adv_y + (adv_length / 2) * torch.sin(adv_theta)
        dx = adv_front_x - ego_front_x
        dy = adv_front_y - ego_front_y
        h_contact = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6) * w["contact_point"]

        # 2. Heading (unified, paper Eq. 28-29): dot ∈ [-1.00, -0.95] (opposite direction)
        h_heading = self._unified_heading_residual(
            ego_theta, adv_theta, ego_x, ego_y, adv_x, adv_y, CollisionType.HEAD_ON
        )

        # 3. Combined speed >= 8 m/s, dead-zone ±1 m/s
        speed_deficit = 8.0 - (adv_v + ego_v)
        h_svt = torch.relu(speed_deficit - dz["svt_speed"]) * w["svt_level"]

        return torch.stack([h_contact, h_heading, h_svt], dim=-1)


def build_collision_config_from_obs(
    ego_idx: int,
    adv_idx: int,
    collision_type: CollisionType,
    T_collision: int,
    obs_dict: Dict,
    conflict_point=None,
) -> Dict:
    """Build a collision_config dict from environment observation data.

    Args:
        ego_idx, adv_idx: batch indices of ego and adversary.
        collision_type: CollisionType enum value.
        T_collision: timestep index at which collision should occur.
        obs_dict: observation dictionary with 'extent' key.
        conflict_point: optional [2] tensor of the lane conflict point.

    Returns:
        collision_config dict suitable for CollisionConstraints.compute_constraints().
    """
    extent = obs_dict.get("extent")
    ego_length = float(extent[ego_idx, 0]) if extent is not None else 4.5
    ego_width = float(extent[ego_idx, 1]) if extent is not None else 1.8
    adv_length = float(extent[adv_idx, 0]) if extent is not None else 4.5
    adv_width = float(extent[adv_idx, 1]) if extent is not None else 1.8

    config = {
        "ego_idx": ego_idx,
        "adv_idx": adv_idx,
        "collision_type": collision_type,
        "T_collision": T_collision,
        "ego_length": ego_length,
        "ego_width": ego_width,
        "adv_length": adv_length,
        "adv_width": adv_width,
    }
    if conflict_point is not None:
        config["conflict_point"] = conflict_point
    return config
