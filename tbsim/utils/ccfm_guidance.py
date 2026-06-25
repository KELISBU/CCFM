"""
CCFM Guidance Function for Flow Matching

Callable that performs periodic Gauss-Newton projection onto the
collision constraint manifold during ODE integration in
RasterizedFMModel._flow_match_sample_actions().

Compatible with the guide_sample_fn(actions, t=, data_batch_for_guidance=, curr_states=)
signature used by the FM model.
"""

import torch
from typing import Dict, Optional, Tuple

from tbsim.utils.collision_constraints import CollisionConstraints, CollisionType


class CCFMGuidanceFunction:
    """Gauss-Newton projection onto collision constraint manifold.

    Called once per ODE step by the FM sampler.  Performs projection only
    every ``projection_freq`` steps (and always on the last step).

    The constraint evaluation uses:
      - The adversary's predicted trajectory (from forward dynamics on actions)
      - An ego reference trajectory derived from the ego lane centerline
      - A conflict point pre-computed by the HCS
    """

    def __init__(
        self,
        collision_config: Dict,
        dynamics_model,
        device: str = "cuda",
        projection_freq: int = 1,
        max_projection_iters: int = 1,
        projection_tolerance: float = 1e-3,
        step_size: float = 0.8,
        damping: float = 1e-4,
        denormalize_fn=None,
        normalize_fn=None,
    ):
        self.collision_config = collision_config
        self.dynamics_model = dynamics_model
        self.device = device
        self.denormalize_fn = denormalize_fn
        self.normalize_fn = normalize_fn
        self.projection_freq = projection_freq
        self.max_projection_iters = max_projection_iters
        self.projection_tolerance = projection_tolerance
        self.step_size = step_size
        self.damping = damping

        self.constraints = CollisionConstraints(
            device=device,
            dt=collision_config.get("dt", 0.1),
        )
        self._call_count = 0
        self._skip_projection = False  # set by _update_T_collision when dynamic T < 5

        # CCFM Eq.(5) OT reverse update cache. Stores the initial noise u_0 of
        # the current sampling trajectory so projections can be linearly blended
        # back to the current flow time tau':
        #     u_hat_tau' = tau' * u_proj + (1 - tau') * u_0
        # Cached lazily on the first __call__ of each sampling; call reset()
        # between samplings to refresh.
        self._actions_prior: Optional[torch.Tensor] = None
        # Skip projection when violation is too large (constraints unsatisfiable).
        # In world frame, initial violations scale with distance (e.g. 45m → ~50),
        # so threshold must be well above typical initial violations.
        self.max_violation_threshold = 500.0

        # Per-step delta clipping to preserve vehicle dynamics feasibility.
        # actions[:,:,0] = accel (m/s²), actions[:,:,1] = yaw_rate (rad/s)
        self.max_delta_accel = 8.0
        self.max_delta_yawrate = 6.0

        # Per-channel absolute clamp in NORMALIZED space.
        # Derived from physical limits and training normalization:
        #   accel:    ±8 m/s²,    mean=0.02483, std=1.092723 → ±7.3
        #   yaw_rate: ±π/2 rad/s, mean=-3e-5,   std=0.0656   → ±24.0
        self.normalized_action_clamp_accel = 10#7.3
        self.normalized_action_clamp_yawrate = 15
    # ------------------------------------------------------------------
    # guide_sample_fn interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear per-sampling state. Call before each new ODE sampling so the
        OT reverse update (CCFM Eq.5) caches the fresh initial noise u_0 and
        the step counter (used to derive tau') restarts from zero.
        """
        self._call_count = 0
        self._actions_prior = None

    def __call__(
        self,
        actions: torch.Tensor,
        t=None,
        data_batch_for_guidance: Optional[Dict] = None,
        curr_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Called per ODE step.  Returns (possibly projected) actions.

        Only the adversary agent indices are projected; all other agents
        keep their regular FM-sampled actions untouched.
        """
        self._call_count += 1
        n_total = (
            data_batch_for_guidance.get("n_total_steps", 20)
            if data_batch_for_guidance is not None
            else 20
        )

        # Lazily cache the initial noise u_0 for OT reverse update (CCFM Eq.5).
        # First call arrives after one Euler step from pure Gaussian noise, so
        # the cached tensor is an excellent approximation of u_0.
        if (
            self._actions_prior is None
            or self._actions_prior.shape != actions.shape
        ):
            self._actions_prior = actions.detach().clone()

        should_project = (
            (self._call_count % self.projection_freq == 0)
            or (self._call_count >= n_total)
        )
        if should_project and curr_states is not None:
            batch_size = data_batch_for_guidance.get("batch_size", None) if data_batch_for_guidance else None
            adv_idx = self.collision_config.get("adv_idx", None)
            ego_idx = self.collision_config.get("ego_idx", None)

            if batch_size is not None and adv_idx is not None and batch_size > 1:
                total = actions.shape[0]
                num_samples = total // batch_size
                # Build index mask for the adversary agent across all samples
                adv_indices = [adv_idx * num_samples + s for s in range(num_samples)]
                # Also extract ego states for building ego reference trajectory
                ego_states = None
                if ego_idx is not None:
                    ego_indices = [ego_idx * num_samples + s for s in range(num_samples)]
                    ego_states = curr_states[ego_indices]

                adv_actions = actions[adv_indices]
                adv_states = curr_states[adv_indices]

                # if self._call_count == self.projection_freq:
                #     print(f"  [CCFM] First projection at ODE step {self._call_count}: "
                #           f"adv_idx={adv_idx}, ego_idx={ego_idx}, batch_size={batch_size}, "
                #           f"num_samples={num_samples}")

                # Dynamically update T_collision based on world-frame positions
                self._update_T_collision(data_batch_for_guidance, horizon=actions.shape[1])

                # Skip projection if dynamic T_collision too small (FM only)
                if self._skip_projection:
                    return actions

                # Mark whether this is the final ODE step (only last-step
                # diagnostics are printed inside _gauss_newton_project).
                self._is_last_step = self._call_count >= n_total
                projected = self._gauss_newton_project(
                    adv_actions, adv_states, data_batch_for_guidance,
                    ego_curr_states=ego_states,
                )
                # Safety: check for NaN/Inf
                if torch.isnan(projected).any() or torch.isinf(projected).any():
                    print(f"  [CCFM] WARNING: NaN/Inf in projected actions at step {self._call_count}, "
                          f"keeping original actions")
                else:
                    # CCFM Eq.(5): OT reverse update via straight-line displacement.
                    # Blends the projected terminal state back to the current flow
                    # time tau' = call_count / n_total, preventing the "yank" from
                    # GN projection when tau' is small (early sampling steps).
                    tau_prime = min(float(self._call_count) / float(max(n_total, 1)), 1.0)
                    u0_adv = self._actions_prior[adv_indices]
                    projected = tau_prime * projected + (1.0 - tau_prime) * u0_adv

                    actions = actions.clone()
                    actions[adv_indices] = projected
            else:
                self._is_last_step = self._call_count >= n_total
                actions = self._gauss_newton_project(
                    actions, curr_states, data_batch_for_guidance
                )
        return actions

    def _project_single(
        self,
        adv_actions: torch.Tensor,
        adv_states: torch.Tensor,
        data_batch_for_guidance: Optional[Dict],
        ego_curr_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Project a single scene's adv actions. Called by MultiSceneCCFMGuidance.

        Returns projected actions or None if projection is skipped/fails.
        """
        self._call_count += 1
        n_total = (
            data_batch_for_guidance.get("n_total_steps", 20)
            if data_batch_for_guidance is not None
            else 20
        )

        # Cache u_0 for OT reverse update (CCFM Eq.5).
        if (
            self._actions_prior is None
            or self._actions_prior.shape != adv_actions.shape
        ):
            self._actions_prior = adv_actions.detach().clone()

        should_project = (
            (self._call_count % self.projection_freq == 0)
            or (self._call_count >= n_total)
        )
        if not should_project:
            return None

        # if self._call_count == self.projection_freq:
        #     cc = self.collision_config
        #     print(f"  [CCFM-multi] First projection: "
        #           f"type={cc.get('collision_type')}, T_collision={cc.get('T_collision')}")

        self._update_T_collision(data_batch_for_guidance, horizon=adv_actions.shape[1])
        if self._skip_projection:
            return None

        self._is_last_step = self._call_count >= n_total
        projected = self._gauss_newton_project(
            adv_actions, adv_states, data_batch_for_guidance,
            ego_curr_states=ego_curr_states,
        )
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            print(f"  [CCFM-multi] WARNING: NaN/Inf in projected actions, keeping original")
            return None

        # CCFM Eq.(5): OT reverse update (see __call__ for rationale).
        tau_prime = min(float(self._call_count) / float(max(n_total, 1)), 1.0)
        projected = tau_prime * projected + (1.0 - tau_prime) * self._actions_prior
        return projected

    # ------------------------------------------------------------------
    # Dynamic T_collision update
    # ------------------------------------------------------------------

    def _update_T_collision(
        self, data_batch_for_guidance: Optional[Dict], horizon: int
    ):
        """Recompute T_collision and conflict_point dynamically.

        For REAR_END, CUT_IN, HEAD_ON: T_collision is fixed at 20.
        For SIDE: dynamically computed from ego_plan (closest waypoint to adv).

        Args:
            data_batch_for_guidance: dict with world_positions, world_speeds,
                                     optionally ego_plan and world_from_agent.
            horizon: trajectory length (T_act)
        """
        # --- CLI override: --hcs_fixed_ttc forces T for ALL types ---
        fixed_ttc = self.collision_config.get("fixed_ttc")
        if fixed_ttc is not None:
            self.collision_config["T_collision"] = min(int(fixed_ttc), horizon - 1)
            self._skip_projection = False
            return

        # --- Per-type T_collision update ---
        ctype = self.collision_config.get("collision_type")
        if isinstance(ctype, str):
            ctype = CollisionType(ctype)

        # REAR_END / CUT_IN / HEAD_ON (dynamic-fixed-TTC, time-aligned):
        #   - T_collision adapts per step using the paper-style time-to-closest-
        #     approach formula (constant-velocity assumption):
        #         t_col = -(dv · d) / ‖dv‖²
        #     where d = ego_pos - adv_pos, dv = ego_vel - adv_vel.
        #     If t_col ≤ 0 (agents parallel/receding), fall back to T_MAX.
        #   - Clamped to [T_MIN, T_MAX] so the constraint window never collapses
        #     or overruns the FM horizon.
        #   - adv@T and ego@T evaluated at the SAME moment (no ego_target_pos
        #     override) → real collision, no ghost / chase-ego-current issues.
        if ctype in {CollisionType.REAR_END, CollisionType.CUT_IN, CollisionType.HEAD_ON, CollisionType.SIDE}:
            T_MIN = 5      # 0.5s  — adv needs some planning headroom
            T_MAX = 10     # 1.0s  — don't push constraint past effective horizon
            T_dyn = 10     # default if geometry / yaws unavailable

            if data_batch_for_guidance is not None:
                world_pos = data_batch_for_guidance.get("world_positions")
                world_spd = data_batch_for_guidance.get("world_speeds")
                world_yaw = data_batch_for_guidance.get("world_yaws")
                ego_idx = self.collision_config.get("ego_idx", 0)
                adv_idx = self.collision_config.get("adv_idx", 1)
                dt = self.collision_config.get("dt", 0.1)

                if (world_pos is not None and world_spd is not None
                        and world_yaw is not None
                        and ego_idx < world_pos.shape[0]
                        and adv_idx < world_pos.shape[0]):
                    # Reconstruct velocity vectors from scalar speed + yaw.
                    ego_yaw_val = world_yaw[ego_idx]
                    adv_yaw_val = world_yaw[adv_idx]
                    ego_v_vec = torch.stack([
                        world_spd[ego_idx] * torch.cos(ego_yaw_val),
                        world_spd[ego_idx] * torch.sin(ego_yaw_val),
                    ])
                    adv_v_vec = torch.stack([
                        world_spd[adv_idx] * torch.cos(adv_yaw_val),
                        world_spd[adv_idx] * torch.sin(adv_yaw_val),
                    ])
                    d = world_pos[ego_idx, :2] - world_pos[adv_idx, :2]
                    dv = ego_v_vec - adv_v_vec
                    dv_sq = (dv * dv).sum().clamp(min=0.5)   # avoid div-by-zero
                    t_col_sec = (-(dv * d).sum() / dv_sq).item()

                    if t_col_sec > 0:
                        T_dyn = int(t_col_sec / dt)
                    else:
                        # Parallel / receding: keep constraint with max window.
                        T_dyn = T_MAX

            T_dyn = max(T_MIN, min(T_dyn, T_MAX))
            self.collision_config["T_collision"] = min(T_dyn, horizon - 1)
            self._skip_projection = False

            # Clear any stale ego_target overrides so the constraint falls back
            # to trajectories[:, ego_idx, T] (same-time-index ego position).
            self.collision_config.pop("ego_target_pos", None)
            self.collision_config.pop("ego_target_yaw", None)
            self.collision_config.pop("ego_target_speed", None)
            return

        # --- SIDE moved to unified CPA branch above. The following block is now
        # only reached for unknown collision types or when ctype is None. ---
        # (Legacy SIDE path with decoupled horizons preserved here as fallback,
        # in case ego_target_* is desired again — currently unreachable.)
        if data_batch_for_guidance is None:
            return
        world_pos = data_batch_for_guidance.get("world_positions")
        world_spd = data_batch_for_guidance.get("world_speeds")
        if world_pos is None or world_spd is None:
            return

        dt = self.collision_config.get("dt", 0.1)
        ego_idx = self.collision_config.get("ego_idx", 0)
        adv_idx = self.collision_config.get("adv_idx", 1)

        if ego_idx >= world_pos.shape[0] or adv_idx >= world_pos.shape[0]:
            return

        adv_pos = world_pos[adv_idx, :2]
        ego_pos = world_pos[ego_idx, :2]

        # --- Priority 1: ego_plan based ---
        use_ego_plan = self.collision_config.get("use_ego_plan", False)
        ego_plan = data_batch_for_guidance.get("ego_plan") if use_ego_plan else None

        if ego_plan is not None and ego_plan.numel() > 0:
            plan_xy = ego_plan[0].to(adv_pos.device) if ego_plan.dim() == 3 else ego_plan.to(adv_pos.device)

            # Transform ego_plan from ego-local to world frame
            wfa = data_batch_for_guidance.get("world_from_agent")
            if wfa is not None and ego_idx < wfa.shape[0]:
                W = wfa[ego_idx].to(adv_pos.device)
                R = W[:2, :2]
                t = W[:2, 2]
                plan_xy = plan_xy @ R.T + t.unsqueeze(0)

            # Find the ego_plan waypoint closest to adv's current position
            # (unchanged — same selection method as before).
            dists = torch.norm(plan_xy - adv_pos.unsqueeze(0), dim=-1)  # [T_plan]
            min_idx = int(torch.argmin(dists).item())
            cp = plan_xy[min_idx]

            # NEW: fix adv's optimization horizon at 15, store the ego waypoint
            # separately so the constraint compares adv_traj[15] against
            # ego_plan[min_idx] (decoupled horizons).
            FIXED_T_ADV = 15
            self.collision_config["T_collision"] = min(FIXED_T_ADV, horizon - 1)
            self.collision_config["conflict_point"] = cp.detach()

            # Tangent heading from neighboring plan waypoints.
            T_plan = plan_xy.shape[0]
            if T_plan >= 2:
                j = min(min_idx + 1, T_plan - 1)
                i = max(j - 1, 0)
                tangent = plan_xy[j] - plan_xy[i]
                target_yaw = torch.atan2(tangent[1], tangent[0])
            else:
                target_yaw = torch.tensor(0.0, device=adv_pos.device)

            # Speed from consecutive waypoint spacing / dt (approximate).
            if T_plan >= 2:
                j = min(min_idx + 1, T_plan - 1)
                i = max(j - 1, 0)
                target_speed = torch.norm(plan_xy[j] - plan_xy[i]) / max(dt, 1e-3)
            else:
                target_speed = torch.tensor(0.0, device=adv_pos.device)

            self.collision_config["ego_target_pos"] = cp.detach()
            self.collision_config["ego_target_yaw"] = target_yaw.detach()
            self.collision_config["ego_target_speed"] = target_speed.detach()

            self._skip_projection = False
            return

        # --- Fallback: distance / closing speed ---
        ego_speed = world_spd[ego_idx].clamp(min=1.0)
        adv_speed = world_spd[adv_idx].clamp(min=1.0)
        dist = torch.norm(ego_pos - adv_pos)
        closing = (ego_speed + adv_speed).clamp(min=2.0)
        t_seconds = (dist / closing).item()

        old_T = self.collision_config["T_collision"]
        new_T = max(1, min(int(t_seconds / dt), horizon - 1))
        self.collision_config["T_collision"] = new_T

        self._skip_projection = False

    # ------------------------------------------------------------------
    # Coordinate transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _local_traj_to_world(
        traj_local: torch.Tensor,
        world_from_agent: torch.Tensor,
    ) -> torch.Tensor:
        """Transform trajectory from agent-local frame to world frame.

        Args:
            traj_local: [B, T, 3] (x, y, yaw) in agent-local frame
            world_from_agent: [3, 3] affine transform matrix

        Returns:
            traj_world: [B, T, 3] (x, y, yaw) in world frame
        """
        B, T, _ = traj_local.shape
        pos_local = traj_local[..., :2]  # [B, T, 2]
        yaw_local = traj_local[..., 2]   # [B, T]

        # Apply affine transform to positions: p_world = R @ p_local + t
        R = world_from_agent[:2, :2]     # [2, 2]
        t = world_from_agent[:2, 2]      # [2]
        # pos_world = pos_local @ R^T + t
        pos_world = torch.matmul(pos_local, R.T) + t.unsqueeze(0).unsqueeze(0)  # [B, T, 2]

        # Rotate yaw by the agent's heading in world frame
        agent_yaw = torch.atan2(world_from_agent[1, 0], world_from_agent[0, 0])
        yaw_world = yaw_local + agent_yaw  # [B, T]

        return torch.cat([pos_world, yaw_world.unsqueeze(-1)], dim=-1)  # [B, T, 3]

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def _adv_traj_to_world(
        self,
        adv_traj_local: torch.Tensor,
        data_batch_for_guidance: Optional[Dict],
    ) -> torch.Tensor:
        """Transform adversary trajectory from adv-local frame to world frame.

        Args:
            adv_traj_local: [B, T, 3] (x, y, yaw) in adv-local frame
            data_batch_for_guidance: must contain world_from_agent [N_agents, 3, 3]

        Returns:
            adv_traj_world: [B, T, 3] (x, y, yaw) in world frame
        """
        adv_idx = self.collision_config.get("adv_idx", 1)
        wfa = data_batch_for_guidance.get("world_from_agent")
        if wfa is None:
            return adv_traj_local  # fallback: no transform available

        # world_from_agent[adv_idx]: [3, 3]
        W = wfa[adv_idx].to(adv_traj_local.device)
        R = W[:2, :2]            # [2, 2] rotation
        t = W[:2, 2]             # [2]    translation
        agent_yaw = torch.atan2(W[1, 0], W[0, 0])  # scalar

        B, T, _ = adv_traj_local.shape
        pos_local = adv_traj_local[..., :2]   # [B, T, 2]
        yaw_local = adv_traj_local[..., 2]    # [B, T]

        # p_world = p_local @ R^T + t
        pos_world = torch.matmul(pos_local, R.T) + t.unsqueeze(0).unsqueeze(0)
        yaw_world = yaw_local + agent_yaw

        return torch.cat([pos_world, yaw_world.unsqueeze(-1)], dim=-1)

    def _gauss_newton_project(
        self,
        actions: torch.Tensor,
        curr_states: torch.Tensor,
        data_batch_for_guidance: Optional[Dict],
        ego_curr_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project actions onto the constraint manifold via Gauss-Newton.

        actions shape: [B, T, 2]  (accel, yaw_rate) - adversary's actions (adv-local frame)
        curr_states shape: [B, 4]  (x, y, yaw, speed) - adversary's states (adv-local)
        ego_curr_states shape: [B, 4] - ego's states (ego-local)

        Constraint evaluation is done in WORLD frame:
          - ego_traj: built from world-frame centerline using world-frame ego position
          - adv_traj: forward dynamics in adv-local, then transformed to world frame
        Actions are updated in adv-local frame (no offroad issue).

        Only the first K_opt timesteps of actions are optimized (default: T_collision).
        This concentrates the GN update on actions that will actually be used
        in closed-loop simulation (n_step_action ≈ 5), instead of spreading
        updates across all 32 timesteps where later ones are discarded.
        """
        B, T_act, _ = actions.shape
        step_time = self.collision_config.get("dt", 0.1)

        # Build ego reference trajectory in WORLD frame
        ego_traj = self._build_ego_reference(B, T_act, ego_curr_states, data_batch_for_guidance)
        # ego_traj: [B, T_act, 3]  (x, y, yaw) in WORLD frame

        actions_proj = actions.detach().clone()

        # Local config for constraint evaluation (0=ego, 1=adv in stacked tensor)
        local_config = dict(self.collision_config)
        local_config["ego_idx"] = 0
        local_config["adv_idx"] = 1

        # Use T_collision from outer _update_T_collision logic.
        T_coll = local_config.get("T_collision", 20)
        T_coll = min(T_coll, T_act - 1)
        local_config["T_collision"] = T_coll
        # Optimize all timesteps up to T_collision.
        K_opt = min(T_coll, T_act)

        for it in range(self.max_projection_iters):
            # ---- Differentiable section --------------------------------
            # _gauss_newton_project is called from get_action, which runs
            # inside torch.no_grad() (see wrappers.py:320). We locally
            # re-enable autograd so the forward graph from a_grad -> h is
            # built and torch.autograd.grad can compute the Jacobian.
            with torch.enable_grad():
                # Fresh grad-tracked copy for autograd Jacobian (per-iter,
                # since actions_proj is updated out-of-graph by the GN step).
                a_grad = actions_proj.detach().clone().requires_grad_(True)

                # 1. Forward dynamics -> adversary trajectory in adv-local frame
                adv_traj_local = self._forward_dynamics(curr_states, a_grad, step_time)

                # 2. Transform adv trajectory to WORLD frame
                adv_traj_world = self._adv_traj_to_world(adv_traj_local, data_batch_for_guidance)

                # 3. Stack in world frame and compute constraints
                trajectories = torch.stack([ego_traj, adv_traj_world], dim=1)  # [B, 2, T, 3]
                h = self.constraints.compute_constraints(trajectories, local_config)  # [B, n_c]

                # 4. Convergence check uses detached values.
                h_det = h.detach()
                if torch.isnan(h_det).any() or torch.isinf(h_det).any():
                    print(f"    [CCFM] NaN/Inf in constraints at iter {it}, aborting projection")
                    actions_proj = actions.detach().clone()
                    break
                violation = torch.norm(h_det, dim=-1).mean().item()
                if it == 0:
                    if getattr(self, "_is_last_step", False):
                        print(f"    [CCFM-diag] T_collision={T_coll}, K_opt={K_opt}, "
                              f"h=[{h_det[0,0].item():.3f}, {h_det[0,1].item():.3f}, {h_det[0,2].item():.3f}], "
                              f"violation={violation:.4f}, "
                              f"ego_ref_pos=({ego_traj[0, min(T_coll, T_act-1), 0].item():.1f}, "
                              f"{ego_traj[0, min(T_coll, T_act-1), 1].item():.1f}), "
                              f"adv_pos_T=({adv_traj_world[0, min(T_coll, T_act-1), 0].detach().item():.1f}, "
                              f"{adv_traj_world[0, min(T_coll, T_act-1), 1].detach().item():.1f})")
                    if violation > self.max_violation_threshold:
                        return actions.detach().clone()
                if violation < self.projection_tolerance:
                    break

                # 5. Autograd Jacobian J[:, c, :] = d h[:, c] / d a_grad[:, :K_opt, :]
                #    One backward pass per constraint row (n_c backward vs the old
                #    (K_opt*2) forward of finite differences — typically ~10x faster).
                n_c = h.shape[1]
                J_rows = []
                for c in range(n_c):
                    grad_c = torch.autograd.grad(
                        outputs=h[:, c].sum(),
                        inputs=a_grad,
                        retain_graph=(c < n_c - 1),
                        create_graph=False,
                        allow_unused=False,
                    )[0]  # [B, T_act, 2]
                    J_rows.append(grad_c[:, :K_opt, :].reshape(B, -1))
                J = torch.stack(J_rows, dim=1).detach()  # [B, n_c, K_opt*2]
            # ---- End differentiable section ----------------------------

            # 6. Gauss-Newton step: delta = J^T (J J^T + lambda I)^{-1} h
            JJT = torch.bmm(J, J.transpose(1, 2))  # [B, n_c, n_c]
            eye = torch.eye(n_c, device=actions.device).unsqueeze(0)
            JJT_reg = JJT + self.damping * eye

            try:
                JJT_inv = torch.linalg.inv(JJT_reg)
            except Exception:
                JJT_inv = torch.linalg.pinv(JJT_reg)

            delta = torch.bmm(
                J.transpose(1, 2),
                torch.bmm(JJT_inv, h_det.unsqueeze(-1)),
            ).squeeze(-1)  # [B, K_opt*2]

            # 7. Apply GN update to first K_opt actions (no per-step delta clamp)
            delta_2d = delta.reshape(B, K_opt, 2)
            actions_proj = actions_proj.clone()
            actions_proj[:, :K_opt] = actions_proj[:, :K_opt] - self.step_size * delta_2d

            if torch.isnan(actions_proj).any() or torch.isinf(actions_proj).any():
                print(f"    [CCFM] NaN/Inf in actions after GN iter {it}, reverting")
                actions_proj = actions.detach().clone()
                break

        # Clamp projected actions in normalized space to ±N std (absolute bound).
        # This prevents GN cumulative updates from pushing actions out of distribution.
        # Per-channel clamp: accel (ch0) and yaw_rate (ch1) have different physical limits.
        pre_clamp = actions_proj.clone()
        actions_proj[..., 0].clamp_(-self.normalized_action_clamp_accel, self.normalized_action_clamp_accel)
        actions_proj[..., 1].clamp_(-self.normalized_action_clamp_yawrate, self.normalized_action_clamp_yawrate)
        n_clamped = (pre_clamp != actions_proj).sum().item()

        # Diagnostic: only print on the final ODE step so log stays readable.
        if getattr(self, "_is_last_step", False):
            delta_total = actions_proj - actions
            print(f"    [CCFM-diag] After {min(it+1, self.max_projection_iters)} GN iters: "
                  f"final_violation={violation:.4f}, "
                  f"n_clamped={n_clamped}, "
                  f"accel_delta[:5]=[{', '.join(f'{delta_total[0,t,0].item():.3f}' for t in range(min(5, T_act)))}], "
                  f"accel_proj[:5]=[{', '.join(f'{actions_proj[0,t,0].item():.3f}' for t in range(min(5, T_act)))}], "
                  f"yawrate_proj[:5]=[{', '.join(f'{actions_proj[0,t,1].item():.3f}' for t in range(min(5, T_act)))}]")

        return actions_proj.detach()

    def _forward_dynamics(
        self, curr_states: torch.Tensor, actions: torch.Tensor, step_time: float
    ) -> torch.Tensor:
        """Run forward dynamics and return trajectory [B, T, 3] (x, y, yaw) in agent-local frame."""
        if self.dynamics_model is not None:
            # Actions may be in normalized space; denormalize before dynamics
            actions_phys = self.denormalize_fn(actions) if self.denormalize_fn is not None else actions
            x_states = self.dynamics_model.forward_dynamics(
                initial_states=curr_states, actions=actions_phys, step_time=step_time,
            )
            if isinstance(x_states, tuple):
                return torch.cat([x_states[1], x_states[2]], dim=-1)  # [B, T, 3]
            else:
                return x_states[..., :3]
        else:
            return actions

    # ------------------------------------------------------------------
    # Ego reference trajectory from centerline
    # ------------------------------------------------------------------

    def _get_ego_world_state(
        self, data_batch_for_guidance: Optional[Dict]
    ) -> Optional[torch.Tensor]:
        """Get ego's world-frame state [x, y, yaw, speed] from guidance data.

        curr_states from the FM model has x=0, y=0 (agent-local frame),
        so we must use data_batch["centroid"], ["yaw"], ["curr_speed"] instead.
        """
        if data_batch_for_guidance is None:
            return None
        ego_idx = self.collision_config.get("ego_idx", 0)
        world_pos = data_batch_for_guidance.get("world_positions")
        world_spd = data_batch_for_guidance.get("world_speeds")
        world_yaw = data_batch_for_guidance.get("world_yaws")
        if world_pos is None or ego_idx >= world_pos.shape[0]:
            return None

        device = world_pos.device
        x = world_pos[ego_idx, 0]
        y = world_pos[ego_idx, 1]
        yaw = world_yaw[ego_idx] if world_yaw is not None else torch.tensor(0.0, device=device)
        speed = world_spd[ego_idx] if world_spd is not None else torch.tensor(5.0, device=device)
        return torch.tensor([x, y, yaw, speed], device=device)

    def _build_ego_reference(
        self,
        B: int,
        T: int,
        curr_states: torch.Tensor,
        data_batch_for_guidance: Optional[Dict],
    ) -> torch.Tensor:
        """Build an ego reference trajectory in WORLD frame.

        Priority order:
          1. ego_plan  — actual planner output (StrivePolicy_trajdata positions)
          2. ego_centerline — lane-centerline extrapolation (world-frame centerline)
          3. Straight-line constant-velocity fallback (world-frame)

        Returns: [B, T, 3] (x, y, yaw) in WORLD frame
        """
        dt = self.collision_config.get("dt", 0.1)
        device = curr_states.device if curr_states is not None else self.device

        # Get ego's world-frame state (NOT agent-local where x=y=0)
        ego_world = self._get_ego_world_state(data_batch_for_guidance)
        if ego_world is None and curr_states is not None:
            # Fallback: use curr_states (will be wrong if agent-local, but better than crash)
            ego_world = curr_states[0]
        elif ego_world is None:
            ego_world = torch.zeros(4, device=device)

        # --- Priority 1: use actual ego plan (only when --egoplan is set) ---
        use_ego_plan = self.collision_config.get("use_ego_plan", False)
        if use_ego_plan and data_batch_for_guidance is not None:
            ego_plan = data_batch_for_guidance.get("ego_plan")
            if ego_plan is not None and ego_plan.numel() > 0:
                try:
                    # if self._call_count <= self.projection_freq:
                    #     print(f"  [CCFM] Using ego_plan for reference trajectory (shape={ego_plan.shape})")
                    return self._ego_plan_trajectory(ego_plan, ego_world, T, B, device, data_batch_for_guidance)
                except Exception as e:
                    print(f"  [CCFM] ego_plan trajectory failed ({e}), falling back")

        # --- Priority 2 (default): lane centerline extrapolation (world frame) ---
        ego_cl = None
        if data_batch_for_guidance is not None:
            ego_cl = data_batch_for_guidance.get("ego_centerline")

        if ego_cl is not None and ego_cl.dim() >= 2 and ego_cl.shape[0] >= 2:
            return self._centerline_trajectory(ego_cl, ego_world, T, dt, B)

        # --- Priority 3: constant-velocity straight line (world frame) ---
        return self._straight_line_trajectory_world(ego_world, T, dt, B)

    def _ego_plan_trajectory(
        self,
        ego_plan: torch.Tensor,
        ego_world_state: torch.Tensor,
        T: int,
        B: int,
        device,
        data_batch_for_guidance: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Use the ego planner's actual output positions as the reference trajectory.

        Args:
            ego_plan: [B_ego, T_plan, 2] in EGO-LOCAL frame (action output).
            ego_world_state: [4] (x, y, yaw, speed) — ego's WORLD-frame state.
            T: desired trajectory length.
            B: batch size (num_samples).
            device: torch device.
            data_batch_for_guidance: contains world_from_agent for coordinate transform.

        Returns: [B, T, 3] (x, y, yaw) in WORLD frame
        """
        if ego_plan.dim() == 3:
            plan_xy = ego_plan[0].to(device)  # [T_plan, 2] ego-local
        else:
            plan_xy = ego_plan.to(device)

        # Transform ego_plan from ego-local to world frame
        ego_idx = self.collision_config.get("ego_idx", 0)
        wfa = data_batch_for_guidance.get("world_from_agent") if data_batch_for_guidance else None
        if wfa is not None and ego_idx < wfa.shape[0]:
            W = wfa[ego_idx].to(device)  # [3, 3]
            R = W[:2, :2]               # [2, 2]
            t = W[:2, 2]                # [2]
            plan_xy = plan_xy @ R.T + t.unsqueeze(0)  # [T_plan, 2] now in world frame

        # Prepend current ego world position
        ego_pos_now = ego_world_state[:2].unsqueeze(0).to(device)  # [1, 2]
        plan_xy = torch.cat([ego_pos_now, plan_xy], dim=0)

        # Pad or truncate to length T
        if plan_xy.shape[0] < T + 1:
            last_dir = plan_xy[-1] - plan_xy[-2]
            for _ in range(T + 1 - plan_xy.shape[0]):
                plan_xy = torch.cat([plan_xy, plan_xy[-1:] + last_dir], dim=0)
        plan_xy = plan_xy[:T + 1]

        dxy = plan_xy[1:] - plan_xy[:-1]
        dxy_norm = torch.norm(dxy, dim=-1, keepdim=True)  # [T, 1]
        # When ego is stationary, dxy ≈ 0 → atan2 gives wrong yaw.
        # Fall back to ego's actual world-frame heading for near-zero displacements.
        ego_yaw_world = ego_world_state[2].to(device)
        raw_yaws = torch.atan2(dxy[:, 1], dxy[:, 0])
        stationary_mask = (dxy_norm.squeeze(-1) < 0.01)  # threshold: 1cm
        yaws = torch.where(stationary_mask, ego_yaw_world.expand_as(raw_yaws), raw_yaws)
        positions = plan_xy[1:]
        traj = torch.cat([positions, yaws.unsqueeze(-1)], dim=-1)  # [T, 3]
        return traj.unsqueeze(0).expand(B, -1, -1)

    def _centerline_trajectory(
        self, centerline: torch.Tensor, ego_world_state: torch.Tensor, T: int, dt: float, B: int
    ) -> torch.Tensor:
        """Sample T points along the WORLD-frame centerline at constant speed.

        Args:
            centerline: [N_pts, 2] in WORLD frame (centerline_world_xy)
            ego_world_state: [4] (x, y, yaw, speed) in WORLD frame
            T: trajectory length
            dt: time step
            B: batch size

        Returns: [B, T, 3] (x, y, yaw) in WORLD frame
        """
        device = ego_world_state.device
        if centerline.dim() == 3:
            centerline = centerline[0]
        centerline = centerline.to(device)

        # Find closest point on centerline to ego's WORLD position
        ego_pos = ego_world_state[:2].unsqueeze(0)  # [1, 2]
        dists = torch.norm(centerline - ego_pos, dim=-1)
        start_idx = torch.argmin(dists).item()

        ego_speed = ego_world_state[3].clamp(min=1.0).item()

        cl_pts = centerline[start_idx:]
        if cl_pts.shape[0] < 2:
            return self._straight_line_trajectory_world(ego_world_state, T, dt, B)

        seg_lens = torch.norm(cl_pts[1:] - cl_pts[:-1], dim=-1)
        cum_len = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg_lens, dim=0)])
        desired_dists = torch.arange(1, T + 1, device=device, dtype=torch.float32) * dt * ego_speed

        positions = []
        yaws = []
        for d in desired_dists:
            idx = torch.searchsorted(cum_len, d.unsqueeze(0)).item()
            idx = min(idx, len(cl_pts) - 1)
            prev_idx = max(idx - 1, 0)

            if idx == prev_idx:
                pos = cl_pts[idx]
            else:
                alpha = ((d - cum_len[prev_idx]) / (cum_len[idx] - cum_len[prev_idx] + 1e-6)).clamp(0, 1)
                pos = cl_pts[prev_idx] + alpha * (cl_pts[idx] - cl_pts[prev_idx])
            positions.append(pos)

            if idx < len(cl_pts) - 1:
                direction = cl_pts[min(idx + 1, len(cl_pts) - 1)] - cl_pts[idx]
            else:
                direction = cl_pts[idx] - cl_pts[max(idx - 1, 0)]
            yaw = torch.atan2(direction[1], direction[0])
            yaws.append(yaw)

        pos_tensor = torch.stack(positions)  # [T, 2]
        yaw_tensor = torch.stack(yaws).unsqueeze(-1)  # [T, 1]
        traj = torch.cat([pos_tensor, yaw_tensor], dim=-1)  # [T, 3]
        return traj.unsqueeze(0).expand(B, -1, -1)

    def _straight_line_trajectory_world(
        self, ego_world_state: torch.Tensor, T: int, dt: float, B: int
    ) -> torch.Tensor:
        """Constant-velocity straight-line extrapolation in WORLD frame.

        Args:
            ego_world_state: [4] (x, y, yaw, speed) in world frame
            T: trajectory length
            dt: time step
            B: batch size

        Returns: [B, T, 3] (x, y, yaw) in world frame
        """
        device = ego_world_state.device
        x0 = ego_world_state[0]
        y0 = ego_world_state[1]
        yaw0 = ego_world_state[2]
        speed = ego_world_state[3].clamp(min=0.1)

        vx = speed * torch.cos(yaw0)
        vy = speed * torch.sin(yaw0)

        t_steps = torch.arange(1, T + 1, device=device, dtype=torch.float32) * dt
        xs = x0 + vx * t_steps     # [T]
        ys = y0 + vy * t_steps     # [T]
        yaws = yaw0.expand(T)      # [T]

        traj = torch.stack([xs, ys, yaws], dim=-1)  # [T, 3]
        return traj.unsqueeze(0).expand(B, -1, -1)  # [B, T, 3]


class MultiSceneCCFMGuidance:
    """Wraps multiple CCFMGuidanceFunction instances for per-scene projection.

    When batch_size > 1 and each scene has its own collision_config (different
    ego_idx, adv_idx, collision_type, etc.), this class creates one
    CCFMGuidanceFunction per scene and dispatches projection to each.
    """

    def __init__(
        self,
        collision_configs: list,
        dynamics_model,
        device: str = "cuda",
        denormalize_fn=None,
        normalize_fn=None,
        **ccfm_params,
    ):
        self.per_scene_fns = []
        self.collision_configs = collision_configs
        for cc in collision_configs:
            fn = CCFMGuidanceFunction(
                collision_config=dict(cc),  # copy to avoid cross-scene mutation
                dynamics_model=dynamics_model,
                device=device,
                denormalize_fn=denormalize_fn,
                normalize_fn=normalize_fn,
                **ccfm_params,
            )
            self.per_scene_fns.append(fn)

    def __call__(
        self,
        actions: torch.Tensor,
        t=None,
        data_batch_for_guidance=None,
        curr_states=None,
    ) -> torch.Tensor:
        """Project each scene's adv independently."""
        if curr_states is None or data_batch_for_guidance is None:
            return actions

        batch_size = data_batch_for_guidance.get("batch_size", None)
        if batch_size is None or batch_size <= 1:
            # Fallback to first fn
            return self.per_scene_fns[0](actions, t, data_batch_for_guidance, curr_states)

        total = actions.shape[0]
        num_samples = total // batch_size
        actions_out = actions.clone()

        for scene_fn in self.per_scene_fns:
            cc = scene_fn.collision_config
            adv_idx = cc.get("adv_idx")
            ego_idx = cc.get("ego_idx")
            if adv_idx is None or adv_idx >= batch_size:
                continue

            # Extract adv slice
            adv_indices = [adv_idx * num_samples + s for s in range(num_samples)]
            ego_indices = [ego_idx * num_samples + s for s in range(num_samples)] if ego_idx is not None else None

            adv_actions = actions_out[adv_indices]
            adv_states = curr_states[adv_indices]
            ego_states = curr_states[ego_indices] if ego_indices is not None else None

            # Build a per-scene data_batch_for_guidance with batch_size=1
            scene_data = dict(data_batch_for_guidance)
            scene_data["batch_size"] = 1  # tell CCFMGuidanceFunction it's single-scene

            # Update world-frame fields for this scene's ego/adv
            if "world_positions" in scene_data and ego_idx < scene_data["world_positions"].shape[0]:
                # Keep only ego and adv positions for dynamic T_collision
                wp = scene_data["world_positions"]
                ws = scene_data.get("world_speeds")
                # Remap: ego->0, adv->1 in the sub-batch
                scene_data["world_positions"] = torch.stack([wp[ego_idx], wp[adv_idx]])
                if ws is not None:
                    scene_data["world_speeds"] = torch.stack([ws[ego_idx], ws[adv_idx]])
                if "world_yaws" in scene_data:
                    wy = scene_data["world_yaws"]
                    scene_data["world_yaws"] = torch.stack([wy[ego_idx], wy[adv_idx]])
                if "world_from_agent" in scene_data:
                    wfa = scene_data["world_from_agent"]
                    scene_data["world_from_agent"] = torch.stack([wfa[ego_idx], wfa[adv_idx]])

            # Provide ego plan if available
            if "ego_plan" in scene_data and scene_data["ego_plan"] is not None:
                ep = scene_data["ego_plan"]
                if ep.dim() >= 2 and ego_idx < ep.shape[0]:
                    scene_data["ego_plan"] = ep[ego_idx:ego_idx+1]

            # Provide centerlines if available
            if "ego_centerline" not in scene_data and "collision_configs" in data_batch_for_guidance:
                extras_cl = data_batch_for_guidance.get("extras", {})

            # Override collision_config indices to local (ego=0, adv=1)
            scene_fn.collision_config["ego_idx"] = 0
            scene_fn.collision_config["adv_idx"] = 1

            # Run single-scene projection (batch_size=1 path)
            projected = scene_fn._project_single(
                adv_actions, adv_states, scene_data, ego_states
            )
            if projected is not None:
                actions_out[adv_indices] = projected

        return actions_out

    def filter(self, actions, state, data_batch_for_guidance):
        """Delegate filtering to each per-scene fn and combine."""
        # Use first fn's filter as they all share the same structure
        if hasattr(self.per_scene_fns[0], "filter"):
            return self.per_scene_fns[0].filter(actions, state, data_batch_for_guidance)
        return actions


class CombinedFMGuidance:
    """Combined guidance for Flow Matching: gradient guidance + CCFM projection.

    Each ODE step:
      1. Gradient guidance (collision + optional causecollision + route) via autograd
      2. CCFM projection (hard collision constraints between ego & adv)

    Collision avoidance guidance masks out the ego-adv pair (adv_mode=True),
    so it only prevents collisions between:
      - ego ↔ other agents
      - adv ↔ other agents
    The CCFM projection handles the ego-adv collision.
    """

    def __init__(
        self,
        ccfm_fn: CCFMGuidanceFunction,
        collision_guidance,
        causecollision_guidance,
        route_guidance,
        dynamics_model,
        step_time: float = 0.1,
        inner_lr: float = 0.02,
        causecollision_lr: float = None,
        route_lr: float = 0.02,
        inner_beta=[0.5, 3.0],
        collision_inner_beta=None,
        causecollision_inner_beta=None,
        route_inner_beta=None,
        denormalize_fn=None,
        skip_ccfm=False,
    ):
        self.ccfm_fn = ccfm_fn
        self.skip_ccfm = skip_ccfm
        self.collision_guidance = collision_guidance
        self.causecollision_guidance = causecollision_guidance
        self.route_guidance = route_guidance
        self.dynamics_model = dynamics_model
        self.denormalize_fn = denormalize_fn
        self.step_time = step_time
        self.inner_lr = inner_lr
        self.causecollision_lr = inner_lr if causecollision_lr is None else causecollision_lr
        self.route_lr = route_lr
        # Per-channel beta: [accel_beta, yawrate_beta] in normalized space.
        # Default: accel=0.5, yawrate=3.0 (≈0.55 m/s² and ≈0.2 rad/s in physical space)
        default_beta = [0.5, 3.0]
        self.inner_beta = self._parse_beta(inner_beta, default_beta)
        self.collision_inner_beta = self._parse_beta(collision_inner_beta, self.inner_beta)
        self.causecollision_inner_beta = self._parse_beta(causecollision_inner_beta, self.inner_beta)
        self.route_inner_beta = self._parse_beta(route_inner_beta, self.inner_beta)

    @staticmethod
    def _parse_beta(beta, default):
        """Parse beta into [accel_beta, yawrate_beta] list."""
        if beta is None:
            return list(default) if isinstance(default, (list, tuple)) else [default, default]
        if isinstance(beta, (list, tuple)):
            return list(beta)
        # Scalar: use same value for both channels (backward compatible)
        return [beta, beta]

    def __call__(
        self,
        actions: torch.Tensor,
        t=None,
        data_batch_for_guidance: Optional[Dict] = None,
        curr_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # --- Step 1: Gradient guidance (collision avoidance + route) ---
        if data_batch_for_guidance is not None and curr_states is not None:
            actions = self._gradient_guidance(
                actions, curr_states, data_batch_for_guidance
            )

        # --- Step 2: CCFM projection (ego-adv hard constraints) ---
        if not self.skip_ccfm:
            actions = self.ccfm_fn(
                actions, t=t,
                data_batch_for_guidance=data_batch_for_guidance,
                curr_states=curr_states,
            )
        return actions

    def _gradient_guidance(
        self,
        actions: torch.Tensor,
        curr_states: torch.Tensor,
        data_batch_for_guidance: Dict,
    ) -> torch.Tensor:
        """Compute gradient from collision avoidance and route losses, apply to actions."""
        actions_original = actions.detach()
        actions_g = actions_original.clone().requires_grad_(True)

        # Forward dynamics to get state trajectory
        if self.dynamics_model is not None:
            # actions_g is in normalized space; denormalize before dynamics
            actions_phys = self.denormalize_fn(actions_g) if self.denormalize_fn is not None else actions_g
            x_states = self.dynamics_model.forward_dynamics(
                initial_states=curr_states,
                actions=actions_phys,
                step_time=self.step_time,
            )
            if isinstance(x_states, tuple):
                # parallel mode: (x_all, x_and_y, yaw) → concat pos + speed + yaw
                state_traj = x_states[0]  # [B, T, 4] = (x, y, speed, yaw)
            else:
                # chain mode: [B, T, 4] = (x, y, speed, yaw)
                state_traj = x_states
        else:
            state_traj = actions_g

        coll_obj = None
        causecoll_obj = None
        route_obj = None

        # Collision avoidance loss (ego↔other, adv↔other, NOT ego↔adv)
        if self.collision_guidance is not None:
            try:
                # Expand per-agent fields to match B*num_samples if needed
                BN = actions_g.shape[0]
                coll_data = dict(data_batch_for_guidance)
                for key in ("world_from_agent", "curr_speed", "yaw"):
                    if key in coll_data and coll_data[key].shape[0] != BN:
                        num_samples = BN // coll_data[key].shape[0]
                        coll_data[key] = coll_data[key].repeat_interleave(num_samples, dim=0)
                coll_data["BN"] = BN
                coll_loss = self.collision_guidance.calculate_loss(
                    actions_g, state_traj, coll_data
                )
                coll_obj = coll_loss.sum() * self.inner_lr
            except Exception as e:
                if not hasattr(self, "_coll_warn_printed"):
                    print(f"  [FM-Guidance] Collision guidance error: {e}")
                    self._coll_warn_printed = True

        # Causecollision loss (targeted ego↔adv interaction for adversary)
        if self.causecollision_guidance is not None:
            try:
                BN = actions_g.shape[0]
                cause_data = dict(data_batch_for_guidance)
                for key in ("world_from_agent", "curr_speed", "yaw"):
                    if key in cause_data and cause_data[key].shape[0] != BN:
                        num_samples = BN // cause_data[key].shape[0]
                        cause_data[key] = cause_data[key].repeat_interleave(num_samples, dim=0)
                cause_data["BN"] = BN
                causecoll_loss = self.causecollision_guidance.calculate_loss(
                    actions_g, state_traj, cause_data
                )
                causecoll_obj = causecoll_loss.sum() * self.causecollision_lr
            except Exception as e:
                if not hasattr(self, "_causecoll_warn_printed"):
                    print(f"  [FM-Guidance] Causecollision guidance error: {e}")
                    self._causecoll_warn_printed = True

        # Route loss
        if self.route_guidance is not None:
            try:
                # Expand guidance data to match B*num_samples if needed
                BN = actions_g.shape[0]  # B * num_samples
                route_data = dict(data_batch_for_guidance)
                if "centerline" in route_data:
                    cl = route_data["centerline"]
                    if cl.shape[0] != BN:
                        num_samples = BN // cl.shape[0]
                        route_data["centerline"] = cl.repeat_interleave(num_samples, dim=0)
                if "lane_avail" in route_data:
                    la = route_data["lane_avail"]
                    if la.shape[0] != BN:
                        num_samples = BN // la.shape[0]
                        route_data["lane_avail"] = la.repeat_interleave(num_samples, dim=0)
                route_loss = self.route_guidance.calculate_loss(
                    actions_g, state_traj, route_data
                )
                route_obj = route_loss.sum() * self.route_lr
            except Exception as e:
                if not hasattr(self, "_route_warn_printed"):
                    print(f"  [FM-Guidance] Route guidance error: {e}")
                    self._route_warn_printed = True

        grad_terms = []
        if coll_obj is not None and getattr(coll_obj, "requires_grad", False):
            grad_terms.append((coll_obj, self.collision_inner_beta))
        if causecoll_obj is not None and getattr(causecoll_obj, "requires_grad", False):
            grad_terms.append((causecoll_obj, self.causecollision_inner_beta))
        if route_obj is not None and getattr(route_obj, "requires_grad", False):
            grad_terms.append((route_obj, self.route_inner_beta))

        if grad_terms:
            delta_total = torch.zeros_like(actions_original)
            for i, (obj, beta) in enumerate(grad_terms):
                grad_i = torch.autograd.grad(
                    -obj,
                    actions_g,
                    retain_graph=(i < len(grad_terms) - 1),
                )[0]
                delta_i = grad_i.detach()
                if beta is not None:
                    # Per-channel clamp: beta = [accel_beta, yawrate_beta]
                    beta_t = torch.tensor(beta, device=delta_i.device, dtype=delta_i.dtype)
                    delta_i = torch.clamp(delta_i, -beta_t, beta_t)
                delta_total = delta_total + delta_i
            actions = actions_original + delta_total

        return actions.detach()

    @staticmethod
    def _gather_best_sample(tensor: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
        """Gather the best sample along dim=1 from a [B, N, ...] tensor."""
        B = tensor.shape[0]
        gather_idx = best_idx.view(B, 1, *([1] * (tensor.ndim - 2))).expand(B, 1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=gather_idx)

    @staticmethod
    def _repeat_interleave_if_needed(value, target_first_dim: int):
        """Repeat a tensor on dim=0 when it is stored per-agent (B) but samples are per-candidate (B*N)."""
        if not torch.is_tensor(value) or value.ndim == 0:
            return value
        if value.shape[0] == target_first_dim:
            return value
        if value.shape[0] <= 0 or target_first_dim % value.shape[0] != 0:
            return value
        return value.repeat_interleave(target_first_dim // value.shape[0], dim=0)

    def _build_filter_guidance_data(self, data_batch_for_guidance: Dict, BN: int) -> Dict:
        """Prepare a guidance-data copy for final filtering on flattened [B*N, ...] candidates.

        Keep scene-level fields (e.g. scene_ids, ego_extents) at B, and only expand
        per-agent fields that must align with flattened candidates.
        """
        filt_data = dict(data_batch_for_guidance)
        for key in ("centerline", "lane_avail", "world_from_agent", "yaw", "curr_speed"):
            if key in filt_data:
                filt_data[key] = self._repeat_interleave_if_needed(filt_data[key], BN)
        return filt_data

    def filter(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
        data_batch_for_guidance: Dict,
        return_indices: bool = False,
    ):
        """Select the best final candidate using only collision + route losses.

        This runs *after* ODE integration/projection and therefore does not affect
        projection itself. It only ranks the final 20 candidates and returns the best.
        """
        if action.ndim != 4:
            raise ValueError(f"Expected action shape [B,N,T,2] for FM filter, got {tuple(action.shape)}")
        if state is None or state.ndim != 4:
            raise ValueError("FM filter requires final state trajectories with shape [B,N,T,D].")

        B, N = action.shape[:2]
        BN = B * N
        action_flat = action.reshape(BN, *action.shape[2:])
        state_flat = state.reshape(BN, *state.shape[2:])

        filt_data = self._build_filter_guidance_data(data_batch_for_guidance, BN)
        total_loss = torch.zeros(action_flat.shape[:2], device=action.device)
        used_any_loss = False

        if self.collision_guidance is not None:
            try:
                coll_loss = self.collision_guidance.calculate_loss(action_flat, state_flat, filt_data)
                total_loss = total_loss + coll_loss
                used_any_loss = True
            except Exception as e:
                if not hasattr(self, "_filter_coll_warn_printed"):
                    print(f"  [FM-Filter] Collision loss error: {e}")
                    self._filter_coll_warn_printed = True

        if self.route_guidance is not None:
            try:
                route_loss = self.route_guidance.calculate_loss(action_flat, state_flat, filt_data)
                total_loss = total_loss + route_loss
                used_any_loss = True
            except Exception as e:
                if not hasattr(self, "_filter_route_warn_printed"):
                    print(f"  [FM-Filter] Route loss error: {e}")
                    self._filter_route_warn_printed = True

        if not used_any_loss:
            best_idx = torch.zeros(B, dtype=torch.long, device=action.device)
            best_actions = action[:, 0:1]
            if return_indices:
                return best_actions, best_idx, None
            return best_actions

        score_matrix = total_loss.sum(dim=-1).reshape(B, N)  # [B, N]
        best_idx = torch.argmin(score_matrix, dim=1)
        best_actions = self._gather_best_sample(action, best_idx)

        if return_indices:
            return best_actions, best_idx, score_matrix
        return best_actions
