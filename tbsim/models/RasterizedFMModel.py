from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from torchcfm import ConditionalFlowMatcher

import tbsim.models.base_models as base_models
from tbsim.models.temporal import TemporalUnet
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import get_stationary_mask


class RasterizedFMModel(nn.Module):
    """Raster-based flow-matching model with the same public API as RasterizedDiffusionModel."""

    def __init__(
        self,
        model_arch: str,
        input_image_shape,
        map_feature_dim: int,
        dynamics_config: tuple,
        history_encoder_config,
        diffnet_config: tuple,
        diffuse_config: Dict,
        weights_scaling: List[float],
        use_spatial_softmax=False,
        spatial_softmax_kwargs=None,
        rasterize_mode="point",
        drop_cond_prob=0,
        do_guidance=False,
        guide_config=None,
        nusc_norm_info=None,
    ) -> None:
        super().__init__()
        if rasterize_mode is None:
            rasterize_mode = "point"
        assert rasterize_mode in ["point", "square"]

        self.rasterize_mode = rasterize_mode
        self.drop_cond_prob = drop_cond_prob
        self.weights_scaling = weights_scaling
        self.do_guidance = do_guidance
        self.guide_config = guide_config

        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=map_feature_dim,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
            output_activation=nn.ReLU,
        )
        self.agent_history_encoder = base_models.HistoryEncoder(history_encoder_config)
        self.other_history_encoder = base_models.HistoryEncoder(history_encoder_config)

        self.diffnet = TemporalUnet(**diffnet_config, dynamics_config=dynamics_config)
        self.diffuse_args = diffuse_config
        # torchcfm (svd env) exposes ConditionalFlowMatcher with a single `sigma`.
        # Use `fm_sigma` if provided, otherwise fall back to `fm_sigma_max` for backwards-compat.
        fm_sigma = float(0.1)#self.diffuse_args.get("fm_sigma", self.diffuse_args.get("fm_sigma_max", 0.0)))
        self.flow_matcher = ConditionalFlowMatcher(sigma=fm_sigma)
        if self.guide_config is not None and hasattr(self.guide_config, "params"):
            for key in ["num_samples", "sample_step", "sampling_mode"]:
                if key in self.guide_config.params:
                    self.diffuse_args[key] = self.guide_config.params[key]
        self.default_chosen_inds = [0, 1]  # accel, yaw_rate
        self._init_norm_buffers(nusc_norm_info)

    def _init_norm_buffers(self, nusc_norm_info):
        add_coeffs = None
        div_coeffs = None

        if nusc_norm_info is not None:
            flow_info = None
            if isinstance(nusc_norm_info, dict):
                flow_info = nusc_norm_info.get("flow_matching", None)

                # Optional direct format support.
                if flow_info is None and "add_coeffs" in nusc_norm_info and "div_coeffs" in nusc_norm_info:
                    add_coeffs = torch.as_tensor(nusc_norm_info["add_coeffs"], dtype=torch.float32)
                    div_coeffs = torch.as_tensor(nusc_norm_info["div_coeffs"], dtype=torch.float32)

            if flow_info is not None:
                # Expected format from FlowmatchingTrafficConfig.nusc_norm_info:
                # {"flow_matching": [mean_tuple, std_tuple], ...}
                if isinstance(flow_info, (list, tuple)) and len(flow_info) >= 2:
                    mean_coeffs = torch.as_tensor(flow_info[0], dtype=torch.float32)
                    std_coeffs = torch.as_tensor(flow_info[1], dtype=torch.float32)
                    add_coeffs = -mean_coeffs
                    div_coeffs = std_coeffs
                elif isinstance(flow_info, dict):
                    if "add_coeffs" in flow_info and "div_coeffs" in flow_info:
                        add_coeffs = torch.as_tensor(flow_info["add_coeffs"], dtype=torch.float32)
                        div_coeffs = torch.as_tensor(flow_info["div_coeffs"], dtype=torch.float32)
                    elif "mean" in flow_info and "std" in flow_info:
                        add_coeffs = -torch.as_tensor(flow_info["mean"], dtype=torch.float32)
                        div_coeffs = torch.as_tensor(flow_info["std"], dtype=torch.float32)

        if add_coeffs is None or div_coeffs is None:
            # No external norm provided: use identity transform (fixed, no EMA updates).
            add_coeffs = torch.zeros(2, dtype=torch.float32)
            div_coeffs = torch.ones(2, dtype=torch.float32)

        if add_coeffs.ndim != 1 or div_coeffs.ndim != 1 or add_coeffs.shape != div_coeffs.shape:
            raise ValueError(
                f"Invalid norm coeffs shape: add={tuple(add_coeffs.shape)}, div={tuple(div_coeffs.shape)}"
            )
        if max(self.default_chosen_inds) >= add_coeffs.shape[0]:
            raise ValueError(
                "flow_matching norm coeffs do not have enough dims for action normalization "
                f"(need indices {self.default_chosen_inds}, got {add_coeffs.shape[0]} dims)"
            )

        self.register_buffer("add_coeffs", add_coeffs)
        self.register_buffer("div_coeffs", div_coeffs.clamp_min(1e-6))

    def _get_norm_coeffs(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        if chosen_inds is None or len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        idx = torch.as_tensor(chosen_inds, device=self.add_coeffs.device, dtype=torch.long)
        dx_add = self.add_coeffs.index_select(0, idx).to(device=target_traj_orig.device, dtype=target_traj_orig.dtype)
        dx_div = self.div_coeffs.index_select(0, idx).to(device=target_traj_orig.device, dtype=target_traj_orig.dtype)
        view_shape = [1] * (target_traj_orig.ndim - 1) + [len(chosen_inds)]
        return dx_add.view(*view_shape), dx_div.view(*view_shape)

    def scale_traj(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        """
        Scale trajectory/features using (x + add_coeffs) / div_coeffs.
        `chosen_inds` selects dimensions from the configured coeff vector.
        """
        dx_add, dx_div = self._get_norm_coeffs(target_traj_orig, chosen_inds)
        return (target_traj_orig + dx_add) / dx_div

    def descale_traj(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        """
        Inverse transform of `scale_traj`: x * div_coeffs - add_coeffs.
        """
        dx_add, dx_div = self._get_norm_coeffs(target_traj_orig, chosen_inds)
        return target_traj_orig * dx_div - dx_add

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.scale_traj(actions, chosen_inds=self.default_chosen_inds)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.descale_traj(actions, chosen_inds=self.default_chosen_inds)

    def _encode_history(self, data_batch):
        map_feat = self.map_encoder(data_batch["image"])
        target_traj_feat = self.agent_history_encoder(data_batch["agent_hist"]).squeeze(1)
        other_traj_feat = self.other_history_encoder(
            data_batch["neigh_hist"].to(data_batch["agent_hist"].device)
        ).squeeze(1)
        return torch.cat([map_feat, target_traj_feat, other_traj_feat], dim=-1)

    def _forward_dynamics(self, actions, curr_states) -> Dict[str, torch.Tensor]:
        if len(actions.shape) == 3:
            traj, states = self.diffnet._forward_dynamics(actions=actions, current_states=curr_states)
        else:
            traj, states = self.diffnet._forward_dynamics(actions=actions, current_states=curr_states)
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:]
        return {
            "states": states,
            "controls": actions,
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
        }

    def _flow_match_sample_actions(
        self,
        cond_feat,
        curr_states,
        num_samples,
        num_steps,
        *,
        guide_sample_fn=None,
        data_batch_for_guidance=None,
        return_all_steps=False,
    ):
        device = cond_feat.device
        batch_size = cond_feat.shape[0]

        horizon = int(self.diffuse_args.get("num_points", getattr(self.diffnet, "num_steps", 0) or 0))
        if horizon <= 0:
            raise ValueError("Unable to infer horizon for FM sampling (missing diffuse_config.num_points).")

        actions = torch.randn(batch_size, num_samples, horizon, 2, device=device)
        actions = actions.reshape(batch_size * num_samples, horizon, 2)

        # Let any stateful guidance (e.g. CCFM OT reverse cache) clear its
        # per-sampling state so the fresh initial noise is re-captured.
        if guide_sample_fn is not None and hasattr(guide_sample_fn, "reset"):
            guide_sample_fn.reset()

        cond_feat_rep = cond_feat.repeat_interleave(num_samples, dim=0)
        curr_states_rep = curr_states.repeat_interleave(num_samples, dim=0)

        def wrapped_velocity(x, t):
            if self.diffnet.dyn is not None:
                # x is in normalized action space; denormalize before dynamics
                x_phys = self.denormalize_actions(x)
                x_t = self.diffnet.dyn.forward_dynamics(
                    initial_states=curr_states_rep,
                    actions=x_phys,
                    step_time=self.diffnet.step_time,
                )[0]
                tau_t = torch.cat([x, x_t], dim=-1)
            else:
                tau_t = x
            return self.diffnet(
                tau_t,
                cond_feat_rep,
                time=t,
                current_states=curr_states_rep,
                output_type="control",
            )

        n_steps = max(int(num_steps), 1)
        dt = 1.0 / float(n_steps)
        all_steps = [] if return_all_steps else None

        for step in range(n_steps):
            # ---- Second-order Heun (RK2) integration ----
            t0_scalar = step / float(n_steps)
            t1_scalar = min((step + 1) / float(n_steps), 1.0)

            t0 = torch.full((actions.shape[0],), t0_scalar, device=device)
            t1 = torch.full((actions.shape[0],), t1_scalar, device=device)

            k1 = wrapped_velocity(actions, t0)           # slope at start
            actions_pred = actions + dt * k1             # Euler prediction
            k2 = wrapped_velocity(actions_pred, t1)      # slope at end
            actions = actions + dt * 0.5 * (k1 + k2)    # trapezoidal correction

            # # DEBUG: print ODE integration diagnostics at a few steps
            # if step in (0, n_steps // 2, n_steps - 1):
            #     with torch.no_grad():
            #         act_phys = self.denormalize_actions(actions)
            #         print(f"  [ODE step {step}/{n_steps}] t={t0_scalar:.3f} "
            #               f"k1_norm={k1.abs().mean().item():.4f} "
            #               f"k2_norm={k2.abs().mean().item():.4f} "
            #               f"actions_norm(mean_abs)={actions.abs().mean().item():.4f} "
            #               f"actions_phys(mean_abs)={act_phys.abs().mean().item():.4f} "
            #               f"accel_phys={act_phys[...,0].mean().item():.4f} "
            #               f"yawrate_phys={act_phys[...,1].mean().item():.4f}")
            # ─────────────────────────────────────────────────────────────────

            # Optional: per-step guidance hook
            if guide_sample_fn is not None:
                t_mid = torch.full((actions.shape[0],), (t0_scalar + t1_scalar) / 2.0, device=device)
                try:
                    actions = guide_sample_fn(
                        actions,
                        t=t_mid,
                        data_batch_for_guidance=data_batch_for_guidance,
                        curr_states=curr_states_rep,
                    )
                except TypeError:
                    actions = guide_sample_fn(actions)

            if return_all_steps:
                all_steps.append(actions.reshape(batch_size, num_samples, horizon, 2))

        actions = actions.reshape(batch_size, num_samples, horizon, 2)
        # Denormalize from normalized action space back to physical units.
        actions = self.denormalize_actions(actions)
        if return_all_steps and all_steps is not None:
            all_steps = [self.denormalize_actions(s) for s in all_steps]
        return actions, all_steps

    def forward(self, data_batch, guide_sample_fn=None, data_batch_for_guidance=None) -> Dict[str, torch.Tensor]:
        stationary_mask = get_stationary_mask(data_batch, disable_control_on_stationary="on_lane")
        cond_feat = self._encode_history(data_batch)
        curr_states = batch_utils().get_current_states(
            data_batch,
            dyn_type=self.diffnet.dyn.type() if self.diffnet.dyn is not None else 0,
        )

        num_samples = int(self.diffuse_args.get("num_samples", 1))
        sample_steps = int(self.diffuse_args.get("sample_step", 20))
        return_all_steps = bool(self.diffuse_args.get("return_integration_steps", False))
        actions, all_steps = self._flow_match_sample_actions(
            cond_feat,
            curr_states,
            num_samples,
            sample_steps,
            return_all_steps=return_all_steps,
            guide_sample_fn=guide_sample_fn,
            data_batch_for_guidance=data_batch_for_guidance,
        )

        n_stationary = stationary_mask.sum().item()
        if n_stationary > 0 and guide_sample_fn is not None:
            pass  # print(f"[FM] stationary_mask: {n_stationary}/{stationary_mask.shape[0]} agents masked as stationary")
        actions = actions * (~stationary_mask).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        out_dict = self._forward_dynamics(actions, curr_states)
        out_dict["denoising_predictions"] = {}
        if return_all_steps and all_steps is not None:
            out_dict["denoising_predictions"]["integration_actions"] = all_steps
        return out_dict

    def sample(self, data_batch, collision_config=None) -> Dict[str, torch.Tensor]:
        """Sample trajectories, optionally with CCFM constraint projection.

        Args:
            data_batch: standard observation batch.
            collision_config: if provided, enables CCFM hard-constraint projection
                during ODE integration.  Expected keys: ego_idx, adv_idx,
                collision_type, T_collision, conflict_point, ego_length, ...
                Can also be a list of per-scene configs (some may be None).
        """
        # Use stored collision_configs (per-scene list) or single config
        if collision_config is None:
            collision_configs = getattr(self, "_collision_configs", None)
            if collision_configs is None:
                # Fallback to single config
                collision_config = getattr(self, "_collision_config", None)
            else:
                # Filter out None entries; use as list
                valid_configs = [c for c in collision_configs if c is not None]
                if len(valid_configs) == 0:
                    collision_config = None
                elif len(valid_configs) == 1:
                    collision_config = valid_configs[0]
                else:
                    collision_config = valid_configs  # pass as list

        if not self.do_guidance or collision_config is None:
            if self.do_guidance and collision_config is None:
                print("[FM] do_guidance=True but no collision_config, using regular forward")
            return self.forward(data_batch)

        # Build the CCFM guidance callable
        from tbsim.utils.ccfm_guidance import CCFMGuidanceFunction, MultiSceneCCFMGuidance, CombinedFMGuidance

        ccfm_params = self._get_ccfm_params()

        # Handle per-scene collision configs (list) vs single config
        if isinstance(collision_config, list):
            # print(f"[FM] CCFM active (multi-scene): {len(collision_config)} configs, "
            #       f"batch_size={data_batch['image'].shape[0]}")
            ccfm_fn = MultiSceneCCFMGuidance(
                collision_configs=collision_config,
                dynamics_model=self.diffnet.dyn,
                device=data_batch["image"].device,
                denormalize_fn=self.denormalize_actions,
                normalize_fn=self.normalize_actions,
                **ccfm_params,
            )
            # Use first valid config for _prepare_fm_guidance_data (shared fields)
            primary_config = collision_config[0]
        else:
            # print(f"[FM] CCFM active: ego_idx={collision_config.get('ego_idx')}, "
            #       f"adv_idx={collision_config.get('adv_idx')}, "
            #       f"batch_size={data_batch['image'].shape[0]}")
            ccfm_fn = CCFMGuidanceFunction(
                collision_config=collision_config,
                dynamics_model=self.diffnet.dyn,
                device=data_batch["image"].device,
                denormalize_fn=self.denormalize_actions,
                normalize_fn=self.normalize_actions,
                **ccfm_params,
            )
            primary_config = collision_config

        data_batch_for_guidance = self._prepare_fm_guidance_data(data_batch, primary_config)
        # Store all configs in guidance data for MultiSceneCCFMGuidance
        if isinstance(collision_config, list):
            data_batch_for_guidance["collision_configs"] = collision_config

        # Build gradient guidance (collision avoidance + causecollision + route)
        collision_guidance, causecollision_guidance, route_guidance = self._build_gradient_guidance(
            data_batch, primary_config
        )
        skip_projection = primary_config.get("skip_projection", False) if isinstance(primary_config, dict) else False
        if collision_guidance is not None or causecollision_guidance is not None or route_guidance is not None:
            shared_inner_beta = self._get_guidance_lr("inner_beta", [0.5, 3.0])
            collision_lr = self._get_guidance_lr("collision_lr", 0.5)
            guide_fn = CombinedFMGuidance(
                ccfm_fn=ccfm_fn,
                collision_guidance=collision_guidance,
                causecollision_guidance=causecollision_guidance,
                route_guidance=route_guidance,
                dynamics_model=self.diffnet.dyn,
                step_time=self.diffnet.step_time if hasattr(self.diffnet, "step_time") else 0.1,
                inner_lr=collision_lr,
                causecollision_lr=self._get_guidance_lr("causecollision_lr", collision_lr),
                route_lr=self._get_guidance_lr("route_lr", 0.5),
                inner_beta=shared_inner_beta,
                collision_inner_beta=self._get_guidance_lr("collision_inner_beta", shared_inner_beta),
                causecollision_inner_beta=self._get_guidance_lr("causecollision_inner_beta", shared_inner_beta),
                route_inner_beta=self._get_guidance_lr("route_inner_beta", shared_inner_beta),
                denormalize_fn=self.denormalize_actions,
                skip_ccfm=skip_projection,
            )
            guide_names = []
            if collision_guidance is not None:
                guide_names.append("collision_avoidance")
            if causecollision_guidance is not None:
                guide_names.append("causecollision")
            if route_guidance is not None:
                guide_names.append("route")
            # print(f"[FM] Combined guidance: {' + '.join(guide_names)} + CCFM projection")
        else:
            guide_fn = ccfm_fn

        result = self.forward(data_batch, guide_sample_fn=guide_fn,
                              data_batch_for_guidance=data_batch_for_guidance)
        # Final candidate selection happens after ODE integration/projection, so it does not
        # interfere with the per-step projection dynamics. We keep all N candidates and only
        # overwrite sample 0 with the filtered best candidate (same convention as diffusion).
        if hasattr(guide_fn, "filter"):
            result = self._apply_final_fm_filter(result, guide_fn, data_batch_for_guidance)
        # Check output
        pos = result.get("predictions", {}).get("positions", None)
        if pos is not None:
            pass
            # pos_norm = pos.abs().mean().item()
            # has_nan = torch.isnan(pos).any().item()
            # print(f"[FM] Output positions: mean_abs={pos_norm:.4f}, has_nan={has_nan}")
        return result

    def _build_gradient_guidance(self, data_batch, collision_config):
        """Build collision/causecollision/route calculators for FM guidance.

        Returns:
            collision_guidance: collision avoidance for all agents (ego-adv masked).
            causecollision_guidance: targeted ego-adv interaction (adv-focused).
            route_guidance: lane/route regularization.
        """
        from tbsim.utils.guidance_utils import (
            CauseCollisionLossCalculator,
            CollisionLossCalculator,
            RouteLossCalculator,
        )
        device = data_batch["image"].device

        # Check which guidance functions to use
        fm_guidance_fns = collision_config.get("fm_guidance_fns", ["collision", "route"]) if isinstance(collision_config, dict) else ["collision", "route"]

        # Collision guidance
        collision_guidance = None
        if "collision" in fm_guidance_fns:
            # Mask ego↔adv avoidance loss (but keep avoidance between ego↔others
            # and adv↔others) when either:
            #   (a) causecollision is enabled (adv actively tries to hit ego), or
            #   (b) CCFM projection is active (CCFM's job is to force collision;
            #       avoidance between ego and the CCFM-selected adv would fight
            #       CCFM).
            ccfm_active = collision_config is not None and (
                (isinstance(collision_config, dict) and collision_config.get("adv_idx") is not None)
                or (isinstance(collision_config, list) and any(cc is not None for cc in collision_config))
            )
            mask_ego_adv_pair = ("causecollision" in fm_guidance_fns) or ccfm_active
            try:
                cg = CollisionLossCalculator(
                    mode="gaussian",
                    sigma=1.0,
                    heading_weight=2.0,
                    buff_dist=-1.0,
                    prediction_mode="multi_agent",
                    adv_mode=mask_ego_adv_pair,
                    loss_scale=1.0,
                )
                cg.to_device(device)
                batch_ctrl_indices = getattr(self, "_batch_ctrl_indices", None)
                batch_ego_indices = getattr(self, "_batch_ego_indices", None)
                if batch_ctrl_indices is not None and batch_ego_indices is not None:
                    cg.update_config({
                        "batch_ctrl_indices": batch_ctrl_indices,
                        "batch_ego_indices": batch_ego_indices,
                    })
                    collision_guidance = cg
            except Exception as e:
                print(f"[FM] Failed to build collision guidance: {e}")

        # Causecollision guidance (adv-focused)
        causecollision_guidance = None
        if "causecollision" in fm_guidance_fns:
            # CauseCollisionLossCalculator: targeted ego-adv collision
            try:
                cg = CauseCollisionLossCalculator(
                    mode="gaussian",
                    sigma=1.0,
                    heading_weight=2.0,
                    buff_dist=-1.0,
                    prediction_mode="multi_agent",
                    loss_scale=1.0,
                    adv_term_weight={
                        "distance": 1.0,
                        "speed_penalty": 0.0,
                        "filtered_distance": 0.1,
                    },
                    interact_mode="distance",
                    adv_bound=30.0,
                    speed_diff=2.0,
                    interact_dist_thresh=100.0,
                )
                cg.to_device(device)
                batch_ctrl_indices = getattr(self, "_batch_ctrl_indices", None)
                batch_ego_indices = getattr(self, "_batch_ego_indices", None)
                if batch_ctrl_indices is not None and batch_ego_indices is not None:
                    cg.update_config({
                        "batch_ctrl_indices": batch_ctrl_indices,
                        "batch_ego_indices": batch_ego_indices,
                    })
                    causecollision_guidance = cg
            except Exception as e:
                print(f"[FM] Failed to build causecollision guidance: {e}")

        # Route guidance
        route_guidance = None
        if "route" in fm_guidance_fns and "extras" in data_batch and "centerline_xy" in data_batch.get("extras", {}):
            try:
                route_guidance = RouteLossCalculator(
                    lane_margin=1.0,
                    nonlinear_factor=5.0,
                    loss_scale=1.0,
                )
                route_guidance.to_device(device)
            except Exception as e:
                print(f"[FM] Failed to build route guidance: {e}")
                route_guidance = None

        return collision_guidance, causecollision_guidance, route_guidance

    def _get_guidance_lr(self, key, default):
        """Get FM guidance scalar from guide_config, including nested ccfm_projection."""
        if self.guide_config is not None:
            if isinstance(self.guide_config, dict):
                # Backward compatibility: direct top-level keys.
                if key in self.guide_config:
                    return self.guide_config.get(key, default)
                # Allow params.* overrides (same style as diffusion CLI params).
                params_cfg = self.guide_config.get("params")
                if isinstance(params_cfg, dict) and key in params_cfg:
                    return params_cfg.get(key, default)
                # Preferred location: ccfm_projection in various config layouts.
                ccfm_cfg = None
                if "ccfm_projection" in self.guide_config:
                    ccfm_cfg = self.guide_config.get("ccfm_projection")
                elif "configs" in self.guide_config and isinstance(self.guide_config["configs"], dict):
                    ccfm_cfg = self.guide_config["configs"].get("ccfm_projection")
                elif "guidance_configs" in self.guide_config and isinstance(self.guide_config["guidance_configs"], dict):
                    ccfm_cfg = self.guide_config["guidance_configs"].get("ccfm_projection")
                if isinstance(ccfm_cfg, dict) and key in ccfm_cfg:
                    return ccfm_cfg[key]
                return default
            # Object-like configs (Dict/EasyDict-style)
            if hasattr(self.guide_config, key):
                return getattr(self.guide_config, key)
            ccfm_cfg = None
            if hasattr(self.guide_config, "configs"):
                ccfm_cfg = self.guide_config.configs.get("ccfm_projection")
            elif hasattr(self.guide_config, "guidance_configs"):
                ccfm_cfg = self.guide_config.guidance_configs.get("ccfm_projection")
            if ccfm_cfg is not None:
                if hasattr(ccfm_cfg, "get"):
                    val = ccfm_cfg.get(key, None)
                    if val is not None:
                        return val
                elif hasattr(ccfm_cfg, key):
                    return getattr(ccfm_cfg, key)
        return default

    @staticmethod
    def _gather_sample_dim1(tensor: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
        """Gather best sample along dim=1 from a [B, N, ...] tensor."""
        B = tensor.shape[0]
        gather_idx = best_idx.view(B, 1, *([1] * (tensor.ndim - 2))).expand(B, 1, *tensor.shape[2:])
        return torch.gather(tensor, dim=1, index=gather_idx)

    def _apply_final_fm_filter(self, result, guide_fn, data_batch_for_guidance):
        """Run final guidance-based filtering and replace sample 0 with the best candidate."""
        controls = result.get("controls")
        states = result.get("states")
        if controls is None or states is None:
            return result
        if controls.ndim != 4 or states.ndim != 4:
            return result

        try:
            _, best_idx, score_matrix = guide_fn.filter(
                controls, states, data_batch_for_guidance, return_indices=True
            )
        except TypeError:
            # Backward-compatible path for guide objects without `return_indices`.
            best_controls = guide_fn.filter(controls, states, data_batch_for_guidance)
            if best_controls is None or best_controls.ndim != 4 or best_controls.shape[1] != 1:
                return result
            # If indices are unavailable, only overwrite controls and leave predictions unchanged.
            result["controls"] = result["controls"].clone()
            result["controls"][:, 0:1] = best_controls
            return result
        except Exception as e:
            print(f"[FM] Final filter skipped due to error: {e}")
            return result

        if best_idx is None:
            return result

        # Overwrite sample 0 with the selected best candidate while preserving all N candidates.
        for key in ("controls", "states", "trajectories"):
            tensor = result.get(key)
            if torch.is_tensor(tensor) and tensor.ndim >= 2 and tensor.shape[1] > 0:
                tensor_best = self._gather_sample_dim1(tensor, best_idx)
                tensor = tensor.clone()
                tensor[:, 0:1] = tensor_best
                result[key] = tensor

        preds = result.get("predictions", {})
        if isinstance(preds, dict):
            for key in ("positions", "yaws"):
                tensor = preds.get(key)
                if torch.is_tensor(tensor) and tensor.ndim >= 2 and tensor.shape[1] > 0:
                    tensor_best = self._gather_sample_dim1(tensor, best_idx)
                    tensor = tensor.clone()
                    tensor[:, 0:1] = tensor_best
                    preds[key] = tensor
            result["predictions"] = preds

        denoise = result.get("denoising_predictions")
        if isinstance(denoise, dict):
            denoise["filter_best_indices"] = best_idx.detach()
            if score_matrix is not None:
                denoise["filter_scores_collision_route"] = score_matrix.detach()
            result["denoising_predictions"] = denoise

        return result

    def _prepare_fm_guidance_data(self, data_batch, collision_config):
        """Prepare guidance data dict for both CCFM projection and gradient guidance."""
        batch_size = data_batch["image"].shape[0]
        num_samples = int(self.diffuse_args.get("num_samples", 1))

        guidance_data = {
            "batch_size": batch_size,
            "BN": batch_size * num_samples,
            "num_samples": num_samples,
            "n_total_steps": int(self.diffuse_args.get("sample_step", 20)),
            "dt": self.diffnet.step_time if hasattr(self.diffnet, "step_time") else 0.1,
            "collision_config": collision_config,
        }
        # --- Fields for CCFM projection ---
        if "extras" in data_batch and "centerline_world_xy" in data_batch.get("extras", {}):
            ego_idx = collision_config.get("ego_idx", 0)
            adv_idx = collision_config.get("adv_idx", 1)
            cl = data_batch["extras"]["centerline_world_xy"]
            if ego_idx < cl.shape[0]:
                guidance_data["ego_centerline"] = cl[ego_idx]
            if adv_idx < cl.shape[0]:
                guidance_data["adv_centerline"] = cl[adv_idx]
        if "conflict_point" in collision_config:
            guidance_data["conflict_point"] = collision_config["conflict_point"]
        if "ego_plan" in data_batch and data_batch["ego_plan"] is not None:
            guidance_data["ego_plan"] = data_batch["ego_plan"]
        # World-frame states for CCFM dynamic T_collision
        if "centroid" in data_batch:
            guidance_data["world_positions"] = data_batch["centroid"]
        if "curr_speed" in data_batch:
            guidance_data["world_speeds"] = data_batch["curr_speed"]
        if "yaw" in data_batch:
            guidance_data["world_yaws"] = data_batch["yaw"]
        if "world_from_agent" in data_batch:
            guidance_data["world_from_agent"] = data_batch["world_from_agent"]

        # --- Fields for gradient guidance (CollisionLossCalculator / RouteLossCalculator) ---
        if "curr_speed" in data_batch:
            guidance_data["curr_speed"] = data_batch["curr_speed"]
        if "yaw" in data_batch:
            guidance_data["yaw"] = data_batch["yaw"]
        if "agent_fut_extent" in data_batch:
            guidance_data["ego_extents"] = data_batch["agent_fut_extent"][:, 0, :2]
        if "world_from_agent" in data_batch:
            guidance_data["world_from_agent"] = data_batch["world_from_agent"]
        guidance_data["scene_ids"] = data_batch.get(
            "scene_index", data_batch.get("scene_ids", torch.zeros(batch_size))
        )
        # Route guidance: centerline in agent-local frame
        if "extras" in data_batch and "centerline_xy" in data_batch.get("extras", {}):
            guidance_data["centerline"] = data_batch["extras"]["centerline_xy"]
            guidance_data["lane_avail"] = data_batch["extras"].get(
                "has_lane", torch.ones(batch_size, device=data_batch["image"].device)
            )
        return guidance_data

    def _get_ccfm_params(self):
        """Extract CCFM projection parameters from guide_config."""
        defaults = {
            "projection_freq": 1,
            "max_projection_iters": 1,
            "projection_tolerance": 1e-3,
            "step_size": 0.8,
            "damping": 1e-4,
        }
        if self.guide_config is not None:
            ccfm_cfg = None
            if isinstance(self.guide_config, dict):
                # For Dict-like configs loaded from checkpoint config.json, prefer direct key lookup.
                if "ccfm_projection" in self.guide_config:
                    ccfm_cfg = self.guide_config.get("ccfm_projection")
                elif "configs" in self.guide_config and isinstance(self.guide_config["configs"], dict):
                    ccfm_cfg = self.guide_config["configs"].get("ccfm_projection")
                elif "guidance_configs" in self.guide_config and isinstance(self.guide_config["guidance_configs"], dict):
                    ccfm_cfg = self.guide_config["guidance_configs"].get("ccfm_projection")
            elif hasattr(self.guide_config, "configs"):
                ccfm_cfg = self.guide_config.configs.get("ccfm_projection")
            elif hasattr(self.guide_config, "guidance_configs"):
                ccfm_cfg = self.guide_config.guidance_configs.get("ccfm_projection")
            if ccfm_cfg is not None:
                defaults.update({k: v for k, v in ccfm_cfg.items() if k in defaults})
        return defaults

    def update_guide_config(self, update_config_dict, device):
        """Update guidance config (called per-step from rollout loop)."""
        if hasattr(update_config_dict, "guide_config"):
            gc = update_config_dict.guide_config
            # Store per-scene collision_configs list
            if isinstance(gc, dict) and "collision_configs" in gc:
                self._collision_configs = gc["collision_configs"]
            elif hasattr(gc, "collision_configs"):
                self._collision_configs = gc.collision_configs
            # Backward-compatible single config
            if hasattr(gc, "collision_config"):
                self._collision_config = gc.collision_config
            elif isinstance(gc, dict) and "collision_config" in gc:
                self._collision_config = gc["collision_config"]
            # Store ctrl/ego indices for gradient-based collision avoidance
            # Use 'in' check to avoid Dict.__missing__ creating empty sub-Dicts
            if isinstance(gc, dict):
                batch_ctrl = gc["batch_ctrl_indices"] if "batch_ctrl_indices" in gc else None
                batch_ego = gc["batch_ego_indices"] if "batch_ego_indices" in gc else None
            else:
                batch_ctrl = getattr(gc, "batch_ctrl_indices", None)
                batch_ego = getattr(gc, "batch_ego_indices", None)
            # Filter out empty Dict objects created by Dict.__missing__
            if isinstance(batch_ctrl, dict) and len(batch_ctrl) == 0:
                batch_ctrl = None
            if isinstance(batch_ego, dict) and len(batch_ego) == 0:
                batch_ego = None
            if batch_ctrl is not None:
                self._batch_ctrl_indices = batch_ctrl
            if batch_ego is not None:
                self._batch_ego_indices = batch_ego

    def compute_training_losses(self, data_batch: Dict[str, torch.Tensor]):
        if "extras" not in data_batch or data_batch["extras"] is None or "actions" not in data_batch["extras"]:
            raise KeyError(
                "Missing `data_batch['extras']['actions']`. Enable trajdata inverse-dynamics actions "
                "by providing extras['actions']=get_actions_inverse_dynamics in the trajdata datamodule."
            )

        gt_actions = data_batch["extras"]["actions"].to(data_batch["image"].device).float()
        batch_size, _, control_dim = gt_actions.shape
        if control_dim != 2:
            raise ValueError(f"Expected actions dim=2 (accel,yaw_rate), got {control_dim}")

        avail = data_batch["target_availabilities"].to(gt_actions.device).float()
        avail_u = avail.unsqueeze(-1)
        denom = avail_u.sum().clamp_min(1.0)

        cond_feat = self._encode_history(data_batch)
        curr_states = batch_utils().get_current_states(
            data_batch, dyn_type=self.diffnet.dyn.type() if self.diffnet.dyn is not None else 0
        )
        gt_actions_norm = self.normalize_actions(gt_actions)
        # In torchcfm, we sample a conditional probability path between x0 (base noise)
        # and x1 (data), and the conditional flow target is u_t = x1 - x0.
        x0 = torch.randn_like(gt_actions_norm)
        t, x_t, v_target = self.flow_matcher.sample_location_and_conditional_flow(x0, gt_actions_norm)

        if self.diffnet.dyn is not None:
            x_t_phys = self.denormalize_actions(x_t)
            state_t = self.diffnet.dyn.forward_dynamics(
                initial_states=curr_states,
                actions=x_t_phys,
                step_time=self.diffnet.step_time,
            )[0]
            tau_t = torch.cat([x_t, state_t], dim=-1)
        else:
            tau_t = x_t

        v_pred = self.diffnet(
            tau_t,
            cond_feat,
            time=t,
            current_states=curr_states,
            output_type="control",
        )

        fm_mse = ((v_pred - v_target) ** 2 * avail_u).sum() / denom
        losses = OrderedDict()
        losses["flow_matching_mse"] = fm_mse
        return losses

    def compute_losses(self, pred_batch: Dict[str, torch.Tensor], data_batch: Dict[str, torch.Tensor]):
        pred_positions = pred_batch["predictions"]["positions"]
        pred_yaws = pred_batch["predictions"]["yaws"]

        if pred_positions.ndim == 4:
            pred_positions = pred_positions[:, 0]
        if pred_yaws.ndim == 4:
            pred_yaws = pred_yaws[:, 0]

        target_positions = data_batch["target_positions"]
        target_yaws = data_batch["target_yaws"]
        target_avail = data_batch["target_availabilities"].float()

        avail = target_avail.unsqueeze(-1)
        denom = avail.sum().clamp_min(1.0)

        pos_loss = ((pred_positions - target_positions) ** 2 * avail).sum() / denom
        yaw_loss = ((pred_yaws - target_yaws) ** 2 * avail).sum() / denom

        losses = OrderedDict()
        losses["prediction_loss"] = pos_loss + yaw_loss
        if "controls" in pred_batch:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses

    def get_loss(self, data_batch: Dict[str, torch.Tensor]):
        pred_batch = self.forward(data_batch)
        return self.compute_losses(pred_batch, data_batch)


# Backward-compat alias
FlowMatchingModel = RasterizedFMModel
