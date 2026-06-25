"""Variants of Conditional Variational Autoencoder (C-VAE)"""
from collections import OrderedDict
from typing import Dict, List
import torch
import torch.nn as nn
from tbsim.utils.config_utils import update_config
import tbsim.models.base_models as base_models

import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.guidance_utils import initialize_loss_calculator
from tbsim.utils.adv_utils import generate_collision_paths
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.trajdata_utils import enlarge_batch_samples, extend_ego_plan, get_stationary_mask

from tbsim.models.temporal import TemporalUnet

from tbsim.models.diffusion import DiffusionTraj,VarianceSchedule



class RasterizedDiffusionModel(nn.Module):
    """Raster-based diffusion model for controllable traffic simulation.
     """
    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            dynamics_config:tuple,
            history_encoder_config,
            diffnet_config:tuple,
            diffuse_config:Dict,
            weights_scaling: List[float],
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            rasterize_mode = "point",
            drop_cond_prob = 0,
            do_guidance = False,
            guide_config = None,
            nusc_norm_info = None,
    ) -> None:
    
        super().__init__()
        if rasterize_mode is None:
            rasterize_mode = "point"
        assert rasterize_mode in ["point","square"]
        self.rasterize_mode = rasterize_mode
        self.drop_cond_prob = drop_cond_prob
        self.weights_scaling = weights_scaling

        self.map_encoder = base_models.RasterizedMapEncoder(
                model_arch=model_arch,
                input_image_shape=input_image_shape,
                feature_dim=map_feature_dim,
                use_spatial_softmax=use_spatial_softmax,
                spatial_softmax_kwargs=spatial_softmax_kwargs,
                output_activation=nn.ReLU
            )
        self.agent_history_encoder = base_models.HistoryEncoder(history_encoder_config)
        self.other_history_encoder =  base_models.HistoryEncoder(history_encoder_config)
        
        self.diffnet = TemporalUnet(**diffnet_config,dynamics_config=dynamics_config) ##TODO config of TemperalNet
        self.diffusion = DiffusionTraj(
            net = self.diffnet,
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='cosine'
            )
        )
        self.diffuse_args = diffuse_config
        self.default_chosen_inds = [0, 1]
        self._init_norm_buffers(nusc_norm_info)
        for key in ["num_samples", "sample_step","sampling_mode"]:
            self.diffuse_args[key] = guide_config.params[key]
        self.do_guidance  = do_guidance
        self.guide_config = guide_config
        self.gen_adv_trajs = guide_config.params.partial_t is not None # Check if partial_t has been set
        if self.do_guidance:
            self.Loss_Calculater = initialize_loss_calculator(self.guide_config)

    def _init_norm_buffers(self, nusc_norm_info):
        add_coeffs = None
        div_coeffs = None

        if nusc_norm_info is not None:
            norm_info = None
            if isinstance(nusc_norm_info, dict):
                for key in ("diffusion", "flow_matching"):
                    if key in nusc_norm_info:
                        norm_info = nusc_norm_info.get(key)
                        break

                if norm_info is None and "add_coeffs" in nusc_norm_info and "div_coeffs" in nusc_norm_info:
                    add_coeffs = torch.as_tensor(nusc_norm_info["add_coeffs"], dtype=torch.float32)
                    div_coeffs = torch.as_tensor(nusc_norm_info["div_coeffs"], dtype=torch.float32)

            if norm_info is not None:
                if isinstance(norm_info, (list, tuple)) and len(norm_info) >= 2:
                    mean_coeffs = torch.as_tensor(norm_info[0], dtype=torch.float32)
                    std_coeffs = torch.as_tensor(norm_info[1], dtype=torch.float32)
                    add_coeffs = -mean_coeffs
                    div_coeffs = std_coeffs
                elif isinstance(norm_info, dict):
                    if "add_coeffs" in norm_info and "div_coeffs" in norm_info:
                        add_coeffs = torch.as_tensor(norm_info["add_coeffs"], dtype=torch.float32)
                        div_coeffs = torch.as_tensor(norm_info["div_coeffs"], dtype=torch.float32)
                    elif "mean" in norm_info and "std" in norm_info:
                        add_coeffs = -torch.as_tensor(norm_info["mean"], dtype=torch.float32)
                        div_coeffs = torch.as_tensor(norm_info["std"], dtype=torch.float32)

        if add_coeffs is None or div_coeffs is None:
            add_coeffs = torch.zeros(2, dtype=torch.float32)
            div_coeffs = torch.ones(2, dtype=torch.float32)

        if add_coeffs.ndim != 1 or div_coeffs.ndim != 1 or add_coeffs.shape != div_coeffs.shape:
            raise ValueError(
                f"Invalid norm coeffs shape: add={tuple(add_coeffs.shape)}, div={tuple(div_coeffs.shape)}"
            )
        if max(self.default_chosen_inds) >= add_coeffs.shape[0]:
            raise ValueError(
                "diffusion norm coeffs do not have enough dims for action normalization "
                f"(need indices {self.default_chosen_inds}, got {add_coeffs.shape[0]} dims)"
            )

        # Config-derived normalization stats should not be checkpoint-required; keep them off state_dict
        # so older checkpoints (without these buffers) still load under strict=True.
        self.register_buffer("add_coeffs", add_coeffs, persistent=False)
        self.register_buffer("div_coeffs", div_coeffs.clamp_min(1e-6), persistent=False)

    def _get_norm_coeffs(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        if chosen_inds is None or len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        idx = torch.as_tensor(chosen_inds, device=self.add_coeffs.device, dtype=torch.long)
        dx_add = self.add_coeffs.index_select(0, idx).to(device=target_traj_orig.device, dtype=target_traj_orig.dtype)
        dx_div = self.div_coeffs.index_select(0, idx).to(device=target_traj_orig.device, dtype=target_traj_orig.dtype)
        view_shape = [1] * (target_traj_orig.ndim - 1) + [len(chosen_inds)]
        return dx_add.view(*view_shape), dx_div.view(*view_shape)

    def scale_traj(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        dx_add, dx_div = self._get_norm_coeffs(target_traj_orig, chosen_inds)
        return (target_traj_orig + dx_add) / dx_div

    def descale_traj(self, target_traj_orig: torch.Tensor, chosen_inds=None):
        dx_add, dx_div = self._get_norm_coeffs(target_traj_orig, chosen_inds)
        return target_traj_orig * dx_div - dx_add

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.scale_traj(actions, chosen_inds=self.default_chosen_inds)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.descale_traj(actions, chosen_inds=self.default_chosen_inds)


    def forward(self, data_batch, guide_sample_fn=None, data_batch_for_guidance=None)-> Dict[str, torch.Tensor]:
        #whether to control stationary agents
        stationary_mask = get_stationary_mask(data_batch, disable_control_on_stationary="on_lane")
        ## Calculating conditioning feature
        cond_feat = self._encode_history(data_batch)
        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.diffnet.dyn.type() if self.diffnet.dyn is not None else 0 )

        actions = self.diffusion.sample(
            cond_feat,
            forward_mode="sample",
            guide_sample_fn=guide_sample_fn,
            data_batch_for_guidance=data_batch_for_guidance,
            current_states=curr_states,
            guide_config=self.guide_config,  # guide_config for controllable simulation
            action_normalize_fn=self.normalize_actions,
            action_denormalize_fn=self.denormalize_actions,
            adv_proposals_dict=(data_batch["adv_proposals_dict"] 
                              if "adv_proposals_dict" in data_batch 
                              else None),  # B * 20 * 12 * 2
            **self.diffuse_args.to_dict(),
        )
        if isinstance(actions, dict):
            actions, denoised_actions = actions["sampled_trajectories"], actions["denoised_trajectories"]
            actions = actions * (~stationary_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if denoised_actions is not None:
                denoised_actions = denoised_actions * (~stationary_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            denoised_actions = None
        if self.diffnet.dyn is not None:
            out_dict = self._forward_dynamics(actions,curr_states)
            if denoised_actions is not None:
                denoised_out_dict = self._forward_dynamics(denoised_actions,curr_states)
                out_dict["denoising_predictions"] = {
                    "positions": denoised_out_dict["predictions"]["positions"],
                    "yaws": denoised_out_dict["predictions"]["yaws"],
                    # "adv_proposals": data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"] if "adv_proposals_dict" in data_batch else None #padded for bug in logger
                }
            else:
                out_dict["denoising_predictions"] = {}
            if "adv_proposals_dict" in data_batch and data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"] is not None:
                out_dict["denoising_predictions"]["adv_proposals"] = data_batch["adv_proposals_dict"]["padded_adv_proposals_pos"]
        else:
            raise NotImplementedError

        return out_dict
       
    def update_guide_config(self, update_guide_config, device):
        update_config(self.guide_config, update_guide_config.guide_config)
        self.Loss_Calculater.to_device(device)
        self.Loss_Calculater.update_config(self.guide_config)

    def sample(self, data_batch) -> Dict[str, torch.Tensor]:
        """Generate trajectory samples, optionally with guidance.
        
        Args:
            data_batch (Dict): Input batch containing scene and agent information
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predicted trajectories and states
        """
        if not self.do_guidance:
            return self.forward(data_batch, guide_sample_fn=None)
        
        # Prepare guidance data
        data_batch_for_guidance = self._prepare_guidance_data(data_batch)
        
        # Generate adversarial trajectories if needed
        if self.gen_adv_trajs:
            adv_proposals_dict = generate_collision_paths(
                data_batch,
                self.guide_config["batch_ego_indices"],
                self.guide_config["batch_ctrl_indices"],
                self.guide_config.params.desired_delta_s,
                self.guide_config.params.normal_offset,
                self.guide_config.params.ref_idx,
                T=self.diffuse_args["num_points"],
                dt=data_batch["dt"][0].item()
            )
            data_batch_for_guidance.update(adv_proposals_dict)
        
        # Run forward pass with guidance
        out_dict = self.forward(
            data_batch,
            guide_sample_fn=self.Loss_Calculater,
            data_batch_for_guidance=data_batch_for_guidance
        )
        
        return out_dict

    def _prepare_guidance_data(self, data_batch) -> Dict:
        """Prepare data needed for guided sampling.
        
        Args:
            data_batch (Dict): Input batch containing scene and agent information
            
        Returns:
            Dict: Processed data batch for guidance
        """
        # Calculate drivable region and distance maps
        drivable_map = batch_utils().get_drivable_region_map(data_batch["image"]).float()
        dis_map = GeoUtils.calc_distance_map(drivable_map)
        
        # Calculate batch dimensions
        batch_size = data_batch["dt"].shape[0]
        BN = batch_size * self.diffuse_args["num_samples"]
        
        guidance_data = {
            "batch_size": batch_size,
            "centroid": data_batch["centroid"],
            "curr_speed": data_batch["curr_speed"],
            "yaw": data_batch["yaw"],
            "raster_from_agent": data_batch["raster_from_agent"],
            "dis_map": dis_map,
            "agent_fut_extent": data_batch["agent_fut_extent"][:,0,:2],
            "centerline": data_batch["extras"]["centerline_xy"],
            "agent_pos": data_batch["world_from_agent"][:,:2,-1],
            "lane_avail": data_batch["extras"]["has_lane"],
            "ego_extents": data_batch["agent_fut_extent"][:,0,:2],
            "raw_types": data_batch["all_other_agents_types"],
            "world_from_agent": data_batch["world_from_agent"],
            "scene_ids": data_batch.get("scene_index", data_batch.get("scene_ids")),
            "ego_plan": data_batch.get("ego_plan"),
            "BN": BN,
            "num_samples": self.diffuse_args["num_samples"],
            "dt": data_batch["dt"][0].item()
        }
        
        # Adjust shapes for batch processing
        self._adjust_batch_shapes(guidance_data, batch_size)
        
        return guidance_data

    def _adjust_batch_shapes(self, guidance_data: Dict, batch_size: int):
        """Adjust shapes of guidance data for batch processing.
        
        Args:
            guidance_data (Dict): Data to be adjusted
            batch_size (int): Base batch size
        """
        keys_to_adjust = ["centerline", "lane_avail","world_from_agent","yaw", "raster_from_agent","dis_map"]
        
        for key in keys_to_adjust:
            guidance_data[key] = enlarge_batch_samples(
                guidance_data[key],
                batch_size,
                num_samples=self.diffuse_args["num_samples"]
            )
        # Extend ego plan if present
        if guidance_data["ego_plan"] is not None:
            guidance_data["ego_plan"] = extend_ego_plan(
                guidance_data["ego_plan"],
                target_length=self.diffuse_args["num_points"]
            )

    
    def _forward_dynamics(self,actions,curr_states) -> Dict[str,torch.Tensor]:
        #TODO if actions is more than 1 sample, need to do in batch
        if  len(actions.shape) == 3:
            traj, x    =  self.diffusion.net._forward_dynamics(actions=actions.squeeze(), current_states=curr_states)
        else:
            traj, x    =  self.diffusion.net._forward_dynamics(actions=actions, current_states=curr_states)
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:]
        
        out_dict = {
            "states": x,
            "controls": actions,
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

        return out_dict

    def _encode_history(self,data_batch):
        ## Calculating conditioning feature
        map_feat = self.map_encoder(data_batch["image"])
        target_traj_feat = self.agent_history_encoder(data_batch["agent_hist"]).squeeze(1)
        other_traj_feat = self.other_history_encoder(data_batch["neigh_hist"].to(data_batch["agent_hist"].device)).squeeze(1)

        cond_feat = torch.cat([map_feat,target_traj_feat,other_traj_feat],dim = -1)
        return cond_feat

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

    def compute_training_losses(self, data_batch: Dict[str, torch.Tensor]):
        """
        Diffusion training objective on inverse-dynamics actions.

        We sample a diffusion step t, add noise to the ground-truth action sequence a_0,
        predict the clean action with the TemporalUnet, and minimize MSE.
        """
        if "extras" not in data_batch or data_batch["extras"] is None or "actions" not in data_batch["extras"]:
            raise KeyError(
                "Missing `data_batch['extras']['actions']`. Enable trajdata inverse-dynamics actions "
                "by providing extras['actions']=get_actions_inverse_dynamics in the trajdata datamodule."
            )

        gt_actions = data_batch["extras"]["actions"].to(data_batch["image"].device).float()  # [B, T, 2]
        B, T, U = gt_actions.shape
        if U != 2:
            raise ValueError(f"Expected actions dim=2 (accel,yaw_rate), got {U}")

        avail = data_batch["target_availabilities"].to(gt_actions.device).float()  # [B, T]
        avail_u = avail.unsqueeze(-1)
        denom = avail_u.sum().clamp_min(1.0)

        cond_feat = self._encode_history(data_batch)
        curr_states = batch_utils().get_current_states(
            data_batch, dyn_type=self.diffnet.dyn.type() if self.diffnet.dyn is not None else 0
        )
        gt_actions_norm = self.normalize_actions(gt_actions)

        # Sample diffusion step per batch element.
        num_steps = int(self.diffusion.var_sched.num_steps)
        t_idx = torch.randint(1, num_steps + 1, (B,), device=gt_actions.device)

        alpha_bar = self.diffusion.var_sched.alpha_bars[t_idx].view(B, 1, 1)
        noise = torch.randn_like(gt_actions_norm)
        noisy_actions = alpha_bar.sqrt() * gt_actions_norm + (1.0 - alpha_bar).sqrt() * noise

        if self.diffnet.dyn is not None:
            noisy_actions_phys = self.denormalize_actions(noisy_actions)
            x_t = self.diffnet.dyn.forward_dynamics(
                initial_states=curr_states, actions=noisy_actions_phys, step_time=self.diffnet.step_time
            )[0]
            tau_t = torch.cat([noisy_actions, x_t], dim=-1)  # [B, T, 6]
        else:
            tau_t = noisy_actions

        beta = self.diffusion.var_sched.betas[t_idx]  # [B]
        pred_clean_actions = self.diffnet(
            tau_t, cond_feat, time=beta, current_states=curr_states, output_type="control"
        )

        action_mse = ((pred_clean_actions - gt_actions_norm) ** 2 * avail_u).sum() / denom

        losses = OrderedDict()
        losses["diffusion_action_mse"] = action_mse
        return losses
