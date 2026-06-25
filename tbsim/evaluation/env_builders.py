from copy import deepcopy
import numpy as np
from collections import defaultdict
import os
from functools import partial
from pathlib import Path

# from l5kit.data import LocalDataManager, ChunkedDataset
# from l5kit.rasterization import build_rasterizer
from trajdata import AgentType, UnifiedDataset

# from tbsim.l5kit.vectorizer import build_vectorizer


from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
from tbsim.envs.env_trajdata import EnvUnifiedSimulation, EnvSplitUnifiedSimulation
from tbsim.utils.config_utils import  translate_trajdata_cfg
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.evaluation.metric_composers import CVAEMetrics, OccupancyMetrics
from tbsim.utils.trajdata_utils import get_full_fut_traj,get_full_fut_valid, get_stationary_mask
# from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed

from trajdata.custom_func.get_lane_info import get_lane_info


class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        # Import collision relative speed metric
        from tbsim.envs.collision_rel_speed_metric import CollisionRelativeSpeedMetric
        from tbsim.envs.collision_rel_heading_metric import CollisionRelativeHeadingMetric
        from tbsim.envs.collision_type_match_metric import CollisionTypeMatchMetric, ActualCollisionTypeMetric
        from tbsim.envs.ego_first_collision_metric import EgoFirstCollisionMetric

        metrics = dict(
            # Paper evaluation metrics
            #all_collision_rate=EnvMetrics.CollisionRate(),           # 1. Collision Rate
            #ego_off_road_rate=EnvMetrics.OffRoadRate(),              # 2. Adv Offroad
            #all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),  # 3. Other Offroad
            ego_collision_rel_speed=CollisionRelativeSpeedMetric(),  # 4. Collision Rel Speed
            ego_collision_rel_heading=CollisionRelativeHeadingMetric(),  # 4b. Collision Rel Heading
            ego_collision_type_match=CollisionTypeMatchMetric(),       # 5. Collision Type Match
            #ego_collision_actual_type=ActualCollisionTypeMetric(),     # 6. Actual Collision Type
            ego_first_collision=EgoFirstCollisionMetric(),               # 7. First Collision Position

            # Additional metrics
            agents_off_road_rate=EnvMetrics.AgentsOffRoadRate(),       # Non-ego offroad
            adv_off_road_rate=EnvMetrics.AdvOffRoadRate(),             # Adv-only offroad (1 if ever offroad)
            ego_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),

            # Per-(collision_type, constraint) infeasibility analysis.
            # mode="selected": only check the type HCS chose at each replan step.
            # Key prefix "all_" ensures it routes to the full-obs dispatch in EnvSplitUnifiedSimulation.
            # every_n_steps=1: record on EVERY metric call (which itself fires every
            # n_step_action=5 sim steps, i.e., once per CCFM replan).
            #all_constraint_infeas=EnvMetrics.ConstraintInfeasibilityMetric(dt=0.1, every_n_steps=1, mode="selected"),

            # 5. Realism metric (requires generating the GT histogram first)
        )
        if getattr(self.eval_cfg, "env", None) == "nuplan":
            real_histogram_file = "path/to/nuplan_gt/hist_stats.json"
        else:
            real_histogram_file = "path/to/nuscene_gt/hist_stats.json"
        if os.path.exists(real_histogram_file):
            metrics["all_realism_deviation"] = EnvMetrics.RealismDeviationMetrics(
                real_histogram_file=real_histogram_file
            )
        return metrics
    def _get_agent2ego_metrics(self):
        metrics = dict(
            agents2ego_ttc = EnvMetrics.TimeToCollisionMetrics(),
            agents2ego_dist = EnvMetrics.DistanceMetrics(),
        )
        return metrics
    
    def _get_learned_metrics(self):
        perturbations = dict()
        if self.eval_cfg.perturb.enabled:
            for sigma in self.eval_cfg.perturb.OU.sigma:
                perturbations["OU_sigma_{}".format(sigma)] = OrnsteinUhlenbeckPerturbation(
                    theta=self.eval_cfg.perturb.OU.theta*np.ones(3),
                    sigma=sigma*np.array(self.eval_cfg.perturb.OU.scale))
        self.eval_cfg.ckpt_root_dir = "path/to/cvae_checkpoint"
        cvae_metrics = CVAEMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        learned_occu_metric = OccupancyMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        metrics = dict(
            all_cvae_metrics=cvae_metrics.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.cvae.rolling,
            rolling_horizon = self.eval_cfg.cvae.rolling_horizon,env=self.eval_cfg.env,),
            # all_occu_likelihood=learned_occu_metric.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.occupancy.rolling,
            # rolling_horizon = self.eval_cfg.occupancy.rolling_horizon,env=self.eval_cfg.env,)
        )
        return metrics

    def get_env(self):
        raise NotImplementedError


def _resolve_nuplan_data_dir(dataset_path: str) -> str:
    """
    Accept either the nuPlan root (containing `maps/` and `nuplan-v1.1/`) or the
    already-narrowed database directory. trajdata's nuPlan adapter expects the
    DBs under `<data_dir>/mini`, which for the standard mini layout means
    `<root>/nuplan-v1.1/splits`.
    """
    path = Path(dataset_path).expanduser()

    if (path / "nuplan-v1.1" / "splits" / "mini").is_dir():
        return str(path / "nuplan-v1.1" / "splits")

    if (path / "splits" / "mini").is_dir():
        return str(path / "splits")

    return str(path)


class EnvNuscBuilder(EnvironmentBuilder):
    def get_env(self,split_ego=False,parse_obs=True,split_dataset=False):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        # add for cache
        exp_cfg.train.load_cache = self.eval_cfg.train.load_cache
        exp_cfg.train.data_filter = self.eval_cfg.train.data_filter
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        # Load val only (split_dataset controls sliding windows, not data source)
        if split_dataset:
            desired_data = ["val-nusc_trainval"]
        else:
            desired_data = ["val"]

        kwargs = dict(
            desired_data=desired_data,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            # history_sec=(10.0, 10.0),
            data_dirs={
                "nusc_trainval": data_cfg.dataset_path,
                "nusc_mini": data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            incl_vector_map = True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "collate": False,
            },
            num_workers=os.cpu_count(),
            # augmentations = [noise_hists],
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            extras={
            "closest_lane_point": partial(get_lane_info, VEC_MAP_PARAMS=self.eval_cfg.vec_map_params),
            "full_fut_valid": get_full_fut_valid,
            "full_fut_traj": get_full_fut_traj,
            },
            obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
            # max_neighbor_num = data_cfg.other_agents_num,
        )
        print(os.cpu_count())
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env
class EnvNuplanBuilder(EnvironmentBuilder):
    
    def get_env(self, split_ego=False, parse_obs=True, split_dataset=False):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        # add for cache
        exp_cfg.train.load_cache = self.eval_cfg.train.load_cache
        exp_cfg.train.data_filter = self.eval_cfg.train.data_filter
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        nuplan_data_dir = _resolve_nuplan_data_dir(data_cfg.dataset_path)

        kwargs = dict(
            desired_data=["nuplan_mini-mini_val"], #["val"]
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            # history_sec=(10.0, 10.0),
            ego_only=True,
            data_dirs={
                "nuplan_mini": nuplan_data_dir,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            incl_vector_map = True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "no_collate": True,
            },
            num_workers=os.cpu_count(),
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            extras={
            "closest_lane_point": get_lane_info,
            "full_fut_valid": get_full_fut_valid,
            "full_fut_traj": get_full_fut_traj,
            # "all_possible_lane_pts": get_refs
            },
            obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
            
            # max_neighbor_num = data_cfg.other_agents_num,
            # rebuild_cache=True,
            # rebuild_maps=True
        )
        print(os.cpu_count())
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        # metrics = {}
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env

class EnvDrivesimBuilder(EnvironmentBuilder):
    def get_env(self,split_ego=False,parse_obs=True):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            desired_data=["main"],
            future_sec=(0.1, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={"drivesim":"home"},
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            incl_vector_map = True,
            vector_map_params = {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "no_collate": False,
            },
            # num_workers=os.cpu_count(),
            num_workers = 0,
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            # max_neighbor_num = data_cfg.other_agents_num,
        )
        # if data_cfg.vectorize_lane!="none":
        #     kwargs["vectorize_lane"] = data_cfg.vectorize_lane
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env
