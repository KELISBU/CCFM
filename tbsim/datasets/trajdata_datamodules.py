import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict as TypingDict, List, Optional, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tbsim.configs.base import TrainConfig

try:
    from trajdata import AgentType, UnifiedDataset
    from trajdata.custom_func.get_lane_info import get_lane_info
    from trajdata.custom_func.get_actions import get_actions_inverse_dynamics
    from tbsim.utils.trajdata_utils import get_full_fut_traj, get_full_fut_valid
except Exception as e:  # pragma: no cover
    AgentType = None
    UnifiedDataset = None
    get_lane_info = None
    get_actions_inverse_dynamics = None
    get_full_fut_traj = None
    get_full_fut_valid = None
    _TRAJDATA_IMPORT_ERROR = e


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _resolve_nuplan_data_dir(dataset_path: str) -> str:
    path = Path(dataset_path).expanduser()

    if (path / "nuplan-v1.1" / "splits" / "mini").is_dir():
        return str(path / "nuplan-v1.1" / "splits")

    if (path / "splits" / "mini").is_dir():
        return str(path / "splits")

    return str(path)


def _infer_data_dirs(
    trajdata_data_dirs: TypingDict[str, str],
    trajdata_source_root: str,
    dataset_path: str,
    sources: Sequence[str],
) -> TypingDict[str, str]:
    data_dirs: TypingDict[str, str] = dict(trajdata_data_dirs or {})
    resolved_dataset_path = dataset_path
    if dataset_path:
        nuplan_sources = [
            s for s in sources if isinstance(s, str) and "nuplan" in s
        ]
        if nuplan_sources:
            resolved_dataset_path = _resolve_nuplan_data_dir(dataset_path)

    if (
        trajdata_source_root
        and resolved_dataset_path
        and trajdata_source_root not in data_dirs
    ):
        data_dirs[trajdata_source_root] = resolved_dataset_path

    def _is_split_token(tok: str) -> bool:
        if tok in {"train", "val", "test"}:
            return True
        return tok.endswith(("_train", "_val", "_test"))

    for s in sources:
        if not isinstance(s, str) or "-" not in s:
            continue
        tokens = [t for t in s.split("-") if t]
        dataset_tokens = [t for t in tokens if not _is_split_token(t)]
        for dataset_id in dataset_tokens:
            target_path = resolved_dataset_path
            if dataset_id and target_path and dataset_id not in data_dirs:
                data_dirs[dataset_id] = target_path

    if "nusc_trainval" in data_dirs and "nusc_mini" not in data_dirs:
        data_dirs["nusc_mini"] = data_dirs["nusc_trainval"]

    return data_dirs


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config: TrainConfig):
        super().__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
        return dict(
            image=(
                3 + self._data_config.history_num_frames + 1,
                self._data_config.raster_size,
                self._data_config.raster_size,
            ),
            static=(3, self._data_config.raster_size, self._data_config.raster_size),
            dynamic=(
                self._data_config.history_num_frames + 1,
                self._data_config.raster_size,
                self._data_config.raster_size,
            ),
        )

    def setup(self, stage: Optional[str] = None):
        if UnifiedDataset is None:  # pragma: no cover
            raise ImportError(
                "trajdata is required for UnifiedDataModule; install the modified trajdata repo."
            ) from _TRAJDATA_IMPORT_ERROR

        dt = float(self._data_config.step_time)
        future_sec = float(self._data_config.future_num_frames) * dt
        history_sec = float(self._data_config.history_num_frames) * dt
        neighbor_distance = float(self._data_config.max_agents_distance)

        train_sources = _as_list(self._data_config.trajdata_source_train)
        valid_sources = _as_list(self._data_config.trajdata_source_valid)
        data_dirs = _infer_data_dirs(
            trajdata_data_dirs=getattr(self._data_config, "trajdata_data_dirs", {}),
            trajdata_source_root=getattr(self._data_config, "trajdata_source_root", ""),
            dataset_path=getattr(self._data_config, "dataset_path", ""),
            sources=train_sources + valid_sources,
        )

        raster_map_params = {
            "px_per_m": int(1 / float(self._data_config.pixel_size)),
            "map_size_px": int(self._data_config.raster_size),
            "return_rgb": False,
            "offset_frac_xy": tuple(self._data_config.raster_center),
            "original_format": True,
        }

        vector_map_params = {
            "incl_road_lanes": True,
            "incl_road_areas": False,
            "incl_ped_crosswalks": False,
            "incl_ped_walkways": False,
            "collate": False,
        }

        extras = {}
        if get_lane_info is not None:
            extras["closest_lane_point"] = get_lane_info
        if get_actions_inverse_dynamics is not None:
            extras["actions"] = get_actions_inverse_dynamics
        if get_full_fut_traj is not None:
            extras["full_fut_traj"] = get_full_fut_traj
        if get_full_fut_valid is not None:
            extras["full_fut_valid"] = get_full_fut_valid

        common_kwargs = dict(
            centric=getattr(self._data_config, "centric", "agent"),
            desired_dt=dt,
            history_sec=(history_sec, history_sec),
            future_sec=(future_sec, future_sec),
            data_dirs=data_dirs,
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params=raster_map_params,
            incl_vector_map=True,
            vector_map_params=vector_map_params,
            standardize_data=bool(getattr(self._data_config, "standardize_data", True)),
            ego_only=bool(getattr(self._train_config, "ego_only", False)),
            max_neighbor_num=int(getattr(self._data_config, "other_agents_num", 0)),
            cache_type=str(getattr(self._data_config, "trajdata_cache_type", "dataframe")),
            cache_location=str(
                getattr(self._data_config, "trajdata_cache_location", "~/.unified_data_cache")
            ),
            rebuild_cache=bool(getattr(self._data_config, "trajdata_rebuild_cache", False)),
            rebuild_maps=bool(getattr(self._data_config, "trajdata_rebuild_maps", False)),
            save_index=bool(getattr(self._data_config, "trajdata_save_index", False)),
            check_cache=bool(getattr(self._data_config, "load_cache", True)),
            # trajdata's `num_workers` controls dataset preprocessing / caching workers.
            # Setting this to `os.cpu_count()` combined with PyTorch DataLoader workers
            # can easily create too many processes and get workers SIGKILL'd (OOM).
            num_workers=int(self._data_config.get("trajdata_num_workers", 0)),
            extras=extras,
            obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
        )

        if self.train_dataset is None and (stage is None or stage == "fit"):
            self.train_dataset = UnifiedDataset(desired_data=train_sources, **common_kwargs)
            self.valid_dataset = UnifiedDataset(desired_data=valid_sources, **common_kwargs)

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        # Important: trajdata's `return_dict=True` collate path uses `dataclasses.asdict`,
        # which strips `StateTensor/StateArray` subclasses into plain `torch.Tensor`.
        # We keep the original dataclass batch (return_dict=False) and then expose its
        # attributes as a dict without deep-copying, so downstream parsing can rely
        # on `StateTensor` semantics without modifying trajdata.
        base_collate = self.train_dataset.get_collate_fn(return_dict=False, pad_format="outside", augment=True)

        def collate_keep_state(batch_elems):
            batch = base_collate(batch_elems)
            return batch if isinstance(batch, dict) else vars(batch)

        return DataLoader(
            self.train_dataset,
            batch_size=int(self._train_config.training.batch_size),
            shuffle=True,
            num_workers=int(self._train_config.training.num_data_workers),
            pin_memory=False,
            drop_last=True,
            persistent_workers=int(self._train_config.training.num_data_workers) > 0,
            collate_fn=collate_keep_state,
        )

    def val_dataloader(self):
        if not bool(self._train_config.validation.enabled):
            return None
        if self.valid_dataset is None:
            self.setup()
        base_collate = self.valid_dataset.get_collate_fn(return_dict=False, pad_format="outside", augment=False)

        def collate_keep_state(batch_elems):
            batch = base_collate(batch_elems)
            return batch if isinstance(batch, dict) else vars(batch)

        return DataLoader(
            self.valid_dataset,
            batch_size=int(self._train_config.validation.batch_size),
            shuffle=False,
            num_workers=int(self._train_config.validation.num_data_workers),
            pin_memory=False,
            drop_last=False,
            persistent_workers=int(self._train_config.validation.num_data_workers) > 0,
            collate_fn=collate_keep_state,
        )
