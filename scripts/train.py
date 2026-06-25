import argparse
import sys
import os
import socket
import re

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.train_utils as TrainUtils
#from tbsim.utils.env_utils import RolloutCallback
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.algos.factory import algo_factory


class _ResumeDiagnosticsCallback(pl.Callback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        lr = None
        if getattr(trainer, "optimizers", None):
            opt = trainer.optimizers[0]
            if opt.param_groups:
                lr = opt.param_groups[0].get("lr", None)
        print(
            f"[resume] on_fit_start: current_epoch={trainer.current_epoch} "
            f"global_step={trainer.global_step} lr={lr}"
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        lr = None
        if getattr(trainer, "optimizers", None):
            opt = trainer.optimizers[0]
            if opt.param_groups:
                lr = opt.param_groups[0].get("lr", None)
        print(
            f"[resume] on_train_epoch_start: epoch={trainer.current_epoch} "
            f"global_step={trainer.global_step} lr={lr}"
        )


def _infer_resume_paths(resume_ckpt_path: str):
    resume_ckpt_path = os.path.abspath(os.path.expanduser(resume_ckpt_path))
    ckpt_dir = os.path.dirname(resume_ckpt_path)
    run_dir = os.path.dirname(ckpt_dir)
    version_key = os.path.basename(run_dir)
    root_dir = os.path.dirname(run_dir)
    log_dir = os.path.join(root_dir, version_key, "logs")
    video_dir = os.path.join(root_dir, version_key, "videos")
    return root_dir, log_dir, ckpt_dir, video_dir, version_key, resume_ckpt_path


def _latest_ckpt_in_run_dir(resume_run_dir: str):
    resume_run_dir = os.path.abspath(os.path.expanduser(resume_run_dir))
    ckpt_dir = os.path.join(resume_run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints dir found at {ckpt_dir}")
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt")
    ]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]

def _parse_step_epoch_from_ckpt_path(ckpt_path: str):
    """
    Best-effort parse of step/epoch from PL checkpoint filenames used in this repo.
    Examples:
      iter94000.ckpt
      iter94000_ep6_valLoss0.06.ckpt
    """
    name = os.path.basename(ckpt_path)
    step = None
    epoch = None
    m = re.search(r"iter(\d+)", name)
    if m:
        step = int(m.group(1))
    m = re.search(r"_ep(\d+)", name)
    if m:
        epoch = int(m.group(1))
    return step, epoch


def main(cfg, auto_remove_exp_dir=False, debug=False, resume_ckpt_path=None):
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.train.datamodule_class.startswith("L5"):
        set_global_batch_type("l5kit")
    elif cfg.train.datamodule_class.startswith("Unified"):
        set_global_batch_type("trajdata")
    else:
        raise NotImplementedError("Unsupported datamodule_class {}".format(cfg.train.datamodule_class))

    print("\n============= New Training Run with Config =============")
    # print(cfg)
    # print("")
    if resume_ckpt_path is not None:
        root_dir, log_dir, ckpt_dir, video_dir, version_key, resume_ckpt_path = _infer_resume_paths(
            resume_ckpt_path
        )
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt_path}")
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        parsed_step, parsed_epoch = _parse_step_epoch_from_ckpt_path(resume_ckpt_path)
        if parsed_step is not None:
            print(f"Checkpoint filename indicates step={parsed_step}")
            if int(cfg.train.training.num_steps) <= int(parsed_step):
                print(
                    "WARNING: cfg.train.training.num_steps ({}) <= checkpoint step ({}). "
                    "Trainer will likely finish immediately or run very few batches. "
                    "Increase num_steps to continue training.".format(
                        int(cfg.train.training.num_steps), int(parsed_step)
                    )
                )
        if parsed_epoch is not None:
            print(f"Checkpoint filename indicates epoch={parsed_epoch}")
        print(f"Using existing run dir: {os.path.join(root_dir, version_key)}")
    else:
        root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
            exp_name=cfg.name,
            output_dir=cfg.root_dir,
            save_checkpoints=cfg.train.save.enabled,
            auto_remove_exp_dir=auto_remove_exp_dir,
        )

    # Save experiment config to the training dir
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    if cfg.train.logging.terminal_output_to_txt and not debug:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []
    if resume_ckpt_path is not None:
        train_callbacks.append(_ResumeDiagnosticsCallback())

    # Training Parallelism
    supported_strategies = {
        None,
        "dp",
        "ddp",
        "ddp2",
        "ddp_spawn",
        "ddp_find_unused_parameters_false",
        "ddp_sharded",
        "ddp_sharded_spawn",
    }
    if cfg.train.parallel_strategy not in supported_strategies:
        raise ValueError(
            "Unsupported parallel strategy: {}. Supported strategies: {}".format(
                cfg.train.parallel_strategy,
                sorted([s for s in supported_strategies if s is not None]) + [None],
            )
        )

    if not cfg.devices.num_gpus > 1:
        # Override strategy when training on a single GPU
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None
    else:
        # Prefer DDP for multi-GPU performance when no strategy is explicitly set
        if cfg.train.parallel_strategy is None:
            with cfg.train.unlocked():
                cfg.train.parallel_strategy = "ddp"

    ddp_like_strategies = {
        "ddp",
        "ddp2",
        "ddp_spawn",
        "ddp_find_unused_parameters_false",
        "ddp_sharded",
        "ddp_sharded_spawn",
    }
    if cfg.train.parallel_strategy in ddp_like_strategies:
        if cfg.train.training.batch_size % cfg.devices.num_gpus != 0:
            raise ValueError(
                "Training batch_size ({}) must be divisible by num_gpus ({}) for strategy '{}'".format(
                    cfg.train.training.batch_size,
                    cfg.devices.num_gpus,
                    cfg.train.parallel_strategy,
                )
            )
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = int(
                cfg.train.training.batch_size / cfg.devices.num_gpus
            )
        if cfg.train.validation.batch_size % cfg.devices.num_gpus != 0:
            raise ValueError(
                "Validation batch_size ({}) must be divisible by num_gpus ({}) for strategy '{}'".format(
                    cfg.train.validation.batch_size,
                    cfg.devices.num_gpus,
                    cfg.train.parallel_strategy,
                )
            )
        with cfg.train.validation.unlocked():
            cfg.train.validation.batch_size = int(
                cfg.train.validation.batch_size / cfg.devices.num_gpus
            )

    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    # Environment for close-loop evaluation
    if cfg.train.rollout.enabled:
        # Run rollout at regular intervals
        rollout_callback = RolloutCallback(
            exp_config=cfg,
            every_n_steps=cfg.train.rollout.every_n_steps,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
            verbose=True,
            save_video=cfg.train.rollout.save_video,
            video_dir=video_dir
        )
        train_callbacks.append(rollout_callback)

    # Model
    model = algo_factory(
        config=cfg,
        modality_shapes=datamodule.modality_shapes
    )

    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
            cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency ("+str(cfg.train.save.every_n_steps)+") needs to be greater than validation frequency ("+str(cfg.train.validation.every_n_steps)+")"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_%s{%s:.2f}" % (metric_name, metric_key),
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)

    if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
        assert (
            cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
        ), "checkpointing frequency needs to be greater than rollout frequency"
        ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
            # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
            auto_insert_metric_name=False,
            save_top_k=cfg.train.save.best_k,  # save the best k models
            monitor="rollout/metrics_ego_ADE",
            mode="min",
            every_n_train_steps=cfg.train.save.every_n_steps,
            verbose=True,
        )
        train_callbacks.append(ckpt_rollout_callback)

    # a ckpt monitor to save at fixed interval
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=10000,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)

    # # NOTE: this is annoying to be able to load in EMA weights at test time,
    # #           implemented in the lightning module class instead.
    # if 'use_ema' in cfg.algo.__dict__ and cfg.algo.use_ema:
    #     from tbsim.utils.ema import EMA
    #     ema_callback = EMA(decay=cfg.algo.ema_decay, every_n_train_steps=cfg.algo.ema_step)
    #     train_callbacks.append(ema_callback)

    # Logging
    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_tb:
        logger = TensorBoardLogger(
            save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
        )
        print("Tensorboard event will be saved at {}".format(logger.log_dir))
    else:
        print("WARNING: not logging training stats")
    if logger is not None:
        train_callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        # device & distributed training setup
        gpus=cfg.devices.num_gpus,
        strategy=cfg.train.parallel_strategy,
        # setting for overfit debugging
        # limit_val_batches=0,
        # overfit_batches=2
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

    parser.add_argument(
        "--on_ngc",
        action="store_true",
        help="whether running the script on ngc (this will change some behaviors like avoid writing into dataset)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="(optional) path to a .ckpt file to resume training (continues the same run directory).",
    )

    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default=None,
        help="(optional) path to an existing run directory (e.g. .../outputs/<exp>/run17). "
        "The latest checkpoint under checkpoints/ will be used.",
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
        print('args.config_name', args.config_name)
        print('default_config', default_config)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.on_ngc:
        ngc_job_id = socket.gethostname()
        default_config.name = default_config.name + "_" + ngc_job_id

    default_config.train.on_ngc = args.on_ngc

    if args.debug:
        # Test policy rollout
        default_config.train.validation.every_n_steps = 5
        default_config.train.save.every_n_steps = 10
        default_config.train.rollout.every_n_steps = 10
        default_config.train.rollout.num_episodes = 1

    # make rollout evaluation config consistent with the rest of the config
    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name
        assert default_config.algo.eval_class is not None, \
            "Please set an eval_class for {}".format(default_config.algo.name)
        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path
        for k in default_config.eval[default_config.eval.env]:  # copy env-specific config to the global-level
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]
        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")
        # default_config.eval.pop("trajdata")

    default_config.lock()  # Make config read-only
    if args.resume_ckpt is not None and args.resume_run_dir is not None:
        raise ValueError("Use only one of --resume_ckpt or --resume_run_dir")
    resume_ckpt_path = args.resume_ckpt
    if resume_ckpt_path is None and args.resume_run_dir is not None:
        resume_ckpt_path = _latest_ckpt_in_run_dir(args.resume_run_dir)

    main(
        default_config,
        auto_remove_exp_dir=args.remove_exp_dir,
        debug=args.debug,
        resume_ckpt_path=resume_ckpt_path,
    )
