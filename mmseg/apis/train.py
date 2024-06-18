import random
import warnings
import os
import geoopt as gt

import numpy as np
import torch
import torch.optim as optim

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from ..mmcv_custom.runner import build_optimizer, build_runner, get_dist_info
from ..mmcv_custom.runner.optimizer import build_optimizer, OPTIMIZERS

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from ..multi_optim import MultipleOptimizer


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            #device_ids=[torch.cuda.current_device()],
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    def get_parameter_groups(model):
        groups = ([], [], [], [], [], [])
        for name, value in model.named_parameters():
            if 'backbone' in name:
                if 'stem' in name or 'layer1' in name:
                    groups[0].append(value)
                elif 'layer2' in name or 'layer3' in name:
                    groups[1].append(value)
                elif 'normals' in name:
                    groups[4].append(value)
                elif 'offsets' in name:
                    groups[5].append(value)   
                else:
                    groups[2].append(value)
            else:
                if 'normals' in name:
                    groups[4].append(value)
                elif 'offsets' in name:
                    groups[5].append(value) 
                else:
                    groups[3].append(value)
        return groups

    param_groups = get_parameter_groups(model)
    hyperbolic_lr = 0.1
  
    optimizer_SGD = optim.SGD([
        {'params': param_groups[0], 'name': 'stem or layer1', 'lr': 0.1*cfg.optimizer.lr},
        {'params': param_groups[1], 'name':'layer2 or layer3', 'lr': 0.15*cfg.optimizer.lr},
        {'params': param_groups[2], 'name': 'in backbone', 'lr': 0.2*cfg.optimizer.lr}, 
        {'params': param_groups[3], 'name': 'not in backbone nor hyperbolic'}],
        lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, 
        weight_decay=cfg.optimizer.weight_decay)

    optimizer_offset = gt.optim.RiemannianSGD(param_groups[5], lr=0.0001, momentum=0.9)
    optimizer_normals = optim.SGD([{'params': param_groups[4], 'lr': 0.001, 'name': 'normals'}])
    optimizer = MultipleOptimizer(optimizer_SGD, optimizer_offset, optimizer_normals)


    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    #with torch.autograd.set_detect_anomaly(True):
    runner.run(data_loaders, cfg.workflow)
