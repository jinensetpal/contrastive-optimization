#!/usr/bin/env python3

from ..data.oxford_iiit_pet import get_generators as oxford_iiit_pet
from torch.distributed.optim import ZeroRedundancyOptimizer
from ..data.imagenet import get_generators as imagenet
from contextlib import nullcontext
from .loss import ContrastiveLoss
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from torch import optim
from .arch import Model
from src import const
from torch import nn
import numpy as np
import mlflow
import torch
import json
import time
import sys
import os


# overrides by model name; schedule multiple jobs w/o reconfiguration overhead
def configure(model_name):
    const.MODEL_NAME = model_name
    const.OPTIMIZER = 'Adam' if 'adam' in const.MODEL_NAME else 'SGD'
    const.USE_ZERO = 'zero' in const.MODEL_NAME and const.DDP
    const.EMA = 'ema' in const.MODEL_NAME
    const.DATASET = 'imagenet' if 'imagenet' in const.MODEL_NAME else 'oxford-iiit'
    if 'ablated_only' in const.MODEL_NAME: const.LAMBDAS[-1] = 0

    if const.DATASET != 'imagenet':
        const.BINARY_CLS = 'multiclass' not in const.MODEL_NAME
        const.N_CLASSES = 2 if const.BINARY_CLS else 37
        const.FINETUNING = 'finetuned' in const.MODEL_NAME
        const.PRETRAINED_BACKBONE = 'pretrained' in const.MODEL_NAME
        const.BBOX_MAP = 'bbox' in const.MODEL_NAME
    else:
        const.N_CLASSES = 1000
        const.USE_CUTMIX = 'cutmixed' in const.MODEL_NAME
        const.AUGMENT = 'augmented' in const.MODEL_NAME
        if 'label_smoothing' not in const.MODEL_NAME: const.LABEL_SMOOTHING = 0
        const.PRETRAINED_BACKBONE = False
        const.FINETUNING = False


def fit(model, optimizer, scheduler, criterion, train, val,
        ema=None, selected=None, init_epoch=0, mlflow_run_id=None):
    model.train()
    start_time = time.time()
    selected = selected or {'last': model.state_dict(),
                            'epoch': init_epoch,
                            'acc': 0.0}
    is_primary_rank = not const.DDP or (const.DDP and const.DEVICE == 0)

    with mlflow.start_run(mlflow_run_id) if is_primary_rank else nullcontext():
        # log hyperparameters
        if is_primary_rank: mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH', 'EPOCHS', 'SELECT_BEST', 'DEVICE', 'TRAIN_CUTOFF', 'PORT'])})

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(init_epoch, const.EPOCHS + int(init_epoch == 0)):
            if not (epoch) % interval: print('-' * 10)
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['contrast_loss', 'acc', 'divergence_loss', 'ablated_ce_loss', 'cse_loss'] for split in const.SPLITS[:2]]}

            for split, dataloader in zip(const.SPLITS[:2], (train, val)):
                for batch_idx, (X, y) in enumerate(dataloader):
                    if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.zero_grad()
                    X = X.to(const.DEVICE)
                    y = [y_i.to(const.DEVICE) for y_i in y]

                    y_pred = model(X)
                    batch_loss = criterion(y_pred, y) if criterion._get_name() != 'CrossEntropyLoss' else criterion(y_pred[0], y[1])

                    metrics[f'{split}_acc'].extend(y[1].argmax(1).eq(y_pred[0].argmax(1)).unsqueeze(1).tolist())
                    metrics[f'{split}_contrast_loss'].append(batch_loss.item())
                    metrics[f'{split}_cse_loss'].append(F.cross_entropy(y_pred[0], y[1]).item())

                    del y_pred, X, y
                    torch.cuda.empty_cache()

                    if criterion._get_name() != 'CrossEntropyLoss': metrics[f'{split}_ablated_ce_loss'].append(criterion.prev)
                    if split == 'train' and epoch > 0:  # epoch 0 is for evaluating performance on initalization
                        batch_loss.backward(inputs=optimizer.param_groups[0]['params'])
                        if is_primary_rank and const.LOG_BATCHWISE: mlflow.log_metric(f'{split}_batchwise_loss', batch_loss.item(), step=(epoch-1) * len(dataloader) + batch_idx)

                        if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.step()

                    if ema and not (batch_idx+1) % const.EMA_STEPS:
                        ema.update_parameters(model)
                        if epoch < const.LR_WARMUP_EPOCHS: ema.n_averaged.fill_(0)

                    del batch_loss
                    torch.cuda.empty_cache()

            metrics = {metric: np.mean(metrics[metric]) for metric in metrics}
            if const.DDP:
                store.set(f'metric_{const.DEVICE}', json.dumps(metrics))
                dist.barrier(device_ids=[const.DEVICE])

            if not is_primary_rank:
                scheduler.step()
                continue

            if const.DDP:
                dist_metrics = [json.loads(store.get(f'metric_{rank}')) for rank in range(int(os.environ['WORLD_SIZE']))]
                metrics = {key: np.mean([metric[key] for metric in dist_metrics]) for key in metrics.keys()}

            metrics['lr'] = scheduler.get_last_lr()[-1]
            mlflow.log_metrics(metrics, step=epoch-1)
            scheduler.step()

            if const.SELECT_BEST and metrics['valid_acc'] > selected['acc']:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['acc'] = metrics['valid_acc']

            if not (epoch) % interval:
                print(f'epoch\t\t\t: {epoch}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if const.TRAIN_CUTOFF is not None and time.time() - start_time >= const.TRAIN_CUTOFF: break

        if is_primary_rank:
            if ema: selected['ema'] = ema.state_dict()
            selected['last'] = deepcopy(model.state_dict())
            if const.SELECT_BEST and 'best' not in selected:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['acc'] = metrics['valid_acc']

            if const.SELECT_BEST:
                mlflow.log_metrics({'selected_epoch': selected['epoch'],
                                    'selected_valid_acc': selected['acc']}, step=epoch)
                model.load_state_dict(selected['best'])
            else:
                selected['epoch'] = epoch
                selected['acc'] = metrics['valid_acc']

        print('-' * 10)
        return epoch, selected


if __name__ == '__main__':
    # Usage: $ python -m path.to.script model_name --nocheckpoint
    configure(sys.argv[1])

    is_contrastive = 'default' not in const.MODEL_NAME

    path = const.MODELS_DIR / const.MODEL_NAME
    (path).mkdir(exist_ok=True, parents=True)

    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    if const.DDP:
        const.DEVICE = int(os.environ['LOCAL_RANK'])
        store = dist.TCPStore('127.0.0.1', const.PORT, is_master=const.DEVICE == 0)
        torch.cuda.set_device(const.DEVICE)
        dist.init_process_group('nccl')
        dist.barrier(device_ids=[const.DEVICE])

    model = Model(const.IMAGE_SHAPE, is_contrastive=is_contrastive).to(const.DEVICE)
    criterion = ContrastiveLoss(model.get_contrastive_cams, is_label_mask=const.USE_CUTMIX) if is_contrastive else nn.CrossEntropyLoss(label_smoothing=const.LABEL_SMOOTHING)
    ema = optim.swa_utils.AveragedModel(model, device=const.DEVICE, avg_fn=optim.swa_utils.get_ema_avg_fn(1 - min(1, (1 - const.EMA_DECAY) * const.BATCH_SIZE * const.EMA_STEPS / const.EPOCHS)), use_buffers=True) if const.EMA else None

    if const.DDP:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[const.DEVICE])
        model.load_state_dict = model.module.load_state_dict
        model.state_dict = model.module.state_dict

    train, val, test = imagenet() if const.DATASET == 'imagenet' else oxford_iiit_pet()

    if const.FINETUNING:
        params = [*model.linear.parameters(),] if is_contrastive else [*model.linear.parameters(),
                                                                       *model.backbone.layer4[0].conv2.parameters(),
                                                                       *model.backbone.layer4[0].downsample[0].parameters()]
    else: params = model.parameters()

    if const.OPTIMIZER == 'SGD':
        if const.DDP and const.USE_ZERO: optimizer = ZeroRedundancyOptimizer(params, optim.SGD, lr=const.LR, momentum=const.MOMENTUM, weight_decay=const.WEIGHT_DECAY)
        else: optimizer = optim.SGD(params, lr=const.LR, momentum=const.MOMENTUM, weight_decay=const.WEIGHT_DECAY)
    else:
        if const.DDP and const.USE_ZERO: optimizer = ZeroRedundancyOptimizer(params, optim.Adam, lr=const.LR, weight_decay=const.WEIGHT_DECAY)
        else: optimizer = optim.Adam(params, lr=const.LR, weight_decay=const.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=const.EPOCHS - const.LR_WARMUP_EPOCHS, eta_min=2E-4)
    if const.LR_WARMUP_EPOCHS:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=const.LR_WARMUP_DECAY, total_iters=const.LR_WARMUP_EPOCHS)
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup, scheduler], optimizer=optimizer)

    checkpoint_args = {'init_epoch': 0,
                       'mlflow_run_id': None}
    selected = None
    if const.CHECKPOINTING and len(sys.argv) == 2:  # add extra sys.argv to signify first checkpointing run
        model.load_state_dict(torch.load(path / 'last.pt', map_location=torch.device(const.DEVICE), weights_only=True))
        optimizer.load_state_dict(torch.load(path / 'optim.pt', map_location=torch.device(const.DEVICE), weights_only=True))
        checkpoint_args = json.load(open(path / 'checkpoint_metadata.json'))
        prev_metrics = mlflow.get_run(checkpoint_args['mlflow_run_id']).data.metrics

        selected = {'best': torch.load(path / 'best.pt', map_location='cpu', weights_only=True),
                    'last': model.state_dict(),
                    'epoch': prev_metrics['selected_epoch'],
                    'acc': prev_metrics['selected_valid_acc']}
        if ema and (path / 'ema.pt').exists():
            selected['ema'] = torch.load(path / 'ema.pt', map_location=torch.device(const.DEVICE), weights_only=False)
            ema = selected['ema']

    completed_epochs, selected = fit(model, optimizer, scheduler, criterion, train, val, ema=ema, selected=selected, **checkpoint_args)

    if const.DDP:
        if const.USE_ZERO: optimizer.consolidate_state_dict()
        dist.destroy_process_group()

    if not const.DDP or (const.DDP and const.DEVICE == 0):
        torch.save(selected['last'], path / 'last.pt')
        if const.SELECT_BEST: torch.save(selected['best'], path / 'best.pt')
        if const.EMA: torch.save(selected['ema'], path / 'ema.pt')

        if const.CHECKPOINTING:
            torch.save(optimizer.state_dict(), path / 'optim.pt')
            json.dump({'init_epoch': completed_epochs+1, 'mlflow_run_id': mlflow.last_active_run().info.run_id}, open(path / 'checkpoint_metadata.json', 'w'))
