#!/usr/bin/env python3

from ..data.oxford_iiit_pet import get_generators
from .loss import PieceWiseLoss
import torch.nn.functional as F
from .arch import Model
from src import const
import numpy as np
import mlflow
import torch
import json
import time
import sys


def fit(model, optimizer, scheduler, criterion, train, val,
        selected=None, init_epoch=1, mlflow_run_id=None):
    model.train()
    start_time = time.time()
    selected = selected or {'last': model.state_dict(),
                            'epoch': init_epoch,
                            'acc': 0.0}

    with mlflow.start_run(mlflow_run_id):
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH', 'SELECT_BEST'])})
        mlflow.log_param('optimizer_fn', 'SGD')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(init_epoch, const.EPOCHS + init_epoch):
            if not (epoch) % interval: print('-' * 10)
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['contrast_loss', 'background_loss', 'foreground_loss', 'acc', 'cse_loss'] for split in const.SPLITS[:2]]}

            for split, dataloader in zip(const.SPLITS[:2], (train, val)):
                for batch_idx, (X, y) in enumerate(dataloader):
                    if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.zero_grad()

                    y_pred = model(X)
                    if model.is_contrastive:
                        cc = model.get_contrastive_cams(y[1], y_pred[1])
                        fg_masks = y[0].repeat((cc.shape[1], 1, 1, 1)).permute(1, 0, 2, 3)
                        out = (-cc * fg_masks + cc.abs() * (1 - fg_masks)).sum(dim=[2, 3])
                    else: out = y_pred[0]
                    batch_loss = criterion(out, y[1])

                    metrics[f'{split}_acc'].extend(y[1].argmax(1).eq(y_pred[0].argmax(1)).unsqueeze(1).tolist())
                    metrics[f'{split}_contrast_loss'].append(batch_loss.item())
                    metrics[f'{split}_cse_loss'].append(criterion(y_pred[0], y[1]).item())

                    del y_pred, X, y
                    torch.cuda.empty_cache()

                    if split == 'train':
                        batch_loss.backward()
                        mlflow.log_metric(f'{split}_batchwise_loss', batch_loss.item(), step=(epoch-1) * len(dataloader) + batch_idx)

                        if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.step()
            scheduler.step()

            metrics = {metric: np.mean(metrics[metric]) for metric in metrics}
            mlflow.log_metrics(metrics, step=epoch-1)

            if const.SELECT_BEST and metrics['valid_acc'] > selected['acc']:
                selected['best'] = model.state_dict()
                selected['epoch'] = epoch
                selected['acc'] = metrics['valid_acc']

            if not (epoch) % interval:
                print(f'epoch\t\t\t: {epoch}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if const.TRAIN_CUTOFF is not None and time.time() - start_time >= const.TRAIN_CUTOFF: break

        selected['last'] = model.state_dict()
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
    const.MODEL_NAME = sys.argv[1]
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    model = Model(const.IMAGE_SHAPE, is_contrastive=const.MODEL_NAME != 'default')
    if const.DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)
        model.is_contrastive = model.module.is_contrastive
        model.get_contrastive_cams = model.module.get_contrastive_cams

    train, val, test = get_generators()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM,
                                weight_decay=const.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_args = {'init_epoch': 1,
                       'mlflow_run_id': None}
    selected = None
    if const.CHECKPOINTING and len(sys.argv) == 2:  # add extra sys.argv to signify first checkpointing run
        model.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}_last.pt', map_location=const.DEVICE))
        optimizer.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt', map_location=const.DEVICE))
        checkpoint_args = json.load(open(const.MODELS_DIR / f'{const.MODEL_NAME}_checkpoint_metadata.json'))
        prev_metrics = mlflow.get_run(checkpoint_args['mlflow_run_id']).data.metrics

        selected = {'best': torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}_best.pt', map_location='cpu'),
                    'last': model.state_dict(),
                    'epoch': prev_metrics['selected_epoch'],
                    'acc': prev_metrics['selected_valid_acc']}

    completed_epochs, selected = fit(model, optimizer, scheduler, criterion, train, val, selected=selected, **checkpoint_args)
    torch.save(selected['last'], const.MODELS_DIR / f'{const.MODEL_NAME}_last.pt')
    if const.SELECT_BEST: torch.save(selected['best'], const.MODELS_DIR / f'{const.MODEL_NAME}_best.pt')

    if const.CHECKPOINTING:
        torch.save(optimizer.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt')
        json.dump({'init_epoch': completed_epochs+1, 'mlflow_run_id': mlflow.last_active_run().info.run_id}, open(const.MODELS_DIR / f'{const.MODEL_NAME}_checkpoint_metadata.json', 'w'))
