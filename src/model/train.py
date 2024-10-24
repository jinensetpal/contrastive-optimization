#!/usr/bin/env python3

from ..data.oxford_iiit_pet import get_generators
from .loss import ContrastiveLoss
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
        selected=None, init_epoch=0, mlflow_run_id=None):
    model.train()
    start_time = time.time()
    selected = selected or {'last': model.state_dict(),
                            'epoch': init_epoch,
                            'acc': 0.0}

    with mlflow.start_run(mlflow_run_id):
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH', 'SELECT_BEST'])})
        mlflow.log_param('optimizer_fn', const.OPTIMIZER)

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(init_epoch, const.EPOCHS + init_epoch + int(init_epoch == 0)):
            if not (epoch) % interval: print('-' * 10)
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['contrast_loss', 'acc', 'divergence_loss', 'ablated_ce_loss', 'cse_loss'] for split in const.SPLITS[:2]]}

            for split, dataloader in zip(const.SPLITS[:2], (train, val)):
                for batch_idx, (X, y) in enumerate(dataloader):
                    if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.zero_grad()

                    y_pred = model(X)
                    batch_loss = criterion(y_pred, y) if criterion._get_name() != 'CrossEntropyLoss' else criterion(y_pred[0], y[1])

                    metrics[f'{split}_acc'].extend(y[1].argmax(1).eq(y_pred[0].argmax(1)).unsqueeze(1).tolist())
                    metrics[f'{split}_contrast_loss'].append(batch_loss.item())
                    metrics[f'{split}_cse_loss'].append(F.cross_entropy(y_pred[0], y[1]).item())

                    del y_pred, X, y
                    torch.cuda.empty_cache()

                    if criterion._get_name() != 'CrossEntropyLoss':
                        metrics[f'{split}_ablated_ce_loss'].append(criterion.prev[0])
                        metrics[f'{split}_divergence_loss'].append(criterion.prev[1])

                    if split == 'train' and epoch > 0:  # epoch 0 is for evaluating performance on initalization
                        batch_loss.backward()
                        mlflow.log_metric(f'{split}_batchwise_loss', batch_loss.item(), step=epoch * len(dataloader) + batch_idx)

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
        if const.SELECT_BEST and 'best' not in selected:
            selected['best'] = model.state_dict()
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
    const.MODEL_NAME = sys.argv[1]
    const.FINETUNING = 'finetuned' in const.MODEL_NAME
    const.OPTIMIZER = 'Adam' if 'adam' in const.MODEL_NAME else 'SGD'
    const.PRETRAINED_BACKBONE = 'pretrained' in const.MODEL_NAME

    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    model = Model(const.IMAGE_SHAPE, is_contrastive='default' not in const.MODEL_NAME)
    train, val, test = get_generators()

    params = [*model.linear.parameters(),
              *model.backbone.layer4[0].conv2.parameters(),
              *model.backbone.layer4[0].downsample[0].parameters()] if const.FINETUNING else model.parameters()
    if const.OPTIMIZER == 'SGD': optimizer = torch.optim.SGD(params, lr=const.LEARNING_RATE, momentum=const.MOMENTUM, weight_decay=const.WEIGHT_DECAY)
    else: optimizer = torch.optim.Adam(params, lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = ContrastiveLoss(model.get_contrastive_cams) if 'default' not in const.MODEL_NAME else torch.nn.CrossEntropyLoss()

    checkpoint_args = {'init_epoch': 0,
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
