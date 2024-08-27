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


def fit(model, optimizer, loss, train, val, best=None, init_epoch=1, mlflow_run_id=None):
    start_time = time.time()
    best = best or {'param': model.state_dict(),
                    'epoch': init_epoch,
                    'acc': 0.0}

    with mlflow.start_run(mlflow_run_id):
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'SGD')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(init_epoch, const.EPOCHS + init_epoch):
            if not (epoch) % interval: print('-' * 10)
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['contrast_loss', 'kld_loss', 'background_loss', 'foreground_loss', 'acc', 'cse_loss'] for split in const.SPLITS[:2]]}

            for split, dataloader in zip(const.SPLITS[:2], (train, val)):
                for batch_idx, (X, y) in enumerate(dataloader):
                    y_pred = model(X)
                    batch_loss = loss(y_pred, y) if loss._get_name() != 'CrossEntropyLoss' else loss(y_pred[0], y[1])

                    metrics[f'{split}_acc'].extend(y[1].argmax(1).eq(y_pred[0].argmax(1)).unsqueeze(1).tolist())
                    metrics[f'{split}_contrast_loss'].append(batch_loss.item())
                    metrics[f'{split}_cse_loss'].append(F.cross_entropy(y_pred[0], y[1]).item())

                    del y_pred, X, y
                    torch.cuda.empty_cache()

                    if loss._get_name() != 'CrossEntropyLoss':
                        metrics[f'{split}_kld_loss'].append(loss.prev[0])
                        metrics[f'{split}_background_loss'].append(loss.prev[1])
                        metrics[f'{split}_foreground_loss'].append(loss.prev[2])

                    if split == 'train':
                        batch_loss.backward()
                        mlflow.log_metric(f'{split}_batchwise_loss', batch_loss.item(), step=(epoch-1) * len(dataloader) + batch_idx)

                        if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS:
                            optimizer.step()
                            optimizer.zero_grad()

            metrics = {metric: np.mean(metrics[metric]) for metric in metrics}
            mlflow.log_metrics(metrics, step=epoch-1)

            if metrics['valid_acc'] > best['acc']:
                best['param'] = model.state_dict()
                best['epoch'] = epoch
                best['acc'] = metrics['valid_acc']

            if not (epoch) % interval:
                print(f'epoch\t\t\t: {epoch}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if const.TRAIN_CUTOFF is not None and time.time() - start_time >= const.TRAIN_CUTOFF: break
        if const.SELECT_BEST:
            mlflow.log_metrics({'selected_epoch': best['epoch'],
                                'selected_valid_acc': best['acc']}, step=epoch-1)
            model.load_state_dict(best['param'])
        print('-' * 10)
        return epoch, best


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1]
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    model = Model(const.IMAGE_SHAPE, is_contrastive=const.MODEL_NAME != 'default')  # initialize before loss functions to ensure accurate cam size configuration
    train, val, test = get_generators()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    loss = ContrastiveLoss(model.get_contrastive_cams) if const.MODEL_NAME != 'default' else torch.nn.CrossEntropyLoss()

    checkpoint_args = {'init_epoch': 1,
                       'mlflow_run_id': None}
    best = None
    if const.CHECKPOINTING and len(sys.argv) == 2:  # add extra sys.argv to signify first checkpointing run
        model.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}.pt', map_location=const.DEVICE))
        optimizer.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt', map_location=const.DEVICE))
        checkpoint_args = json.load(open(const.MODELS_DIR / f'{const.MODEL_NAME}_checkpoint_metadata.json'))
        prev_metrics = mlflow.get_run(checkpoint_args['mlflow_run_id']).data.metrics
        best = {'param': model.state_dict(),
                'epoch': prev_metrics['selected_epoch'],
                'acc': prev_metrics['selected_valid_acc']}

    completed_epochs, best = fit(model, optimizer, loss, train, val, best=best, **checkpoint_args)
    torch.save(model.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}.pt')

    if const.CHECKPOINTING:
        torch.save(optimizer.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt')
        json.dump({'init_epoch': completed_epochs+1, 'mlflow_run_id': mlflow.last_active_run().info.run_id}, open(const.MODELS_DIR / f'{const.MODEL_NAME}_checkpoint_metadata.json', 'w'))
