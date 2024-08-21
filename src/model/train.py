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
                    'epoch': 0,
                    'acc': 0.0}

    with mlflow.start_run(mlflow_run_id):
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(init_epoch, const.EPOCHS + init_epoch):
            if not (epoch) % interval: print('-' * 10)
            metrics = {metric: [] for metric in ['train_contrast_loss', 'train_kld_loss', 'train_background_loss', 'train_foreground_loss', 'train_acc', 'cse_loss',
                                                 'valid_contrast_loss', 'valid_kld_loss', 'valid_background_loss', 'valid_foreground_loss', 'valid_acc']}

            for batch_idx, ((X_train, y_train), (X_valid, y_valid)) in enumerate(zip(train, val)):
                y_pred_train = model(X_train)
                y_pred_valid = model(X_valid)

                metrics['train_acc'].extend((torch.argmax(y_train[1], dim=1) == torch.argmax(y_pred_train[0], dim=1)).unsqueeze(1).tolist())
                metrics['valid_acc'].extend((torch.argmax(y_valid[1], dim=1) == torch.argmax(y_pred_valid[0], dim=1)).unsqueeze(1).tolist())

                train_batch_loss = loss(y_pred_train, y_train) if loss._get_name() != 'CrossEntropyLoss' else loss(y_pred_train[0], y_train[1])
                metrics['train_contrast_loss'].append(train_batch_loss.item())
                metrics['cse_loss'].append(F.cross_entropy(y_pred_train[0], y_train[1]).item())
                if loss._get_name() != 'CrossEntropyLoss':
                    metrics['train_kld_loss'].append(loss.prev[0])
                    metrics['train_background_loss'].append(loss.prev[1])
                    metrics['train_foreground_loss'].append(loss.prev[2])

                metrics['valid_contrast_loss'].append((loss(y_pred_valid, y_valid) if loss._get_name() != 'CrossEntropyLoss' else loss(y_pred_valid[0], y_valid[1])).item())
                if loss._get_name() != 'CrossEntropyLoss':
                    metrics['valid_kld_loss'].append(loss.prev[0])
                    metrics['valid_background_loss'].append(loss.prev[1])
                    metrics['valid_foreground_loss'].append(loss.prev[2])

                train_batch_loss.backward()
                if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS:
                    optimizer.step()
                    optimizer.zero_grad()

            metrics = {metric: np.mean(metrics[metric]) for metric in metrics}
            mlflow.log_metrics(metrics, step=epoch)

            if metrics['valid_acc'] > best['acc']:
                best['param'] = model.state_dict()
                best['epoch'] = epoch + 1
                best['acc'] = metrics['valid_acc']

            if not (epoch) % interval:
                print(f'epoch\t\t\t: {epoch}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if const.TRAIN_CUTOFF is not None and time.time() - start_time >= const.TRAIN_CUTOFF: break
        if const.SELECT_BEST:
            mlflow.log_metrics({'selected_epoch': best['epoch'],
                                'selected_valid_acc': best['acc']}, step=epoch)
            model.load_state_dict(best['param'])
        print('-' * 10)
        return epoch, best


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1]
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    model = Model(const.IMAGE_SHAPE, is_contrastive=const.MODEL_NAME != 'default')  # initialize before loss functions to ensure accurate cam size configuration
    train, val, test = get_generators()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.LEARNING_RATE)
    loss = ContrastiveLoss(model.get_contrastive_cams) if const.MODEL_NAME != 'default' else torch.nn.CrossEntropyLoss()

    checkpoint_args = {'init_epoch': 1,
                       'mlflow_run_id': None}
    best = None
    if const.CHECKPOINTING and len(sys.argv) == 2:  # add extra sys.argv to signify first checkpointing run
        model.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}.pt', map_location=const.DEVICE))
        optimizer.load_state_dict(torch.load(const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt', map_location=const.DEVICE))
        checkpoint_args = json.load(open(const.MODELS_DIR / 'checkpoint_metadata.json'))
        prev_metrics = mlflow.get_run(checkpoint_args['mlflow_run_id']).data.metrics
        best = {'param': model.state_dict(),
                'epoch': prev_metrics['selected_epoch'],
                'acc': prev_metrics['selected_valid_acc']}

    completed_epochs, best = fit(model, optimizer, loss, train, val, best=best, **checkpoint_args)
    torch.save(model.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}.pt')

    if const.CHECKPOINTING:
        torch.save(optimizer.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}_optim.pt')
        json.dump({'init_epoch': completed_epochs+1, 'mlflow_run_id': mlflow.last_active_run().info.run_id}, open(const.MODELS_DIR / 'checkpoint_metadata.json', 'w'))
