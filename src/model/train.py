#!/usr/bin/env python3

from ..data.oxford_iiit_pet import get_generators
from .loss import ContrastiveLoss
import torch.nn.functional as F
from .arch import Model
from src import const
import mlflow
import torch
import sys


def fit(model, optimizer, loss, train, val):
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    best = {'param': model.state_dict(),
            'epoch': 0,
            'acc': 0.0}

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            train_loss = torch.empty(1, device=const.DEVICE)
            valid_loss = torch.empty(1, device=const.DEVICE)
            train_acc = torch.empty(1, device=const.DEVICE)
            valid_acc = torch.empty(1, device=const.DEVICE)
            cse_loss = torch.empty(1, device=const.DEVICE)

            for train_batch, valid_batch in zip(train, val):
                optimizer.zero_grad()

                X_train, y_train, X_valid, y_valid = [*train_batch, *valid_batch]
                y_pred_train = model(X_train)
                y_pred_valid = model(X_valid)

                train_acc = torch.vstack([train_acc, (torch.argmax(y_train[1], dim=1) == torch.argmax(y_pred_train[0], dim=1)).unsqueeze(1)])
                valid_acc = torch.vstack([valid_acc, (torch.argmax(y_valid[1], dim=1) == torch.argmax(y_pred_valid[0], dim=1)).unsqueeze(1)])

                train_batch_loss = loss(y_pred_train, y_train) if loss._get_name() != 'CrossEntropyLoss' else loss(y_pred_train[0], y_train[1])
                train_loss = torch.vstack([train_loss, torch.tensor(train_batch_loss)])
                cse_loss = torch.vstack([cse_loss, F.cross_entropy(y_pred_train[0], y_train[1])])

                valid_batch_loss = loss(y_pred_valid, y_valid) if loss._get_name() != 'CrossEntropyLoss' else loss(y_pred_valid[0], y_valid[1])
                valid_loss = torch.vstack([valid_loss, torch.tensor(valid_batch_loss)])

                train_batch_loss.backward()
                optimizer.step()

            train_acc = train_acc[1:]
            valid_acc = valid_acc[1:]
            train_loss = train_loss[1:].mean()
            valid_loss = valid_loss[1:].mean()
            cse_loss = cse_loss[1:].mean()
            metrics = {'train_contrast_loss': train_loss.item(),
                       'val_contrast_loss': valid_loss.item(),
                       'benchmark_cse_loss': cse_loss.item(),
                       'train_acc': (train_acc[1:].sum() / train_acc.shape[0]).item(),
                       'valid_acc': (valid_acc[1:].sum() / valid_acc.shape[0]).item()}
            mlflow.log_metrics(metrics, step=epoch)

            if metrics['valid_acc'] > best['acc']:
                best['param'] = model.state_dict()
                best['epoch'] = epoch + 1
                best['acc'] = metrics['valid_acc']

            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')
        if const.SELECT_BEST:
            mlflow.log_param('selected_epoch', best['epoch'])
            model.load_state_dict(best['param'])
        print('-' * 10)


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1]

    model = Model(const.IMAGE_SHAPE)  # initialize before loss functions to ensure accurate cam size configuration
    train, val, test = get_generators()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.LEARNING_RATE)
    loss = ContrastiveLoss(model.get_contrastive_cams) if const.MODEL_NAME != 'default' else torch.nn.CrossEntropyLoss()
    fit(model, optimizer, loss, train, val)
    torch.save(model.state_dict(), const.MODELS_DIR / f'{const.MODEL_NAME}.pt')
