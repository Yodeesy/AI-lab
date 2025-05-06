import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import autocast, GradScaler


def evaluate_model(model, loader, device, scaler_y):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).cpu().numpy()
            targets = batch_y.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    preds_real = scaler_y.inverse_transform(all_preds)
    targets_real = scaler_y.inverse_transform(all_targets)

    mae = mean_absolute_error(targets_real, preds_real)
    rmse = np.sqrt(mean_squared_error(targets_real, preds_real))
    return mae, rmse


def train(model, train_dataset, test_dataset, optimizer, scheduler, config, device, scaler_y=None, use_amp=True):
    # 获取优化器名称用于提取子配置
    opt_name = optimizer.__class__.__name__.lower()
    opt_config = config.get(opt_name, {})

    # 获取损失函数配置，默认为 MSE
    loss_type = opt_config.get("loss_funciton", "MSE")
    criterion = nn.HuberLoss() if loss_type == 'Huber' else nn.MSELoss()

    scaler = GradScaler(enabled=use_amp)
    generator = torch.Generator().manual_seed(config.get('seed', 42))

    batch_size = opt_config.get('batch_size', 64)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=config.get('num_workers', 0), pin_memory=config.get('pin_memory', False),
                              generator=generator)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=config.get('num_workers', 0), pin_memory=config.get('pin_memory', False))

    all_losses, all_test_maes, all_test_rmses = [], [], []
    best_mae = float('inf')
    best_epoch = 0
    patience = opt_config.get('patience', config.get('patience', 10))
    counter = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mae, _ = evaluate_model(model, test_loader, device, scaler_y)
            scheduler.step(mae)
        else:
            scheduler.step()

        if (epoch + 1) % config.get('log_interval', 10) == 0 or epoch == 0:
            mae, rmse = evaluate_model(model, test_loader, device, scaler_y)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Test MAE={mae:.2f} | Test RMSE={rmse:.2f}")
            all_test_maes.append(mae)
            all_test_rmses.append(rmse)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch + 1
                counter = 0
            else:
                counter += 1
                if config.get('early_stopping', False) and counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    final_mae, final_rmse = evaluate_model(model, test_loader, device, scaler_y)
    print(f"\n📊 Final Test Evaluation ({optimizer.__class__.__name__}):")
    print(f"Final Test MAE  : {final_mae:.2f}")
    print(f"Final Test RMSE : {final_rmse:.2f}")
    print(f"✅ Best MAE: {best_mae:.2f} at epoch {best_epoch}")

    return all_losses, all_test_maes, all_test_rmses
