import torch
import random
import numpy as np
from torch.utils.data import TensorDataset
from model import MLP
from trainer import train
from utils import load_yaml_config
from dataset import load_data
from visualizer import plot_single_curve, plot_compare_curve
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = load_yaml_config("configs.yaml")

    set_seed(config.get('seed', 42))

    # 动态选择设备：GPU / CPU
    device = torch.device("cuda" if (torch.cuda.is_available() and config.get('use_gpu', True)) else "cpu")
    print(f"🚀 Using device: {device}")

    # 是否使用 AMP（自动混合精度）
    use_amp = config.get('use_amp', True)

    # 数据加载
    X, y, X_scaler, y_scaler = load_data(config['data_path'])

    # 数据集划分
    train_size = int(len(X) * config['train_ratio'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 创建训练和测试数据集
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # ========= 训练 SGD =========
    sgd_config = config['sgd']
    model_sgd = MLP(input_dim=X.shape[1], hidden_dims=config['hidden_dims'], dropout=sgd_config.get('dropout', 0.3)).to(device)
    optimizer_sgd = getattr(torch.optim, sgd_config['optimizer'])(
        model_sgd.parameters(),
        lr=sgd_config['lr'],
        weight_decay=sgd_config.get('weight_decay', 0.0)
    )
    if sgd_config.get('lr_scheduler') == 'ReduceLROnPlateau':
        scheduler_sgd = ReduceLROnPlateau(optimizer_sgd, mode='min',
                                          factor=sgd_config.get('factor', 0.5),
                                          patience=sgd_config.get('patience', 20))
    else:
        scheduler_sgd = StepLR(optimizer_sgd,
                               step_size=sgd_config.get('lr_step_size', 100),
                               gamma=sgd_config.get('lr_gamma', 0.95))

    print(f"\n========= Training with SGD (Loss: {sgd_config.get('loss_function', 'MSE')})=========")
    losses_sgd, maes_sgd, rmses_sgd = train(
        model_sgd,
        train_dataset,
        test_dataset,
        optimizer_sgd,
        scheduler_sgd,
        config,
        device,
        scaler_y=y_scaler,
        use_amp=use_amp
    )

    # ========= 是否继续训练 Adam？ =========
    run_adam = input("\n是否继续用Adam优化器训练？(Y/N): ").strip().lower()

    if run_adam == 'y':
        adam_config = config['adam']
        model_adam = MLP(input_dim=X.shape[1], hidden_dims=config['hidden_dims'], dropout=adam_config.get('dropout', 0.4)).to(device)
        optimizer_adam = getattr(torch.optim, adam_config['optimizer'])(
            model_adam.parameters(),
            lr=adam_config['lr'],
            weight_decay=adam_config.get('weight_decay', 0.0)
        )
        if adam_config.get('lr_scheduler') == 'ReduceLROnPlateau':
            scheduler_adam = ReduceLROnPlateau(optimizer_sgd, mode='min',
                                              factor=adam_config.get('factor', 0.5),
                                              patience=adam_config.get('patience', 10))
        else:
            scheduler_adam = StepLR(optimizer_sgd,
                                   step_size=adam_config.get('lr_step_size', 100),
                                   gamma=adam_config.get('lr_gamma', 0.95))

        print(f"\n========= Training with Adam (Loss: {adam_config.get('loss_function', 'MSE')})=========")
        losses_adam, maes_adam, rmses_adam = train(
            model_adam,
            train_dataset,
            test_dataset,
            optimizer_adam,
            scheduler_adam,
            config,
            device,
            scaler_y=y_scaler,
            use_amp=use_amp
        )

        # ========= 可视化 SGD vs Adam 比较 =========
        plot_compare_curve(losses_sgd, losses_adam)

    else:
        # 只可视化 SGD 单独曲线
        plot_single_curve(losses_sgd, sgd_config['optimizer'])

if __name__ == "__main__":
    main()
