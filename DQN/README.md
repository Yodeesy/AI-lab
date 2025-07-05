# Deep Q-Network (DQN) for CartPole

## 项目概述

本项目实现了一个基于深度 Q 网络（DQN）的智能体，用于解决 OpenAI Gym 中的 CartPole 环境。DQN 是一种结合了深度学习和 Q 学习的算法，用于在强化学习中进行最优策略的学习。该项目支持使用 GPU 进行训练，采用了混合精度训练（AMP）以提高训练效率，同时实现了早停策略和线性 ε 衰减探索策略。

## 安装依赖

在运行项目之前，需要安装必要的 Python 库。可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 代码结构

- main.py: 主程序入口，包含训练和测试的逻辑。
- agent_dir/agent_dqn.py: 定义了 DQN 智能体的类，包括 Q 网络、经验回放缓冲区和训练逻辑。
- agent_dir/agent.py: 定义了智能体的基类。
- argument.py: 定义了 DQN 训练所需的参数。

## 使用方法

### 训练模型

要训练 DQN 模型，可以运行以下命令：

```bash
python main.py --train_dqn
```

训练过程中，模型会在达到一定的训练轮数或满足早停条件时停止。训练完成后，会在 `./model` 目录下保存最优模型，并在 `./log` 目录下保存损失曲线。

### 测试模型

要测试训练好的模型，可以运行以下命令：

```bash
python main.py --test
```

测试过程中，智能体将使用训练好的模型在 CartPole 环境中进行交互，并输出平均奖励、最大奖励和最小奖励。

## 参数配置

可以在 argument.py 文件中调整 DQN 训练的参数，以下是一些重要的参数：

- `--env_name`: Gym 环境名称，默认为 `CartPole-v0`。
- `--use_cuda`: 是否使用 CUDA 进行训练，默认为 `True`。
- `--use_amp`: 是否使用混合精度训练，默认为 `True`。
- `--n_iter`: 总训练步数，默认为 `10000`。
- `--max_episode_len`: 每个 episode 的最大步数，默认为 `2000`。
- `--update_step`: 每隔多少步更新一次 Q 网络，默认为 `8`。
- `--patience`: 早停策略的耐心值，默认为 `6000`。
- `--hidden_size`: Q 网络隐藏层维度，默认为 `256`。
- `--lr`: 学习率，默认为 `1e-3`。
- `--gamma`: 折扣因子，默认为 `0.95`。
- `--tau`: 目标网络软更新参数，默认为 `0.005`。
- `--epsilon_start`: 初始 epsilon 值，默认为 `1.0`。
- `--epsilon_end`: 最终 epsilon 值，默认为 `0.01`。
- `--epsilon_decay`: epsilon 衰减步数，默认为 `3000`。
- `--buffer_size`: 经验回放缓冲区大小，默认为 `100000`。
- `--batch_size`: 训练 batch 大小，默认为 `64`。
- `--log_path`: 日志保存路径，默认为 `./log/log.txt`。

## 注意事项

- 确保你的环境中已经安装了所需的库，可以通过 `pip install -r requirements.txt` 进行安装。
- 如果要使用 GPU 进行训练，请确保你的系统中已经安装了 CUDA 和相应的驱动。
- 在测试模型之前，请确保已经训练好了模型，否则会提示模型文件不存在的错误。

## 贡献

如果你发现任何问题或有改进建议，请随时提交 issue 或 pull request。