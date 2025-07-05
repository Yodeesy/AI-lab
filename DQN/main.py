import argparse
import gymnasium as gym
from argument import dqn_arguments
import os
from agent_dir.agent_dqn import AgentDQN
from tqdm import tqdm
import torch  # 导入 torch 用于加载模型
import numpy as np
import random
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="dqn for cartpole")
parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
parser.add_argument('--test', action='store_true', help='whether test model')
parser = dqn_arguments(parser)


def do_train(args):
    env_name = args.env_name
    env = gym.make(env_name)

    # 设置随机种子
    seed = args.seed
    env.reset(seed=seed)
    env.action_space.seed(seed) # For discrete action spaces
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    agent = AgentDQN(env, args)
    loss_history = agent.train()

    # 绘制损失曲线
    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('DQN Loss Curve')
        loss_plot_path = f"./log/loss_{env_name}.png"
        os.makedirs("./log", exist_ok=True)
        plt.savefig(loss_plot_path)
        print(f"Loss curve saved to {loss_plot_path}")
        plt.show()
    else:
        print("No loss data to plot.")


def do_test(args):
    env_name = args.env_name
    # render_mode='human' 用于可视化，如果不需要可以改为 None
    env = gym.make(env_name, render_mode='human')

    # 重新初始化 AgentDQN 实例
    agent = AgentDQN(env, args)

    model_path = f"model/best_{env_name}.pt"
    # 确保模型路径存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first for the {env_name} environment.")
        env.close()
        return

    try:
        # 加载模型状态字典
        device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        agent.behavior_network.load_state_dict(torch.load(model_path, map_location=device))
        # 将模型设置为评估模式
        agent.behavior_network.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return
    num_test_episodes = 10  # 运行更多 episode 以获得更可靠的平均奖励
    all_episode_rewards = []

    print(f"\nStarting {num_test_episodes} test episodes for {env_name}...")
    for i in tqdm(range(num_test_episodes), desc="Testing Episodes"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        # 限制每个 episode 的步数
        max_steps_per_episode = env.spec.max_episode_steps if env.spec.max_episode_steps is not None else 500

        for step in range(max_steps_per_episode):
            action = agent.make_action(state, test=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            done = terminated or truncated

            if done:
                break

        all_episode_rewards.append(episode_reward)

    env.close()

    average_reward = sum(all_episode_rewards) / num_test_episodes
    print(f'\nEnv: {env_name}')
    print(f'Total test episodes: {num_test_episodes}')
    print(f'Average reward for test: {average_reward:.4f}')
    print(f'Max reward in test: {max(all_episode_rewards):.4f}')
    print(f'Min reward in test: {min(all_episode_rewards):.4f}')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.train_dqn:
        print("--- Starting DQN Training ---")
        print(args)
        do_train(args)
        print("--- DQN Training Finished ---")
    if args.test:
        print("--- Starting DQN Testing ---")
        do_test(args)
        print("--- DQN Testing Finished ---")