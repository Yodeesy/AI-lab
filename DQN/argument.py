def dqn_arguments(parser):
    """
    优化后的 DQN 参数配置，适用于中小规模环境如 CartPole。
    支持 AMP、GPU、日志与模型保存路径等自定义。
    """

    # 环境与设备
    parser.add_argument('--env_name', default="CartPole-v0", help='Gym 环境名称')
    parser.add_argument("--use_cuda", type=bool, default=True, help="是否使用 CUDA")
    parser.add_argument("--use_amp", type=bool, default=True, help="是否使用混合精度训练（建议在较大模型中开启）")

    # 训练基本参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_iter", type=int, default=10000, help="总训练步数，建议增加以充分训练")
    parser.add_argument("--max_episode_len", type=int, default=2000, help="每个 episode 的最大步数（v1 默认最大500）")
    parser.add_argument("--update_step", type=int, default=8, help="每隔多少步更新一次 Q 网络")   # v1环境建议4-6

    # 早停策略
    parser.add_argument("--patience", type=int, default=6000, help="在x轮中都未显著提升，就提前终止训练")

    # 模型结构
    parser.add_argument("--hidden_size", type=int, default=256, help="Q 网络隐藏层维度，适当增加提升模型容量")

    # 优化器与学习率
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新参数")
    parser.add_argument("--grad_clip_norm", type=float, default=10.0, help="梯度裁剪上限，适度放宽防止梯度爆炸")

    # epsilon 探索策略
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始 epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="最终 epsilon")
    parser.add_argument("--epsilon_decay", type=int, default=3000, help="epsilon 衰减步数，延长衰减周期，保证充分探索")  # v1环境建议4000-5000

    # 经验回放
    parser.add_argument("--buffer_size", type=int, default=100000, help="经验回放缓冲区大小，增大样本多样性")
    parser.add_argument("--batch_size", type=int, default=64, help="训练 batch 大小")

    # 路径
    # 模型保存路径放在agent_dqn.py中
    parser.add_argument("--log_path", type=str, default="./log/log.txt", help="日志保存路径")

    return parser