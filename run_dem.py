import argparse
import random
import gym
import d4rl.gym_mujoco
import numpy as np
import torch
from nets.mlp import MLP
from modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from dynamics import EnsembleDynamics
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.load_dataset import qlearning_dataset
from buffer import ReplayBuffer
from utils.logger import Logger, make_log_dirs
from policy_trainer import PolicyTrainer
from policy.dem import DEMPolicy
from copy import deepcopy

"""
suggested hypers

python run_dem.py --task halfcheetah-medium-v2 --rollout-length 5  --penalty-coef 0.5 --kappa-ori 300
python run_dem.py --task hopper-medium-v2 --rollout-length 5  --penalty-coef 5.0 --kappa-ori 300
python run_dem.py --task walker2d-medium-v2 --rollout-length 5  --penalty-coef 0.5 --kappa-ori 100
python run_dem.py --task halfcheetah-medium-replay-v2 --rollout-length 5  --penalty-coef 0.5 --kappa-ori 200
python run_dem.py --task hopper-medium-replay-v2 --rollout-length 5  --penalty-coef 2.5 --kappa-ori 300
python run_dem.py --task walker2d-medium-replay-v2 --rollout-length 1  --penalty-coef 2.5 --kappa-ori 20
python run_dem.py --task halfcheetah-medium-expert-v2 --rollout-length 5  --penalty-coef 2.5 --kappa-ori 300
python run_dem.py --task hopper-medium-expert-v2 --rollout-length 5  --penalty-coef 5.0 --kappa-ori 300
python run_dem.py --task walker2d-medium-expert-v2 --rollout-length 1  --penalty-coef 2.5 --kappa-ori 200
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="dem")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument(
        "--dynamics-hidden-dims", type=int, nargs="*", default=[200, 200, 200, 200]
    )
    parser.add_argument(
        "--dynamics-weight-decay",
        type=float,
        nargs="*",
        default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
    )
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--kappa-ori", type=int, default=200)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default="")

    parser.add_argument("--online-batch-size", type=int, default=2000)
    parser.add_argument("--model-update-epochs", type=int, default=5)
    parser.add_argument("--short-rollout-length", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.001)
    parser.add_argument("--t0", type=int, default=5000)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(
        input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims
    )
    critic1_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims,
    )
    critic2_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims,
    )
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action,
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = (
            args.target_entropy
            if args.target_entropy
            else -np.prod(env.action_space.shape)
        )

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device,
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args.dynamics_lr)
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=args.penalty_coef,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"])
        + args.online_batch_size * args.model_update_epochs * args.short_rollout_length,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device,
    )
    real_buffer.add_batch(
        dataset["observations"],
        dataset["next_observations"],
        dataset["actions"],
        dataset["rewards"].reshape(-1, 1),
        dataset["terminals"].reshape(-1, 1),
    )
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size
        * args.rollout_length
        * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device,
    )

    # log
    log_dirs = make_log_dirs(
        args.task,
        args.algo_name,
        args.seed,
        vars(args),
        record_params=["rollout_length", "penalty_coef", "kappa_ori"],
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    policy_dem = DEMPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        env=env,
        task=args.task,
        real_buffer=real_buffer,
        logger=logger,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
    )

    # create policy trainer
    policy_trainer = PolicyTrainer(
        policy=policy_dem,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(
            args.rollout_freq,
            args.rollout_batch_size,
            args.rollout_length,
        ),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
    )

    """
    FIXME: 总的来说这个实验的流程大概是
    offline-rl: train dynamic1, train policy1 (phase 1) -> online-rl: train dynamic2, train policy2 (phase 2)
    其实这两个 dynamic 和 policy 可以没有关联（代码里 policy2 是继承 policy1 的权重，但也可以重新训练），
    只是 online-rl 的部分需要 offline-rl 的 policy1 进行在线训练，而 policy 的训练又是依靠 dynamic model 的。

    一开始实验的时候 phase 1 和 phase 2 是分开执行的（为了节省时间以及方便调试），
    在 phase 2 阶段导入了需要的 dynamic1 和 policy1 。

    现在的问题的是：整合代码的表现与分开实验不一致。

    思考方向：
    phase 1 应该是没有问题的, 因为基本没有做出改动，并且是 phase 1 基本是 MOPO 算法的流程。
    我的工作基本是在 phase 2 的 online interaction (train dynamic2) 部分，其实提升不大。
    所以主要还是看 phase 1, 我有查看实验的日志，有问题的实验在 phase 1 阶段就不太好，但日志方面的处理不知道是否正确。

    描述的不是很好，有些类型没有补上，抱歉。
    """

    """ phase 1 """
    # train dynamics
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)

    # train policy offline
    policy_trainer.train()

    """ phase 2 """
    # save offline policy
    saved_actor = deepcopy(actor)
    saved_critic1 = deepcopy(critic1)
    saved_critic2 = deepcopy(critic2)
    saved_actor_optim = deepcopy(actor_optim)
    saved_critic1_optim = deepcopy(critic1_optim)
    saved_critic2_optim = deepcopy(critic2_optim)

    if args.auto_alpha:
        saved_log_alpha = deepcopy(log_alpha)
        saved_alpha_optim = deepcopy(alpha_optim)
    else:
        saved_alpha = alpha

    saved_lr_scheduler = deepcopy(lr_scheduler)

    # online interaction
    policy_trainer.policy.online_iteration(
        args.model_update_epochs,
        args.online_batch_size,
        args.short_rollout_length,
        args.c,
        args.t0,
        args.kappa_ori,
    )

    # restore policy
    policy_trainer.policy.actor = saved_actor
    policy_trainer.policy.critic1 = saved_critic1
    policy_trainer.policy.critic2 = saved_critic2
    policy_trainer.policy.actor_optim = saved_actor_optim
    policy_trainer.policy.critic1_optim = saved_critic1_optim
    policy_trainer.policy.critic2_optim = saved_critic2_optim

    if args.auto_alpha:
        policy_trainer.policy.alpha = (
            args.target_entropy,
            saved_log_alpha,
            saved_alpha_optim,
        )
    else:
        policy_trainer.policy.alpha = saved_alpha

    policy_trainer.lr_scheduler = saved_lr_scheduler

    # change dynamics uncertainty mode
    fake_buffer.clear()
    dynamics._uncertainty_mode = "epistemic"

    # update policy
    policy_trainer.train()

    logger.close()


if __name__ == "__main__":
    train()
