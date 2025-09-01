import numpy as np
import torch
import torch.nn as nn
import gym
from tqdm import tqdm
from collections import defaultdict
from torch.nn import functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from buffer import ReplayBuffer
from utils.logger import Logger
from policy.sac import SACPolicy


class DEMPolicy(SACPolicy):

    def __init__(
        self,
        dynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        env: gym.Env,
        task: "hopper-medium-replay-v2",
        real_buffer: ReplayBuffer,
        logger: Logger,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
        )
        self.env = env
        self.task = task
        self.real_buffer = real_buffer
        self.logger = logger
        self.dynamics = dynamics

    def rollout(
        self, init_obss: np.ndarray, rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(
                observations, actions
            )
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, {
            "num_transitions": num_transitions,
            "reward_mean": rewards_arr.mean(),
        }

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {
            k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()
        }
        return super().learn(mix_batch)

    def online_iteration(
        self,
        model_update_epochs,
        online_batch_size,
        short_rollout_length,
        c,
        t0,
        kappa_ori,
    ):
        self.train()
        self.env.reset()
        for epoch in range(model_update_epochs):
            with tqdm(
                total=online_batch_size,
                desc=f"Epoch #{epoch + 1}/{model_update_epochs}",
            ) as t:
                while t.n < t.total:
                    kappa = 1 / (1 + np.exp(c * (epoch * online_batch_size + t.n - t0)))
                    short_rollout = defaultdict(list)
                    if t.n % 200 == 0:  # 每 200 轮更新一次候选池
                        high_uncertainty_pool = {}
                        sampled_states, sampled_actions, _, _, _ = (
                            self.real_buffer.sample(10000).values()
                        )
                        sampled_states = sampled_states.cpu().numpy()
                        sampled_actions = sampled_actions.cpu().numpy()
                        uncertainties = self.dynamics.step(
                            sampled_states, sampled_actions, pred=False
                        ).flatten()
                        top_indices = np.argsort(uncertainties)[-500:]
                        high_uncertainty_pool = sampled_states[top_indices]

                    observation = high_uncertainty_pool[np.random.randint(500)]
                    nq = self.env.model.nq
                    qpos = np.concatenate(
                        [[self.env.init_qpos[0]], observation[: nq - 1]]
                    )
                    qvel = observation[nq - 1 :]
                    self.env.set_state(qpos, qvel)
                    self.env.sim.forward()

                    # observation = self.env.reset()
                    for _ in range(short_rollout_length):
                        action = self.select_action(
                            observation.reshape(1, -1), deterministic=True
                        )
                        next_observation, reward, terminal, _ = self.env.step(
                            action.flatten()
                        )
                        self.real_buffer.add(
                            observation,
                            next_observation,
                            action.flatten(),
                            reward,
                            terminal,
                        )
                        uncertainty = self.dynamics.step(
                            observation.reshape(1, -1), action, pred=False
                        ).flatten()
                        for key, value in zip(
                            [
                                "observations",
                                "actions",
                                "next_observations",
                                "rewards",
                                "terminals",
                                "uncertainties",
                            ],
                            [
                                observation,
                                action.flatten(),
                                next_observation,
                                [reward],
                                [terminal],
                                uncertainty,
                            ],
                        ):
                            short_rollout[key].append(value)
                        observation = (
                            next_observation if not terminal else self.env.reset()
                        )
                    loss = self.learn_online(short_rollout, kappa * kappa_ori)
                    # t.set_postfix({'loss': '{0:1.3f}'.format(loss.item())})
                    t.set_postfix(**loss)
                    t.update(1)
            # update dynamics model
            self.dynamics.train(
                self.real_buffer.sample_all(), self.logger, max_epochs_since_update=5
            )
