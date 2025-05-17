from typing import Any, Callable, cast, override

import torch
import torch.nn as nn

from agent.interfaces import Policy
from agent._network import CNNActorNetwork, CNNCriticNetwork, NewCNNEncoder
from tracking.stats import MaskedCategorical, TensorStats
from agent.tensor_util import convert_tensors
from training.replay import make_batches_from_data
from tracking.stats import maskd_entropy

TRAIN_SPEC = {
    "state": ((25,), torch.long),
    "valid_actions": ((4,), torch.bool),
    "action": ((), torch.int8),
    "action_log_prob": ((), torch.float32),
    "reward": ((), torch.float32),
    "adv": ((), torch.float32),
    "next_state": ((25,), torch.long),
    "next_valid_actions": ((4,), torch.bool),
    "terminated": ((), torch.bool),
    "step": ((), torch.float32),
}


class NewBaseActorCriticPolicy(Policy):
    """
    Mask-supporting version of BaseActorCriticPolicy.
    """

    _critic: nn.Module
    _actor: nn.Module

    def _actor_logits(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def action_logits(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        return self._actor_logits(state, valid_actions)

    def _critic_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def _compute_actor_ppo_adv_loss(
        self,
        states: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        actions: torch.LongTensor,
        action_log_probs: torch.FloatTensor,
        adv: torch.FloatTensor,
        *,
        step: torch.FloatTensor,
        epsilon: float,
        entropy_coef: float,
        tensor_stats: TensorStats,
    ): 
        """
        Given advantage values, compute the actor loss with PPO, and the entropy loss
        """
        logits = self._actor_logits(states, valid_actions)
        min_real = torch.finfo(logits.dtype).min
        masked_logits = torch.where(valid_actions, logits, min_real)
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - action_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        entropy = dist.entropy()
        entropy_loss = -torch.mean(entropy) * entropy_coef

        tensor_stats.update("adv", adv)
        tensor_stats.update("policy_loss", policy_loss)
        tensor_stats.update("entropy_loss", entropy_loss)
        
        return policy_loss, entropy_loss

    def _compute_critic_loss(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        reward: torch.FloatTensor,
        next_state: torch.LongTensor,
        next_valid_actions: torch.BoolTensor,
        terminated: torch.BoolTensor | None,
        *,
        gamma: float,
        critic_coef: float,
        tensor_stats: TensorStats,
    ) -> tuple[torch.Tensor]:
        """
        Compute the critic loss
        """
        v0 = self._critic_value(state, valid_actions)
        with torch.no_grad():
            v1 = self._critic_value(next_state, next_valid_actions)
        if terminated is None:
            v1 = torch.where(torch.any(next_valid_actions, dim=-1), v1, 0)
        else:
            v1 = torch.where(terminated, 0, v1)
        q0 = gamma * v1 + reward
        loss = critic_coef * loss_fn(q0, v0)
        tensor_stats.update("critic_loss", loss)
        return (loss,)


class NewCNNActorCriticPolicy(NewBaseActorCriticPolicy):
    """
    Actor-Critic policy for static 5x5 board with masking support. Only shared encoder mode is supported.
    """
    def __init__(
        self,
        encoder_features: int = 1250,
        share_encoder: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.encoder_shared = share_encoder
        if not share_encoder:
            raise ValueError("Only share encoder is tested")
        self._encoder = NewCNNEncoder(encoder_features)
        self._actor = CNNActorNetwork(encoder_features, 256, 64)
        self._critic = CNNCriticNetwork(encoder_features, 256, 64)
        self._critic_loss_fn = nn.MSELoss(reduction="mean")
        self.seed = seed

    def _actor_logits(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        batch_shape = state.shape[:-1]
        state = torch.reshape(state, (-1, 25))
        x = self._encoder(state)
        logits = self._actor(x, valid_actions)
        logits = torch.reshape(logits, batch_shape + (4,))
        return logits

    def _critic_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        batch_shape = state.shape[:-1]
        state = torch.reshape(state, (-1, 25))
        x = self._encoder(state)
        value = self._critic(x, valid_actions)
        value = torch.reshape(value, batch_shape)
        return value


    def eval_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        return self._critic_value(state, valid_actions)

    def sample_actions(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        """
        Efficiently sample actions and log_probs for a batch, avoiding unnecessary computation.
        - Uses in-place ops where possible.
        - Returns actions as torch.int8 for memory savings.
        """
        state = state.reshape(-1, 25)
        x = self._encoder(state)
        logits = self._actor(x, valid_actions)
        # Robust masking: set invalid actions to min_real
        min_real = torch.finfo(logits.dtype).min
        masked_logits = torch.where(valid_actions, logits, min_real)
        dist = torch.distributions.Categorical(logits=masked_logits)
        with torch.no_grad():
            actions = dist.sample().to(torch.int8)
            log_probs = dist.log_prob(actions)
        return actions, log_probs

    def _learn_shared(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
    ) -> dict[str, Any]:
        assert self.encoder_shared
        gamma = params["gamma"]
        epsilon = params["ppo_epsilon"]
        actor_lr = params["actor_lr"]
        critic_lr = params["actor_lr"]
        batch_size = params["actor_batch_size"]
        entropy_coef = params["entropy_coef"]
        critic_coef = params.get("critic_coef", 1.0)

        params_list = [
            {"params": self._encoder.parameters(), "lr": min(actor_lr, critic_lr)},
            {"params": self._actor.parameters(), "lr": actor_lr},
            {"params": self._critic.parameters(), "lr": critic_lr},
        ]
        optimizer = torch.optim.Adam(params_list, lr=min(actor_lr, critic_lr))

        keys = {
            "state",
            "valid_actions",
            "action",
            "action_log_prob",
            "reward",
            "next_state",
            "next_valid_actions",
            "adv",
            "terminated",
            "step",
        }
        spec = {k: TRAIN_SPEC[k] for k in keys}

        losses = []
        for idx, batch in enumerate(make_batches_from_data(data, batch_size, seed=self.seed)):
            batch = convert_tensors(spec, batch, device=device)
            state = cast(torch.LongTensor, batch["state"])
            valid_actions = cast(torch.BoolTensor, batch["valid_actions"])
            action = cast(torch.LongTensor, batch["action"])
            action_log_prob = cast(torch.FloatTensor, batch["action_log_prob"])
            reward = cast(torch.FloatTensor, batch["reward"])
            next_state = cast(torch.LongTensor, batch["next_state"])
            next_valid_actions = cast(torch.BoolTensor, batch["next_valid_actions"])
            adv = cast(torch.FloatTensor, batch["adv"])
            terminated = cast(torch.BoolTensor, batch["terminated"])
            step = cast(torch.FloatTensor, batch["step"])

            # Policy and entropy loss
            policy_loss, entropy_loss = self._compute_actor_ppo_adv_loss(
                state,
                valid_actions,
                action,
                action_log_prob,
                adv,
                step=step,
                tensor_stats=tensor_stats,
                epsilon=epsilon,
                entropy_coef=entropy_coef,
            )
            # Entropy and entropy2 update (see _compute_actor_ppo_adv_loss)
            # Recompute dist to get entropy for logging
            logits = self._actor_logits(state, valid_actions)
            dist = MaskedCategorical(
                logits=logits,
                valid_actions=valid_actions,
            )
            entropy = maskd_entropy(dist)

            # entropy2: match base code (could be a placeholder if not used)
            entropy2 = entropy * 0 + entropy.mean() * 0.000121
            (critic_loss,) = self._compute_critic_loss(
                self._critic_loss_fn,
                state,
                valid_actions,
                reward,
                next_state,
                next_valid_actions,
                terminated,
                tensor_stats=tensor_stats,
                gamma=gamma,
                critic_coef=critic_coef,
            )
            optimizer.zero_grad()
            loss = policy_loss + entropy_loss + critic_loss
            loss.backward()
            optimizer.step()

            # update all stats for printout
            tensor_stats.update("batch_combine_loss", loss)


            losses.append((policy_loss.detach(), entropy_loss.detach(), critic_loss.detach()))
        losses = torch.tensor(losses, device="cpu")
        losses = torch.mean(losses, dim=0)
        
        return {
            "policy_loss": losses[0],
            "entropy_loss": losses[1],
            "critic_losses": losses[2:3],
        }

    @override

    def learn(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
    ) -> dict[str, Any]:
        fn = self._learn_shared

        return fn(
            params,
            data,
            tensor_stats=tensor_stats,
            device=device,
        )
        
