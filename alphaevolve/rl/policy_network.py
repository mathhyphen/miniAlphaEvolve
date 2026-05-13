"""Policy network for RL-based code patch generation.

The policy network takes code state and execution history as input,
and outputs patch actions for modifying code to solve graph theory problems.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for policy network.

    Args:
        code_embed_dim: Dimension of code embedding
        history_embed_dim: Dimension of execution history embedding
        hidden_dim: Hidden layer dimension
        num_actions: Number of possible patch actions
        num_layers: Number of transformer/MLP layers
        dropout: Dropout probability
        graph_state_dim: Dimension for graph structure features
    """
    code_embed_dim: int = 256
    history_embed_dim: int = 128
    hidden_dim: int = 512
    num_actions: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    graph_state_dim: int = 64


class CodeEncoder(nn.Module):
    """Encodes code into dense embeddings using simple tokenization."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Initialize code encoder.

        Args:
            vocab_size: Size of code vocabulary
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, code_tokens: Tensor) -> Tensor:
        """Encode code tokens to embeddings.

        Args:
            code_tokens: Tokenized code of shape (batch, seq_len)

        Returns:
            Encoded embeddings of shape (batch, seq_len, embed_dim)
        """
        embedded = self.embedding(code_tokens)
        return self.projection(embedded)


class ExecutionHistoryEncoder(nn.Module):
    """Encodes execution history into embeddings."""

    def __init__(self, history_dim: int, embed_dim: int) -> None:
        """Initialize history encoder.

        Args:
            history_dim: Dimension of raw history features
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.history_fc = nn.Linear(history_dim, embed_dim)

    def forward(self, history: Tensor) -> Tensor:
        """Encode execution history.

        Args:
            history: History features of shape (batch, history_len, history_dim)

        Returns:
            Encoded history of shape (batch, history_len, embed_dim)
        """
        return torch.tanh(self.history_fc(history))


class GraphStateEncoder(nn.Module):
    """Encodes graph structure features for Steiner Tree problems."""

    def __init__(self, graph_feature_dim: int, embed_dim: int) -> None:
        """Initialize graph state encoder.

        Args:
            graph_feature_dim: Dimension of graph features
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.graph_fc = nn.Linear(graph_feature_dim, embed_dim)

    def forward(self, graph_features: Tensor) -> Tensor:
        """Encode graph state features.

        Args:
            graph_features: Graph features of shape (batch, graph_feature_dim)

        Returns:
            Encoded graph state of shape (batch, embed_dim)
        """
        return torch.tanh(self.graph_fc(graph_features))


class PolicyNetwork(nn.Module):
    """Policy network for generating code patches.

    Takes current code state, execution history, and graph features as input,
    and outputs a distribution over patch actions for policy gradient training.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        """Initialize policy network.

        Args:
            cfg: Policy configuration
        """
        super().__init__()
        self.cfg = cfg

        # Code encoder
        self.code_encoder = CodeEncoder(
            vocab_size=10000,
            embed_dim=cfg.code_embed_dim
        )

        # History encoder
        self.history_encoder = ExecutionHistoryEncoder(
            history_dim=32,
            embed_dim=cfg.history_embed_dim
        )

        # Graph state encoder
        self.graph_encoder = GraphStateEncoder(
            graph_feature_dim=64,
            embed_dim=cfg.graph_state_dim
        )

        # Combined state encoder
        total_input_dim = cfg.code_embed_dim + cfg.history_embed_dim + cfg.graph_state_dim

        encoder_layers = []
        for _ in range(cfg.num_layers):
            encoder_layers.extend([
                nn.Linear(total_input_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            total_input_dim = cfg.hidden_dim

        self.state_encoder = nn.Sequential(*encoder_layers)

        # Action head - outputs patch action probabilities
        self.action_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_actions),
        )

        # Value head - outputs state value for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(
        self,
        code_tokens: Tensor,
        code_mask: Optional[Tensor] = None,
        execution_history: Optional[Tensor] = None,
        graph_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through policy network.

        Args:
            code_tokens: Tokenized code of shape (batch, seq_len)
            code_mask: Mask for padded code tokens
            execution_history: Execution history features
            graph_features: Graph structure features

        Returns:
            Tuple of (action_logits, value_estimate)
        """
        self._validate_batch_shapes(
            code_tokens,
            code_mask,
            execution_history,
            graph_features,
        )
        batch_size = code_tokens.size(0)

        # Encode code
        code_emb = self.code_encoder(code_tokens)

        # Apply mask if provided
        if code_mask is not None:
            code_emb = code_emb * code_mask.unsqueeze(-1)

        # Pool code embeddings (mean pooling)
        if code_mask is not None:
            denom = code_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            code_pooled = (code_emb * code_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            code_pooled = code_emb.mean(dim=1)

        # Encode history
        if execution_history is not None:
            history_emb = self.history_encoder(execution_history)
            history_pooled = history_emb.mean(dim=1)
        else:
            history_pooled = torch.zeros(batch_size, self.cfg.history_embed_dim, device=code_tokens.device)

        # Encode graph features
        if graph_features is not None:
            graph_emb = self.graph_encoder(graph_features)
        else:
            graph_emb = torch.zeros(batch_size, self.cfg.graph_state_dim, device=code_tokens.device)

        # Combine all features
        state_repr = torch.cat([code_pooled, history_pooled, graph_emb], dim=-1)

        # Encode combined state
        state_encoded = self.state_encoder(state_repr)

        # Get action logits and value estimate
        action_logits = self.action_head(state_encoded)
        value_estimate = self.value_head(state_encoded).squeeze(-1)

        if not torch.isfinite(action_logits).all():
            raise ValueError("PolicyNetwork produced non-finite action logits.")
        if not torch.isfinite(value_estimate).all():
            raise ValueError("PolicyNetwork produced non-finite value estimates.")

        return action_logits, value_estimate

    def _validate_batch_shapes(
        self,
        code_tokens: Tensor,
        code_mask: Optional[Tensor],
        execution_history: Optional[Tensor],
        graph_features: Optional[Tensor],
    ) -> None:
        """Fail fast on inconsistent tensor ranks or batch sizes."""
        batch_size = code_tokens.size(0)
        if code_tokens.dim() != 2:
            raise ValueError(f"code_tokens must have shape (batch, seq_len), got {tuple(code_tokens.shape)}")

        if code_mask is not None:
            if code_mask.dim() != 2:
                raise ValueError(f"code_mask must have shape (batch, seq_len), got {tuple(code_mask.shape)}")
            if code_mask.shape != code_tokens.shape:
                raise ValueError(
                    f"code_mask shape {tuple(code_mask.shape)} must match code_tokens shape {tuple(code_tokens.shape)}"
                )

        if execution_history is not None:
            if execution_history.dim() != 3:
                raise ValueError(
                    "execution_history must have shape (batch, history_len, history_dim), "
                    f"got {tuple(execution_history.shape)}"
                )
            if execution_history.size(0) != batch_size:
                raise ValueError(
                    f"execution_history batch {execution_history.size(0)} does not match code_tokens batch {batch_size}"
                )

        if graph_features is not None:
            if graph_features.dim() != 2:
                raise ValueError(
                    f"graph_features must have shape (batch, graph_dim), got {tuple(graph_features.shape)}"
                )
            if graph_features.size(0) != batch_size:
                raise ValueError(
                    f"graph_features batch {graph_features.size(0)} does not match code_tokens batch {batch_size}"
                )

    def get_action(
        self,
        code_tokens: Tensor,
        code_mask: Optional[Tensor] = None,
        execution_history: Optional[Tensor] = None,
        graph_features: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample an action from the policy.

        Args:
            code_tokens: Tokenized code
            code_mask: Mask for padded code tokens
            execution_history: Execution history features
            graph_features: Graph structure features
            deterministic: If True, take most likely action

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value_estimate = self.forward(
            code_tokens, code_mask, execution_history, graph_features
        )

        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value_estimate

    def evaluate_actions(
        self,
        code_tokens: Tensor,
        actions: Tensor,
        code_mask: Optional[Tensor] = None,
        execution_history: Optional[Tensor] = None,
        graph_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate log prob and entropy for given actions.

        Args:
            code_tokens: Tokenized code
            actions: Actions to evaluate
            code_mask: Mask for padded code tokens
            execution_history: Execution history features
            graph_features: Graph structure features

        Returns:
            Tuple of (log_probs, entropy, value)
        """
        action_logits, value_estimate = self.forward(
            code_tokens, code_mask, execution_history, graph_features
        )

        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value_estimate


def compute_advantages(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute GAE (Generalized Advantage Estimation).

    Args:
        rewards: List of rewards
        values: List of value estimates
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    return advantages, returns


def ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_epsilon: float = 0.2,
) -> Tensor:
    """Compute PPO (Proximal Policy Optimization) clip loss.

    Args:
        log_probs: Current log probabilities
        old_log_probs: Old log probabilities from previous policy
        advantages: Advantage estimates
        clip_epsilon: PPO clipping epsilon

    Returns:
        PPO loss value
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    return -torch.min(surr1, surr2).mean()
