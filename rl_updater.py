#!/usr/bin/env python3
"""
Learnable Updater Policy

Follows the exact framework:
- Inputs: permutation-invariant summaries of current trust + simple aggregates of today's GNN scores
- Outputs: step scale [0,1] per participating robot and track
- No raw graph, only summaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from robot_track_classes import Robot, Track


@dataclass
class EgoGraphSummary:
    """Permutation-invariant summary for one ego graph"""
    # Current trust summaries (for nodes in this ego graph only)
    robot_trust_mean: float      # Mean of robot trust means in ego graph
    robot_trust_std: float       # Std of robot trust means in ego graph
    robot_trust_count: int       # Number of robots in ego graph

    track_trust_mean: float      # Mean of track trust means in ego graph
    track_trust_std: float       # Std of track trust means in ego graph
    track_trust_count: int       # Number of tracks in ego graph

    # Simple aggregates of today's GNN scores
    robot_score_mean: float      # Mean of today's robot scores
    robot_score_std: float       # Std of today's robot scores
    track_score_mean: float      # Mean of today's track scores
    track_score_std: float       # Std of today's track scores


@dataclass
class UpdateDecision:
    """Step scales for participating nodes in this ego graph"""
    robot_steps: Dict[int, float]   # robot_id -> step_scale [0,1]
    track_steps: Dict[str, float]   # track_id -> step_scale [0,1]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional bias support"""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, d_model]
            key: [B, N_k, d_model]
            value: [B, N_v, d_model] (N_v = N_k)
            mask: [B, N_k] binary mask (1=valid, 0=padding)
            attn_bias: [B, N_q, N_k] or [N_q, N_k] attention bias (e.g., confidence)
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        # Linear projections
        Q = self.w_q(query).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N_q, d_k]
        K = self.w_k(key).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)    # [B, H, N_k, d_k]
        V = self.w_v(value).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N_k, d_k]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N_q, N_k]

        # Apply attention bias (confidence weighting)
        if attn_bias is not None:
            if attn_bias.dim() == 2:  # [N_q, N_k]
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_k]
            elif attn_bias.dim() == 3:  # [B, N_q, N_k]
                attn_bias = attn_bias.unsqueeze(1)  # [B, 1, N_q, N_k]
            scores = scores + attn_bias

        # Apply padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N_k]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # [B, H, N_q, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)  # [B, N_q, d_model]

        return self.w_o(out)


class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB) - self-attention within a set"""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                conf_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
            mask: [B, N] binary mask
            conf_bias: [B, N] confidence scores for attention bias
        """
        # Create attention bias from confidence if provided
        attn_bias = None
        if conf_bias is not None:
            # Apply confidence bias to attention logits: bias_scale * (score_conf - 0.5)
            B, N = conf_bias.shape
            bias_scale = 2.0  # Scale factor for confidence bias
            confidence_bias = bias_scale * (conf_bias - 0.5)  # Center around 0.5
            attn_bias = confidence_bias.unsqueeze(1).expand(B, N, N)  # [B, N, N]

        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask, attn_bias)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PoolingByMultiHeadAttention(nn.Module):
    """PMA - Pooling by Multihead Attention with learned seed vectors"""

    def __init__(self, d_model: int, num_seeds: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(num_seeds, d_model))
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                conf_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model] input set
            mask: [B, N] binary mask
            conf_weights: [B, N] confidence weights for value weighting
        Returns:
            [B, num_seeds, d_model] pooled representation
        """
        B = x.shape[0]

        # Expand seeds for batch
        seeds = self.seeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_seeds, d_model]

        # Apply confidence weighting to values if provided
        values = x
        if conf_weights is not None:
            conf_weights = conf_weights.unsqueeze(-1)  # [B, N, 1]
            values = x * conf_weights  # Weight the values by confidence

        # Attention: seeds attend to input set
        pooled = self.attention(seeds, x, values, mask)  # [B, num_seeds, d_model]
        pooled = self.norm(pooled)

        return pooled


class CrossSetAttention(nn.Module):
    """Bidirectional cross-attention between robot and track sets"""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.robot_to_track_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.track_to_robot_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.robot_norm = nn.LayerNorm(d_model)
        self.track_norm = nn.LayerNorm(d_model)

    def forward(self, robot_features: torch.Tensor, track_features: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None, track_mask: Optional[torch.Tensor] = None,
                robot_conf: Optional[torch.Tensor] = None, track_conf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            robot_features: [B, N_R, d_model]
            track_features: [B, N_T, d_model]
            robot_mask: [B, N_R]
            track_mask: [B, N_T]
            robot_conf: [B, N_R] confidence for attention bias
            track_conf: [B, N_T] confidence for attention bias
        Returns:
            (updated_robot_features, updated_track_features)
        """
        # Create attention biases from confidence
        robot_bias = None
        track_bias = None

        if track_conf is not None:
            # For robot->track attention, bias by track confidence
            B, N_R, N_T = robot_features.shape[0], robot_features.shape[1], track_features.shape[1]
            robot_bias = track_conf.unsqueeze(1).expand(B, N_R, N_T)  # [B, N_R, N_T]

        if robot_conf is not None:
            # For track->robot attention, bias by robot confidence
            B, N_T, N_R = track_features.shape[0], track_features.shape[1], robot_features.shape[1]
            track_bias = robot_conf.unsqueeze(1).expand(B, N_T, N_R)  # [B, N_T, N_R]

        # Robots attend to tracks
        robot_cross = self.robot_to_track_attn(
            robot_features, track_features, track_features,
            track_mask, robot_bias
        )
        robot_out = self.robot_norm(robot_features + robot_cross)

        # Tracks attend to robots
        track_cross = self.track_to_robot_attn(
            track_features, robot_features, robot_features,
            robot_mask, track_bias
        )
        track_out = self.track_norm(track_features + track_cross)

        return robot_out, track_out




class SetTransformerCritic(nn.Module):
    """
    SetTransformer-based centralized critic following the recommended architecture

    Uses the same SetTransformer backbone as the actor but with a different output head
    for value estimation. Supports both Option A (fast) and Option B (richer) architectures.
    """

    def __init__(self,
                 robot_input_dim: int = 6,     # trust_mean, strength, Δtrust, agent_score, score_conf, degree
                 track_input_dim: int = 7,     # trust_mean, strength, Δtrust, maturity, track_score, score_conf, degree
                 tier0_dim: int = 34,          # Global/Tier-0 summary dimension
                 d_model: int = 128,           # Embedding dimension
                 num_heads: int = 4,           # Number of attention heads
                 num_sab_layers: int = 2,      # Number of SAB layers per set
                 pma_seeds: int = 2,           # Number of PMA seed vectors
                 dropout: float = 0.1,
                 use_rich_critic: bool = False):  # Use Option B (richer) vs Option A (fast)
        super().__init__()
        self.d_model = d_model
        self.use_rich_critic = use_rich_critic

        # Shared SetTransformer backbone (can be shared with actor if desired)
        # 1) Per-set projectors
        self.robot_projector = nn.Sequential(
            nn.Linear(robot_input_dim, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.track_projector = nn.Sequential(
            nn.Linear(track_input_dim, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 2) Within-set self-attention (SAB blocks)
        self.robot_sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout) for _ in range(num_sab_layers)
        ])
        self.track_sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout) for _ in range(num_sab_layers)
        ])

        # 3) Cross-set attention
        self.cross_attention = CrossSetAttention(d_model, num_heads, dropout)

        # 4) PMA pooling
        self.robot_pma = PoolingByMultiHeadAttention(d_model, pma_seeds, num_heads, dropout)
        self.track_pma = PoolingByMultiHeadAttention(d_model, pma_seeds, num_heads, dropout)

        # Value estimation heads
        if use_rich_critic:
            # Option B: Latent bottleneck that cross-attends to Z_R, Z_T
            self.critic_seeds = nn.Parameter(torch.randn(4, d_model))  # 4 seeds for rich representation
            self.critic_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

            critic_input_dim = 4 * d_model + tier0_dim  # 4 seeds + tier0
        else:
            # Option A: Fast MLP on pooled features
            critic_input_dim = 2 * d_model + tier0_dim  # g_R + g_T + s0

        # Final value MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # NULL tokens for empty sets
        self.robot_null_token = nn.Parameter(torch.randn(d_model))
        self.track_null_token = nn.Parameter(torch.randn(d_model))

    def forward(self, tier0_features: torch.Tensor,
                robot_features: torch.Tensor,
                track_features: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                track_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for value estimation

        Args:
            tier0_features: [B, tier0_dim] global summary
            robot_features: [B, N_R, robot_input_dim] robot features (padded)
            track_features: [B, N_T, track_input_dim] track features (padded)
            robot_mask: [B, N_R] mask for valid robots (1=valid, 0=padding)
            track_mask: [B, N_T] mask for valid tracks (1=valid, 0=padding)
        Returns:
            values: [B] state values
        """
        B = tier0_features.shape[0]
        device = tier0_features.device

        # Handle empty sets with NULL tokens
        if robot_features.shape[1] == 0:
            robot_embeddings = self.robot_null_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            if robot_mask is None:
                robot_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)  # zeros for NULL tokens
        else:
            # 1) Project robot features
            robot_embeddings = self.robot_projector(robot_features)
            if robot_mask is None:
                robot_mask = torch.ones(B, robot_features.shape[1], device=device, dtype=torch.bool)

        if track_features.shape[1] == 0:
            track_embeddings = self.track_null_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            if track_mask is None:
                track_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)  # zeros for NULL tokens
        else:
            # 1) Project track features
            track_embeddings = self.track_projector(track_features)
            if track_mask is None:
                track_mask = torch.ones(B, track_features.shape[1], device=device, dtype=torch.bool)

        # Extract confidence features for attention bias (check dimensions first)
        robot_conf = None
        track_conf = None
        if robot_features.dim() == 3 and robot_features.shape[1] > 0 and robot_features.shape[-1] >= 5:
            robot_conf = robot_features[:, :, 4]  # score_conf feature
        if track_features.dim() == 3 and track_features.shape[1] > 0 and track_features.shape[-1] >= 6:
            track_conf = track_features[:, :, 5]  # score_conf feature

        # 2) Within-set self-attention (SAB blocks)
        robot_h = robot_embeddings
        for sab_layer in self.robot_sab_layers:
            robot_h = sab_layer(robot_h, robot_mask, robot_conf)

        track_h = track_embeddings
        for sab_layer in self.track_sab_layers:
            track_h = sab_layer(track_h, track_mask, track_conf)

        # 3) Cross-set attention
        robot_z, track_z = self.cross_attention(
            robot_h, track_h, robot_mask, track_mask, robot_conf, track_conf
        )

        if self.use_rich_critic:
            # Option B: Rich critic with latent bottleneck
            # Combine robot and track representations
            combined_z = torch.cat([robot_z, track_z], dim=1)  # [B, N_R + N_T, d_model]
            combined_mask = None
            if robot_mask is not None and track_mask is not None:
                combined_mask = torch.cat([robot_mask, track_mask], dim=1)  # [B, N_R + N_T]

            # Cross-attend critic seeds to combined representations
            critic_seeds = self.critic_seeds.unsqueeze(0).expand(B, -1, -1)  # [B, 4, d_model]
            critic_features = self.critic_cross_attn(
                critic_seeds, combined_z, combined_z, combined_mask
            )  # [B, 4, d_model]

            # Flatten and combine with tier0
            critic_features_flat = critic_features.view(B, -1)  # [B, 4*d_model]
            critic_input = torch.cat([critic_features_flat, tier0_features], dim=1)

        else:
            # Option A: Fast critic using PMA pooling
            # 4) PMA pooling to global summaries
            robot_g = self.robot_pma(robot_z, robot_mask, robot_conf)  # [B, pma_seeds, d_model]
            track_g = self.track_pma(track_z, track_mask, track_conf)  # [B, pma_seeds, d_model]

            # Reduce PMA outputs to single vectors
            robot_g_mean = robot_g.mean(dim=1)  # [B, d_model]
            track_g_mean = track_g.mean(dim=1)  # [B, d_model]

            # Combine with tier0 features
            critic_input = torch.cat([robot_g_mean, track_g_mean, tier0_features], dim=1)

        # Final value estimation
        values = self.value_mlp(critic_input).squeeze(-1)  # [B]

        return values


class SetTransformerActor(nn.Module):
    """
    SetTransformer-based actor network with cross-attention and PMA pooling

    Follows the ego-centric architecture:
    - Per-set projectors to d=128
    - Within-set self-attention (SAB blocks)
    - Cross-set attention (bidirectional)
    - PMA pooling to local summaries
    - Local fusion with ego graph summary
    - Decentralized action heads per element
    """

    def __init__(self,
                 robot_input_dim: int = 6,     # trust_mean, strength, Δtrust, agent_score, score_conf, degree
                 track_input_dim: int = 7,     # trust_mean, strength, Δtrust, maturity, track_score, score_conf, degree
                 ego_summary_dim: int = 10,    # Ego graph summary dimension (local stats only)
                 d_model: int = 128,           # Embedding dimension
                 num_heads: int = 4,           # Number of attention heads
                 num_sab_layers: int = 2,      # Number of SAB layers per set
                 pma_seeds: int = 2,           # Number of PMA seed vectors
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 1) Per-set projectors
        self.robot_projector = nn.Sequential(
            nn.Linear(robot_input_dim, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.track_projector = nn.Sequential(
            nn.Linear(track_input_dim, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 2) Within-set self-attention (SAB blocks)
        self.robot_sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout) for _ in range(num_sab_layers)
        ])
        self.track_sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout) for _ in range(num_sab_layers)
        ])

        # 3) Cross-set attention
        self.cross_attention = CrossSetAttention(d_model, num_heads, dropout)

        # 4) PMA pooling
        self.robot_pma = PoolingByMultiHeadAttention(d_model, pma_seeds, num_heads, dropout)
        self.track_pma = PoolingByMultiHeadAttention(d_model, pma_seeds, num_heads, dropout)

        # 5) Local ego fusion
        fusion_input_dim = 2 * d_model + ego_summary_dim  # g_R + g_T + ego_summary
        self.ego_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 6) Decentralized action heads
        # FiLM conditioning from ego context
        self.robot_film_mlp = nn.Linear(256, 2 * d_model)  # gamma, beta
        self.track_film_mlp = nn.Linear(256, 2 * d_model)  # gamma, beta

        # Action heads (outputs alpha and beta parameters for Beta distribution)
        self.robot_action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # Alpha and Beta parameters
        )
        self.track_action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # Alpha and Beta parameters
        )

        # Beta distribution uses alpha/beta parameters from action heads directly

        # NULL tokens for empty sets
        self.robot_null_token = nn.Parameter(torch.randn(d_model))
        self.track_null_token = nn.Parameter(torch.randn(d_model))

    def forward(self, ego_summary: torch.Tensor,
                robot_features: torch.Tensor,
                track_features: torch.Tensor,
                robot_mask: Optional[torch.Tensor] = None,
                track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for action generation

        Args:
            ego_summary: [B, ego_summary_dim] ego graph summary (local stats only)
            robot_features: [B, N_R, robot_input_dim] robot features (padded)
            track_features: [B, N_T, track_input_dim] track features (padded)
            robot_mask: [B, N_R] mask for valid robots (1=valid, 0=padding)
            track_mask: [B, N_T] mask for valid tracks (1=valid, 0=padding)
        Returns:
            robot_actions: [B, N_R] robot step scales
            track_actions: [B, N_T] track step scales
        """
        B = ego_summary.shape[0]
        device = ego_summary.device

        # Handle empty sets with NULL tokens
        if robot_features.shape[1] == 0:
            robot_embeddings = self.robot_null_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            if robot_mask is None:
                robot_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)  # zeros for NULL tokens
        else:
            # 1) Project robot features
            robot_embeddings = self.robot_projector(robot_features)
            if robot_mask is None:
                robot_mask = torch.ones(B, robot_features.shape[1], device=device, dtype=torch.bool)

        if track_features.shape[1] == 0:
            track_embeddings = self.track_null_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            if track_mask is None:
                track_mask = torch.zeros(B, 1, device=device, dtype=torch.bool)  # zeros for NULL tokens
        else:
            # 1) Project track features
            track_embeddings = self.track_projector(track_features)
            if track_mask is None:
                track_mask = torch.ones(B, track_features.shape[1], device=device, dtype=torch.bool)

        # Extract confidence features for attention bias (check dimensions first)
        robot_conf = None
        track_conf = None
        if robot_features.dim() == 3 and robot_features.shape[1] > 0 and robot_features.shape[-1] >= 5:
            robot_conf = robot_features[:, :, 4]  # score_conf feature
        if track_features.dim() == 3 and track_features.shape[1] > 0 and track_features.shape[-1] >= 6:
            track_conf = track_features[:, :, 5]  # score_conf feature

        # 2) Within-set self-attention (SAB blocks)
        robot_h = robot_embeddings
        for sab_layer in self.robot_sab_layers:
            robot_h = sab_layer(robot_h, robot_mask, robot_conf)

        track_h = track_embeddings
        for sab_layer in self.track_sab_layers:
            track_h = sab_layer(track_h, track_mask, track_conf)

        # 3) Cross-set attention
        robot_z, track_z = self.cross_attention(
            robot_h, track_h, robot_mask, track_mask, robot_conf, track_conf
        )

        # 4) PMA pooling to global summaries
        robot_global = self.robot_pma(robot_z, robot_mask)  # [B, pma_seeds, d_model]
        track_global = self.track_pma(track_z, track_mask)  # [B, pma_seeds, d_model]

        # Mean pooling over seeds
        robot_summary = robot_global.mean(dim=1)  # [B, d_model]
        track_summary = track_global.mean(dim=1)  # [B, d_model]

        # 5) Local ego fusion
        ego_context = torch.cat([robot_summary, track_summary, ego_summary], dim=1)  # [B, 2*d_model + ego_summary_dim]
        fused_context = self.ego_fusion(ego_context)  # [B, 256]

        # 6) FiLM conditioning for decentralized actions
        robot_film = self.robot_film_mlp(fused_context)  # [B, 2*d_model]
        track_film = self.track_film_mlp(fused_context)  # [B, 2*d_model]

        robot_gamma_raw, robot_beta_raw = robot_film.chunk(2, dim=-1)  # [B, d_model] each
        track_gamma_raw, track_beta_raw = track_film.chunk(2, dim=-1)  # [B, d_model] each

        # Bound FiLM scales/shifts to prevent representation explosion
        robot_gamma = 0.5 + 1.5 * torch.sigmoid(robot_gamma_raw)  # Scale: (0.5, 2.0) around 1.25
        robot_beta = torch.tanh(robot_beta_raw) * 0.5             # Shift: [-0.5, 0.5]
        track_gamma = 0.5 + 1.5 * torch.sigmoid(track_gamma_raw)  # Scale: (0.5, 2.0) around 1.25
        track_beta = torch.tanh(track_beta_raw) * 0.5             # Shift: [-0.5, 0.5]

        # Apply FiLM to element representations
        robot_conditioned = robot_gamma.unsqueeze(1) * robot_z + robot_beta.unsqueeze(1)  # [B, N_R, d_model]
        track_conditioned = track_gamma.unsqueeze(1) * track_z + track_beta.unsqueeze(1)  # [B, N_T, d_model]

        # 7) Decentralized action heads - output (μ, κ) parameters for stable Beta distribution
        robot_params = self.robot_action_head(robot_conditioned)  # [B, N_R, 2]
        track_params = self.track_action_head(track_conditioned)  # [B, N_T, 2]

        # Parameterize Beta(α,β) via (μ, κ) for stability
        # μ = sigmoid(h_μ) ∈ (0,1), κ = κ_max * sigmoid(h_κ) + κ_min
        kappa_min = 1e-3  # Minimum concentration
        kappa_max = 50.0  # Maximum concentration (soft ceiling)

        # Robot Beta parameters (μ, κ)
        robot_mu = torch.sigmoid(robot_params[..., 0])  # [B, N_R] ∈ (0,1)
        robot_kappa = kappa_max * torch.sigmoid(robot_params[..., 1]) + kappa_min  # [B, N_R]

        # Track Beta parameters (μ, κ)
        track_mu = torch.sigmoid(track_params[..., 0])  # [B, N_T] ∈ (0,1)
        track_kappa = kappa_max * torch.sigmoid(track_params[..., 1]) + kappa_min  # [B, N_T]

        return (robot_mu, robot_kappa), (track_mu, track_kappa)

    def sample_actions(self, ego_summary: torch.Tensor,
                      robot_features: torch.Tensor,
                      track_features: torch.Tensor,
                      robot_mask: Optional[torch.Tensor] = None,
                      track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                         Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample actions with log probabilities and entropy for training

        Args:
            ego_summary: [B, ego_summary_dim] ego graph summary (local stats only)
            robot_features: [B, N_R, robot_input_dim] robot features (padded)
            track_features: [B, N_T, track_input_dim] track features (padded)
            robot_mask: [B, N_R] mask for valid robots (1=valid, 0=padding)
            track_mask: [B, N_T] mask for valid tracks (1=valid, 0=padding)
        Returns:
            robot_actions: [B, N_R] sampled robot actions
            track_actions: [B, N_T] sampled track actions
            robot_log_probs: [B, N_R] log probabilities of robot actions
            track_log_probs: [B, N_T] log probabilities of track actions
            entropy: [B] total entropy of the action distributions
            robot_alpha_beta: (robot_alpha, robot_beta) for diagnostics
            track_alpha_beta: (track_alpha, track_beta) for diagnostics
        """
        # Get mu and kappa parameters from forward pass
        (robot_mu, robot_kappa), (track_mu, track_kappa) = self.forward(
            ego_summary, robot_features, track_features, robot_mask, track_mask)

        # Convert (μ, κ) to (α, β) for Beta distribution with small-ε guard
        eps = 1e-4
        robot_mu = robot_mu.clamp(eps, 1.0 - eps)
        track_mu = track_mu.clamp(eps, 1.0 - eps)

        robot_alpha = robot_mu * robot_kappa      # α = μ·κ
        robot_beta = (1 - robot_mu) * robot_kappa # β = (1-μ)·κ
        track_alpha = track_mu * track_kappa      # α = μ·κ
        track_beta = (1 - track_mu) * track_kappa # β = (1-μ)·κ

        # Create Beta distributions for sampling
        robot_dist = torch.distributions.Beta(robot_alpha, robot_beta)
        track_dist = torch.distributions.Beta(track_alpha, track_beta)

        # Sample actions (automatically in [0, 1] range - no clamping needed)
        robot_actions = robot_dist.rsample()  # [B, N_R]
        track_actions = track_dist.rsample()  # [B, N_T]

        # Compute log probabilities
        robot_log_probs = robot_dist.log_prob(robot_actions)  # [B, N_R]
        track_log_probs = track_dist.log_prob(track_actions)  # [B, N_T]

        # Apply masks to log probs (set masked entries to 0)
        if robot_mask is not None:
            robot_log_probs = robot_log_probs * robot_mask.float()
        if track_mask is not None:
            track_log_probs = track_log_probs * track_mask.float()

        # Compute entropy with regularization on κ to prevent exploration collapse
        robot_entropy = robot_dist.entropy()  # [B, N_R]
        track_entropy = track_dist.entropy()  # [B, N_T]

        # Add entropy bonus on κ (concentration) to prevent exploration death
        # Bonus = -log(κ) encourages lower concentration (higher exploration)
        kappa_entropy_coef = 0.01  # Small coefficient for κ regularization
        # κ is already available directly from forward pass (no need to compute from α+β)

        robot_kappa_entropy = -torch.log(robot_kappa + 1e-8) * kappa_entropy_coef  # [B, N_R]
        track_kappa_entropy = -torch.log(track_kappa + 1e-8) * kappa_entropy_coef  # [B, N_T]

        # Combine Beta entropy with κ entropy bonus
        robot_entropy_total = robot_entropy + robot_kappa_entropy  # [B, N_R]
        track_entropy_total = track_entropy + track_kappa_entropy  # [B, N_T]

        if robot_mask is not None:
            robot_entropy_total = robot_entropy_total * robot_mask.float()
        if track_mask is not None:
            track_entropy_total = track_entropy_total * track_mask.float()

        # Sum entropy over valid elements and normalize by count of valid elements
        robot_entropy_sum = robot_entropy_total.sum(dim=1)  # [B]
        track_entropy_sum = track_entropy_total.sum(dim=1)  # [B]

        # Count valid elements for normalization
        robot_count = robot_mask.float().sum(dim=1) if robot_mask is not None else torch.full((robot_entropy_total.shape[0],), robot_entropy_total.shape[1], device=robot_entropy_total.device)
        track_count = track_mask.float().sum(dim=1) if track_mask is not None else torch.full((track_entropy_total.shape[0],), track_entropy_total.shape[1], device=track_entropy_total.device)

        # Normalize entropy by element count (avoid division by zero)
        robot_entropy_norm = robot_entropy_sum / torch.clamp(robot_count, min=1.0)
        track_entropy_norm = track_entropy_sum / torch.clamp(track_count, min=1.0)

        total_entropy = robot_entropy_norm + track_entropy_norm  # [B]

        return robot_actions, track_actions, robot_log_probs, track_log_probs, total_entropy, \
               (robot_alpha, robot_beta), (track_alpha, track_beta)

    def evaluate_actions(self, ego_summary: torch.Tensor,
                        robot_features: torch.Tensor,
                        track_features: torch.Tensor,
                        robot_actions: torch.Tensor,
                        track_actions: torch.Tensor,
                        robot_mask: Optional[torch.Tensor] = None,
                        track_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions to compute log probabilities and entropy

        Args:
            ego_summary: [B, ego_summary_dim] ego graph summary (local stats only)
            robot_features: [B, N_R, robot_input_dim] robot features (padded)
            track_features: [B, N_T, track_input_dim] track features (padded)
            robot_actions: [B, N_R] robot actions to evaluate
            track_actions: [B, N_T] track actions to evaluate
            robot_mask: [B, N_R] mask for valid robots (1=valid, 0=padding)
            track_mask: [B, N_T] mask for valid tracks (1=valid, 0=padding)
        Returns:
            robot_log_probs: [B, N_R] log probabilities of robot actions
            track_log_probs: [B, N_T] log probabilities of track actions
            entropy: [B] total entropy of the action distributions
        """
        # Get (μ, κ) parameters from forward pass
        (robot_mu, robot_kappa), (track_mu, track_kappa) = self.forward(
            ego_summary, robot_features, track_features, robot_mask, track_mask)

        # Convert (μ, κ) to (α, β) for Beta distribution with small-ε guard
        eps = 1e-4
        robot_mu = robot_mu.clamp(eps, 1.0 - eps)
        track_mu = track_mu.clamp(eps, 1.0 - eps)

        robot_alpha = robot_mu * robot_kappa
        robot_beta = (1 - robot_mu) * robot_kappa
        track_alpha = track_mu * track_kappa
        track_beta = (1 - track_mu) * track_kappa

        # Create Beta distributions
        robot_dist = torch.distributions.Beta(robot_alpha, robot_beta)
        track_dist = torch.distributions.Beta(track_alpha, track_beta)

        # Actions are already in [0, 1] range (no clamping needed for Beta distribution)
        # Compute log probabilities
        robot_log_probs = robot_dist.log_prob(robot_actions)  # [B, N_R]
        track_log_probs = track_dist.log_prob(track_actions)  # [B, N_T]

        # Apply masks to log probs (set masked entries to 0)
        if robot_mask is not None:
            robot_log_probs = robot_log_probs * robot_mask.float()
        if track_mask is not None:
            track_log_probs = track_log_probs * track_mask.float()

        # Compute entropy with regularization on κ to prevent exploration collapse
        robot_entropy = robot_dist.entropy()  # [B, N_R]
        track_entropy = track_dist.entropy()  # [B, N_T]

        # Add entropy bonus on κ (concentration) to prevent exploration death
        # Bonus = -log(κ) encourages lower concentration (higher exploration)
        kappa_entropy_coef = 0.01  # Small coefficient for κ regularization
        # κ is already available directly from forward pass (no need to compute from α+β)

        robot_kappa_entropy = -torch.log(robot_kappa + 1e-8) * kappa_entropy_coef  # [B, N_R]
        track_kappa_entropy = -torch.log(track_kappa + 1e-8) * kappa_entropy_coef  # [B, N_T]

        # Combine Beta entropy with κ entropy bonus
        robot_entropy_total = robot_entropy + robot_kappa_entropy  # [B, N_R]
        track_entropy_total = track_entropy + track_kappa_entropy  # [B, N_T]

        if robot_mask is not None:
            robot_entropy_total = robot_entropy_total * robot_mask.float()
        if track_mask is not None:
            track_entropy_total = track_entropy_total * track_mask.float()

        # Sum entropy over valid elements and normalize by count of valid elements
        robot_entropy_sum = robot_entropy_total.sum(dim=1)  # [B]
        track_entropy_sum = track_entropy_total.sum(dim=1)  # [B]

        # Count valid elements for normalization
        robot_count = robot_mask.float().sum(dim=1) if robot_mask is not None else torch.full((robot_entropy_total.shape[0],), robot_entropy_total.shape[1], device=robot_entropy_total.device)
        track_count = track_mask.float().sum(dim=1) if track_mask is not None else torch.full((track_entropy_total.shape[0],), track_entropy_total.shape[1], device=track_entropy_total.device)

        # Normalize entropy by element count (avoid division by zero)
        robot_entropy_norm = robot_entropy_sum / torch.clamp(robot_count, min=1.0)
        track_entropy_norm = track_entropy_sum / torch.clamp(track_count, min=1.0)

        total_entropy = robot_entropy_norm + track_entropy_norm  # [B]

        return robot_log_probs, track_log_probs, total_entropy

class LearnableUpdater:
    """Wrapper for the learnable updater policy with centralized critic for MAPPO"""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)

        # Initialize SetTransformer-based actor and critic
        self.policy = SetTransformerActor(
            robot_input_dim=6,  # trust_mean, strength, Δtrust, agent_score, score_conf, degree
            track_input_dim=7,  # trust_mean, strength, Δtrust, maturity, track_score, score_conf, degree
            ego_summary_dim=10,  # ego graph summary (local stats only)
            d_model=128,
            num_heads=4,
            num_sab_layers=2,
            pma_seeds=2,
            dropout=0.1
        ).to(self.device)

        self.critic = SetTransformerCritic(
            robot_input_dim=6,
            track_input_dim=7,
            tier0_dim=34,
            d_model=128,
            num_heads=4,
            num_sab_layers=2,
            pma_seeds=2,
            dropout=0.1,
            use_rich_critic=False  # Start with Option A (fast)
        ).to(self.device)

        if model_path:
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded updater policy from {model_path}")
            except Exception as e:
                print(f"Failed to load updater policy: {e}")

        # Always initialize a fresh critic for training
        print("Initialized fresh centralized critic for training")

        self.policy.eval()
        self.critic.eval()

    def create_ego_graph_summary(self,
                                ego_robots: List[Robot],      # robots in ego graph
                                ego_tracks: List[Track],      # tracks in ego graph
                                robot_scores: Dict[int, float],
                                track_scores: Dict[str, float]) -> EgoGraphSummary:
        """Create permutation-invariant summary of ego graph"""

        # Current trust summaries
        if ego_robots:
            robot_means = [robot.trust_value for robot in ego_robots]
            robot_trust_mean = np.mean(robot_means)
            robot_trust_std = np.std(robot_means) if len(robot_means) > 1 else 0.0
        else:
            robot_trust_mean = 0.5
            robot_trust_std = 0.0

        if ego_tracks:
            track_means = [track.trust_value for track in ego_tracks]
            track_trust_mean = np.mean(track_means)
            track_trust_std = np.std(track_means) if len(track_means) > 1 else 0.0
        else:
            track_trust_mean = 0.5
            track_trust_std = 0.0

        # Today's GNN score aggregates
        ego_robot_scores = [robot_scores.get(robot.id, 0.5) for robot in ego_robots]
        ego_track_scores = [track_scores.get(track.track_id, 0.5) for track in ego_tracks]

        if ego_robot_scores:
            robot_score_mean = np.mean(ego_robot_scores)
            robot_score_std = np.std(ego_robot_scores) if len(ego_robot_scores) > 1 else 0.0
        else:
            robot_score_mean = 0.5
            robot_score_std = 0.0

        if ego_track_scores:
            track_score_mean = np.mean(ego_track_scores)
            track_score_std = np.std(ego_track_scores) if len(ego_track_scores) > 1 else 0.0
        else:
            track_score_mean = 0.5
            track_score_std = 0.0

        return EgoGraphSummary(
            robot_trust_mean=robot_trust_mean,
            robot_trust_std=robot_trust_std,
            robot_trust_count=len(ego_robots),
            track_trust_mean=track_trust_mean,
            track_trust_std=track_trust_std,
            track_trust_count=len(ego_tracks),
            robot_score_mean=robot_score_mean,
            robot_score_std=robot_score_std,
            track_score_mean=track_score_mean,
            track_score_std=track_score_std
        )

    def get_step_scales(self,
                       ego_robots: List[Robot],        # robots in ego graph
                       ego_tracks: List[Track],        # tracks in ego graph
                       participating_robots: List[Robot],  # robots that detected something
                       participating_tracks: List[Track],  # tracks that were detected
                       robot_scores: Dict[int, float],
                       track_scores: Dict[str, float]) -> UpdateDecision:
        """
        Get step scales for participating robots and tracks in this ego graph
        """

        if not participating_robots and not participating_tracks:
            return UpdateDecision({}, {})

        # Create permutation-invariant summary
        summary = self.create_ego_graph_summary(ego_robots, ego_tracks, robot_scores, track_scores)

        with torch.no_grad():
            # Convert summary to tensor
            summary_tensor = torch.tensor([
                summary.robot_trust_mean, summary.robot_trust_std, summary.robot_trust_count,
                summary.track_trust_mean, summary.track_trust_std, summary.track_trust_count,
                summary.robot_score_mean, summary.robot_score_std,
                summary.track_score_mean, summary.track_score_std
            ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 10]

            # Build per-node features for participating nodes only (match actor input dimensions)
            robot_features = []
            robot_ids = []
            for robot in participating_robots:
                trust_mean = robot.trust_value
                strength = robot.trust_alpha + robot.trust_beta
                delta_trust = 0.0  # No previous step info in inference mode
                agent_score = robot_scores.get(robot.id, 0.5)
                score_conf = 2 * abs(agent_score - 0.5)  # Confidence as distance from 0.5
                degree = len(robot.get_current_timestep_tracks())  # Number of tracks this robot detected
                robot_features.append([trust_mean, strength, delta_trust, agent_score, score_conf, degree])
                robot_ids.append(robot.id)

            track_features = []
            track_ids = []
            for track in participating_tracks:
                trust_mean = track.trust_value
                strength = track.trust_alpha + track.trust_beta
                delta_trust = 0.0  # No previous step info in inference mode
                maturity = min(1.0, track.observation_count / 10.0)
                track_score = track_scores.get(track.track_id, 0.5)
                score_conf = 2 * abs(track_score - 0.5)  # Confidence as distance from 0.5
                # Degree: number of robots that detected this track
                degree = sum(1 for robot in participating_robots
                           if any(t.track_id == track.track_id for t in robot.get_current_timestep_tracks()))
                track_features.append([trust_mean, strength, delta_trust, maturity, track_score, score_conf, degree])
                track_ids.append(track.track_id)

            robot_features_tensor = torch.tensor(robot_features, dtype=torch.float32, device=self.device) if robot_features else torch.zeros(0, 6, device=self.device)
            track_features_tensor = torch.tensor(track_features, dtype=torch.float32, device=self.device) if track_features else torch.zeros(0, 7, device=self.device)

            # Get (μ, κ) parameters from policy (ego-centric)
            # Ensure proper batching for SetTransformer (needs 3D tensors)
            robot_features_batched = robot_features_tensor.unsqueeze(0) if robot_features_tensor.dim() == 2 else robot_features_tensor
            track_features_batched = track_features_tensor.unsqueeze(0) if track_features_tensor.dim() == 2 else track_features_tensor

            (robot_mu, robot_kappa), (track_mu, track_kappa) = self.policy(summary_tensor, robot_features_batched, track_features_batched)

            # Use μ directly as action (deterministic inference)
            robot_steps = {}
            if robot_mu.size(1) > 0:  # Check actual number of robots (batch=0, robots=1)
                robot_mus = robot_mu[0, :]  # [N_R] - remove batch dimension
                # μ is already the mean of the Beta distribution, use directly
                for i, robot_id in enumerate(robot_ids):
                    robot_steps[robot_id] = float(robot_mus[i])

            track_steps = {}
            if track_mu.size(1) > 0:  # Check actual number of tracks (batch=0, tracks=1)
                track_mus = track_mu[0, :]  # [N_T] - remove batch dimension
                # μ is already the mean of the Beta distribution, use directly
                for i, track_id in enumerate(track_ids):
                    track_steps[track_id] = float(track_mus[i])

            return UpdateDecision(robot_steps, track_steps)