from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    embed_dim: int = 64
    hidden_dim: int = 64
    n_blocks: int = 4
    geo_dim: int = 4
    shared_stem_layers: int = 2
    gate_stem_layers: int = 2
    gate_hidden_dim: int = 64
    gate_n_blocks: int = 1
    gate_init_bias: float = 0.0
    encode_gate_thresh: float = 0.2
    assignment_gate_thresh: float = 0.1
    fixed_gate_threshold: float | None = None
    gate_pos_weight_scale: float = 1.0
    gate_use_deep: bool = False
    gate_spatial_dim: int = 5
    gate_shared_backbone: bool = False
    encoder_route: str = "peak_only"
    spatial_pool_hidden_dim: int = 64
    spatial_gamma_init: float = 0.05
    footprint_stats_dim: int = 4


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    fusion_gate: torch.Tensor
    spike_logit: torch.Tensor
    spike_prob: torch.Tensor
    encoded_mask: torch.Tensor
    fused_feature: torch.Tensor | None = None
    spatial_alpha: torch.Tensor | None = None


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


def _group_count(channels: int) -> int:
    channels = max(1, int(channels))
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class PeakWaveTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class PeakWaveEncoder(nn.Module):
    """Peak-channel waveform encoder for recording-local graph clustering."""

    def __init__(
        self,
        *,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_blocks: int = 4,
        stem_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        stem: List[nn.Module] = []
        in_ch = 1
        for layer in range(max(1, int(stem_layers))):
            kernel = 7 if layer == 0 else 3
            stem.extend(
                [
                    nn.Conv1d(in_ch, self.hidden_dim, kernel_size=kernel, padding=kernel // 2),
                    nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
                    nn.GELU(),
                ]
            )
            in_ch = self.hidden_dim
        self.shared_stem = nn.Sequential(*stem)
        self.blocks = nn.ModuleList(
            [PeakWaveTCNBlock(self.hidden_dim, dilation=2**i) for i in range(max(1, int(n_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    @staticmethod
    def _to_bct(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PeakWaveEncoder input must be 3D, got shape={tuple(x.shape)}")
        if x.shape[1] == 1:
            return x
        if x.shape[2] == 1:
            return x.permute(0, 2, 1).contiguous()
        raise ValueError(
            "PeakWaveEncoder only supports single-channel peak waveforms; "
            f"got shape={tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch = self._to_bct(x)
        h = self.shared_stem(x_ch)
        for block in self.blocks:
            h = block(h)
        pooled = torch.cat([torch.mean(h, dim=-1), torch.amax(h, dim=-1)], dim=1)
        return F.normalize(self.head(pooled), p=2, dim=1)


class MultiChannelWaveEncoder(nn.Module):
    """Multi-channel waveform encoder used by config-driven ablations."""

    def __init__(
        self,
        *,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_blocks: int = 4,
        stem_layers: int = 1,
        use_spatial: bool = False,
        footprint_stats_dim: int = 4,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_spatial = bool(use_spatial)
        self.footprint_stats_dim = max(0, int(footprint_stats_dim)) if self.use_spatial else 0
        stem: List[nn.Module] = []
        in_ch = 1
        for layer in range(max(1, int(stem_layers))):
            kernel = 7 if layer == 0 else 3
            stem.extend(
                [
                    nn.Conv1d(in_ch, self.hidden_dim, kernel_size=kernel, padding=kernel // 2),
                    nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
                    nn.GELU(),
                ]
            )
            in_ch = self.hidden_dim
        self.shared_stem = nn.Sequential(*stem)
        self.blocks = nn.ModuleList(
            [PeakWaveTCNBlock(self.hidden_dim, dilation=2**i) for i in range(max(1, int(n_blocks)))]
        )
        self.channel_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        attn_in = self.hidden_dim + (3 if self.use_spatial else 0)
        self.channel_score = nn.Sequential(
            nn.Linear(attn_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear((self.hidden_dim * 2) + self.footprint_stats_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    @staticmethod
    def _to_btk(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MultiChannelWaveEncoder input must be 3D, got shape={tuple(x.shape)}")
        if x.shape[1] == 1 and x.shape[2] > 1:
            return x.permute(0, 2, 1).contiguous()
        return x

    def _footprint_stats(self, x_btk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.footprint_stats_dim <= 0:
            return x_btk.new_zeros((x_btk.shape[0], 0))
        peak = torch.amax(torch.abs(x_btk), dim=1)
        mask_f = mask.to(dtype=x_btk.dtype)
        valid_count = torch.sum(mask_f, dim=1, keepdim=True).clamp_min(1.0)
        peak_masked = peak * mask_f
        max_amp = torch.amax(peak_masked, dim=1, keepdim=True).clamp_min(1e-6)
        mean_amp = torch.sum(peak_masked, dim=1, keepdim=True) / valid_count
        spread = mean_amp / max_amp
        active_frac = valid_count / max(1.0, float(x_btk.shape[2]))
        stats = torch.cat([max_amp / 30.0, mean_amp / 30.0, spread, active_frac], dim=1)
        if stats.shape[1] < self.footprint_stats_dim:
            pad = x_btk.new_zeros((x_btk.shape[0], self.footprint_stats_dim - int(stats.shape[1])))
            stats = torch.cat([stats, pad], dim=1)
        return stats[:, : self.footprint_stats_dim]

    def forward(self, x: torch.Tensor, meta: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_btk = self._to_btk(x)
        b, t, k = x_btk.shape
        if b == 0:
            return x_btk.new_zeros((0, self.embed_dim)), x_btk.new_zeros((0, k))
        mask = _normalize_channel_mask(meta.get("channel_mask"), batch_size=b, channels=k, x=x_btk)
        x_ch = x_btk.permute(0, 2, 1).reshape(b * k, 1, t)
        h = self.shared_stem(x_ch)
        for block in self.blocks:
            h = block(h)
        ch_pooled = torch.cat([torch.mean(h, dim=-1), torch.amax(h, dim=-1)], dim=1).reshape(
            b, k, self.hidden_dim * 2
        )
        ch_feat = self.channel_proj(ch_pooled)

        score_parts = [ch_feat]
        if self.use_spatial:
            rel_pos = _normalize_rel_pos(meta.get("rel_pos"), geo=meta.get("geo"), batch_size=b, channels=k, x=x_btk)
            peak = torch.amax(torch.abs(x_btk), dim=1, keepdim=True).permute(0, 2, 1)
            peak = peak / torch.amax(peak.masked_fill(~mask.unsqueeze(-1), 0.0), dim=1, keepdim=True).clamp_min(1e-6)
            score_parts.extend([rel_pos, peak])
        score_in = torch.cat(score_parts, dim=2)
        score = self.channel_score(score_in).squeeze(-1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        alpha = torch.softmax(score, dim=1)
        weighted = torch.sum(ch_feat * alpha.unsqueeze(-1), dim=1)
        max_feat = ch_feat.masked_fill(~mask.unsqueeze(-1), torch.finfo(ch_feat.dtype).min).max(dim=1).values
        head_in = torch.cat([weighted, max_feat, self._footprint_stats(x_btk, mask)], dim=1)
        return F.normalize(self.head(head_in), p=2, dim=1), alpha


def _build_temporal_stem(*, stem_layers: int, out_channels: int) -> nn.Sequential:
    stem: List[nn.Module] = []
    in_ch = 1
    for i in range(max(1, int(stem_layers))):
        kernel = 7 if i == 0 else 3
        pad = kernel // 2
        stem.extend([nn.Conv1d(in_ch, out_channels, kernel_size=kernel, padding=pad), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)])
        in_ch = out_channels
    return nn.Sequential(*stem)


class GateNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_dim = max(8, int(cfg.gate_hidden_dim))
        self.gate_spatial_dim = max(0, int(cfg.gate_spatial_dim))
        self.stem = _build_temporal_stem(stem_layers=max(1, int(cfg.gate_stem_layers)), out_channels=self.hidden_dim)
        self.blocks = nn.ModuleList([ResidualTCNBlock(self.hidden_dim, dilation=2**i) for i in range(max(0, int(cfg.gate_n_blocks)))])
        gate_in_dim = (self.hidden_dim * 2) + self.gate_spatial_dim
        self.spike_noise_head = nn.Sequential(nn.Linear(gate_in_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, 1))

    @property
    def spike_head(self) -> nn.Sequential:
        return self.spike_noise_head

    def _features(
        self,
        x: torch.Tensor,
        *,
        spatial_stats: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, k = x.shape
        if b == 0:
            return x.new_zeros((0, (self.hidden_dim * 2) + self.gate_spatial_dim))
        x_ch = x.permute(0, 2, 1).reshape(b * k, 1, t)
        h = self.stem(x_ch)
        for block in self.blocks:
            h = block(h)
        pooled = torch.mean(h, dim=-1).reshape(b, k, self.hidden_dim)
        if channel_mask is None:
            feats: List[torch.Tensor] = [torch.mean(pooled, dim=1), torch.max(pooled, dim=1).values]
        else:
            mask = _normalize_channel_mask(channel_mask, batch_size=b, channels=k, x=x)
            mask_f = mask.to(dtype=pooled.dtype).unsqueeze(-1)
            valid_count = torch.clamp(torch.sum(mask_f, dim=1), min=1.0)
            pooled_mean = torch.sum(pooled * mask_f, dim=1) / valid_count
            pooled_max = pooled.masked_fill(~mask.unsqueeze(-1), torch.finfo(pooled.dtype).min).max(dim=1).values
            feats = [pooled_mean, pooled_max]
        if self.gate_spatial_dim > 0:
            if spatial_stats is None:
                spatial_stats = x.new_zeros((b, self.gate_spatial_dim))
            elif spatial_stats.shape[1] < self.gate_spatial_dim:
                pad = x.new_zeros((b, self.gate_spatial_dim - int(spatial_stats.shape[1])))
                spatial_stats = torch.cat([spatial_stats, pad], dim=1)
            feats.append(spatial_stats[:, : self.gate_spatial_dim])
        return torch.cat(feats, dim=1)

    def forward_heads(
        self,
        x: torch.Tensor,
        *,
        spatial_stats: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            empty = x.new_zeros((0,))
            return empty, empty
        gate_feature = self._features(x, spatial_stats=spatial_stats, channel_mask=channel_mask)
        spike_logit = self.spike_noise_head(gate_feature).squeeze(1)
        spike_prob = torch.sigmoid(spike_logit)
        return spike_logit, spike_prob

    def forward(
        self,
        x: torch.Tensor,
        *,
        spatial_stats: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spike_logit, spike_prob = self.forward_heads(
            x,
            spatial_stats=spatial_stats,
            channel_mask=channel_mask,
        )
        return spike_logit, spike_prob


def _normalize_channel_mask(
    channel_mask: torch.Tensor | None,
    *,
    batch_size: int,
    channels: int,
    x: torch.Tensor,
) -> torch.Tensor:
    if channel_mask is None:
        if channels <= 1:
            return torch.ones((batch_size, channels), device=x.device, dtype=torch.bool)
        inferred = torch.any(torch.abs(x) > 0.0, dim=1)
        inferred = inferred.to(device=x.device, dtype=torch.bool)
        inferred[:, 0] = True
        return inferred
    mask = channel_mask.to(device=x.device)
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = mask.to(dtype=torch.bool)
    if mask.shape[0] != batch_size:
        raise ValueError(f"channel_mask batch mismatch: expected {batch_size}, got {mask.shape[0]}")
    if mask.shape[1] < channels:
        pad = torch.zeros((batch_size, channels - int(mask.shape[1])), device=x.device, dtype=torch.bool)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > channels:
        mask = mask[:, :channels]
    no_valid = ~torch.any(mask, dim=1)
    if torch.any(no_valid):
        mask = mask.clone()
        mask[no_valid, 0] = True
    return mask


def _normalize_rel_pos(
    rel_pos: torch.Tensor | None,
    *,
    geo: torch.Tensor | None,
    batch_size: int,
    channels: int,
    x: torch.Tensor,
) -> torch.Tensor:
    if rel_pos is None and geo is not None:
        rel_pos = geo[..., :2]
    if rel_pos is None:
        out = x.new_zeros((batch_size, channels, 2))
    else:
        out = rel_pos.to(device=x.device, dtype=x.dtype)
        if out.ndim != 3:
            raise ValueError(f"rel_pos must have shape [B,K,D], got {tuple(out.shape)}")
        if out.shape[0] != batch_size:
            raise ValueError(f"rel_pos batch mismatch: expected {batch_size}, got {out.shape[0]}")
        if out.shape[1] < channels:
            pad = x.new_zeros((batch_size, channels - int(out.shape[1]), out.shape[2]))
            out = torch.cat([out, pad], dim=1)
        elif out.shape[1] > channels:
            out = out[:, :channels]
        if out.shape[2] < 2:
            pad = x.new_zeros((batch_size, channels, 2 - int(out.shape[2])))
            out = torch.cat([out, pad], dim=2)
        elif out.shape[2] > 2:
            out = out[:, :, :2]
    return out


class EncoderNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_dim = int(cfg.hidden_dim)
        self.embed_dim = int(cfg.embed_dim)
        self.encoder_route = _normalize_encoder_route(str(cfg.encoder_route))
        if self.encoder_route == "peak_only":
            self.encoder = PeakWaveEncoder(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                n_blocks=int(cfg.n_blocks),
                stem_layers=max(1, int(cfg.shared_stem_layers)),
            )
        else:
            self.encoder = MultiChannelWaveEncoder(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                n_blocks=int(cfg.n_blocks),
                stem_layers=max(1, int(cfg.shared_stem_layers)),
                use_spatial=self.encoder_route == "multi_channel_spatial",
                footprint_stats_dim=int(cfg.footprint_stats_dim),
            )

    @property
    def peak_encoder(self) -> nn.Module:
        return self.encoder

    @property
    def shared_stem(self) -> nn.Sequential:
        return self.encoder.shared_stem

    @property
    def blocks(self) -> nn.ModuleList:
        return self.encoder.blocks

    @property
    def embed_head(self) -> nn.Sequential:
        return self.encoder.head

    def forward(
        self,
        x: torch.Tensor,
        meta: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = meta
        b = int(x.shape[0])
        k = int(x.shape[2]) if x.ndim == 3 and x.shape[1] != 1 else 1
        if b == 0:
            return (
                x.new_zeros((0, self.embed_dim)),
                x.new_zeros((0,)),
                x.new_zeros((0, self.hidden_dim)),
                x.new_zeros((0, k)),
            )
        if self.encoder_route == "peak_only" and x.ndim == 3 and x.shape[2] > 1:
            x = x[:, :, :1]
            k = 1
        if self.encoder_route == "peak_only":
            embedding = self.encoder(x)
            spatial_alpha = x.new_ones((b, k))
        else:
            embedding, spatial_alpha = self.encoder(x, meta)
        fusion_gate = x.new_zeros((b,))
        fused_feature = x.new_zeros((b, self.hidden_dim))
        return embedding, fusion_gate, fused_feature, spatial_alpha


def _normalize_encoder_route(route: str) -> str:
    normalized = str(route or "peak_only").strip().lower()
    if normalized in ("peak", "main", "main_channel", "single_channel", "single", "peak_only"):
        return "peak_only"
    if normalized in ("multi", "multichannel", "multi_channel", "multi-channel", "multi_channel_waveform"):
        return "multi_channel"
    if normalized in (
        "spatial",
        "spatial_pool",
        "spatial-pool",
        "dense",
        "high_density",
        "footprint",
        "multi_channel_spatial",
        "multi-channel-spatial",
    ):
        return "multi_channel_spatial"
    raise ValueError(
        f"unsupported encoder_route={route!r}; use peak_only | multi_channel | multi_channel_spatial"
    )


class SpikeEmbeddingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate_net = GateNet(cfg)
        self.encoder_net = EncoderNet(cfg)
        self._init_gate_head()

    def _init_gate_head(self) -> None:
        head = self.gate_net.spike_noise_head
        first = head[0]
        last = head[-1]
        if isinstance(first, nn.Linear):
            nn.init.kaiming_uniform_(first.weight, a=math.sqrt(5.0))
            nn.init.zeros_(first.bias)
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight)
            nn.init.constant_(last.bias, float(self.cfg.gate_init_bias))

    @property
    def shared_stem(self) -> nn.Sequential:
        return self.encoder_net.shared_stem

    @property
    def blocks(self) -> nn.ModuleList:
        return self.encoder_net.blocks

    @property
    def temp_proj(self) -> nn.Linear:
        return self.encoder_net.temp_proj

    @property
    def spatial_score(self) -> nn.Sequential:
        return self.encoder_net.spatial_score

    @property
    def spatial_proj(self) -> nn.Sequential:
        return self.encoder_net.spatial_proj

    @property
    def embed_head(self) -> nn.Linear:
        return self.encoder_net.embed_head

    @property
    def spike_head(self) -> nn.Sequential:
        return self.gate_net.spike_head

    @property
    def spike_noise_head(self) -> nn.Sequential:
        return self.gate_net.spike_noise_head

    def freeze_theta(self) -> None:
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("encoder_net.")

    @staticmethod
    def _derive_spatial_stats(x: torch.Tensor) -> torch.Tensor:
        peak_per_ch = torch.amax(torch.abs(x), dim=1)
        amp_proxy = torch.amax(peak_per_ch, dim=1, keepdim=True) / 30.0
        mean_amp = torch.mean(peak_per_ch, dim=1, keepdim=True)
        max_amp = torch.amax(peak_per_ch, dim=1, keepdim=True)
        spatial_proxy = mean_amp / (max_amp + 1e-6)
        return torch.cat([amp_proxy, spatial_proxy], dim=1)

    def _gate_spatial_stats(self, x: torch.Tensor, meta: dict) -> torch.Tensor | None:
        spatial_stats = meta.get("spatial_stats")
        if spatial_stats is None and self.cfg.gate_spatial_dim > 0:
            spatial_stats = self._derive_spatial_stats(x)
        return spatial_stats

    def _select_encoder_meta(self, meta: dict, keep_idx: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        keys = ("geo", "rel_pos", "channel_mask")
        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            value = meta.get(key)
            if value is None:
                continue
            if keep_idx.numel() == batch_size:
                out[key] = value
            else:
                out[key] = value.index_select(0, keep_idx)
        return out

    def _encode_selected(
        self,
        x: torch.Tensor,
        meta: dict,
        keep_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_meta = self._select_encoder_meta(meta, keep_idx, x.shape[0])
        if keep_idx.numel() == x.shape[0]:
            return self.encoder_net(x, encoder_meta)
        return self.encoder_net(x.index_select(0, keep_idx), encoder_meta)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict,
        apply_encode_gate: bool = True,
    ) -> ModelOutput:
        b = x.shape[0]
        spike_logit, spike_prob = self.forward_gate_heads(x, meta)
        encoded_mask = spike_prob >= float(self.cfg.encode_gate_thresh) if apply_encode_gate else torch.ones_like(spike_prob, dtype=torch.bool)
        embedding = x.new_zeros((b, self.cfg.embed_dim))
        fusion_gate = x.new_zeros((b,))
        fused_feature = x.new_zeros((b, self.cfg.hidden_dim))
        spatial_alpha = x.new_zeros((b, x.shape[2]))
        if torch.any(encoded_mask):
            keep_idx = torch.nonzero(encoded_mask, as_tuple=False).flatten()
            embed_keep, fusion_keep, fused_keep, alpha_keep = self._encode_selected(x, meta, keep_idx)
            embedding.index_copy_(0, keep_idx, embed_keep)
            fusion_gate.index_copy_(0, keep_idx, fusion_keep)
            fused_feature.index_copy_(0, keep_idx, fused_keep)
            if alpha_keep.shape[1] < spatial_alpha.shape[1]:
                alpha_pad = x.new_zeros((alpha_keep.shape[0], spatial_alpha.shape[1] - int(alpha_keep.shape[1])))
                alpha_keep = torch.cat([alpha_keep, alpha_pad], dim=1)
            spatial_alpha.index_copy_(0, keep_idx, alpha_keep[:, : spatial_alpha.shape[1]])
        return ModelOutput(
            embedding=embedding,
            fusion_gate=fusion_gate,
            spike_logit=spike_logit,
            spike_prob=spike_prob,
            encoded_mask=encoded_mask,
            fused_feature=fused_feature,
            spatial_alpha=spatial_alpha,
        )

    def forward_gate_heads(self, x: torch.Tensor, meta: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gate_net.forward_heads(
            x,
            spatial_stats=self._gate_spatial_stats(x, meta),
            channel_mask=meta.get("channel_mask"),
        )

    def forward_gate_only(self, x: torch.Tensor, meta: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gate_net(
            x,
            spatial_stats=self._gate_spatial_stats(x, meta),
            channel_mask=meta.get("channel_mask"),
        )

    def forward_encoder_only(
        self,
        x: torch.Tensor,
        meta: dict,
    ) -> ModelOutput:
        b = x.shape[0]
        embedding, fusion_gate, fused_feature, spatial_alpha = self.encoder_net(
            x,
            self._select_encoder_meta(meta, torch.arange(b, device=x.device), b),
        )
        return ModelOutput(
            embedding=embedding,
            fusion_gate=fusion_gate,
            spike_logit=x.new_zeros((b,)),
            spike_prob=x.new_zeros((b,)),
            encoded_mask=torch.ones((b,), device=x.device, dtype=torch.bool),
            fused_feature=fused_feature,
            spatial_alpha=spatial_alpha,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)
        current_state = super().state_dict()
        mapped: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        legacy_encoder_prefixes = (
            "shared_stem.",
            "blocks.",
            "q_proj.",
            "k_proj.",
            "v_proj.",
            "temp_proj.",
            "fusion_gate.",
            "embed_head.",
        )
        legacy_removed_prefix = "".join(("ad", "aptor."))
        for key, value in state_dict.items():
            if key.startswith(legacy_removed_prefix):
                continue
            if key.startswith("gate_net.overlap_head.") or key.startswith("overlap_head."):
                continue
            new_key = key
            if key.startswith("spike_head."):
                new_key = f"gate_net.spike_noise_head.{key[len('spike_head.'):]}"
            elif key.startswith("gate_net.spike_head."):
                new_key = f"gate_net.spike_noise_head.{key[len('gate_net.spike_head.'):]}"
            elif key.startswith("encoder_net.shared_stem."):
                new_key = f"encoder_net.encoder.shared_stem.{key[len('encoder_net.shared_stem.'):]}"
            elif key.startswith("encoder_net.blocks."):
                new_key = f"encoder_net.encoder.blocks.{key[len('encoder_net.blocks.'):]}"
            elif key.startswith("encoder_net.embed_head."):
                new_key = f"encoder_net.encoder.head.{key[len('encoder_net.embed_head.'):]}"
            elif key.startswith("encoder_net.peak_encoder."):
                new_key = f"encoder_net.encoder.{key[len('encoder_net.peak_encoder.'):]}"
            elif key.startswith(legacy_encoder_prefixes):
                if key.startswith("embed_head."):
                    new_key = f"encoder_net.encoder.head.{key[len('embed_head.'):]}"
                else:
                    new_key = f"encoder_net.encoder.{key}"
            if not strict:
                if new_key not in current_state or current_state[new_key].shape != value.shape:
                    continue
            mapped[new_key] = value
        return super().load_state_dict(mapped, strict=strict)
