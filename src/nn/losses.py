from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _as_cfg_dict(cfg: Any) -> Dict[str, Any]:
    return cfg if isinstance(cfg, dict) else {}


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    cur: Any = _as_cfg_dict(cfg)
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _cfg_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _zero_with_stats(ref: torch.Tensor, stats: Dict[str, Any] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    return ref.new_zeros(()), dict(stats or {})


def _mean_by_group(loss_vec: torch.Tensor, group_labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if loss_vec.numel() == 0:
        return loss_vec.new_zeros(()), 0
    labels = group_labels.to(device=loss_vec.device, dtype=torch.long)
    if labels.numel() != loss_vec.numel():
        raise ValueError("group_labels must match loss_vec length")
    parts = []
    for label in torch.unique(labels).tolist():
        mask = labels == int(label)
        if torch.any(mask):
            parts.append(torch.mean(loss_vec[mask]))
    if not parts:
        return loss_vec.new_zeros(()), 0
    return torch.mean(torch.stack(parts)), len(parts)


def build_stage2_valid_mask(
    batch: Dict[str, torch.Tensor],
    cfg: Any,
    gate_prob: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    unit_id = batch["unit_id"].to(dtype=torch.long)
    spike_label = batch.get("spike_label", (unit_id >= 0).to(dtype=torch.float32)).to(
        device=unit_id.device,
        dtype=torch.float32,
    )
    valid = (unit_id >= 0) & (spike_label > 0.5)

    source = batch.get("sample_source")
    if source is not None:
        source = source.to(device=unit_id.device)
        allowed_raw = _cfg_get(cfg, "stage2.allowed_sources", None)
        if allowed_raw is None:
            source_mode = str(_cfg_get(cfg, "train.stage2_encoder.source_mode", "matched_positives")).lower()
            allowed_raw = (
                ["primary_matched_positive"]
                if source_mode == "primary_only"
                else ["primary_matched_positive", "secondary_matched_positive"]
            )
        source_map = {"primary_matched_positive": 0, "secondary_matched_positive": 1, "detector_negative": 2}
        allowed_codes = []
        for value in allowed_raw:
            allowed_codes.append(int(source_map.get(value, -999)) if isinstance(value, str) else int(value))
        allowed = torch.zeros_like(valid)
        for code in allowed_codes:
            allowed = allowed | (source.to(dtype=torch.long) == int(code))
        valid = valid & allowed
    else:
        allowed = torch.ones_like(valid)

    if "gate_pass" in batch:
        gate_pass = batch["gate_pass"].to(device=unit_id.device, dtype=torch.bool)
    else:
        if gate_prob is None:
            gate_prob = batch.get("gate_confidence")
        if gate_prob is None:
            gate_pass = torch.ones_like(valid)
        else:
            threshold = float(_cfg_get(cfg, "stage2.gate_threshold", 0.5))
            gate_pass = gate_prob.to(device=unit_id.device, dtype=torch.float32) >= threshold
    valid = valid & gate_pass

    strict_overlap = batch.get("strict_overlap_label")
    if strict_overlap is not None:
        valid = valid & (~strict_overlap.to(device=unit_id.device, dtype=torch.bool))
    overlap_label = batch.get("overlap_label")
    if overlap_label is not None:
        valid = valid & (overlap_label.to(device=unit_id.device, dtype=torch.float32) <= 0.5)

    qc_allowed = batch.get("qc_allowed")
    if qc_allowed is not None:
        qc = qc_allowed.to(device=unit_id.device, dtype=torch.bool)
    else:
        qc = torch.ones_like(valid)
        spatial = batch.get("event_spatial_score")
        if spatial is not None:
            qc = qc & (
                spatial.to(device=unit_id.device, dtype=torch.float32)
                >= float(_cfg_get(cfg, "stage2.qc.min_spatial_score", 0.0))
            )
        amp = batch.get("event_amp_norm")
        if amp is not None:
            qc = qc & (
                amp.to(device=unit_id.device, dtype=torch.float32)
                >= float(_cfg_get(cfg, "stage2.qc.min_amp_norm", 0.0))
            )
    strict_amb = batch.get("strict_matching_ambiguous")
    relaxed_amb = batch.get("relaxed_matching_ambiguous")
    if strict_amb is not None:
        qc = qc & (~strict_amb.to(device=unit_id.device, dtype=torch.bool))
    if relaxed_amb is not None:
        qc = qc & (~relaxed_amb.to(device=unit_id.device, dtype=torch.bool))
    valid = valid & qc

    stats = {
        "stage2_valid_count": int(torch.sum(valid).detach().cpu().item()),
        "stage2_valid_ratio": float(torch.mean(valid.to(dtype=torch.float32)).detach().cpu().item())
        if valid.numel()
        else 0.0,
        "gate_pass_count": int(torch.sum(gate_pass).detach().cpu().item()),
        "source_allowed_count": int(torch.sum(allowed).detach().cpu().item()) if source is not None else int(valid.numel()),
        "qc_pass_count": int(torch.sum(qc).detach().cpu().item()),
    }
    if strict_overlap is not None:
        strict_bool = strict_overlap.to(device=unit_id.device, dtype=torch.bool)
        stats["strict_overlap_count"] = int(torch.sum(strict_bool).detach().cpu().item())
        stats["strict_overlap_ratio"] = (
            float(torch.mean(strict_bool.to(dtype=torch.float32)).detach().cpu().item()) if strict_bool.numel() else 0.0
        )
    else:
        stats["strict_overlap_count"] = 0
        stats["strict_overlap_ratio"] = 0.0
    return valid, stats


def _get_embedding(student_out: Any) -> torch.Tensor:
    if isinstance(student_out, dict):
        return student_out["embedding"]
    return student_out.embedding


def local_proto_weight_for_epoch(
    stage2_epoch: int,
    *,
    graph_warmup_epochs: int = 0,
    hold_epochs: int = 0,
    ramp_epochs: int = 0,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
) -> float:
    epoch0 = int(max(0, stage2_epoch))
    if epoch0 < int(max(0, graph_warmup_epochs)):
        return 0.0
    pos = epoch0 - int(max(0, graph_warmup_epochs))
    if pos < int(max(0, hold_epochs)):
        return float(start_weight)
    ramp = int(max(0, ramp_epochs))
    if ramp <= 0:
        return float(end_weight)
    frac = float(min(1.0, max(0.0, float(pos - int(max(0, hold_epochs)) + 1) / float(ramp))))
    return float(start_weight + frac * (float(end_weight) - float(start_weight)))


LOCAL_PROTO_STAT_KEYS = (
    "local_proto_count",
    "local_proto_query_count",
    "local_proto_anchor_unit_count",
    "local_proto_query_unit_count",
    "local_proto_pos_terms",
    "local_proto_neg_terms",
    "local_proto_split_attempts",
    "local_proto_split_accepts",
    "local_proto_split_reject_small_child",
    "local_proto_split_reject_low_gain",
    "local_proto_split_reject_low_residual",
    "local_proto_split_reject_child_close",
    "local_proto_split_reject_child_far",
    "local_proto_residual_p90_before",
    "local_proto_residual_p90_after",
    "local_proto_split_gain_mean",
    "local_proto_units_p1",
    "local_proto_units_p2",
    "local_proto_units_p3",
    "local_proto_units_p4",
    "mean_proto_per_anchor_unit",
    "local_proto_nce_loss",
    "local_proto_same_loss",
    "local_proto_chain_loss",
    "same_proto_cos_mean",
    "same_proto_cos_min",
    "same_proto_pair_count",
    "temporal_chain_cos_mean",
    "temporal_chain_cos_min",
    "temporal_chain_pair_count",
    "local_proto_warmup_active",
    "local_proto_temporal_fallback",
)


def _local_proto_zero_stats(**updates: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {key: 0 for key in LOCAL_PROTO_STAT_KEYS}
    stats.update(updates)
    return stats


def _normalized_mean(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] == 0:
        return z.new_zeros((1, z.shape[1]))
    return F.normalize(torch.mean(F.normalize(z, p=2, dim=1), dim=0, keepdim=True), p=2, dim=1)


def _farthest_point_indices(z_class: torch.Tensor, max_proto: int) -> torch.Tensor:
    z_detached = F.normalize(z_class.detach(), p=2, dim=1)
    n = int(z_detached.shape[0])
    m = int(max(1, min(max_proto, math.ceil(n / 128.0), 4, n)))
    if m == 1:
        center = F.normalize(torch.mean(z_detached, dim=0, keepdim=True), p=2, dim=1)
        sim = torch.matmul(z_detached, center.T).squeeze(1)
        return torch.argmax(sim).view(1)
    selected = [int(torch.argmax(torch.norm(z_detached - torch.mean(z_detached, dim=0, keepdim=True), dim=1)).item())]
    min_dist = 1.0 - torch.matmul(z_detached, z_detached[selected[0]].view(-1, 1)).squeeze(1)
    for _ in range(1, m):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        dist = 1.0 - torch.matmul(z_detached, z_detached[nxt].view(-1, 1)).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    return torch.tensor(selected, device=z_class.device, dtype=torch.long)


def local_proto_nce_loss(
    z_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    z_loss: torch.Tensor,
    y_loss: torch.Tensor,
    *,
    t_anchor: torch.Tensor | None = None,
    proto_strategy: str = "farthest",
    stage2_epoch: int = 0,
    max_proto_per_cluster: int = 4,
    top_pos: int = 2,
    hard_neg: int = 12,
    tau: float = 0.10,
    delta: float = 0.05,
    adaptive_proto_warmup_epochs: int = 0,
    min_child_samples: int = 12,
    min_child_fraction: float = 0.20,
    split_residual_quantile: float = 0.90,
    split_residual_min: float = 0.12,
    split_gain_min: float = 0.025,
    split_child_distance_min: float = 0.05,
    split_child_cosine_min: float = 0.70,
    same_proto_weight: float = 0.0,
    same_proto_margin: float = 0.70,
    temporal_chain_weight: float = 0.0,
    temporal_chain_margin: float = 0.80,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    del t_anchor, stage2_epoch, adaptive_proto_warmup_epochs
    del min_child_samples, min_child_fraction, split_residual_quantile
    del split_residual_min, split_gain_min, split_child_distance_min, split_child_cosine_min
    if z_anchor.shape[0] == 0 or z_loss.shape[0] == 0:
        return _zero_with_stats(z_loss if z_loss.numel() else z_anchor, _local_proto_zero_stats())

    ya = y_anchor.to(device=z_anchor.device, dtype=torch.long)
    yq = y_loss.to(device=z_loss.device, dtype=torch.long)
    classes = torch.unique(ya[ya >= 0])
    proto_parts = []
    proto_labels = []
    class_proto_ranges: list[Tuple[int, int]] = []
    class_chain_edges: list[Tuple[int, int]] = []
    stats_total = _local_proto_zero_stats()
    strategy = str(proto_strategy or "farthest").strip().lower()

    for c in classes.tolist():
        idx = torch.nonzero(ya == int(c), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        start = sum(int(p.shape[0]) for p in proto_parts)
        if strategy in ("support_mean", "single_support_mean", "mean"):
            proto_c = _normalized_mean(z_anchor[idx])
        else:
            selected = _farthest_point_indices(z_anchor[idx], int(max_proto_per_cluster))
            proto_c = F.normalize(z_anchor[idx][selected], p=2, dim=1)
        if proto_c.shape[0] == 0:
            continue
        n_proto_c = int(proto_c.shape[0])
        stats_total["local_proto_anchor_unit_count"] += 1
        stats_total["local_proto_units_p1"] += int(n_proto_c == 1)
        stats_total["local_proto_units_p2"] += int(n_proto_c == 2)
        stats_total["local_proto_units_p3"] += int(n_proto_c == 3)
        stats_total["local_proto_units_p4"] += int(n_proto_c >= 4)
        end = start + n_proto_c
        class_proto_ranges.append((start, end))
        for proto_i in range(start, max(start, end - 1)):
            class_chain_edges.append((proto_i, proto_i + 1))
        proto_parts.append(proto_c)
        proto_labels.append(torch.full((proto_c.shape[0],), int(c), device=z_anchor.device, dtype=torch.long))

    if len(proto_parts) < 2:
        stats_total["local_proto_count"] = sum(int(p.shape[0]) for p in proto_parts)
        anchor_units = float(stats_total.get("local_proto_anchor_unit_count", 0.0))
        if anchor_units > 0.0:
            stats_total["mean_proto_per_anchor_unit"] = float(stats_total["local_proto_count"] / anchor_units)
        return _zero_with_stats(z_loss, stats_total)

    protos_train = F.normalize(torch.cat(proto_parts, dim=0), p=2, dim=1)
    protos = protos_train.detach()
    proto_y = torch.cat(proto_labels, dim=0)
    zq = F.normalize(z_loss, p=2, dim=1)
    sim = torch.matmul(zq, protos.T)
    tau_eff = max(float(tau), 1e-8)

    losses = []
    loss_labels = []
    pos_terms_used = 0
    neg_terms_used = 0
    for i in range(zq.shape[0]):
        if int(yq[i].detach().cpu().item()) < 0:
            continue
        pos_mask = proto_y == yq[i]
        neg_mask = proto_y != yq[i]
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            continue
        pos_sim_all = sim[i, pos_mask]
        pos_k = min(int(max(1, top_pos)), int(pos_sim_all.numel()))
        pos_sim = torch.topk(pos_sim_all, k=pos_k, largest=True).values
        neg_sim_all = sim[i, neg_mask]
        neg_k = min(int(max(1, hard_neg)), int(neg_sim_all.numel()))
        neg_sim = torch.topk(neg_sim_all, k=neg_k, largest=True).values
        q = torch.softmax(pos_sim / tau_eff, dim=0).detach()
        pos_num = torch.sum(q * torch.exp(pos_sim / tau_eff))
        neg_num = torch.sum(torch.exp((neg_sim + float(delta)) / tau_eff))
        losses.append(-(torch.log(pos_num.clamp_min(1e-8)) - torch.log((pos_num + neg_num).clamp_min(1e-8))))
        loss_labels.append(yq[i])
        pos_terms_used += int(pos_k)
        neg_terms_used += int(neg_k)

    if not losses:
        stats_total["local_proto_count"] = int(protos.shape[0])
        anchor_units = float(stats_total.get("local_proto_anchor_unit_count", 0.0))
        if anchor_units > 0.0:
            stats_total["mean_proto_per_anchor_unit"] = float(stats_total["local_proto_count"] / anchor_units)
        return _zero_with_stats(z_loss, stats_total)

    loss_vec = torch.stack(losses)
    label_vec = torch.stack(loss_labels)
    nce_loss, query_unit_count = _mean_by_group(loss_vec, label_vec)

    same_terms = []
    same_cos = []
    for start, end in class_proto_ranges:
        if int(end - start) <= 1:
            continue
        pc = protos_train[start:end]
        sim_c = torch.matmul(pc, pc.T)
        tri = torch.triu(torch.ones_like(sim_c, dtype=torch.bool), diagonal=1)
        vals = sim_c[tri]
        if vals.numel() > 0:
            same_cos.append(vals.detach())
            same_terms.append(F.relu(float(same_proto_margin) - vals).pow(2))
    same_loss = torch.mean(torch.cat(same_terms)) if same_terms else z_anchor.new_zeros(())
    same_vals = torch.cat(same_cos) if same_cos else None

    chain_terms = []
    chain_cos = []
    for a, b in class_chain_edges:
        val = torch.sum(protos_train[int(a)] * protos_train[int(b)])
        chain_cos.append(val.detach().view(1))
        chain_terms.append(F.relu(float(temporal_chain_margin) - val).pow(2).view(1))
    chain_loss = torch.mean(torch.cat(chain_terms)) if chain_terms else z_anchor.new_zeros(())
    chain_vals = torch.cat(chain_cos) if chain_cos else None

    loss = nce_loss + float(same_proto_weight) * same_loss + float(temporal_chain_weight) * chain_loss
    stats_total.update(
        {
            "local_proto_count": int(protos.shape[0]),
            "local_proto_query_count": int(loss_vec.numel()),
            "local_proto_query_unit_count": int(query_unit_count),
            "local_proto_pos_terms": int(pos_terms_used),
            "local_proto_neg_terms": int(neg_terms_used),
            "local_proto_nce_loss": float(nce_loss.detach().cpu().item()),
            "local_proto_same_loss": float(same_loss.detach().cpu().item()),
            "local_proto_chain_loss": float(chain_loss.detach().cpu().item()),
            "same_proto_pair_count": int(same_vals.numel()) if same_vals is not None else 0,
            "same_proto_cos_mean": float(torch.mean(same_vals).cpu().item()) if same_vals is not None and same_vals.numel() else 0.0,
            "same_proto_cos_min": float(torch.min(same_vals).cpu().item()) if same_vals is not None and same_vals.numel() else 0.0,
            "temporal_chain_pair_count": int(chain_vals.numel()) if chain_vals is not None else 0,
            "temporal_chain_cos_mean": float(torch.mean(chain_vals).cpu().item()) if chain_vals is not None and chain_vals.numel() else 0.0,
            "temporal_chain_cos_min": float(torch.min(chain_vals).cpu().item()) if chain_vals is not None and chain_vals.numel() else 0.0,
        }
    )
    anchor_units = float(stats_total.get("local_proto_anchor_unit_count", 0.0))
    if anchor_units > 0.0:
        stats_total["mean_proto_per_anchor_unit"] = float(stats_total["local_proto_count"] / anchor_units)
    return loss, stats_total


def _zero_stage2_stats(z: torch.Tensor) -> Dict[str, Any]:
    zero = z.new_zeros(())
    stats: Dict[str, Any] = {
        "supervised_graph_pos_edges": 0.0,
        "supervised_graph_neg_edges": 0.0,
        "supervised_graph_hard_neg_edges": 0.0,
        "supervised_graph_mean_q_pos": 0.0,
        "supervised_graph_mean_q_neg": 0.0,
        "supervised_graph_pos_loss": 0.0,
        "supervised_graph_neg_loss": 0.0,
        "boundary_anchor_count": 0.0,
        "boundary_unit_count": 0.0,
        "boundary_top_h": 0.0,
        "boundary_active_ratio": 0.0,
        "boundary_s_pos_mean": 0.0,
        "boundary_s_neg_mean": 0.0,
        "boundary_violation_mean": 0.0,
        "supcon_anchor_count": 0,
        "supcon_valid_anchor_count": 0,
        "supcon_skipped_anchor_count": 0,
        "proto_ce_way": 0,
        "proto_batch_classes": 0,
        "proto_batch_count": 0,
        "lambda_umap_eff": 0.0,
        "lambda_supervised_graph_eff": 0.0,
        "lambda_boundary_eff": 0.0,
        "lambda_supcon_eff": 0.0,
        "teacher_graph_pos_edges": 0,
        "teacher_graph_neg_edges": 0,
        "teacher_graph_random_neg_edges": 0,
        "teacher_graph_hard_neg_edges": 0,
        "teacher_graph_hard_neg_fraction": 0.0,
        "teacher_graph_classes_used": 0,
        "teacher_graph_singleton_classes_skipped": 0,
        "teacher_graph_mean_p": 0.0,
        "teacher_graph_mean_q_pos": 0.0,
        "teacher_graph_mean_q_neg": 0.0,
        "teacher_graph_pos_loss": 0.0,
        "teacher_graph_neg_loss": 0.0,
        "teacher_graph_neg_weight": 0.0,
        "teacher_graph_pos_unit_count": 0,
        "teacher_graph_neg_unit_count": 0,
        "supervised_graph_classes_used": 0,
        "supervised_graph_singleton_classes_skipped": 0,
        "supervised_graph_neg_weight": 0.0,
        "supcon": zero.detach(),
        "proto": zero.detach(),
        "proto_ce": zero.detach(),
        "umap_teacher_graph": zero.detach(),
        "supervised_graph": zero.detach(),
        "boundary": zero.detach(),
        "channel": zero.detach(),
    }
    stats.update(_local_proto_zero_stats())
    return stats


def stage2_total_loss(
    student_out: Any,
    teacher_feature: torch.Tensor | None,
    batch: Dict[str, torch.Tensor],
    episode: Dict[str, torch.Tensor],
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    del teacher_feature
    z = _get_embedding(student_out)
    bsz = int(batch["unit_id"].shape[0])
    two_view = bool(z.shape[0] == 2 * bsz)
    z1 = z[:bsz]
    z2 = z[bsz:] if two_view else None
    y_base = batch["unit_id"].to(device=z.device, dtype=torch.long)
    recording_id = batch.get("recording_id")
    if recording_id is not None:
        recording_id = recording_id.to(device=z.device, dtype=torch.long)
    else:
        recording_id = torch.zeros((bsz,), device=z.device, dtype=torch.long)

    valid_mask = batch.get("valid_mask")
    if valid_mask is None:
        valid_mask, valid_stats = build_stage2_valid_mask(batch, cfg, gate_prob=batch.get("gate_confidence"))
    else:
        valid_mask = valid_mask.to(device=z.device, dtype=torch.bool)
        valid_stats = {"stage2_valid_count": int(torch.sum(valid_mask).detach().cpu().item())}

    stage2_epoch = (
        int(batch.get("stage2_epoch", torch.tensor(0, device=z.device)).detach().cpu().item())
        if isinstance(batch.get("stage2_epoch"), torch.Tensor)
        else int(batch.get("stage2_epoch", 0))
    )
    losses_cfg = _as_cfg_dict(_cfg_get(cfg, "stage2.losses", {}))
    graph_cfg = _as_cfg_dict(losses_cfg.get("graph", losses_cfg.get("recording_graph", {})))
    proto_cfg = _as_cfg_dict(losses_cfg.get("local_proto_nce", losses_cfg.get("local_proto", {})))
    boundary_cfg = _as_cfg_dict(losses_cfg.get("boundary", losses_cfg.get("hard_negative_margin", {})))

    graph_enabled = _cfg_bool(graph_cfg.get("enabled", False), False) and float(graph_cfg.get("weight", 0.0)) > 0.0
    boundary_enabled = _cfg_bool(boundary_cfg.get("enabled", False), False) and float(boundary_cfg.get("weight", 0.0)) > 0.0
    if graph_enabled:
        raise ValueError("stage2 graph loss is not included in the public final-loss build")
    if boundary_enabled:
        raise ValueError("stage2 boundary loss is not included in the public final-loss build")

    lambda_proto_eff = (
        local_proto_weight_for_epoch(
            stage2_epoch,
            graph_warmup_epochs=int(_cfg_get(cfg, "stage2.graph_warmup_epochs", proto_cfg.get("graph_warmup_epochs", 0))),
            hold_epochs=int(_cfg_get(cfg, "stage2.local_proto_hold_epochs", proto_cfg.get("hold_epochs", 0))),
            ramp_epochs=int(_cfg_get(cfg, "stage2.local_proto_ramp_epochs", proto_cfg.get("ramp_epochs", 0))),
            start_weight=float(proto_cfg.get("start_weight", 1.0)),
            end_weight=float(proto_cfg.get("weight", proto_cfg.get("end_weight", 1.0))),
        )
        if _cfg_bool(proto_cfg.get("enabled", True), True)
        else 0.0
    )

    all_idx = episode.get("all_idx")
    if all_idx is None:
        all_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    all_idx = all_idx.to(device=z.device, dtype=torch.long)
    anchor_idx = episode.get("anchor_idx", episode.get("support_idx"))
    loss_idx = episode.get("loss_idx", episode.get("query_idx"))
    if anchor_idx is None:
        anchor_idx = all_idx
    if loss_idx is None:
        loss_idx = all_idx
    anchor_idx = anchor_idx.to(device=z.device, dtype=torch.long)
    loss_idx = loss_idx.to(device=z.device, dtype=torch.long)
    all_idx = all_idx[valid_mask[all_idx]]
    anchor_idx = anchor_idx[valid_mask[anchor_idx]]
    loss_idx = loss_idx[valid_mask[loss_idx]]

    stats = _zero_stage2_stats(z)
    stats.update(valid_stats)
    proto_losses = []
    proto_stat_groups = 0
    proto_stats_total = _local_proto_zero_stats()
    views = [z1] if z2 is None else [z1, z2]
    rec_iter = torch.unique(recording_id[all_idx]).tolist() if all_idx.numel() else []
    for rec_value in rec_iter:
        rec_anchor = anchor_idx[recording_id[anchor_idx] == int(rec_value)]
        rec_loss = loss_idx[recording_id[loss_idx] == int(rec_value)]
        if rec_anchor.numel() == 0 or rec_loss.numel() == 0:
            continue
        for z_view in views:
            if lambda_proto_eff <= 0.0:
                continue
            event_time_sample = batch.get("event_time_sample")
            t_anchor = event_time_sample.to(device=z.device)[rec_anchor] if isinstance(event_time_sample, torch.Tensor) else None
            loss_type = str(proto_cfg.get("loss_type", "nce")).strip().lower()
            if loss_type not in ("nce", "local_proto_nce", "contrastive"):
                raise ValueError("public final-loss build supports only stage2.losses.local_proto_nce.loss_type=nce")
            p_loss, p_stats = local_proto_nce_loss(
                z_view[rec_anchor],
                y_base[rec_anchor],
                z_view[rec_loss],
                y_base[rec_loss],
                t_anchor=t_anchor,
                proto_strategy=str(proto_cfg.get("proto_strategy", "farthest")),
                stage2_epoch=int(stage2_epoch),
                max_proto_per_cluster=int(proto_cfg.get("max_proto_per_cluster", 4)),
                top_pos=int(proto_cfg.get("top_pos", 2)),
                hard_neg=int(proto_cfg.get("hard_neg", 12)),
                tau=float(proto_cfg.get("tau", 0.10)),
                delta=float(proto_cfg.get("delta", 0.05)),
                same_proto_weight=float(proto_cfg.get("same_proto_weight", 0.0)),
                same_proto_margin=float(proto_cfg.get("same_proto_margin", 0.70)),
                temporal_chain_weight=float(proto_cfg.get("temporal_chain_weight", 0.0)),
                temporal_chain_margin=float(proto_cfg.get("temporal_chain_margin", 0.80)),
            )
            proto_losses.append(p_loss)
            proto_stat_groups += 1
            for key in proto_stats_total:
                proto_stats_total[key] += float(p_stats.get(key, 0.0))

    if proto_stat_groups > 0:
        for key in proto_stats_total:
            proto_stats_total[key] /= float(proto_stat_groups)
    l_proto = torch.mean(torch.stack(proto_losses)) if proto_losses else z.new_zeros(())
    loss = float(lambda_proto_eff) * l_proto
    stats.update(proto_stats_total)
    stats.update(
        {
            "proto": l_proto.detach(),
            "lambda_proto_eff": float(lambda_proto_eff),
            "proto_ce_support_count": int(anchor_idx.numel()),
            "proto_ce_query_count": int(loss_idx.numel()),
            "episode_effective_n_way": int(torch.unique(y_base[all_idx]).numel()) if all_idx.numel() else 0,
            "supervised_graph_classes_used": int(torch.unique(y_base[all_idx]).numel()) if all_idx.numel() else 0,
        }
    )
    return loss, stats


def gate_detection_loss(
    spike_logits: torch.Tensor,
    spike_labels: torch.Tensor,
    *,
    gate_pos_weight: Optional[torch.Tensor] = None,
    gate_sample_weights: Optional[torch.Tensor] = None,
    gate_loss_type: str = "bce",
    gate_focal_gamma: float = 2.0,
    lambda_gate: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del gate_focal_gamma
    loss_type = str(gate_loss_type).lower()
    if loss_type not in ("bce", "binary_cross_entropy", "binary_cross_entropy_with_logits"):
        raise ValueError("public final-loss build supports only gate BCE loss")
    spike_target = spike_labels.to(dtype=spike_logits.dtype)
    pos_weight = None if gate_pos_weight is None else gate_pos_weight.to(device=spike_logits.device, dtype=spike_logits.dtype)
    sample_weights = (
        None
        if gate_sample_weights is None
        else gate_sample_weights.to(device=spike_logits.device, dtype=spike_logits.dtype)
    )
    if sample_weights is None:
        l_gate = F.binary_cross_entropy_with_logits(spike_logits, spike_target, pos_weight=pos_weight)
    else:
        gate_loss_terms = F.binary_cross_entropy_with_logits(
            spike_logits,
            spike_target,
            pos_weight=pos_weight,
            reduction="none",
        )
        denom = torch.sum(sample_weights).clamp_min(1e-8)
        l_gate = torch.sum(gate_loss_terms * sample_weights) / denom
    total = float(lambda_gate) * l_gate
    return total, {
        "gate": l_gate.detach(),
        "gate_spike_noise": l_gate.detach(),
    }
