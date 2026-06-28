from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from ..utils import Event
from .types import InitializedUnit, normalize_embedding_matrix


REASON_ACCEPTED = 0
REASON_NO_UNITS = 1
REASON_INVALID_EMBEDDING = 2
REASON_NO_CANDIDATE = 3
REASON_DISTANCE = 4
REASON_MARGIN = 5

REASON_TEXT = {
    REASON_ACCEPTED: "accepted",
    REASON_NO_UNITS: "no_units",
    REASON_INVALID_EMBEDDING: "invalid_embedding",
    REASON_NO_CANDIDATE: "no_candidate",
    REASON_DISTANCE: "distance",
    REASON_MARGIN: "margin",
}


@dataclass
class PrototypeAssignConfig:
    max_center_distance: float = 0.35
    margin_min: float = 0.03
    candidate_hop: int = 2
    confidence_temperature: float = 0.15


@dataclass
class AssignmentBatchResult:
    unit_id: np.ndarray
    accepted: np.ndarray
    confidence: np.ndarray
    distance: np.ndarray
    margin: np.ndarray
    reason_code: np.ndarray


@dataclass
class UnitPrototypeSet:
    unit_id: int
    centers: np.ndarray
    dominant_channel: int
    n_events: int = 0


def _build_hop_matrix(neighbors: np.ndarray | None) -> np.ndarray | None:
    if neighbors is None:
        return None
    neigh = np.asarray(neighbors, dtype=np.int32)
    if neigh.ndim != 2 or neigh.shape[0] == 0:
        return None
    n = int(neigh.shape[0])
    out = np.full((n, n), 32767, dtype=np.int16)
    for src in range(n):
        dist = np.full((n,), -1, dtype=np.int16)
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in neigh[u].tolist():
                vv = int(v)
                if vv < 0 or vv >= n or dist[vv] >= 0:
                    continue
                dist[vv] = int(dist[u]) + 1
                queue.append(vv)
        missing = dist < 0
        if np.any(missing):
            idx = np.nonzero(missing)[0]
            dist[idx] = np.abs(idx - src).astype(np.int16)
        out[src] = dist
    return out


def _hop_distance(ch_a: int, ch_b: int, hop_matrix: np.ndarray | None) -> int:
    if hop_matrix is None:
        return abs(int(ch_a) - int(ch_b))
    n = int(hop_matrix.shape[0])
    a = int(np.clip(int(ch_a), 0, n - 1))
    b = int(np.clip(int(ch_b), 0, n - 1))
    return int(hop_matrix[a, b])


def _empty_result(n: int, reason: int) -> AssignmentBatchResult:
    return AssignmentBatchResult(
        unit_id=np.full((int(n),), -1, dtype=np.int32),
        accepted=np.zeros((int(n),), dtype=np.bool_),
        confidence=np.zeros((int(n),), dtype=np.float32),
        distance=np.full((int(n),), np.nan, dtype=np.float32),
        margin=np.full((int(n),), np.nan, dtype=np.float32),
        reason_code=np.full((int(n),), int(reason), dtype=np.int16),
    )


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8 or not np.isfinite(norm):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def _stable_center(z: np.ndarray, trim_quantile: float = 0.90) -> np.ndarray:
    arr = normalize_embedding_matrix(np.asarray(z, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[0] == 0:
        width = int(arr.shape[1]) if arr.ndim == 2 else 0
        return np.zeros((width,), dtype=np.float32)
    center = _normalize_vector(np.mean(arr, axis=0))
    if arr.shape[0] <= 2:
        return center
    dist = 1.0 - np.clip(arr @ center, -1.0, 1.0)
    cutoff = float(np.quantile(dist, float(np.clip(trim_quantile, 0.0, 1.0))))
    keep = dist <= cutoff
    if int(np.sum(keep)) < max(2, int(np.ceil(0.5 * arr.shape[0]))):
        keep = np.ones((arr.shape[0],), dtype=np.bool_)
    return _normalize_vector(np.mean(arr[keep], axis=0))


def build_unit_prototype_set(
    unit: InitializedUnit,
    embeddings: np.ndarray,
    *,
    max_prototypes: int = 5,
    min_events_per_prototype: int = 8,
    trim_quantile: float = 0.90,
    lloyd_iters: int = 3,
) -> UnitPrototypeSet:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {emb.shape}")
    valid = np.all(np.isfinite(emb), axis=1) & (np.linalg.norm(emb, axis=1) > 1e-8)
    z = normalize_embedding_matrix(emb[valid])
    n_valid = int(z.shape[0])
    width = int(emb.shape[1])
    max_count = max(1, int(max_prototypes))
    min_events = max(1, int(min_events_per_prototype))
    if n_valid == 0:
        centers = normalize_embedding_matrix(np.asarray(unit.center, dtype=np.float32).reshape(1, -1))
    elif n_valid < min_events or max_count == 1:
        centers = _stable_center(z, trim_quantile=float(trim_quantile)).reshape(1, width)
    else:
        n_proto = min(max_count, n_valid)
        centers_list = [_stable_center(z, trim_quantile=float(trim_quantile))]
        while len(centers_list) < n_proto:
            current = normalize_embedding_matrix(np.stack(centers_list, axis=0))
            nearest_dist = 1.0 - np.max(np.clip(z @ current.T, -1.0, 1.0), axis=1)
            centers_list.append(z[int(np.argmax(nearest_dist))].astype(np.float32, copy=True))
        centers = normalize_embedding_matrix(np.stack(centers_list, axis=0))
        for _ in range(max(0, int(lloyd_iters))):
            labels = np.argmax(z @ centers.T, axis=1)
            updated: list[np.ndarray] = []
            for proto_idx in range(int(centers.shape[0])):
                members = z[labels == proto_idx]
                if members.shape[0] == 0:
                    updated.append(centers[proto_idx])
                else:
                    updated.append(_normalize_vector(np.mean(members, axis=0)))
            centers = normalize_embedding_matrix(np.stack(updated, axis=0))
    return UnitPrototypeSet(
        unit_id=int(unit.unit_id),
        centers=np.asarray(centers, dtype=np.float32).reshape(-1, width),
        dominant_channel=int(unit.dominant_channel),
        n_events=n_valid,
    )


def _ordered_prototype_sets(
    prototypes: Dict[int, UnitPrototypeSet] | Sequence[UnitPrototypeSet],
) -> List[UnitPrototypeSet]:
    if isinstance(prototypes, dict):
        return [prototypes[key] for key in sorted(prototypes.keys())]
    return sorted(list(prototypes), key=lambda proto: int(proto.unit_id))


def _flatten_prototypes(
    prototypes: Dict[int, UnitPrototypeSet] | Sequence[UnitPrototypeSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proto_list = _ordered_prototype_sets(prototypes)
    centers: list[np.ndarray] = []
    unit_ids: list[int] = []
    unit_channels: list[int] = []
    proto_local: list[int] = []
    for proto_set in proto_list:
        c = normalize_embedding_matrix(np.asarray(proto_set.centers, dtype=np.float32))
        if c.ndim != 2 or c.shape[0] == 0:
            continue
        for idx in range(int(c.shape[0])):
            centers.append(c[idx])
            unit_ids.append(int(proto_set.unit_id))
            unit_channels.append(int(proto_set.dominant_channel))
            proto_local.append(int(idx))
    if not centers:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        normalize_embedding_matrix(np.stack(centers, axis=0)),
        np.asarray(unit_ids, dtype=np.int32),
        np.asarray(unit_channels, dtype=np.int32),
        np.asarray(proto_local, dtype=np.int32),
    )


def assign_to_unit_prototypes(
    events: Sequence[Event],
    embeddings: np.ndarray,
    prototypes: Dict[int, UnitPrototypeSet] | Sequence[UnitPrototypeSet],
    *,
    cfg: PrototypeAssignConfig | None = None,
    neighbors_lookup: np.ndarray | None = None,
) -> AssignmentBatchResult:
    cfg = cfg or PrototypeAssignConfig()
    n_events = int(len(events))
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != n_events:
        raise ValueError(f"event/embedding length mismatch: events={n_events} embeddings={emb.shape}")

    centers, unit_ids, unit_channels, _ = _flatten_prototypes(prototypes)
    if centers.shape[0] == 0:
        return _empty_result(n_events, REASON_NO_UNITS)

    raw_norm = np.linalg.norm(emb, axis=1)
    valid = np.all(np.isfinite(emb), axis=1) & (raw_norm > 1e-8)
    z = normalize_embedding_matrix(emb)
    hop_matrix = _build_hop_matrix(neighbors_lookup)

    result = _empty_result(n_events, REASON_INVALID_EMBEDDING)
    max_distance = float(cfg.max_center_distance)
    margin_min = float(cfg.margin_min)
    candidate_hop = int(cfg.candidate_hop)
    temp = max(float(cfg.confidence_temperature), 1e-6)

    for idx, event in enumerate(events):
        if not bool(valid[int(idx)]):
            continue
        if candidate_hop >= 0:
            candidate_mask = np.asarray(
                [
                    _hop_distance(int(event.c0), int(ch), hop_matrix) <= int(candidate_hop)
                    for ch in unit_channels.tolist()
                ],
                dtype=np.bool_,
            )
        else:
            candidate_mask = np.ones((centers.shape[0],), dtype=np.bool_)
        candidate_idx = np.flatnonzero(candidate_mask)
        if candidate_idx.size == 0:
            result.reason_code[idx] = REASON_NO_CANDIDATE
            continue

        sims = centers[candidate_idx] @ z[int(idx)]
        dists = 1.0 - np.clip(sims, -1.0, 1.0)
        order = np.argsort(dists, kind="stable")
        best_local = int(order[0])
        best_dist = float(dists[best_local])
        best_proto_idx = int(candidate_idx[best_local])
        best_unit_id = int(unit_ids[best_proto_idx])
        second_dist = float("inf")
        for local_idx in order[1:].tolist():
            proto_idx = int(candidate_idx[int(local_idx)])
            if int(unit_ids[proto_idx]) != best_unit_id:
                second_dist = float(dists[int(local_idx)])
                break
        margin = float(second_dist - best_dist) if np.isfinite(second_dist) else float("inf")
        result.distance[idx] = np.float32(best_dist)
        result.margin[idx] = np.float32(margin if np.isfinite(margin) else 1e6)

        if best_dist > max_distance:
            result.reason_code[idx] = REASON_DISTANCE
            continue
        if margin < margin_min:
            result.reason_code[idx] = REASON_MARGIN
            continue

        margin_factor = 1.0 if not np.isfinite(margin) else 1.0 - float(np.exp(-max(0.0, margin) / temp))
        distance_factor = float(np.exp(-max(0.0, best_dist) / temp))
        result.unit_id[idx] = best_unit_id
        result.accepted[idx] = True
        result.confidence[idx] = np.float32(np.clip(distance_factor * max(0.0, margin_factor), 0.0, 1.0))
        result.reason_code[idx] = REASON_ACCEPTED
    return result


def update_nearest_unit_prototype(
    prototypes: Dict[int, UnitPrototypeSet] | Sequence[UnitPrototypeSet],
    unit_id: int,
    embedding: np.ndarray,
    *,
    alpha_mu: float,
) -> bool:
    proto_set: UnitPrototypeSet | None = None
    if isinstance(prototypes, dict):
        proto_set = prototypes.get(int(unit_id))
    else:
        for candidate in prototypes:
            if int(candidate.unit_id) == int(unit_id):
                proto_set = candidate
                break
    if proto_set is None:
        return False
    centers = normalize_embedding_matrix(np.asarray(proto_set.centers, dtype=np.float32))
    if centers.ndim != 2 or centers.shape[0] == 0:
        return False
    z = _normalize_vector(np.asarray(embedding, dtype=np.float32))
    if not np.any(z):
        return False
    dists = 1.0 - np.clip(centers @ z, -1.0, 1.0)
    proto_idx = int(np.argmin(dists))
    alpha = float(np.clip(alpha_mu, 0.0, 1.0))
    centers[proto_idx] = _normalize_vector((1.0 - alpha) * centers[proto_idx] + alpha * z)
    proto_set.centers = centers.astype(np.float32, copy=False)
    proto_set.n_events = int(proto_set.n_events) + 1
    return True

__all__ = [
    "AssignmentBatchResult",
    "PrototypeAssignConfig",
    "REASON_ACCEPTED",
    "REASON_DISTANCE",
    "REASON_INVALID_EMBEDDING",
    "REASON_MARGIN",
    "REASON_NO_CANDIDATE",
    "REASON_NO_UNITS",
    "REASON_TEXT",
    "UnitPrototypeSet",
    "assign_to_unit_prototypes",
    "build_unit_prototype_set",
    "update_nearest_unit_prototype",
]
