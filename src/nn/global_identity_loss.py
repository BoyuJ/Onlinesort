"""Recording-local Identity Discrimination Loss used by the paper."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def global_identity_connection_loss(z, batch, episode, cfg):
    settings = cfg['stage2']['losses']['temporal_connection']
    weight = float(settings.get('weight', 0.0)) if settings.get('enabled', False) else 0.0
    zero = z.sum() * 0.0
    stats = dict(temporal_conn_loss=0.0, lambda_temporal_conn_eff=weight,
                 temporal_conn_pair_count=0.0, temporal_conn_active_ratio=0.0,
                 compact_loss=0.0, separation_loss=0.0, positive_distance=0.0,
                 hard_negative_distance=0.0, eligible_anchors=0, total_anchors=0,
                 missing_positive_anchors=0, missing_negative_anchors=0,
                 eligible_fraction=0.0, eligible_identities=0, eligible_recordings=0)
    if weight <= 0:
        return zero, stats
    assert float(settings['relative_margin']) == 0.08
    assert int(settings['negative_events']) == 16
    all_idx = episode.get('all_idx')
    if all_idx is None or all_idx.numel() == 0:
        return zero, stats
    device = z.device
    all_idx = all_idx.to(device=device, dtype=torch.long)
    labels = batch['unit_id'].to(device=device, dtype=torch.long)
    records = batch['recording_id'].to(device=device, dtype=torch.long)
    bsz = len(labels)
    assert z.shape[0] in (bsz, 2 * bsz), (z.shape, bsz)
    assert all_idx.unique().numel() == all_idx.numel()
    assert (labels[all_idx] >= 0).all(), 'Episode contains invalid identity labels'
    views = [z[:bsz], z[bsz:]] if len(z) == 2 * bsz else [z]
    # Index construction does not participate in autograd and does not draw RNG.
    groups = []
    for rec in records[all_idx].unique(sorted=True):
        ix = all_idx[records[all_idx] == rec]
        y = labels[ix]
        n = len(ix)
        assert n <= 512, 'New loss must not increase the original episode budget'
        positive = (y[:, None] == y[None, :]) & ~torch.eye(n, dtype=torch.bool, device=device)
        negative = y[:, None] != y[None, :]
        pc, nc = positive.sum(1), negative.sum(1)
        eligible = (pc > 0) & (nc > 0)
        stats['total_anchors'] += n
        stats['missing_positive_anchors'] += int((pc == 0).sum())
        stats['missing_negative_anchors'] += int((nc == 0).sum())
        stats['eligible_anchors'] += int(eligible.sum())
        if not eligible.any():
            continue
        identities = [eligible & (y == unit) for unit in y[eligible].unique(sorted=True)]
        stats['eligible_identities'] += len(identities)
        stats['eligible_recordings'] += 1
        stats['temporal_conn_pair_count'] += int(pc[eligible].sum())
        groups.append((ix, positive, negative, pc, nc, eligible, identities))
    stats['eligible_fraction'] = stats['eligible_anchors'] / max(1, stats['total_anchors'])
    if not groups:
        return zero, stats
    view_terms, view_metrics = [], []
    for view in views:
        normalized = F.normalize(view, p=2, dim=1)
        record_terms, record_metrics = [], []
        for ix, positive, negative, pc, nc, eligible, identities in groups:
            d = 1.0 - normalized[ix] @ normalized[ix].T
            # Only the discrete top-k choice is detached; gathered distances are live.
            order = torch.argsort(d.detach().masked_fill(~negative, float('inf')),
                                  dim=1, stable=True)[:, :min(16, len(ix))]
            nd = d.gather(1, order)
            negative_valid = negative.gather(1, order)
            count_neg = negative_valid.sum(1).clamp_min(1)
            compact = (d.square() * positive).sum(1) / pc.clamp_min(1)
            violation = F.relu(d[:, :, None] - nd[:, None, :] + 0.08)
            pair_mask = positive[:, :, None] & negative_valid[:, None, :]
            separated = (violation.square() * pair_mask).sum((1, 2)) / (pc.clamp_min(1) * count_neg)
            pd = (d * positive).sum(1) / pc.clamp_min(1)
            hd = (nd * negative_valid).sum(1) / count_neg
            active = ((violation > 0) & pair_mask).sum((1, 2)) / (pc.clamp_min(1) * count_neg)
            per_anchor = torch.stack([compact, separated, pd, hd, active], dim=1)
            # Mean anchor within identity, mean identity within recording.
            metrics = torch.stack([per_anchor[mask].mean(0) for mask in identities]).mean(0)
            record_terms.append(metrics[0] + metrics[1])
            record_metrics.append(metrics.detach())
        view_terms.append(torch.stack(record_terms).mean())
        view_metrics.append(torch.stack(record_metrics).mean(0))
    loss = torch.stack(view_terms).mean()
    metrics = torch.stack(view_metrics).mean(0).cpu().tolist()
    assert torch.isfinite(loss), 'Nonfinite global identity loss'
    stats.update(dict(zip(['compact_loss', 'separation_loss', 'positive_distance',
                          'hard_negative_distance', 'temporal_conn_active_ratio'], metrics)))
    stats['temporal_conn_loss'] = float(loss.detach())
    return loss, stats
