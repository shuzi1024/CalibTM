import math
from dataclasses import dataclass

import numpy as np
import torch


MASK_TYPES = ("random", "internal_block", "burst", "edge", "mixed_structured")


@dataclass(frozen=True)
class StructuredMaskConfig:
    mask_type: str = "random"
    block_len: int = 8
    block_len_frac: float = 0.25
    num_bursts: int = 3
    burst_min_len: int = 3
    burst_max_len: int = 8
    edge_len_frac: float = 0.25
    preserve_mask_rate: bool = True
    anchor_centric: bool = True


def observed_budget(length, observed_rate):
    count = int(math.ceil(float(observed_rate) * int(length)))
    count = min(int(length), max(1, count))
    return count


def _rng(seed):
    return np.random.default_rng(int(seed) % (2 ** 32))


def _clamp_len(value, length, min_value=1):
    return max(min_value, min(int(value), max(1, int(length) - 2)))


def _sample_remaining(gen, observed, candidates, target_count, length):
    if len(observed) >= target_count:
        return sorted(observed)[:target_count]
    candidates = [idx for idx in candidates if idx not in observed]
    need = target_count - len(observed)
    if need > 0 and candidates:
        take = min(need, len(candidates))
        observed.update(int(v) for v in gen.choice(candidates, size=take, replace=False))
    if len(observed) < target_count:
        fallback = [idx for idx in range(int(length)) if idx not in observed]
        if fallback:
            take = min(target_count - len(observed), len(fallback))
            observed.update(int(v) for v in gen.choice(fallback, size=take, replace=False))
    return sorted(observed)


def _internal_indices(length, budget, gen, cfg):
    block_len = cfg.block_len if cfg.block_len > 0 else round(length * cfg.block_len_frac)
    block_len = _clamp_len(block_len, length, min_value=1)
    if length < 3 or budget < 2:
        return sorted(gen.choice(length, size=budget, replace=False).tolist())
    max_start = max(1, length - block_len - 1)
    if max_start <= 1:
        start = 1
    else:
        start = int(gen.integers(1, max_start))
    end = min(length - 2, start + block_len - 1)
    left = start - 1
    right = end + 1
    observed = {left, right}
    forbidden = set(range(start, end + 1))
    candidates = [idx for idx in range(length) if idx not in forbidden and idx not in observed]
    return _sample_remaining(gen, observed, candidates, budget, length)


def _burst_indices(length, budget, gen, cfg):
    if length < 3 or budget < 2:
        return sorted(gen.choice(length, size=budget, replace=False).tolist())
    min_len = max(1, int(cfg.burst_min_len))
    max_len = max(min_len, int(cfg.burst_max_len))
    requested = max(1, int(cfg.num_bursts))
    feasible_bursts = max(1, min(requested, budget - 1))
    burst_lengths = [int(gen.integers(min_len, max_len + 1)) for _ in range(feasible_bursts)]
    total_span = sum(burst_lengths) + feasible_bursts + 1
    if total_span >= length:
        scale = max(1, length - feasible_bursts - 2)
        each = max(1, scale // feasible_bursts)
        burst_lengths = [min(each, max_len) for _ in range(feasible_bursts)]
        total_span = sum(burst_lengths) + feasible_bursts + 1
    slack = max(0, length - total_span)
    start_anchor = int(gen.integers(0, slack + 1)) if slack else 0
    anchors = [start_anchor]
    cursor = start_anchor
    forbidden = set()
    for burst_len in burst_lengths:
        gap_start = cursor + 1
        gap_end = min(length - 2, gap_start + burst_len - 1)
        forbidden.update(range(gap_start, gap_end + 1))
        cursor = gap_end + 1
        if cursor >= length:
            cursor = length - 1
        anchors.append(cursor)
    observed = {idx for idx in anchors if 0 <= idx < length}
    candidates = [idx for idx in range(length) if idx not in forbidden and idx not in observed]
    return _sample_remaining(gen, observed, candidates, budget, length)


def _edge_indices(length, budget, gen, cfg, side=None):
    if length < 2:
        return [0]
    edge_len = _clamp_len(round(length * float(cfg.edge_len_frac)), length, min_value=1)
    if side is None:
        side = "left" if int(gen.integers(0, 2)) == 0 else "right"
    if side == "left":
        anchor = min(length - 1, edge_len)
        forbidden = set(range(0, anchor))
        candidates = [idx for idx in range(anchor, length) if idx != anchor]
    else:
        anchor = max(0, length - edge_len - 1)
        forbidden = set(range(anchor + 1, length))
        candidates = [idx for idx in range(0, anchor + 1) if idx != anchor]
    observed = {anchor}
    return _sample_remaining(gen, observed, candidates, budget, length)


def _random_indices(length, budget, gen):
    return sorted(gen.choice(length, size=budget, replace=False).tolist())


def structured_indices(length, observed_rate, seed, cfg, sample_index=0, flow_index=0):
    budget = observed_budget(length, observed_rate)
    gen = _rng(seed + 1000003 * int(sample_index) + 9176 * int(flow_index))
    mask_type = str(cfg.mask_type or "random")
    if mask_type == "mixed_structured":
        choices = ("internal_block", "burst", "edge", "random")
        mask_type = choices[int(gen.integers(0, len(choices)))]
    if mask_type == "random":
        return _random_indices(length, budget, gen), mask_type
    if mask_type == "internal_block":
        return _internal_indices(length, budget, gen, cfg), mask_type
    if mask_type == "burst":
        return _burst_indices(length, budget, gen, cfg), mask_type
    if mask_type == "edge":
        return _edge_indices(length, budget, gen, cfg), mask_type
    raise ValueError(f"Unsupported mask_type: {cfg.mask_type}")


def generate_structured_observed_mask(shape, observed_rate, device, seed, cfg):
    bsz, steps, flows = [int(v) for v in shape]
    mask = torch.zeros((bsz, steps, flows), device=device, dtype=torch.float32)
    pattern_counts = {"random": 0, "internal_block": 0, "burst": 0, "edge": 0}
    for b in range(bsz):
        for n in range(flows):
            indices, actual_type = structured_indices(steps, observed_rate, seed, cfg, b, n)
            if indices:
                mask[b, torch.tensor(indices, device=device, dtype=torch.long), n] = 1.0
            pattern_counts[actual_type] = pattern_counts.get(actual_type, 0) + 1
    return mask, pattern_counts


def contiguous_runs(indices):
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = start
    for item in indices[1:]:
        item = int(item)
        if item == prev + 1:
            prev = item
            continue
        runs.append((start, prev))
        start = item
        prev = item
    runs.append((start, prev))
    return runs


def mask_statistics(mask, mask_type="random", bucket_min_count=500, bucket_min_fraction=0.01, cfg=None):
    m = np.asarray(mask) > 0.5
    if m.ndim != 3:
        raise ValueError(f"mask must be [B,T,N], got {m.shape}")
    bsz, steps, flows = m.shape
    missing = ~m
    total_missing = int(missing.sum())
    gap_lengths = []
    support = {
        "gap_1_2": 0,
        "gap_3_4": 0,
        "gap_5_8": 0,
        "gap_gt_8": 0,
        "edge": 0,
        "left_edge": 0,
        "right_edge": 0,
        "burst": 0,
        "internal_block": 0,
        "internal": 0,
        "top10_uncertainty": 0,
    }
    all_missing_flows = 0
    no_valid_internal_anchors = 0
    anchor_covered = 0
    burst_lengths = []
    internal_lengths = []
    q_values = np.zeros_like(m, dtype=np.float32)
    for b in range(bsz):
        for n in range(flows):
            obs = np.flatnonzero(m[b, :, n])
            miss = np.flatnonzero(~m[b, :, n])
            if len(obs) == 0:
                all_missing_flows += 1
            runs = contiguous_runs(miss)
            has_internal = False
            for start, end in runs:
                run_len = end - start + 1
                gap_lengths.append(run_len)
                if run_len <= 2:
                    support["gap_1_2"] += run_len
                elif run_len <= 4:
                    support["gap_3_4"] += run_len
                elif run_len <= 8:
                    support["gap_5_8"] += run_len
                else:
                    support["gap_gt_8"] += run_len
                left_obs = obs[obs < start]
                right_obs = obs[obs > end]
                left = int(left_obs[-1]) if len(left_obs) else None
                right = int(right_obs[0]) if len(right_obs) else None
                is_edge = left is None or right is None
                if is_edge:
                    support["edge"] += run_len
                    if left is None:
                        support["left_edge"] += run_len
                    if right is None:
                        support["right_edge"] += run_len
                else:
                    has_internal = True
                    support["internal"] += run_len
                    anchor_covered += run_len
                    internal_lengths.append(run_len)
                    if run_len > 8 or str(mask_type) == "internal_block":
                        support["internal_block"] += run_len
                    if run_len <= max(8, int(getattr(cfg, "burst_max_len", 8) if cfg else 8)):
                        support["burst"] += run_len
                        burst_lengths.append(run_len)
                for t in range(start, end + 1):
                    dist_left = (t - left) if left is not None else steps
                    dist_right = (right - t) if right is not None else steps
                    edge_component = 1.0 if is_edge else 0.0
                    gap_component = math.log1p(run_len) / math.log1p(max(steps, 1))
                    dist_component = min(dist_left, dist_right) / float(max(steps, 1))
                    q_values[b, t, n] = (gap_component + edge_component + dist_component) / 3.0
            if not has_internal:
                no_valid_internal_anchors += 1
    q_missing = q_values[missing]
    if q_missing.size:
        threshold = np.quantile(q_missing, 0.90)
        support["top10_uncertainty"] = int(np.sum(missing & (q_values >= threshold)))
    valid = {}
    for key, count in support.items():
        valid[key] = int(count >= bucket_min_count or (total_missing > 0 and count / float(total_missing) >= bucket_min_fraction))
    gap_arr = np.asarray(gap_lengths, dtype=np.float64)
    mean_gap = float(np.mean(gap_arr)) if gap_arr.size else 0.0
    max_gap = float(np.max(gap_arr)) if gap_arr.size else 0.0
    hist = {
        "gap_hist_1_2": support["gap_1_2"],
        "gap_hist_3_4": support["gap_3_4"],
        "gap_hist_5_8": support["gap_5_8"],
        "gap_hist_gt_8": support["gap_gt_8"],
    }
    return {
        "actual_observed_density": float(m.mean()),
        "actual_missing_density": float(1.0 - m.mean()),
        "observed_budget": int(round(float(m.sum(axis=1).mean()))) if flows and bsz else 0,
        "anchor_coverage_ratio": float(anchor_covered / max(total_missing, 1)),
        "mean_gap_length": mean_gap,
        "max_gap_length": max_gap,
        "long_gap_ratio": float(support["gap_gt_8"] / max(total_missing, 1)),
        "edge_gap_ratio": float(support["edge"] / max(total_missing, 1)),
        "internal_gap_ratio": float(support["internal"] / max(total_missing, 1)),
        "burst_gap_ratio": float(support["burst"] / max(total_missing, 1)),
        "support_gap_1_2": int(support["gap_1_2"]),
        "support_gap_3_4": int(support["gap_3_4"]),
        "support_gap_5_8": int(support["gap_5_8"]),
        "support_gap_gt_8": int(support["gap_gt_8"]),
        "support_edge": int(support["edge"]),
        "support_left_edge": int(support["left_edge"]),
        "support_right_edge": int(support["right_edge"]),
        "support_burst": int(support["burst"]),
        "support_internal_controlled_block": int(support["internal_block"]),
        "support_top10_uncertainty": int(support["top10_uncertainty"]),
        "all_missing_flows": int(all_missing_flows),
        "no_valid_internal_anchors": int(no_valid_internal_anchors),
        "all_missing_flow_ratio": float(all_missing_flows / max(bsz * flows, 1)),
        "bucket_valid_gap_gt_8": valid["gap_gt_8"],
        "bucket_valid_edge": valid["edge"],
        "bucket_valid_burst": valid["burst"],
        "bucket_valid_internal_block": valid["internal_block"],
        "bucket_valid_top10_uncertainty": valid["top10_uncertainty"],
        **hist,
    }
