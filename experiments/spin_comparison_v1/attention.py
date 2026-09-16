"""Exact dense/chunked algebra for the pinned SPIN additive attention.

Each spatial edge has its own temporal softmax; messages are then summed over
incoming spatial edges. This is not dot-product attention or a replacement GAT.
Chunking/checkpointing changes only execution and floating-point summation order.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .primitives import MLP, PyGLinear, TSLNorm


def masked_softmax(logits, allowed):
    if allowed is None:
        allowed = torch.ones_like(logits, dtype=torch.bool)
    allowed = allowed.expand_as(logits)
    active = allowed.any(-2, keepdim=True)
    masked = logits.masked_fill(~allowed, -torch.inf)
    maximum = torch.where(active, masked.amax(-2, keepdim=True), torch.zeros_like(masked[..., :1, :]))
    exp = torch.exp(masked - maximum)
    # Exact pinned TSL epsilon, with zero fallback for an empty source set.
    return exp / (exp.sum(-2, keepdim=True) + 5e-8)


class TemporalAdditiveAttention(nn.Module):
    def __init__(self, input_size, output_size, msg_size=None, msg_layers=1,
                 root_weight=False, reweight="softmax", norm=False, dropout=0., **kwargs):
        super().__init__()
        if reweight != "softmax" or dropout != 0:
            raise ValueError("This bounded adapter implements official spin.yaml only")
        source, target = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.lin_src = PyGLinear(source, output_size, weight_initializer="glorot", bias_initializer="zeros")
        self.lin_tgt = PyGLinear(target, output_size, weight_initializer="glorot", bias=False)
        if root_weight:
            self.lin_skip = PyGLinear(target, output_size, bias=False)
        else:
            self.register_parameter("lin_skip", None)
        self.msg_nn = nn.Sequential(nn.PReLU(init=.2),
            MLP(output_size, msg_size or output_size, output_size,
                n_layers=msg_layers, activation="prelu", dropout=dropout))
        self.msg_gate = nn.Linear(output_size, 1, bias=False)
        self.norm = nn.LayerNorm(output_size) if norm else None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def pair_read(self, source, target, source_mask=None, temporal_mask=None):
        """Projected source/target [..., time, channels] -> target read."""
        message = self.msg_nn(source.unsqueeze(-3) + target.unsqueeze(-2))
        allowed = None if source_mask is None else source_mask.unsqueeze(-3)
        if temporal_mask is not None:
            shape = [1] * (message.ndim - 3) + [*temporal_mask.shape, 1]
            temporal_mask = temporal_mask.reshape(shape)
            allowed = temporal_mask if allowed is None else allowed & temporal_mask
        weights = masked_softmax(self.msg_gate(message), allowed)
        return (message * weights).sum(-2)

    def forward(self, x, mask=None, temporal_mask=None):
        source, target = (x, x) if isinstance(x, torch.Tensor) else x
        # Official public shape [B,T,N,D]; independent temporal groups become N.
        src = self.lin_src(source).transpose(1, 2)
        tgt = self.lin_tgt(target).transpose(1, 2)
        allowed = None if mask is None else mask.bool().transpose(1, 2)
        out = self.pair_read(src, tgt, allowed, temporal_mask).transpose(1, 2)
        if self.lin_skip is not None:
            out = out + self.lin_skip(target)
        return self.norm(out) if self.norm is not None else out


class TemporalGraphAdditiveAttention(nn.Module):
    def __init__(self, input_size, output_size, msg_size=None, msg_layers=1,
                 root_weight=True, reweight=None, temporal_self_attention=True,
                 mask_temporal=True, mask_spatial=True, norm=True, dropout=0., **kwargs):
        super().__init__()
        args = dict(input_size=input_size, output_size=output_size, msg_size=msg_size,
                    msg_layers=msg_layers, reweight=reweight, dropout=dropout,
                    root_weight=False, norm=False)
        self.mask_temporal, self.mask_spatial = mask_temporal, mask_spatial
        self.self_attention = TemporalAdditiveAttention(**args) if temporal_self_attention else None
        self.cross_attention = TemporalAdditiveAttention(**args)
        target = input_size if isinstance(input_size, int) else input_size[1]
        self.lin_skip = PyGLinear(target, output_size, bias_initializer="zeros") if root_weight else None
        self.norm = TSLNorm(output_size) if norm else None
        self.node_chunk = 8
        self.gradient_checkpointing = True
        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        if self.self_attention is not None:
            self.self_attention.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def _cross_chunk(self, source, target, source_mask, valid_edges):
        out = self.cross_attention.pair_read(source, target.unsqueeze(2), source_mask)
        return (out * valid_edges[None, :, :, None, None]).sum(2)

    def _self_chunk(self, source, target, source_mask, temporal_mask):
        return self.self_attention.pair_read(source, target, source_mask, temporal_mask)

    def forward(self, x, edge_index, edge_weight=None, mask=None):
        if not isinstance(x, torch.Tensor) or edge_weight is not None:
            raise ValueError("Bounded SPIN adapter uses one full unweighted graph")
        # Adapter passes a padded incoming-neighbor table, excluding self.
        incoming = edge_index
        if incoming.ndim != 2 or incoming.shape[0] != x.shape[2]:
            raise ValueError("Expected incoming-neighbor table [N,K]")
        source = self.cross_attention.lin_src(x).transpose(1, 2)
        target = self.cross_attention.lin_tgt(x).transpose(1, 2)
        chunks = []
        do_checkpoint = self.training and self.gradient_checkpointing and torch.is_grad_enabled()
        for first in range(0, x.shape[2], self.node_chunk):
            ids = incoming[first:first+self.node_chunk]
            valid = ids >= 0
            safe = ids.clamp_min(0)
            src = source[:, safe]
            tgt = target[:, first:first+len(ids)]
            allowed = valid[None, :, :, None, None].expand(x.shape[0], -1, -1, x.shape[1], 1)
            if self.mask_spatial and mask is not None:
                allowed = allowed & mask.transpose(1, 2)[:, safe].bool()
            if do_checkpoint:
                out = checkpoint(self._cross_chunk, src, tgt, allowed, valid, use_reentrant=False)
            else:
                out = self._cross_chunk(src, tgt, allowed, valid)
            chunks.append(out)
        out = torch.cat(chunks, dim=1).transpose(1, 2)
        if self.self_attention is not None:
            source = self.self_attention.lin_src(x).transpose(1, 2)
            target = self.self_attention.lin_tgt(x).transpose(1, 2)
            temporal = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
            chunks = []
            for first in range(0, x.shape[2], self.node_chunk):
                src, tgt = source[:, first:first+self.node_chunk], target[:, first:first+self.node_chunk]
                allowed = None
                if self.mask_temporal and mask is not None:
                    allowed = mask.transpose(1, 2)[:, first:first+self.node_chunk].bool()
                if do_checkpoint:
                    result = checkpoint(self._self_chunk, src, tgt, allowed, temporal, use_reentrant=False)
                else:
                    result = self._self_chunk(src, tgt, allowed, temporal)
                chunks.append(result)
            out = out + torch.cat(chunks, dim=1).transpose(1, 2)
        if self.lin_skip is not None:
            out = out + self.lin_skip(x)
        return self.norm(out) if self.norm is not None else out
