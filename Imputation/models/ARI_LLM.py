from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from models.care_modules import GlobalCouplingExpert
from models.geoanchor_modules import (
    GEOANCHOR_VARIANTS,
    GeoFlow2Vec,
    ObservationGeometryExtractor,
    is_geoanchor_variant,
)
from models.geoanchor_v2_modules import (
    ACIL_VARIANTS,
    AnchorConditionedInterpolationLayer,
    ObservationGeometryExtractorV2,
    is_acil_variant,
    parameter_delta_from_state,
    preserve_torch_rng,
    tensor_mean_dict,
)
from models.geoattn_modules import GeoAttnBlock, is_geoattn_variant
from models.salt_modules import (
    GlobalResidualHead,
    LatentTrafficCoupler,
    SameParamMLPControl,
    StageAwareUncertaintyGate,
    stage_observed_mask_like,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class FlattenHead(nn.Module):
    def __init__(self, d_model, Tstep, zero_init=False):
        super().__init__()
        self.Tstep = Tstep
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(384, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, self.Tstep),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        B, N, d_model = x.shape
        x = x.reshape(B * N, d_model)
        x = self.mlp(x)
        return x.reshape(B, N, self.Tstep)


class LowRankExpertHead(nn.Module):
    def __init__(self, d_model, t_step, rank, experts, hidden, dropout, zero_init=False):
        super().__init__()
        self.t_step = int(t_step)
        self.rank = max(1, int(rank))
        self.experts = max(1, int(experts))
        hidden = max(1, int(hidden))
        out_dim = self.experts * self.rank

        self.token_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)
        self.coeff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, out_dim),
        )
        self.basis = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, out_dim * self.t_step),
        )
        self.gate = nn.Linear(d_model, self.experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if zero_init:
            nn.init.zeros_(self.basis[-1].weight)
            nn.init.zeros_(self.basis[-1].bias)

    def forward(self, x):
        bsz, n_tokens, _ = x.shape
        token_features = self.token_norm(x)
        global_feature = self.global_norm(x.mean(dim=1))
        coeff = self.coeff(token_features).view(bsz, n_tokens, self.experts, self.rank)
        basis = self.basis(global_feature).view(bsz, self.experts, self.rank, self.t_step)
        expert_out = torch.einsum('bner,bert->bnet', coeff, basis)
        gate = torch.softmax(self.gate(global_feature), dim=-1).view(bsz, 1, self.experts, 1)
        return torch.sum(gate * expert_out, dim=2)


class MultiScaleTemporalAdapter(nn.Module):
    def __init__(self, hidden, kernel_sizes, dropout, scale):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        invalid = [k for k in self.kernel_sizes if k <= 0 or k % 2 == 0]
        if invalid:
            raise ValueError(f"temporal_adapter_kernels must be positive odd integers, got {invalid}")

        hidden = max(1, int(hidden))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Conv1d(hidden * len(self.kernel_sizes), 1, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        bsz, steps, flows = x.shape
        per_flow = x.permute(0, 2, 1).contiguous().view(bsz * flows, 1, steps)
        features = torch.cat([branch(per_flow) for branch in self.branches], dim=1)
        residual = self.proj(self.dropout(features))
        residual = residual.view(bsz, flows, steps).permute(0, 2, 1).contiguous()
        return x + self.scale * residual


class DepthRouter(nn.Module):
    def __init__(self, feature_dim, num_layers, hidden, dropout):
        super().__init__()
        hidden = max(1, int(hidden))
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, num_layers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return torch.softmax(self.net(features), dim=-1)


class FlowBiasAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, scale, zero_init=False, use_ffn=True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(d_model, d_model)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(d_ff, d_model),
            )
        else:
            self.norm2 = None
            self.ffn = None
        self.residual_scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            if self.ffn is not None:
                nn.init.zeros_(self.ffn[-1].weight)
                nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, attention_bias):
        bsz, n_tokens, d_model = x.shape
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).view(bsz, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            scores = scores + attention_bias.unsqueeze(0)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, n_tokens, d_model)
        x = residual + self.residual_scale * self.out_proj(context)
        if self.ffn is not None:
            x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x


class FlowBiasAttentionAdapter(nn.Module):
    NUM_RELATIONS = 6
    STAGE_TARGETS = (4.0, 8.0, 16.0, 32.0, 64.0, 100.0)

    def __init__(self, configs, router_count, relation_matrix):
        super().__init__()
        self.router_count = router_count
        self.n_heads = int(getattr(configs, 'flow_attention_heads', 4))
        self.use_stage_bias = bool(getattr(configs, 'flow_attention_stage_bias', 1))
        self.local_only = bool(getattr(configs, 'flow_attention_local_only', 0))
        self.register_buffer('relation_matrix', relation_matrix.long(), persistent=False)
        self.relation_bias = nn.Parameter(torch.zeros(self.NUM_RELATIONS, self.n_heads))
        bias_init = float(getattr(configs, 'flow_attention_bias_init', 0.05))
        with torch.no_grad():
            self.relation_bias[1].fill_(bias_init)
            self.relation_bias[2].fill_(bias_init)
            self.relation_bias[3].fill_(bias_init)
            self.relation_bias[4].fill_(bias_init * 0.5)
            self.relation_bias[5].fill_(-bias_init * 0.25)

        if self.use_stage_bias:
            self.stage_bias = nn.Parameter(torch.zeros(len(self.STAGE_TARGETS), self.NUM_RELATIONS, self.n_heads))
        else:
            self.stage_bias = None

        self.layers = nn.ModuleList([
            FlowBiasAttentionLayer(
                d_model=configs.d_model,
                n_heads=self.n_heads,
                d_ff=configs.d_ff,
                dropout=getattr(configs, 'flow_attention_dropout', 0.1),
                scale=getattr(configs, 'flow_attention_init', 0.1),
                zero_init=bool(getattr(configs, 'flow_attention_zero_init', 0)),
                use_ffn=bool(getattr(configs, 'flow_attention_ffn', 1)),
            )
            for _ in range(int(getattr(configs, 'flow_attention_layers', 1)))
        ])

    def _stage_index(self, target_rate):
        target = float(target_rate)
        distances = [abs(target - item) for item in self.STAGE_TARGETS]
        return min(range(len(distances)), key=distances.__getitem__)

    def _attention_bias(self, n_tokens, target_rate, device):
        rel = self.relation_matrix[:n_tokens, :n_tokens].to(device)
        bias = self.relation_bias[rel].permute(2, 0, 1)
        if self.stage_bias is not None:
            stage = self._stage_index(target_rate)
            bias = bias + self.stage_bias[stage][rel].permute(2, 0, 1)
        if self.local_only:
            bias = bias.masked_fill((rel == 5).unsqueeze(0), -1e4)
        return bias

    def forward(self, x, target_rate):
        attention_bias = self._attention_bias(x.shape[1], target_rate, x.device)
        for layer in self.layers:
            x = layer(x, attention_bias)
        return x


class ReliabilityAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        scale,
        key_bias_init,
        coobs_bias_init,
        zero_init=False,
        use_ffn=True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(d_model, d_model)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(d_ff, d_model),
            )
        else:
            self.norm2 = None
            self.ffn = None
        self.residual_scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))
        self.key_bias_scale = nn.Parameter(torch.tensor(float(key_bias_init), dtype=torch.float32))
        self.coobs_bias_scale = nn.Parameter(torch.tensor(float(coobs_bias_init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            if self.ffn is not None:
                nn.init.zeros_(self.ffn[-1].weight)
                nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x, reliability, coobs):
        bsz, n_tokens, d_model = x.shape
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).view(bsz, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel = reliability.clamp(0.0, 1.0)
        uncertain_query = 1.0 - rel
        reliability_bias = uncertain_query.unsqueeze(-1) * rel.unsqueeze(-2)
        scores = scores + self.key_bias_scale * reliability_bias.unsqueeze(1)
        if coobs is not None:
            scores = scores + self.coobs_bias_scale * coobs.unsqueeze(1)

        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, n_tokens, d_model)
        x = residual + self.residual_scale * self.out_proj(context)
        if self.ffn is not None:
            x = x + self.residual_scale * self.ffn(self.norm2(x))
        return x


class ReliabilityAttentionAdapter(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.use_coobs = bool(getattr(configs, 'reliability_attention_coobs', 1))
        self.layers = nn.ModuleList([
            ReliabilityAttentionLayer(
                d_model=configs.d_model,
                n_heads=int(getattr(configs, 'reliability_attention_heads', 4)),
                d_ff=configs.d_ff,
                dropout=float(getattr(configs, 'reliability_attention_dropout', 0.1)),
                scale=float(getattr(configs, 'reliability_attention_init', 0.1)),
                key_bias_init=float(getattr(configs, 'reliability_attention_key_bias_init', 0.05)),
                coobs_bias_init=float(getattr(configs, 'reliability_attention_coobs_bias_init', 0.02)),
                zero_init=bool(getattr(configs, 'reliability_attention_zero_init', 0)),
                use_ffn=bool(getattr(configs, 'reliability_attention_ffn', 1)),
            )
            for _ in range(int(getattr(configs, 'reliability_attention_layers', 1)))
        ])

    def _stats(self, observed_mask, n_tokens, device, dtype):
        mask = observed_mask.to(device=device, dtype=dtype)
        if mask.shape[-1] < n_tokens:
            mask = F.pad(mask, (0, n_tokens - mask.shape[-1]))
        elif mask.shape[-1] > n_tokens:
            mask = mask[..., :n_tokens]
        flow_mask = mask.transpose(1, 2).contiguous()
        reliability = flow_mask.mean(dim=-1)
        if not self.use_coobs:
            return reliability, None
        intersection = torch.bmm(flow_mask, flow_mask.transpose(1, 2))
        counts = flow_mask.sum(dim=-1)
        union = counts.unsqueeze(-1) + counts.unsqueeze(-2) - intersection
        coobs = intersection / (union + 1e-6)
        return reliability, coobs

    def forward(self, x, observed_mask):
        reliability, coobs = self._stats(observed_mask, x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, reliability, coobs)
        return x


class StageAwareResidualRefiner(nn.Module):
    NUM_RELATIONS = 6
    STAGE_TARGETS = (4.0, 8.0, 16.0, 32.0, 64.0, 100.0)

    def __init__(self, configs, relation_matrix):
        super().__init__()
        self.t_step = int(configs.enc_in)
        self.n_heads = int(getattr(configs, 'residual_refine_topology_heads', 4))
        self.use_stage = bool(getattr(configs, 'residual_refine_stage', 1))
        self.use_gate = bool(getattr(configs, 'residual_refine_gate', 1))
        self.anchor_observed = bool(getattr(configs, 'residual_refine_anchor_observed', 1))
        self.local_only = bool(getattr(configs, 'residual_refine_topology_local_only', 1))
        self.use_topology_stage_bias = bool(getattr(configs, 'residual_refine_topology_stage_bias', 1))
        self.register_buffer('relation_matrix', relation_matrix.long(), persistent=False)

        if self.use_stage:
            self.stage_proj = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model),
            )
        else:
            self.stage_proj = None

        self.flow_stat_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

        topology_layers = int(getattr(configs, 'residual_refine_topology_layers', 0))
        if topology_layers > 0:
            self.relation_bias = nn.Parameter(torch.zeros(self.NUM_RELATIONS, self.n_heads))
            bias_init = float(getattr(configs, 'residual_refine_topology_bias_init', 0.03))
            with torch.no_grad():
                self.relation_bias[1].fill_(bias_init)
                self.relation_bias[2].fill_(bias_init)
                self.relation_bias[3].fill_(bias_init)
                self.relation_bias[4].fill_(bias_init * 0.5)
                self.relation_bias[5].fill_(-bias_init * 0.25)
            if self.use_topology_stage_bias:
                self.stage_bias = nn.Parameter(torch.zeros(len(self.STAGE_TARGETS), self.NUM_RELATIONS, self.n_heads))
            else:
                self.stage_bias = None
            self.topology_layers = nn.ModuleList([
                FlowBiasAttentionLayer(
                    d_model=configs.d_model,
                    n_heads=self.n_heads,
                    d_ff=configs.d_ff,
                    dropout=float(getattr(configs, 'residual_refine_topology_dropout', 0.1)),
                    scale=float(getattr(configs, 'residual_refine_topology_init', 0.05)),
                    zero_init=bool(getattr(configs, 'residual_refine_topology_zero_init', 1)),
                    use_ffn=bool(getattr(configs, 'residual_refine_topology_ffn', 1)),
                )
                for _ in range(topology_layers)
            ])
        else:
            self.relation_bias = None
            self.stage_bias = None
            self.topology_layers = None

        self.out_norm = nn.LayerNorm(configs.d_model)
        self.residual_head = FlattenHead(configs.d_model, self.t_step, zero_init=True)
        if self.use_gate:
            gate_hidden = max(32, configs.d_model // 2)
            self.gate_norm = nn.LayerNorm(configs.d_model)
            self.gate_head = nn.Sequential(
                nn.Linear(configs.d_model, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.t_step),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.zeros_(self.gate_head[-1].bias)
        else:
            self.gate_norm = None
            self.gate_head = None

    def _observed_indices_for_rate(self, length, rate, device):
        if float(rate) >= 100.0:
            return torch.arange(length, device=device)
        count = int(torch.ceil(torch.tensor(length * float(rate) / 100.0)).item())
        if length > 1:
            count = max(2, count)
        count = min(length, max(1, count))
        idx = torch.linspace(0, length - 1, count, device=device).round().long().unique(sorted=True)
        if idx[-1].item() != length - 1:
            idx = torch.cat([idx, torch.tensor([length - 1], device=device)])
        if idx[0].item() != 0:
            idx = torch.cat([torch.tensor([0], device=device), idx])
        return idx.unique(sorted=True)

    def _mask_for_stats(self, observed_mask, x_current, known_rate):
        bsz, steps, flows = x_current.shape
        if observed_mask is None:
            mask = x_current.new_zeros((1, steps, 1))
            idx = self._observed_indices_for_rate(steps, known_rate, x_current.device)
            mask[:, idx, :] = 1.0
            return mask.expand(bsz, steps, flows)
        mask = observed_mask.to(device=x_current.device, dtype=x_current.dtype)
        if mask.shape[1] < steps:
            mask = F.pad(mask, (0, 0, 0, steps - mask.shape[1]))
        elif mask.shape[1] > steps:
            mask = mask[:, :steps, :]
        if mask.shape[2] < flows:
            mask = F.pad(mask, (0, flows - mask.shape[2]))
        elif mask.shape[2] > flows:
            mask = mask[:, :, :flows]
        return mask

    def _stage_index(self, target_rate):
        target = float(target_rate)
        distances = [abs(target - item) for item in self.STAGE_TARGETS]
        return min(range(len(distances)), key=distances.__getitem__)

    def _attention_bias(self, n_tokens, target_rate, device):
        if self.relation_bias is None:
            return None
        rel = self.relation_matrix[:n_tokens, :n_tokens].to(device)
        bias = self.relation_bias[rel].permute(2, 0, 1)
        if self.stage_bias is not None:
            stage = self._stage_index(target_rate)
            bias = bias + self.stage_bias[stage][rel].permute(2, 0, 1)
        if self.local_only:
            bias = bias.masked_fill((rel == 5).unsqueeze(0), -1e4)
        return bias

    def forward(self, hidden, x_current, observed_mask, known_rate, target_rate):
        bsz, n_tokens, _ = hidden.shape
        mask = self._mask_for_stats(observed_mask, x_current, known_rate)
        flow_mask = mask.transpose(1, 2).contiguous()
        current = x_current.transpose(1, 2).contiguous()
        reliability = flow_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        uncertainty = 1.0 - reliability
        abs_mean = current.detach().abs().mean(dim=-1, keepdim=True)
        scale = current.detach().std(dim=-1, unbiased=False, keepdim=True)
        hidden = hidden + self.flow_stat_proj(torch.cat([reliability, uncertainty, abs_mean, scale], dim=-1))

        if self.stage_proj is not None:
            density = mask.mean(dim=(1, 2), keepdim=True).view(bsz, 1)
            known = torch.full((bsz, 1), float(known_rate) / 100.0, device=hidden.device, dtype=hidden.dtype)
            target = torch.full((bsz, 1), float(target_rate) / 100.0, device=hidden.device, dtype=hidden.dtype)
            delta = (target - known).clamp_min(0.0)
            stage = self.stage_proj(torch.cat([known, target, delta, density.to(hidden.dtype)], dim=-1))
            hidden = hidden + stage.unsqueeze(1)

        if self.topology_layers is not None:
            attention_bias = self._attention_bias(n_tokens, target_rate, hidden.device)
            for layer in self.topology_layers:
                hidden = layer(hidden, attention_bias)

        residual = self.residual_head(self.out_norm(hidden))
        if self.gate_head is not None:
            gate = 2.0 * torch.sigmoid(self.gate_head(self.gate_norm(hidden)))
            residual = residual * gate
        if self.anchor_observed:
            residual = residual * (1.0 - flow_mask)
        return residual


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.configs = configs
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_roots = {
            'gpt2': self.project_root / 'GPT2',
            'deepseek_R1': self.project_root / 'deepseek_R1_1.5b',
            'Llama_3': self.project_root / 'Llama',
        }

        self.variant = str(getattr(configs, 'variant', 'default'))
        self.geoattn_variant = is_geoattn_variant(self.variant)
        self.geoattn_enabled = bool(getattr(configs, 'geoattn_enable', 0)) or self.geoattn_variant
        if self.variant == 'acil_geoattn_true':
            configs.geoattn_bias_type = 'true'
        elif self.variant == 'acil_geoattn_shuffled_bias':
            configs.geoattn_bias_type = 'shuffled'
        elif self.variant == 'acil_geoattn_random_bias':
            configs.geoattn_bias_type = 'random'
        elif self.variant == 'acil_geoattn_zero_bias':
            configs.geoattn_bias_type = 'zero'
        elif self.variant == 'acil_geoattn_same_param_control':
            configs.geoattn_bias_type = 'true'
            configs.geoattn_same_param_control = 1
        self.acil_prior_only_variants = {
            'acil_prior_only',
            'linearinterp_prior_diagnostic',
            'acil_prior_only_full',
            'acil_prior_only_value_sameparam',
            'acil_prior_only_shuffled_gap_geometry',
            'acil_prior_only_no_anchor_values',
        }
        self.geoanchor_variant = is_geoanchor_variant(self.variant)
        self.acil_variant = is_acil_variant(self.variant)
        self.use_acil = (
            bool(getattr(configs, 'acil_enable', 0))
            or bool(getattr(configs, 'geoanchor_v2_enable', 0))
            or self.acil_variant
            or self.geoattn_variant
            or self.geoattn_enabled
        )
        self.acil_prior_only = bool(getattr(configs, 'acil_prior_only', 0)) or self.variant in self.acil_prior_only_variants
        self.acil_freeze_to_linear = (
            bool(getattr(configs, 'acil_freeze_to_linear', 0))
            or self.variant == 'acil_frozen_linear'
            or self.variant == 'linearinterp_prior_diagnostic'
        )
        self.acil_shuffle_gap_geometry = (
            bool(getattr(configs, 'acil_shuffle_gap_geometry', 0))
            or self.variant == 'acil_shuffled_gap_geometry'
            or self.variant == 'acil_prior_only_shuffled_gap_geometry'
        )
        self.acil_shuffle_anchor_values = (
            bool(getattr(configs, 'acil_shuffle_anchor_values', 0))
            or self.variant == 'acil_shuffled_anchor_values'
        )
        self.acil_feature_set = str(getattr(configs, 'acil_feature_set', 'full'))
        if self.variant in {'acil_value_only_sameparam', 'acil_prior_only_value_sameparam'}:
            self.acil_feature_set = 'value_only'
        elif self.variant in {'acil_no_anchor_values', 'acil_prior_only_no_anchor_values'}:
            self.acil_feature_set = 'no_anchor_values'
        elif self.variant in ACIL_VARIANTS or self.geoattn_variant:
            self.acil_feature_set = 'full'
        self.use_geoanchor = bool(getattr(configs, 'geoanchor_enable', 0)) or self.geoanchor_variant
        self.geoanchor_shuffle_geometry = (
            bool(getattr(configs, 'geoanchor_shuffle_geometry', 0))
            or self.variant == 'agt_shuffled_geometry'
        )
        self.geoanchor_feature_corruption = str(getattr(configs, 'geoanchor_feature_corruption', 'none'))
        self.geoanchor_feature_set = str(getattr(configs, 'geoanchor_feature_set', 'full'))
        if self.variant == 'value_extra_control':
            self.geoanchor_feature_set = 'value_extra'
        elif self.variant == 'mask_only_control':
            self.geoanchor_feature_set = 'mask_only'
        elif self.variant == 'agt_no_anchor_values':
            self.geoanchor_feature_set = 'no_anchor_values'
        self.care_global_variant = self.variant in {
            'care_global',
            'global_expert_only',
            'care_same_param_control',
            'care_global_control',
        }
        self.care_local_variant = (
            self.variant in {'care_local', 'stagegate_local', 'stagegate_baseline'}
            or self.variant in GEOANCHOR_VARIANTS
            or (self.variant in ACIL_VARIANTS and self.variant not in self.acil_prior_only_variants)
            or self.geoattn_variant
            or self.geoattn_enabled
        )
        self.care_use_same_param_control = self.variant in {
            'care_same_param_control',
            'care_global_control',
        }
        self.salt_global_variant = self.variant in {
            'salt_latent_only',
            'salt_gate',
            'salt_full',
            'same_param_mlp_control',
        } or bool(getattr(configs, 'salt_same_param_control', 0))
        self.salt_no_global_variant = self.variant == 'salt_no_global'
        self.use_residual_refine_block = (
            bool(getattr(configs, 'residual_refine_block', 0))
            or self.salt_global_variant
            or self.salt_no_global_variant
            or self.care_local_variant
        )
        self.use_residual_refine = bool(getattr(configs, 'residual_refine', 0)) or self.use_residual_refine_block
        self.use_residual_refine_blend = self.use_residual_refine_block and bool(getattr(configs, 'residual_refine_blend', 0))
        self.residual_refine_scale = None
        if self.use_residual_refine:
            self.residual_refine_scale = nn.Parameter(
                torch.tensor(float(getattr(configs, 'residual_refine_scale', 1.0)), dtype=torch.float32)
            )
        head_zero_init = bool(getattr(configs, 'head_zero_init', 0)) or (
            self.use_residual_refine and not self.use_residual_refine_blend
        )
        self.flattenhead = FlattenHead(configs.d_model, self.configs.enc_in, zero_init=head_zero_init)
        self.use_low_rank_head = bool(getattr(configs, 'low_rank_head', 0))
        self.low_rank_replace = bool(getattr(configs, 'low_rank_replace', 0))
        if self.use_low_rank_head:
            self.low_rank_head = LowRankExpertHead(
                d_model=configs.d_model,
                t_step=self.configs.enc_in,
                rank=getattr(configs, 'low_rank_dim', 16),
                experts=getattr(configs, 'low_rank_experts', 1),
                hidden=getattr(configs, 'low_rank_hidden', 256),
                dropout=getattr(configs, 'low_rank_dropout', 0.1),
                zero_init=not self.low_rank_replace,
            )
            self.low_rank_scale = nn.Parameter(
                torch.tensor(float(getattr(configs, 'low_rank_scale', 1.0)), dtype=torch.float32)
            )
        else:
            self.low_rank_head = None
            self.low_rank_scale = None
        self.use_depth_router = bool(getattr(configs, 'depth_router', 0))
        self.depth_router_layers = self._parse_depth_router_layers(
            getattr(configs, 'depth_router_layers', '1,2,3,6')
        )
        if self.use_depth_router:
            if not self.depth_router_layers:
                raise ValueError("depth_router_layers must contain at least one layer when depth_router=1")
            max_layer = max(self.depth_router_layers)
            if max_layer > int(configs.gpt_layers):
                raise ValueError(
                    f"depth_router_layers={self.depth_router_layers} require gpt_layers >= {max_layer}, "
                    f"got {configs.gpt_layers}"
                )
            self.depth_router = DepthRouter(
                feature_dim=6,
                num_layers=len(self.depth_router_layers),
                hidden=getattr(configs, 'depth_router_hidden', 64),
                dropout=getattr(configs, 'depth_router_dropout', 0.1),
            )
            self.last_depth_router_weights = None
        else:
            self.depth_router = None
            self.last_depth_router_weights = None
        self.use_residual_head = bool(getattr(configs, 'residual_head', 0))
        if self.use_residual_head:
            self.residual_head = FlattenHead(configs.d_model, self.configs.enc_in, zero_init=True)
            self.residual_scale = nn.Parameter(
                torch.tensor(float(getattr(configs, 'residual_scale', 1.0)), dtype=torch.float32)
            )
        else:
            self.residual_head = None
            self.residual_scale = None
        if bool(getattr(configs, 'temporal_adapter', 0)):
            kernel_sizes = self._parse_temporal_kernels(
                getattr(configs, 'temporal_adapter_kernels', '3,5,7')
            )
            self.temporal_adapter = MultiScaleTemporalAdapter(
                hidden=getattr(configs, 'temporal_adapter_hidden', 8),
                kernel_sizes=kernel_sizes,
                dropout=getattr(configs, 'temporal_adapter_dropout', 0.1),
                scale=getattr(configs, 'temporal_adapter_scale', 1.0),
            )
        else:
            self.temporal_adapter = None
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            flow_encoder=getattr(configs, 'flow_encoder', 'lstm'),
            flow_encoder_layers=getattr(configs, 'flow_encoder_layers', 1),
            flow_patch_len=getattr(configs, 'flow_patch_len', 10),
            flow_patch_stride=getattr(configs, 'flow_patch_stride', 5),
            flow_kernel_size=getattr(configs, 'flow_kernel_size', 5),
            flow_encoder_heads=getattr(configs, 'flow_encoder_heads', 4),
        )
        if self.use_geoanchor:
            self.geoanchor_extractor = ObservationGeometryExtractor(
                feature_set=self.geoanchor_feature_set,
                uncertainty=getattr(configs, 'geoanchor_uncertainty', 'equal_weight'),
            )
            self.geo_flow_embedding = GeoFlow2Vec(
                feature_dim=self.geoanchor_extractor.feature_dim,
                d_model=configs.d_model,
                t_steps=configs.enc_in,
                dropout=configs.dropout,
                geo_hidden=getattr(configs, 'geoanchor_hidden', 64),
            )
        else:
            self.geoanchor_extractor = None
            self.geo_flow_embedding = None
        self.last_geoanchor_hidden = None
        self.use_mask_aware = bool(getattr(configs, 'mask_aware', 0))
        self.mask_fusion = getattr(configs, 'mask_fusion', 'add')
        if self.mask_fusion not in ('add', 'concat', 'gate'):
            raise ValueError(f"Unsupported mask_fusion: {self.mask_fusion}")
        if self.use_mask_aware:
            if self.mask_fusion == 'concat':
                self.mask_concat_embedding = DataEmbedding(
                    configs.enc_in * 2,
                    configs.d_model,
                    configs.embed,
                    configs.freq,
                    configs.dropout,
                    flow_encoder=getattr(configs, 'flow_encoder', 'lstm'),
                    flow_encoder_layers=getattr(configs, 'flow_encoder_layers', 1),
                    flow_patch_len=getattr(configs, 'flow_patch_len', 10),
                    flow_patch_stride=getattr(configs, 'flow_patch_stride', 5),
                    flow_kernel_size=getattr(configs, 'flow_kernel_size', 5),
                    flow_encoder_heads=getattr(configs, 'flow_encoder_heads', 4),
                )
                self.mask_embedding = None
                self.mask_scale = None
                self.mask_gate = None
            else:
                self.mask_concat_embedding = None
                self.mask_embedding = DataEmbedding(
                    configs.enc_in,
                    configs.d_model,
                    configs.embed,
                    configs.freq,
                    configs.dropout,
                    flow_encoder=getattr(configs, 'flow_encoder', 'lstm'),
                    flow_encoder_layers=getattr(configs, 'flow_encoder_layers', 1),
                    flow_patch_len=getattr(configs, 'flow_patch_len', 10),
                    flow_patch_stride=getattr(configs, 'flow_patch_stride', 5),
                    flow_kernel_size=getattr(configs, 'flow_kernel_size', 5),
                    flow_encoder_heads=getattr(configs, 'flow_encoder_heads', 4),
                )
                self.mask_scale = nn.Parameter(
                    torch.tensor(float(getattr(configs, 'mask_embed_scale', 0.05)), dtype=torch.float32)
                )
                if self.mask_fusion == 'gate':
                    self.mask_gate = nn.Linear(configs.d_model, configs.d_model)
                    nn.init.zeros_(self.mask_gate.weight)
                    nn.init.zeros_(self.mask_gate.bias)
                else:
                    self.mask_gate = None
        else:
            self.mask_concat_embedding = None
            self.mask_embedding = None
            self.mask_scale = None
            self.mask_gate = None

        self.mask_prompt_tokens = int(getattr(configs, 'mask_prompt_tokens', 0))
        self.mask_prompt_mode = getattr(configs, 'mask_prompt_mode', 'mask')
        if self.mask_prompt_mode not in ('mask', 'learned'):
            raise ValueError(f"Unsupported mask_prompt_mode: {self.mask_prompt_mode}")
        if self.mask_prompt_tokens > 0:
            self.mask_prompt_base = nn.Parameter(torch.zeros(1, self.mask_prompt_tokens, configs.d_model))
            self.mask_prompt_scale = nn.Parameter(
                torch.tensor(float(getattr(configs, 'mask_prompt_scale', 0.05)), dtype=torch.float32)
            )
            self.mask_prompt_norm = nn.LayerNorm(configs.d_model)
            if self.mask_prompt_mode == 'mask':
                self.mask_prompt_encoder = nn.Sequential(
                    nn.Linear(configs.c_out, configs.d_model),
                    nn.GELU(),
                    nn.Dropout(float(getattr(configs, 'mask_prompt_dropout', 0.1))),
                    nn.Linear(configs.d_model, configs.d_model),
                )
            else:
                self.mask_prompt_encoder = None
        else:
            self.mask_prompt_base = None
            self.mask_prompt_scale = None
            self.mask_prompt_norm = None
            self.mask_prompt_encoder = None
        self.use_topology_embed = bool(getattr(configs, 'topology_embed', 0))
        self.topology_scale_value = float(getattr(configs, 'topology_init', 0.1))
        self.topology_router_count = self._infer_router_count()
        self.topology_exclude_self = self._infer_exclude_self_flows()
        topology_flow_count = self._topology_flow_count(self.topology_router_count, self.topology_exclude_self)
        self.max_flow_count = max(int(configs.c_out), topology_flow_count, 1)
        if self.use_topology_embed and self.topology_router_count > 0:
            self.src_embedding = nn.Embedding(self.topology_router_count, configs.d_model)
            self.dst_embedding = nn.Embedding(self.topology_router_count, configs.d_model)
            self.pair_embedding = nn.Embedding(self.max_flow_count, configs.d_model)
            self.topology_norm = nn.LayerNorm(configs.d_model)
            self.topology_scale = nn.Parameter(torch.tensor(self.topology_scale_value, dtype=torch.float32))
        else:
            self.src_embedding = None
            self.dst_embedding = None
            self.pair_embedding = None
            self.topology_norm = None
            self.topology_scale = None

        if bool(getattr(configs, 'flow_attention_layers', 0)) and self.topology_router_count > 0:
            self.flow_attention = FlowBiasAttentionAdapter(
                configs,
                self.topology_router_count,
                self._flow_relation_matrix(max(int(configs.c_out), 1)),
            )
        else:
            self.flow_attention = None

        if int(getattr(configs, 'reliability_attention_layers', 0)) > 0:
            self.reliability_attention = ReliabilityAttentionAdapter(configs)
        else:
            self.reliability_attention = None

        if self.use_residual_refine_block:
            self.residual_refine_block = StageAwareResidualRefiner(
                configs,
                self._flow_relation_matrix(max(int(configs.c_out), 1)),
            )
            if self.use_residual_refine_blend:
                self.residual_blend_use_stage = bool(getattr(configs, 'residual_refine_blend_stage', 1))
                self.residual_blend_flow_stat_proj = nn.Sequential(
                    nn.LayerNorm(4),
                    nn.Linear(4, configs.d_model),
                    nn.GELU(),
                    nn.Linear(configs.d_model, configs.d_model),
                )
                if self.residual_blend_use_stage:
                    self.residual_blend_stage_proj = nn.Sequential(
                        nn.LayerNorm(4),
                        nn.Linear(4, configs.d_model),
                        nn.GELU(),
                        nn.Linear(configs.d_model, configs.d_model),
                    )
                else:
                    self.residual_blend_stage_proj = None
                blend_hidden = max(32, configs.d_model // 2)
                self.residual_blend_norm = nn.LayerNorm(configs.d_model)
                self.residual_blend_head = nn.Sequential(
                    nn.Linear(configs.d_model, blend_hidden),
                    nn.GELU(),
                    nn.Linear(blend_hidden, self.configs.enc_in),
                )
                init = float(getattr(configs, 'residual_refine_blend_init', 0.5))
                init = min(max(init, 1e-4), 1.0 - 1e-4)
                init_logit = torch.logit(torch.tensor(init, dtype=torch.float32))
                nn.init.zeros_(self.residual_blend_head[-1].weight)
                nn.init.constant_(self.residual_blend_head[-1].bias, float(init_logit))
            else:
                self.residual_blend_use_stage = False
                self.residual_blend_flow_stat_proj = None
                self.residual_blend_stage_proj = None
                self.residual_blend_norm = None
                self.residual_blend_head = None
        else:
            self.residual_refine_block = None
            self.residual_blend_use_stage = False
            self.residual_blend_flow_stat_proj = None
            self.residual_blend_stage_proj = None
            self.residual_blend_norm = None
            self.residual_blend_head = None

        self.salt_enabled = bool(self.salt_global_variant)
        self.salt_use_same_param_control = self.salt_enabled and (
            self.variant == 'same_param_mlp_control' or bool(getattr(configs, 'salt_same_param_control', 0))
        )
        self.salt_use_gate = self.salt_enabled and self.variant != 'salt_latent_only' and not bool(
            getattr(configs, 'salt_disable_gate', 0)
        )
        self.salt_cf_loss_weight = float(getattr(configs, 'salt_cf_loss_weight', 0.03))
        self.salt_gate_loss_weight = float(getattr(configs, 'salt_gate_loss_weight', 0.01))
        if self.variant in ('salt_latent_only', 'salt_gate'):
            self.salt_cf_loss_weight = 0.0
        if self.variant != 'salt_full' and not self.salt_use_same_param_control:
            self.salt_gate_loss_weight = 0.0
        if bool(getattr(configs, 'salt_disable_cf_loss', 0)):
            self.salt_cf_loss_weight = 0.0
        self.salt_flow_drop_rate = float(getattr(configs, 'salt_flow_drop_rate', 0.15))
        if self.salt_enabled:
            mixer_type = getattr(configs, 'salt_latent_mixer', 'tiny_l1')
            if bool(getattr(configs, 'salt_disable_latent_mixer', 0)):
                mixer_type = 'none'
            if self.salt_use_same_param_control:
                self.salt_coupler = SameParamMLPControl(
                    configs.d_model,
                    dropout=float(getattr(configs, 'salt_dropout', 0.1)),
                )
            else:
                self.salt_coupler = LatentTrafficCoupler(
                    d_model=configs.d_model,
                    latent_slots=int(getattr(configs, 'salt_latent_slots', 16)),
                    num_heads=int(getattr(configs, 'salt_num_heads', 4)),
                    dropout=float(getattr(configs, 'salt_dropout', 0.1)),
                    mixer_type=mixer_type,
                )
            self.salt_global_head = GlobalResidualHead(
                configs.d_model,
                self.configs.enc_in,
                dropout=float(getattr(configs, 'salt_dropout', 0.1)),
                zero_init=True,
            )
            self.salt_alpha = nn.Parameter(torch.tensor(float(getattr(configs, 'salt_alpha_init', 0.05))))
            if self.salt_use_gate:
                self.salt_gate = StageAwareUncertaintyGate(
                    configs.d_model,
                    gate_bias=float(getattr(configs, 'salt_gate_bias', -2.0)),
                )
            else:
                self.salt_gate = None
        else:
            self.salt_coupler = None
            self.salt_global_head = None
            self.salt_alpha = None
            self.salt_gate = None
        self._last_salt_cache = None
        self._salt_diag_totals = {}
        self._salt_diag_count = 0
        self._salt_aux_totals = {}
        self._salt_aux_count = 0

        if self.care_global_variant:
            mixer_type = getattr(configs, 'care_latent_mixer', 'tiny_l1')
            self.care_global_expert = GlobalCouplingExpert(
                d_model=configs.d_model,
                t_step=self.configs.enc_in,
                latent_slots=int(getattr(configs, 'care_latent_slots', 16)),
                num_heads=int(getattr(configs, 'care_num_heads', 4)),
                latent_mixer=mixer_type,
                dropout=float(getattr(configs, 'care_dropout', 0.1)),
                flow_drop_rate=float(getattr(configs, 'care_global_flow_drop_rate', 0.15)),
                same_param_control=self.care_use_same_param_control,
                anchor_observed=bool(getattr(configs, 'care_anchor_observed', 1)),
            )
        else:
            self.care_global_expert = None

        refine_layers = int(getattr(configs, 'flow_refine_layers', 0))
        if refine_layers > 0:
            refine_layer = nn.TransformerEncoderLayer(
                d_model=configs.d_model,
                nhead=int(getattr(configs, 'flow_refine_heads', 4)),
                dim_feedforward=configs.d_ff,
                dropout=float(getattr(configs, 'flow_refine_dropout', 0.1)),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.flow_refiner = nn.TransformerEncoder(refine_layer, num_layers=refine_layers)
            self.flow_refine_scale = nn.Parameter(
                torch.tensor(float(getattr(configs, 'flow_refine_init', 0.1)), dtype=torch.float32)
            )
        else:
            self.flow_refiner = None
            self.flow_refine_scale = None

        self.geoattn_insert_location = str(getattr(configs, 'geoattn_insert_location', 'pre_decoder'))
        if self.geoattn_insert_location == 'llm_attention':
            print("geoattn_llm_attention_patch_unavailable_using_pre_decoder_fallback:1")
            self.geoattn_insert_location = 'pre_decoder'
        if self.geoattn_insert_location not in {'pre_decoder', 'post_flow2vec'}:
            raise ValueError(
                "geoattn_insert_location must be one of: llm_attention, pre_decoder, post_flow2vec"
            )
        self.geoattn_block = GeoAttnBlock(configs) if self.geoattn_enabled else None

        if configs.llm_model == 'gpt2':
            model_path = self._require_model_path('gpt2')
            self.llm_model = GPT2Model.from_pretrained(
                model_path,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                local_files_only=True,
            )
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        elif configs.llm_model == 'deepseek_R1':
            model_path = self._require_model_path('deepseek_R1')
            self.llm_config = AutoConfig.from_pretrained(model_path)
            self.llm_config.num_hidden_layers = configs.gpt_layers
            self.llm_config.output_attentions = False
            self.llm_config.output_hidden_states = False
            self.llm_config.use_cache = False
            self.llm_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                config=self.llm_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        elif configs.llm_model == 'Llama_3':
            model_path = self._require_model_path('Llama_3')
            self.llm_config = AutoConfig.from_pretrained(model_path)
            self.llm_config.num_hidden_layers = configs.gpt_layers
            self.llm_config.output_attentions = False
            self.llm_config.output_hidden_states = False
            self.llm_config.use_cache = False
            self.llm_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                config=self.llm_config,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            raise ValueError(f"Unsupported llm_model: {configs.llm_model}")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for name, param in self.llm_model.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            self.llm_model.to(device=torch.device('cuda:0'))

        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)

        self.acil_extractor = None
        self.acil_layer = None
        self._acil_initial_state = {}
        self._acil_diag_totals = {}
        self._acil_diag_count = 0
        self._acil_grad_totals = {}
        self._acil_grad_count = 0
        if self.use_acil:
            # Keep the downstream stagegate/ARI initialization identical to the
            # matched baseline; ACIL is an input-prior module, not a backbone swap.
            with preserve_torch_rng():
                self.acil_extractor = ObservationGeometryExtractorV2(
                    feature_set=self.acil_feature_set,
                )
                self.acil_layer = AnchorConditionedInterpolationLayer(
                    feature_dim=self.acil_extractor.feature_dim,
                    hidden=int(getattr(configs, 'acil_hidden', 64)),
                    beta_r=float(getattr(configs, 'acil_beta_r', 0.25)),
                    beta_o=float(getattr(configs, 'acil_beta_o', 0.10)),
                    beta_e=float(getattr(configs, 'acil_beta_e', 0.10)),
                    use_edge_extrapolation=bool(getattr(configs, 'acil_use_edge_extrapolation', 1)),
                )
            prior_ckpt = str(getattr(configs, 'acil_load_prior_ckpt', '') or '')
            if prior_ckpt:
                loaded = self._load_acil_prior_checkpoint(prior_ckpt)
                print(f"acil_prior_checkpoint_loaded:{prior_ckpt}:tensors={loaded}")
            if (
                bool(getattr(configs, 'acil_freeze_prior', 0))
                or self.variant in {'acil_full_prior_init_freeze', 'acil_prior_freeze_baseline'}
            ):
                for param in self.acil_layer.parameters():
                    param.requires_grad = False
                print("acil_prior_frozen:1")
            self._acil_initial_state = {
                name: value.detach().cpu().clone()
                for name, value in self.acil_layer.state_dict().items()
            }

    def _load_acil_prior_checkpoint(self, path):
        if self.acil_layer is None:
            return 0
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            checkpoint = checkpoint['state_dict']
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported ACIL prior checkpoint format: {path}")
        target_keys = set(self.acil_layer.state_dict().keys())
        layer_state = {}
        for key, value in checkpoint.items():
            clean = str(key)
            if clean.startswith('module.'):
                clean = clean[len('module.'):]
            if clean.startswith('acil_layer.'):
                clean = clean[len('acil_layer.'):]
            if clean in target_keys:
                layer_state[clean] = value
        if not layer_state:
            raise ValueError(f"No acil_layer tensors found in checkpoint: {path}")
        current = self.acil_layer.state_dict()
        current.update(layer_state)
        self.acil_layer.load_state_dict(current, strict=True)
        return len(layer_state)

    def _parse_temporal_kernels(self, value):
        if isinstance(value, (list, tuple)):
            kernels = [int(item) for item in value]
        else:
            kernels = [int(item.strip()) for item in str(value).split(',') if item.strip()]
        return tuple(kernels) if kernels else (3, 5, 7)

    def _parse_depth_router_layers(self, value):
        if isinstance(value, (list, tuple)):
            layers = [int(item) for item in value]
        else:
            layers = [int(item.strip()) for item in str(value).split(',') if item.strip()]
        layers = sorted(set(layers))
        invalid = [layer for layer in layers if layer <= 0]
        if invalid:
            raise ValueError(f"depth_router_layers must be positive GPT layer indices, got {invalid}")
        return tuple(layers)

    def _infer_router_count(self):
        if self.configs.data_path == 'abilene.csv':
            return 12
        if self.configs.data_path == 'geant.csv':
            return 22
        return 0

    def _infer_exclude_self_flows(self):
        return self.configs.data_path == 'geant.csv'

    @staticmethod
    def _topology_flow_count(router_count, exclude_self):
        if router_count <= 0:
            return 0
        if exclude_self:
            return router_count * max(router_count - 1, 0)
        return router_count * router_count

    def _flow_endpoints(self, n_tokens, device=None):
        if self.topology_router_count <= 0:
            flow_idx = torch.arange(n_tokens, device=device)
            return flow_idx, flow_idx
        flow_idx = torch.arange(n_tokens, device=device) % self.max_flow_count
        if self.topology_exclude_self:
            src = (flow_idx // max(self.topology_router_count - 1, 1)).clamp(max=self.topology_router_count - 1)
            dst = flow_idx % max(self.topology_router_count - 1, 1)
            dst = dst + (dst >= src).long()
            dst = dst.clamp(max=self.topology_router_count - 1)
        else:
            src = (flow_idx // self.topology_router_count).clamp(max=self.topology_router_count - 1)
            dst = (flow_idx % self.topology_router_count).clamp(max=self.topology_router_count - 1)
        return src, dst

    def _flow_relation_matrix(self, n_tokens):
        src, dst = self._flow_endpoints(n_tokens)
        same_src = src[:, None] == src[None, :]
        same_dst = dst[:, None] == dst[None, :]
        reverse = (src[:, None] == dst[None, :]) & (dst[:, None] == src[None, :])
        cross = ((src[:, None] == dst[None, :]) | (dst[:, None] == src[None, :])) & (~reverse)
        relation = torch.full((n_tokens, n_tokens), 5, dtype=torch.long)
        relation[cross] = 4
        relation[reverse] = 3
        relation[same_dst] = 2
        relation[same_src] = 1
        relation.fill_diagonal_(0)
        return relation

    def _require_model_path(self, key):
        path = self.model_roots[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing local {key} model at {path}. "
                "Run `python setup/download_assets.py --gpt2` for GPT2 or place the model there."
            )
        return str(path)

    def _run_llm(self, inputs_embeds, router_features=None):
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=self.use_depth_router,
            use_cache=False,
        )
        if self.depth_router is not None:
            if router_features is None:
                raise ValueError("router_features are required when depth_router=1")
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("LLM did not return hidden_states for depth_router=1")
            weights = self.depth_router(router_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            selected = torch.stack([hidden_states[layer] for layer in self.depth_router_layers], dim=1)
            routed = torch.sum(weights[:, :, None, None] * selected, dim=1)
            self.last_depth_router_weights = weights.detach()
            return routed
        return outputs.last_hidden_state

    def _depth_router_features(self, x_enc, means, stdev, observed_mask, known_rate, target_rate):
        bsz = x_enc.shape[0]
        device = x_enc.device
        dtype = x_enc.dtype
        known = torch.full((bsz, 1), float(known_rate) / 100.0, device=device, dtype=dtype)
        target = torch.full((bsz, 1), float(target_rate) / 100.0, device=device, dtype=dtype)
        if observed_mask is None:
            density = known
        else:
            density = observed_mask.to(device=device, dtype=dtype).mean(dim=(1, 2), keepdim=True).view(bsz, 1)
        original_mean = means.to(device=device, dtype=dtype).mean(dim=(1, 2), keepdim=True).view(bsz, 1)
        original_scale = stdev.to(device=device, dtype=dtype).mean(dim=(1, 2), keepdim=True).view(bsz, 1)
        normalized_abs = x_enc.detach().abs().mean(dim=(1, 2), keepdim=True).view(bsz, 1)
        normalized_scale = x_enc.detach().std(dim=(1, 2), unbiased=False, keepdim=True).view(bsz, 1)
        features = torch.cat(
            [known, target, density, original_mean, original_scale, normalized_abs + normalized_scale],
            dim=-1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _topology_embeddings(self, n_tokens, device):
        if not self.use_topology_embed or self.topology_router_count <= 0:
            return None
        flow_idx = torch.arange(n_tokens, device=device) % self.max_flow_count
        src, dst = self._flow_endpoints(n_tokens, device=device)
        topo = (
            self.src_embedding(src)
            + self.dst_embedding(dst)
            + self.pair_embedding(flow_idx)
        )
        return self.topology_norm(topo).unsqueeze(0)

    def _mask_prompt_embeddings(self, observed_mask, batch_size, device, dtype):
        if self.mask_prompt_tokens <= 0:
            return None
        prompt = self.mask_prompt_base.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        if self.mask_prompt_encoder is not None and observed_mask is not None:
            mask_series = observed_mask.to(device=device, dtype=dtype)
            feature_count = mask_series.shape[-1]
            if feature_count < self.configs.c_out:
                mask_series = F.pad(mask_series, (0, self.configs.c_out - feature_count))
            elif feature_count > self.configs.c_out:
                mask_series = mask_series[..., : self.configs.c_out]
            encoded = self.mask_prompt_encoder(mask_series)
            pooled = F.adaptive_avg_pool1d(
                encoded.transpose(1, 2), self.mask_prompt_tokens
            ).transpose(1, 2)
            prompt = prompt + pooled
        return self.mask_prompt_scale * self.mask_prompt_norm(prompt)

    def _geoanchor_observed_indices_for_rate(self, length, rate, device):
        if float(rate) >= 100.0:
            return torch.arange(length, device=device)
        count = int(torch.ceil(torch.tensor(length * float(rate) / 100.0, device=device)).item())
        if length > 1:
            count = max(2, count)
        count = min(length, max(1, count))
        idx = torch.linspace(0, length - 1, count, device=device).round().long().unique(sorted=True)
        if idx[-1].item() != length - 1:
            idx = torch.cat([idx, torch.tensor([length - 1], device=device)])
        if idx[0].item() != 0:
            idx = torch.cat([torch.tensor([0], device=device), idx])
        return idx.unique(sorted=True)

    def _geoanchor_mask_like(self, x_current, observed_mask, known_rate):
        bsz, steps, flows = x_current.shape
        if observed_mask is None:
            mask = x_current.new_zeros((1, steps, 1))
            idx = self._geoanchor_observed_indices_for_rate(steps, known_rate, x_current.device)
            mask[:, idx, :] = 1.0
            return mask.expand(bsz, steps, flows)
        mask = observed_mask.to(device=x_current.device, dtype=x_current.dtype)
        if mask.shape[1] < steps:
            mask = F.pad(mask, (0, 0, 0, steps - mask.shape[1]))
        elif mask.shape[1] > steps:
            mask = mask[:, :steps, :]
        if mask.shape[2] < flows:
            mask = F.pad(mask, (0, flows - mask.shape[2]))
        elif mask.shape[2] > flows:
            mask = mask[:, :, :flows]
        return mask

    def _shuffle_geoanchor_geometry(self, features):
        if features.shape[-1] <= 1:
            return features
        bsz, steps, flows, channels = features.shape
        value = features[..., :1]
        geometry = features[..., 1:].permute(0, 2, 1, 3).contiguous().view(bsz * flows, steps, channels - 1)
        if geometry.shape[0] <= 1:
            return features
        perm = torch.randperm(geometry.shape[0], device=features.device)
        geometry = geometry[perm].view(bsz, flows, steps, channels - 1).permute(0, 2, 1, 3).contiguous()
        return torch.cat([value, geometry], dim=-1)

    def _corrupt_geoanchor_features(self, features, mode):
        mode = str(mode or 'none')
        if mode == 'none':
            return features
        if mode == 'shuffled_geometry':
            return self._shuffle_geoanchor_geometry(features)
        value = features[..., :1]
        geometry = features[..., 1:]
        if mode == 'zero_geometry':
            geometry = torch.zeros_like(geometry)
        elif mode == 'random_geometry':
            geometry = torch.rand_like(geometry)
        elif mode == 'scaled_geometry_x10':
            geometry = geometry * 10.0
        else:
            raise ValueError(f"Unsupported GeoAnchor feature corruption: {mode}")
        return torch.cat([value, geometry], dim=-1)

    def _residual_blend_gate(self, hidden, x_current, observed_mask, known_rate, target_rate):
        if self.residual_blend_head is None or self.residual_refine_block is None:
            return None
        bsz = hidden.shape[0]
        mask = self.residual_refine_block._mask_for_stats(observed_mask, x_current, known_rate)
        flow_mask = mask.transpose(1, 2).contiguous()
        current = x_current.transpose(1, 2).contiguous()
        reliability = flow_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        uncertainty = 1.0 - reliability
        abs_mean = current.detach().abs().mean(dim=-1, keepdim=True)
        scale = current.detach().std(dim=-1, unbiased=False, keepdim=True)
        gate_hidden = hidden + self.residual_blend_flow_stat_proj(
            torch.cat([reliability, uncertainty, abs_mean, scale], dim=-1)
        )
        if self.residual_blend_stage_proj is not None:
            density = mask.mean(dim=(1, 2), keepdim=True).view(bsz, 1)
            known = torch.full((bsz, 1), float(known_rate) / 100.0, device=hidden.device, dtype=hidden.dtype)
            target = torch.full((bsz, 1), float(target_rate) / 100.0, device=hidden.device, dtype=hidden.dtype)
            delta = (target - known).clamp_min(0.0)
            stage = self.residual_blend_stage_proj(
                torch.cat([known, target, delta, density.to(hidden.dtype)], dim=-1)
            )
            gate_hidden = gate_hidden + stage.unsqueeze(1)
        return torch.sigmoid(self.residual_blend_head(self.residual_blend_norm(gate_hidden))).permute(0, 2, 1)

    def _record_salt_diagnostics(self, diagnostics):
        if not self.salt_enabled:
            return
        self._salt_diag_count += 1
        for key, value in diagnostics.items():
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            self._salt_diag_totals[key] = self._salt_diag_totals.get(key, 0.0) + value

    def _record_salt_aux(self, diagnostics):
        if not self.salt_enabled:
            return
        self._salt_aux_count += 1
        for key, value in diagnostics.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            self._salt_aux_totals[key] = self._salt_aux_totals.get(key, 0.0) + value

    def get_salt_diagnostics(self):
        if not self.salt_enabled:
            return {}
        out = {}
        if self._salt_diag_count:
            for key, value in sorted(self._salt_diag_totals.items()):
                out[key] = value / self._salt_diag_count
        if self._salt_aux_count:
            for key, value in sorted(self._salt_aux_totals.items()):
                out[key] = value / self._salt_aux_count
        out["diagnostic_forwards"] = self._salt_diag_count
        out["aux_updates"] = self._salt_aux_count
        return out

    def reset_salt_diagnostics(self):
        self._salt_diag_totals = {}
        self._salt_diag_count = 0
        self._salt_aux_totals = {}
        self._salt_aux_count = 0

    def get_care_diagnostics(self):
        if self.care_global_expert is None:
            return {}
        return self.care_global_expert.get_diagnostics()

    def reset_care_diagnostics(self):
        if self.care_global_expert is not None:
            self.care_global_expert.reset_diagnostics()

    def record_geoattn_gradient_norms(self):
        if self.geoattn_block is None:
            return {}
        total_sq = 0.0
        gamma_sq = 0.0
        for name, param in self.geoattn_block.named_parameters():
            if param.grad is None:
                continue
            value = float(param.grad.detach().square().sum().cpu().item())
            total_sq += value
            if name == "gamma":
                gamma_sq += value
        diagnostics = {
            "geoattn_grad_norm": math.sqrt(total_sq),
            "geoattn_gamma_grad_norm": math.sqrt(gamma_sq),
        }
        self.geoattn_block.record_diagnostics(diagnostics)
        return diagnostics

    def get_geoattn_diagnostics(self):
        if self.geoattn_block is None:
            return {}
        return self.geoattn_block.get_diagnostics()

    def reset_geoattn_diagnostics(self):
        if self.geoattn_block is not None:
            self.geoattn_block.reset_diagnostics()

    def _record_acil_diagnostics(self, diagnostics):
        if self.acil_layer is None:
            return
        values = tensor_mean_dict(diagnostics)
        if not values:
            return
        self._acil_diag_count += 1
        for key, value in values.items():
            if math.isfinite(value):
                self._acil_diag_totals[key] = self._acil_diag_totals.get(key, 0.0) + value

    def record_acil_gradient_norms(self):
        if self.acil_layer is None:
            return {}
        total_sq = 0.0
        last_sq = 0.0
        for name, param in self.acil_layer.named_parameters():
            if param.grad is None:
                continue
            value = float(param.grad.detach().square().sum().cpu().item())
            total_sq += value
            if "delta_r_head" in name or "offset_head" in name or "edge_offset_head" in name:
                last_sq += value
        grad = math.sqrt(total_sq)
        last = math.sqrt(last_sq)
        self._acil_grad_count += 1
        self._acil_grad_totals["ACIL_grad_norm"] = self._acil_grad_totals.get("ACIL_grad_norm", 0.0) + grad
        self._acil_grad_totals["ACIL_last_layer_grad_norm"] = (
            self._acil_grad_totals.get("ACIL_last_layer_grad_norm", 0.0) + last
        )
        return {"ACIL_grad_norm": grad, "ACIL_last_layer_grad_norm": last}

    def get_acil_diagnostics(self):
        if self.acil_layer is None:
            return {}
        out = {}
        if self._acil_diag_count:
            for key, value in sorted(self._acil_diag_totals.items()):
                out[key] = value / self._acil_diag_count
        if self._acil_grad_count:
            for key, value in sorted(self._acil_grad_totals.items()):
                out[key] = value / self._acil_grad_count
        out["ACIL_parameter_delta_from_init"] = parameter_delta_from_state(
            self.acil_layer,
            self._acil_initial_state,
        )
        out["ACIL_diagnostic_forwards"] = self._acil_diag_count
        out["ACIL_gradient_updates"] = self._acil_grad_count
        return out

    def reset_acil_diagnostics(self):
        self._acil_diag_totals = {}
        self._acil_diag_count = 0
        self._acil_grad_totals = {}
        self._acil_grad_count = 0

    def _load_acil_state_temporarily(self, state):
        if self.acil_layer is None:
            return None
        current = {name: value.detach().clone() for name, value in self.acil_layer.state_dict().items()}
        self.acil_layer.load_state_dict(
            {name: value.to(next(self.acil_layer.parameters()).device) for name, value in state.items()},
            strict=True,
        )
        return current

    def _restore_acil_state(self, state):
        if self.acil_layer is not None and state is not None:
            self.acil_layer.load_state_dict(state, strict=True)

    def apply_acil_prior(
        self,
        b_linear,
        observed_mask,
        x_obs=None,
        record=True,
        return_details=False,
        use_initial=False,
        corruption="none",
    ):
        if self.acil_layer is None or self.acil_extractor is None:
            details = None
            if return_details:
                zeros = torch.zeros_like(b_linear)
                details = {
                    "delta_r": zeros,
                    "r_hat": zeros,
                    "offset": zeros,
                    "edge_offset": zeros,
                    "uncertainty_q": zeros,
                    "gap_length": zeros,
                    "gap_type": zeros,
                    "relative_position": zeros,
                    "distance_to_nearest_anchor": zeros,
                }
            return (b_linear, details) if return_details else b_linear
        if x_obs is None:
            x_obs = b_linear
        mask = observed_mask.to(device=b_linear.device, dtype=b_linear.dtype)
        x_observed = torch.where(mask > 0.5, x_obs.to(device=b_linear.device, dtype=b_linear.dtype), torch.zeros_like(b_linear))
        restore_state = None
        if use_initial:
            restore_state = self._load_acil_state_temporarily(self._acil_initial_state)
        try:
            geometry = self.acil_extractor(
                x_observed,
                mask,
                b_linear=b_linear,
                shuffle_gap_geometry=self.acil_shuffle_gap_geometry,
                shuffle_anchor_values=self.acil_shuffle_anchor_values,
                corruption=corruption,
            )
            out, diagnostics = self.acil_layer(
                b_linear,
                x_observed,
                mask,
                geometry,
                freeze_to_linear=self.acil_freeze_to_linear,
            )
        finally:
            self._restore_acil_state(restore_state)
        if record and not use_initial:
            self._record_acil_diagnostics(diagnostics)
        if not return_details:
            return out
        details = {
            "delta_r": diagnostics["delta_r"].detach(),
            "r_hat": diagnostics["r_hat"].detach(),
            "offset": diagnostics["offset"].detach(),
            "edge_offset": diagnostics["edge_offset"].detach(),
            "uncertainty_q": geometry["uncertainty_q"].detach(),
            "gap_length": geometry["gap_length_raw"].detach(),
            "gap_type": geometry["gap_type"].detach(),
            "relative_position": geometry["relative_position"].detach(),
            "distance_to_nearest_anchor": geometry["distance_to_nearest_anchor"].detach(),
            "is_edge_gap": geometry["is_edge_gap"].detach(),
        }
        return out, details

    def _salt_update(self, hidden, x_current, observed_mask, known_rate, target_rate, local_residual, means, stdev):
        if not self.salt_enabled:
            self._last_salt_cache = None
            return None
        mask = stage_observed_mask_like(x_current, observed_mask, known_rate)
        density = mask.mean(dim=(1, 2), keepdim=True).view(x_current.shape[0], 1)
        c_flow, attn_diag = self.salt_coupler(hidden, known_rate, target_rate, density)
        global_residual = self.salt_global_head(c_flow)
        if self.salt_gate is None:
            gate = torch.ones(
                global_residual.shape[0],
                global_residual.shape[1],
                1,
                device=global_residual.device,
                dtype=global_residual.dtype,
            )
        else:
            gate, _ = self.salt_gate(hidden, c_flow, x_current, observed_mask, known_rate, target_rate)
        flow_missing = (1.0 - mask.transpose(1, 2).contiguous()).to(global_residual.dtype)
        gated_global = self.salt_alpha * gate * global_residual * flow_missing
        local_norm = torch.sqrt(local_residual.detach().square().mean().clamp_min(1e-12))
        global_norm = torch.sqrt(global_residual.detach().square().mean().clamp_min(1e-12))
        update_norm = torch.sqrt(gated_global.detach().square().mean().clamp_min(1e-12))
        diagnostics = {
            "alpha_final": float(self.salt_alpha.detach().item()),
            "gate_mean": float(gate.detach().mean().item()),
            "gate_std": float(gate.detach().std(unbiased=False).item()),
            "global_residual_norm": float(global_norm.item()),
            "local_residual_norm": float(local_norm.item()),
            "global_local_norm_ratio": float((global_norm / local_norm.clamp_min(1e-8)).item()),
            "gated_global_update_norm": float(update_norm.item()),
        }
        diagnostics.update(attn_diag)
        self._record_salt_diagnostics(diagnostics)
        self._last_salt_cache = {
            "x_current": x_current,
            "target_mask": mask,
            "global_residual": global_residual,
            "gate": gate,
            "means": means,
            "stdev": stdev,
            "alpha": self.salt_alpha,
        }
        return gated_global

    def salt_auxiliary_loss(self, criterion, target):
        if not self.salt_enabled or self._last_salt_cache is None:
            return target.new_tensor(0.0)
        cache = self._last_salt_cache
        x_current = cache["x_current"]
        means = cache["means"]
        stdev = cache["stdev"]
        gate = cache["gate"]
        global_residual = cache["global_residual"]
        mask = cache["target_mask"]
        target = target.to(device=x_current.device, dtype=x_current.dtype)

        aux_loss = target.new_tensor(0.0)
        diagnostics = {}
        cf_weight = float(self.salt_cf_loss_weight)
        if cf_weight > 0.0:
            bsz, steps, flows = x_current.shape
            flow_sample = torch.rand((bsz, flows), device=x_current.device) < self.salt_flow_drop_rate
            uncertain = (1.0 - mask).bool()
            omega = uncertain & flow_sample.unsqueeze(1)
            if not bool(omega.any()):
                omega = uncertain
            cf_update = cache["alpha"] * gate * global_residual
            cf_pred_norm = x_current + cf_update.permute(0, 2, 1)
            cf_pred = cf_pred_norm * stdev + means
            if bool(omega.any()):
                cf_loss = criterion(cf_pred[omega], target[omega])
            else:
                cf_loss = target.new_tensor(0.0)
            aux_loss = aux_loss + cf_weight * cf_loss
            diagnostics["cf_loss"] = float(cf_loss.detach().item())

        gate_weight = float(self.salt_gate_loss_weight)
        if gate_weight > 0.0 and self.salt_gate is not None:
            baseline = x_current * stdev + means
            error = (target - baseline).detach().abs().mean(dim=1)
            min_error = error.min(dim=1, keepdim=True).values
            max_error = error.max(dim=1, keepdim=True).values
            gate_target = (error - min_error) / (max_error - min_error).clamp_min(1e-6)
            gate_loss = F.mse_loss(gate.squeeze(-1), gate_target)
            aux_loss = aux_loss + gate_weight * gate_loss
            diagnostics["gate_loss"] = float(gate_loss.detach().item())

        if diagnostics:
            diagnostics["salt_aux_total"] = float(aux_loss.detach().item())
            self._record_salt_aux(diagnostics)
        return aux_loss

    def forward(self, x_enc, x_mark_enc, known_rate, target_rate, observed_mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, known_rate, target_rate, observed_mask)
        return None

    def imputation(self, x_enc, x_mark_enc, known_rate, target_rate, observed_mask=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        if self.temporal_adapter is not None:
            x_enc = self.temporal_adapter(x_enc)

        if self.geo_flow_embedding is not None:
            geo_mask = self._geoanchor_mask_like(x_enc, observed_mask, known_rate)
            geo_features = self.geoanchor_extractor(x_enc, geo_mask)
            geo_features = self._corrupt_geoanchor_features(geo_features, self.geoanchor_feature_corruption)
            if self.geoanchor_shuffle_geometry:
                geo_features = self._shuffle_geoanchor_geometry(geo_features)
            enc_out = self.geo_flow_embedding(geo_features)
            self.last_geoanchor_hidden = enc_out.detach()
        elif self.mask_concat_embedding is not None and observed_mask is not None:
            mask_channel = observed_mask.to(device=x_enc.device, dtype=x_enc.dtype)
            enc_out = self.mask_concat_embedding(torch.cat([x_enc, mask_channel], dim=1), x_mark_enc)
        else:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self.mask_embedding is not None and observed_mask is not None:
            mask_token = self.mask_embedding(observed_mask.to(device=x_enc.device, dtype=x_enc.dtype), x_mark_enc)
            if self.mask_gate is not None:
                enc_out = enc_out * (1.0 + self.mask_scale * torch.tanh(self.mask_gate(mask_token)))
            else:
                enc_out = enc_out + self.mask_scale * mask_token
        topo = self._topology_embeddings(enc_out.shape[1], enc_out.device)
        if topo is not None:
            enc_out = enc_out + self.topology_scale * topo
        if self.flow_attention is not None:
            enc_out = self.flow_attention(enc_out, target_rate)
        if self.reliability_attention is not None and observed_mask is not None:
            enc_out = self.reliability_attention(enc_out, observed_mask)
        if self.geoattn_block is not None and self.geoattn_insert_location == 'post_flow2vec':
            geoattn_mask = observed_mask
            if geoattn_mask is None:
                geoattn_mask = self._geoanchor_mask_like(x_enc, observed_mask, known_rate)
            enc_out = self.geoattn_block(enc_out, geoattn_mask)
        prompt = []
        for _ in range(x_enc.shape[0]):
            if self.configs.data_path == 'abilene.csv':
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:12 routers (v1-v12) with 144 OD flows (x1-x144), "
                    f"Each flow xij represents traffic from vi to vj, indexed as (i-1)*12+j (i,j in 1..12)"
                    f"- Temporal sequence length: {self.seq_len} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"Perform progressive matrix completion through autoregressive imputation:"
                    f"Stage: 2%->4%->8%->16%->32%->64%->100%"
                    f"Current Task: Advance from {known_rate}% to {target_rate}% completion"
                )
            elif self.configs.data_path == 'wsdream.csv':
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:550 OD flows (x1-x550)"
                    f"- Temporal sequence length: 64 time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"Perform progressive matrix completion through autoregressive imputation:"
                    f"Stage: 2%->4%->8%->16%->32%->64%->100%"
                    f"Current Task: Advance from {known_rate}% to {target_rate}% completion"
                )
            elif self.configs.data_path == 'geant.csv':
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:GEANT traffic matrix with {self.configs.c_out} flow features"
                    f"- Temporal sequence length: {self.seq_len} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"Perform progressive matrix completion through autoregressive imputation:"
                    f"Stage: 2%->4%->8%->16%->32%->64%->100%"
                    f"Current Task: Advance from {known_rate}% to {target_rate}% completion"
                )
            else:
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"- Temporal sequence length: {self.seq_len} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"Current Task: Advance from {known_rate}% to {target_rate}% completion"
                )
            prompt.append(prompt_)

        prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        mask_prompt_embeddings = self._mask_prompt_embeddings(
            observed_mask,
            x_enc.shape[0],
            x_enc.device,
            enc_out.dtype,
        )
        if mask_prompt_embeddings is not None:
            combined_input = torch.cat((prompt_embeddings, mask_prompt_embeddings, enc_out), dim=1)
        else:
            combined_input = torch.cat((prompt_embeddings, enc_out), dim=1)
        router_features = None
        if self.depth_router is not None:
            router_features = self._depth_router_features(
                x_enc,
                means,
                stdev,
                observed_mask,
                known_rate,
                target_rate,
            )
        outputs = self._run_llm(combined_input, router_features)

        outputs = self.ln_proj(outputs[:, -self.configs.c_out:, :])
        if self.flow_refiner is not None:
            outputs = outputs + self.flow_refine_scale * self.flow_refiner(outputs)
        if self.geoattn_block is not None and self.geoattn_insert_location == 'pre_decoder':
            geoattn_mask = observed_mask
            if geoattn_mask is None:
                geoattn_mask = self._geoanchor_mask_like(x_enc, observed_mask, known_rate)
            outputs = self.geoattn_block(outputs, geoattn_mask)
        if self.care_global_expert is not None:
            global_residual, _ = self.care_global_expert(
                outputs,
                x_enc,
                observed_mask,
                known_rate,
                target_rate,
            )
            dec_out = x_enc + global_residual.permute(0, 2, 1)
            return dec_out * stdev + means
        if self.residual_refine_block is not None:
            assert self.residual_refine_scale is not None
            residual = self.residual_refine_block(outputs, x_enc, observed_mask, known_rate, target_rate)
            salt_update = self._salt_update(
                outputs,
                x_enc,
                observed_mask,
                known_rate,
                target_rate,
                residual,
                means,
                stdev,
            )
            if salt_update is None:
                residual_out = x_enc + self.residual_refine_scale * residual.permute(0, 2, 1)
            else:
                residual_out = (
                    x_enc
                    + self.residual_refine_scale * residual.permute(0, 2, 1)
                    + salt_update.permute(0, 2, 1)
                )
            if self.use_residual_refine_blend:
                direct_out = self.flattenhead(outputs).permute(0, 2, 1)
                blend = self._residual_blend_gate(outputs, x_enc, observed_mask, known_rate, target_rate)
                dec_out = direct_out + blend * (residual_out - direct_out)
            else:
                dec_out = residual_out
            return dec_out * stdev + means
        if self.low_rank_head is not None and self.low_rank_replace:
            token_out = self.low_rank_head(outputs)
        else:
            token_out = self.flattenhead(outputs)
            if self.low_rank_head is not None:
                token_out = token_out + self.low_rank_scale * self.low_rank_head(outputs)
        if self.residual_head is not None:
            token_out = token_out + self.residual_scale * self.residual_head(outputs)
        dec_out = token_out.permute(0, 2, 1)
        if self.use_residual_refine:
            assert self.residual_refine_scale is not None
            dec_out = x_enc + self.residual_refine_scale * dec_out
        return dec_out * stdev + means
