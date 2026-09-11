import os
import json
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
from torch import optim
from tqdm import tqdm
from data_provider.data_factory import data_provider
from utils.structured_masks import StructuredMaskConfig, generate_structured_observed_mask
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class SMAPE(nn.Module):
    def __init__(self, eps=1e-8):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        return torch.mean(2.0 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + self.eps))


class NMAELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NMAELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true)).clamp_min(self.eps)


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.pred_len = 0
        self.seq_len = self.args.seq_len
        self.memory_bank = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        if self._acil_prior_only():
            model = self._unwrap_model()
            acil_layer = getattr(model, 'acil_layer', None)
            params = []
            freeze_linear = bool(getattr(self.args, 'acil_freeze_to_linear', 0)) or str(getattr(self.args, 'variant', '')) == 'linearinterp_prior_diagnostic'
            if acil_layer is not None and not freeze_linear:
                params = [param for param in acil_layer.parameters() if param.requires_grad]
            if not params:
                self._dummy_optimizer_param = nn.Parameter(torch.zeros((), device=self.device))
                params = [self._dummy_optimizer_param]
            return optim.Adam(params, lr=self.args.learning_rate, weight_decay=self.args.weight)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)

    def _select_criterion(self):
        loss_name = getattr(self.args, 'train_loss', 'smape').lower()
        if loss_name == 'smape':
            return SMAPE()
        if loss_name == 'mae':
            return nn.L1Loss()
        if loss_name == 'mse':
            return nn.MSELoss()
        if loss_name == 'huber':
            return nn.SmoothL1Loss()
        if loss_name == 'nmae':
            return NMAELoss()
        raise ValueError(f"Unsupported train_loss: {loss_name}")

    def _flatten_by_sample(self, values):
        if values.dim() <= 1:
            return values.reshape(1, -1)
        return values.reshape(values.shape[0], -1)

    def _positive_distribution(self, values):
        eps = float(getattr(self.args, 'distribution_loss_eps', 1e-8))
        positive = torch.relu(values) + eps
        flat = self._flatten_by_sample(positive)
        return flat / flat.sum(dim=1, keepdim=True).clamp_min(eps)

    def _distribution_loss(self, pred, true):
        eps = float(getattr(self.args, 'distribution_loss_eps', 1e-8))
        p = self._positive_distribution(true)
        q = self._positive_distribution(pred)
        return torch.sum(p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps))), dim=1).mean()

    def _mass_loss(self, pred, true):
        eps = float(getattr(self.args, 'distribution_loss_eps', 1e-8))
        pred_flat = self._flatten_by_sample(torch.relu(pred))
        true_flat = self._flatten_by_sample(torch.relu(true))
        pred_mass = pred_flat.sum(dim=1)
        true_mass = true_flat.sum(dim=1)
        return torch.mean(torch.abs(pred_mass - true_mass) / true_mass.clamp_min(eps))

    def _training_objective(self, criterion, pred, true):
        loss = criterion(pred, true)
        distribution_weight = float(getattr(self.args, 'distribution_loss_weight', 0.0))
        mass_weight = float(getattr(self.args, 'mass_loss_weight', 0.0))
        if distribution_weight:
            loss = loss + distribution_weight * self._distribution_loss(pred, true)
        if mass_weight:
            loss = loss + mass_weight * self._mass_loss(pred, true)
        return loss

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _salt_auxiliary_loss(self, criterion, target):
        model = self._unwrap_model()
        if hasattr(model, 'salt_auxiliary_loss'):
            return model.salt_auxiliary_loss(criterion, target)
        return target.new_tensor(0.0)

    def _salt_diagnostics(self):
        model = self._unwrap_model()
        if hasattr(model, 'get_salt_diagnostics'):
            return model.get_salt_diagnostics()
        return {}

    def _care_diagnostics(self):
        model = self._unwrap_model()
        if hasattr(model, 'get_care_diagnostics'):
            return model.get_care_diagnostics()
        return {}

    def _geoattn_diagnostics(self):
        model = self._unwrap_model()
        if hasattr(model, 'get_geoattn_diagnostics'):
            return model.get_geoattn_diagnostics()
        return {}

    def _geoanchor_active(self):
        variant = str(getattr(self.args, 'variant', ''))
        return bool(getattr(self.args, 'geoanchor_enable', 0)) or variant in {
            'value_extra_control',
            'mask_only_control',
            'agt_full',
            'agt_shuffled_geometry',
            'agt_no_anchor_values',
        }

    def _acil_active(self):
        variant = str(getattr(self.args, 'variant', ''))
        return (
            bool(getattr(self.args, 'acil_enable', 0))
            or bool(getattr(self.args, 'geoanchor_v2_enable', 0))
            or bool(getattr(self.args, 'geoattn_enable', 0))
            or variant in {
                'acil_frozen_linear',
                'acil_prior_freeze_baseline',
                'acil_prior_only',
                'linearinterp_prior_diagnostic',
                'acil_prior_only_full',
                'acil_prior_only_value_sameparam',
                'acil_prior_only_shuffled_gap_geometry',
                'acil_prior_only_no_anchor_values',
                'acil_value_only_sameparam',
                'acil_shuffled_gap_geometry',
                'acil_shuffled_anchor_values',
                'acil_no_anchor_values',
                'acil_full',
                'acil_full_scratch',
                'acil_full_prior_init_freeze',
                'acil_full_prior_init_finetune',
                'acil_geoattn_true',
                'acil_geoattn_shuffled_bias',
                'acil_geoattn_random_bias',
                'acil_geoattn_zero_bias',
                'acil_geoattn_same_param_control',
            }
        )

    def _acil_prior_only(self):
        return bool(getattr(self.args, 'acil_prior_only', 0)) or str(getattr(self.args, 'variant', '')) in {
            'acil_prior_only',
            'linearinterp_prior_diagnostic',
            'acil_prior_only_full',
            'acil_prior_only_value_sameparam',
            'acil_prior_only_shuffled_gap_geometry',
            'acil_prior_only_no_anchor_values',
        }

    def _apply_acil_prior(self, baseline, observed_mask, true_x=None, record=True, return_details=False, use_initial=False):
        model = self._unwrap_model()
        if not self._acil_active() or not hasattr(model, 'apply_acil_prior'):
            if return_details:
                return baseline, {}
            return baseline
        x_obs = true_x if true_x is not None else baseline
        return model.apply_acil_prior(
            baseline,
            observed_mask,
            x_obs=x_obs,
            record=record,
            return_details=return_details,
            use_initial=use_initial,
        )

    def _record_acil_gradient_norms(self):
        model = self._unwrap_model()
        if hasattr(model, 'record_acil_gradient_norms'):
            return model.record_acil_gradient_norms()
        return {}

    def _record_geoattn_gradient_norms(self):
        model = self._unwrap_model()
        if hasattr(model, 'record_geoattn_gradient_norms'):
            return model.record_geoattn_gradient_norms()
        return {}

    def _acil_diagnostics(self):
        model = self._unwrap_model()
        if hasattr(model, 'get_acil_diagnostics'):
            return model.get_acil_diagnostics()
        return {}

    def prepare_batch(self, batch_x):
        return batch_x.float().to(self.device)

    def _initial_rate_percent(self):
        rate = float(getattr(self.args, 'initial_observed_rate', 0.02))
        return rate * 100.0 if rate <= 1.0 else rate

    def _stage_rates(self, start_rate=None):
        rate = self._initial_rate_percent() if start_rate is None else float(start_rate)
        rate = max(rate, 1e-6)
        rates = [rate]
        while rates[-1] < 100.0:
            next_rate = min(rates[-1] * 2.0, 100.0)
            if next_rate == rates[-1]:
                break
            rates.append(next_rate)
        return rates

    def _observed_indices_for_rate(self, length, rate, device):
        if rate >= 100.0:
            return torch.arange(length, device=device)
        count = int(np.ceil(length * rate / 100.0))
        if length > 1:
            count = max(2, count)
        count = min(length, max(1, count))
        idx = torch.linspace(0, length - 1, count, device=device).round().long().unique(sorted=True)
        if idx[-1].item() != length - 1:
            idx = torch.cat([idx, torch.tensor([length - 1], device=device)])
        if idx[0].item() != 0:
            idx = torch.cat([torch.tensor([0], device=device), idx])
        return idx.unique(sorted=True)

    def _linear_interpolate_indices(self, batch_x, indices):
        B, T, N = batch_x.shape
        if indices.numel() >= T:
            return batch_x.clone()
        if indices.numel() == 1:
            return batch_x[:, indices[0]:indices[0] + 1, :].expand(B, T, N).clone()

        out = torch.empty_like(batch_x)
        idx = [int(v.item()) for v in indices]
        out[:, :idx[0] + 1, :] = batch_x[:, idx[0]:idx[0] + 1, :]
        for left, right in zip(idx[:-1], idx[1:]):
            alpha = torch.linspace(0.0, 1.0, right - left + 1, device=batch_x.device, dtype=batch_x.dtype)
            alpha = alpha.view(1, -1, 1)
            out[:, left:right + 1, :] = (
                batch_x[:, left:left + 1, :] * (1.0 - alpha)
                + batch_x[:, right:right + 1, :] * alpha
            )
        out[:, idx[-1]:, :] = batch_x[:, idx[-1]:idx[-1] + 1, :]
        return out

    def _build_training_stages(self, batch_x):
        stages = []
        B, T, N = batch_x.shape
        for rate in self._stage_rates():
            if rate >= 100.0:
                stages.append(batch_x.clone())
            else:
                indices = self._observed_indices_for_rate(T, rate, batch_x.device)
                stages.append(self._linear_interpolate_indices(batch_x, indices))
        return stages

    def _stage_loss_weights(self, num_steps, device):
        gamma = float(getattr(self.args, 'stage_loss_gamma', 1.0))
        if num_steps <= 0:
            return torch.ones(0, device=device)
        raw = torch.tensor([gamma ** i for i in range(num_steps)], device=device, dtype=torch.float32)
        return raw / raw.mean().clamp_min(1e-8)

    def _stage_observed_mask_for_rate(self, shape, rate, device):
        B, T, N = shape
        mask = torch.zeros((1, T, 1), device=device, dtype=torch.bool)
        indices = self._observed_indices_for_rate(T, rate, device)
        mask[:, indices, :] = True
        return mask.expand(B, T, N)

    def _stage_loss_mask(self, shape, known_rate, target_rate, device):
        scope = getattr(self.args, 'stage_loss_scope', 'full')
        if scope == 'full':
            return None

        target_mask = self._stage_observed_mask_for_rate(shape, target_rate, device)
        if scope == 'target_observed':
            return target_mask
        if scope == 'new_observed':
            known_mask = self._stage_observed_mask_for_rate(shape, known_rate, device)
            new_mask = target_mask & (~known_mask)
            return new_mask if bool(new_mask.any()) else target_mask
        raise ValueError(f"Unsupported stage_loss_scope: {scope}")

    def _structured_mask_config(self):
        return StructuredMaskConfig(
            mask_type=str(getattr(self.args, 'mask_type', 'random') or 'random'),
            block_len=int(getattr(self.args, 'structured_block_len', 8)),
            block_len_frac=float(getattr(self.args, 'structured_block_len_frac', 0.25)),
            num_bursts=int(getattr(self.args, 'structured_num_bursts', 3)),
            burst_min_len=int(getattr(self.args, 'structured_burst_min_len', 3)),
            burst_max_len=int(getattr(self.args, 'structured_burst_max_len', 8)),
            edge_len_frac=float(getattr(self.args, 'structured_edge_len_frac', 0.25)),
            preserve_mask_rate=bool(getattr(self.args, 'structured_preserve_mask_rate', 1)),
            anchor_centric=bool(getattr(self.args, 'structured_anchor_centric', 0)),
        )

    def _build_observed_mask(self, shape, observed_rate, device, seed_offset):
        B, T, N = shape
        mask_type = str(getattr(self.args, 'mask_type', 'random') or 'random')
        if mask_type != 'random':
            base_seed = getattr(self.args, 'structured_mask_seed', None)
            if base_seed is None:
                base_seed = getattr(self.args, 'mask_seed', 2024)
            seed = int(base_seed) + int(seed_offset)
            mask, _ = generate_structured_observed_mask(
                shape,
                observed_rate,
                device,
                seed,
                self._structured_mask_config(),
            )
            return mask
        observed = torch.zeros((B, T, N), device=device)
        n_observed = max(1, int(np.ceil(observed_rate * T)))
        n_observed = min(T, n_observed)
        seed = int(getattr(self.args, 'mask_seed', 2024)) + int(seed_offset)
        generator = torch.Generator()
        generator.manual_seed(seed)
        for col in range(N):
            idx = torch.randperm(T, generator=generator)[:n_observed].to(device)
            observed[:, idx, col] = 1.0
        return observed

    def _interpolate_series(self, values, indices):
        T = values.shape[0]
        if indices.numel() == 0:
            return values.mean().expand(T)
        if indices.numel() == 1:
            return values[indices[0]].expand(T)

        strategy = getattr(self.args, 'fill_strategy', 'linear')
        if strategy == 'mean':
            return values[indices].mean().expand(T)
        if strategy == 'zero':
            return torch.zeros_like(values)
        if strategy == 'nearest':
            t = torch.arange(T, device=values.device).view(-1, 1)
            dist = torch.abs(t - indices.view(1, -1))
            nearest = indices[torch.argmin(dist, dim=1)]
            return values[nearest]

        out = torch.empty_like(values)
        idx = [int(v.item()) for v in indices]
        out[:idx[0] + 1] = values[idx[0]]
        for left, right in zip(idx[:-1], idx[1:]):
            alpha = torch.linspace(0.0, 1.0, right - left + 1, device=values.device, dtype=values.dtype)
            out[left:right + 1] = values[left] * (1.0 - alpha) + values[right] * alpha
        out[idx[-1]:] = values[idx[-1]]
        return out

    def _prepare_memory_bank(self, train_data):
        if getattr(self.args, 'fill_strategy', 'linear') != 'memory':
            return
        data = getattr(train_data, 'data_x', None)
        if data is None:
            return
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or len(data) < self.seq_len:
            return
        if data.shape[-1] > getattr(self.args, 'c_out', data.shape[-1]):
            data = data[:, :self.args.c_out]
        max_start = len(data) - self.seq_len
        memory_k = max(1, int(getattr(self.args, 'memory_fill_k', 128)))
        rng = np.random.default_rng(int(getattr(self.args, 'seed', 2021)) + 1701)
        if max_start <= 0:
            starts = np.zeros(memory_k, dtype=np.int64)
        else:
            starts = rng.integers(0, max_start + 1, size=memory_k, dtype=np.int64)
        windows = np.stack([data[int(start):int(start) + self.seq_len] for start in starts], axis=0)
        self.memory_bank = torch.from_numpy(windows).float().to(self.device)
        print(f'memory_fill_bank: {tuple(self.memory_bank.shape)}')

    def _linear_fill_from_mask(self, batch_x, observed_mask):
        if getattr(self.args, 'fill_strategy', 'linear') == 'linear':
            mask = observed_mask.to(device=batch_x.device, dtype=batch_x.dtype).clamp(0.0, 1.0)
            mask_bool = mask > 0.5
            B, T, N = batch_x.shape
            pos = torch.arange(T, device=batch_x.device).view(1, T, 1).expand(B, T, N)
            source = torch.where(mask_bool, batch_x, torch.zeros_like(batch_x))

            left_seed = torch.where(mask_bool, pos, torch.full_like(pos, -1))
            left_idx = torch.cummax(left_seed, dim=1).values
            left_exists = left_idx >= 0
            right_seed = torch.where(mask_bool, pos, torch.full_like(pos, T))
            right_idx = torch.flip(torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values, dims=(1,))
            right_exists = right_idx < T

            left_gather = left_idx.clamp(0, T - 1)
            right_gather = right_idx.clamp(0, T - 1)
            left_value = torch.gather(source, dim=1, index=left_gather)
            right_value = torch.gather(source, dim=1, index=right_gather)
            denom = (right_idx - left_idx).to(dtype=batch_x.dtype).clamp_min(1.0)
            alpha = ((pos - left_idx).to(dtype=batch_x.dtype) / denom).clamp(0.0, 1.0)
            interp = left_value * (1.0 - alpha) + right_value * alpha
            interp = torch.where(left_exists & right_exists, interp, interp)
            interp = torch.where(left_exists & (~right_exists), left_value, interp)
            interp = torch.where((~left_exists) & right_exists, right_value, interp)
            mean_value = batch_x.mean(dim=1, keepdim=True).expand_as(batch_x)
            interp = torch.where((~left_exists) & (~right_exists), mean_value, interp)
            return torch.where(mask_bool, batch_x, interp)

        B, T, N = batch_x.shape
        inp = torch.empty_like(batch_x)
        for b in range(B):
            for n in range(N):
                idx = torch.nonzero(observed_mask[b, :, n] > 0, as_tuple=True)[0]
                inp[b, :, n] = self._interpolate_series(batch_x[b, :, n], idx)
        return inp

    def _memory_fill_from_mask(self, batch_x, observed_mask):
        linear = self._linear_fill_from_mask(batch_x, observed_mask)
        if self.memory_bank is None or self.memory_bank.numel() == 0:
            return linear
        bank = self.memory_bank.to(batch_x.device, dtype=batch_x.dtype)
        if bank.shape[-1] != batch_x.shape[-1]:
            bank = bank[:, :, :batch_x.shape[-1]]
        diff = (bank.unsqueeze(0) - batch_x.unsqueeze(1)) * observed_mask.unsqueeze(1)
        denom = observed_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0).squeeze(-1)
        mse = diff.square().sum(dim=(2, 3)) / denom
        topk = min(max(1, int(getattr(self.args, 'memory_fill_topk', 4))), bank.shape[0])
        vals, idx = torch.topk(mse, k=topk, dim=1, largest=False)
        temp = max(float(getattr(self.args, 'memory_fill_temperature', 0.03)), 1e-6)
        weights = torch.softmax(-vals / temp, dim=1)
        selected = bank[idx]
        prior = torch.sum(weights.view(weights.shape[0], topk, 1, 1) * selected, dim=1)
        blend = min(max(float(getattr(self.args, 'memory_fill_blend', 1.0)), 0.0), 1.0)
        filled = blend * prior + (1.0 - blend) * linear
        return filled * (1.0 - observed_mask) + batch_x * observed_mask

    def _initial_fill_from_mask(self, batch_x, observed_mask):
        if getattr(self.args, 'fill_strategy', 'linear') == 'memory':
            return self._memory_fill_from_mask(batch_x, observed_mask)
        return self._linear_fill_from_mask(batch_x, observed_mask)

    def _progressive_impute(self, inp, observed_mask=None, true_x=None, known_rate_start=None):
        known_rate = float(known_rate_start if known_rate_start is not None else self.args.mask_rate * 100.0)
        outputs = inp
        while known_rate < 100.0:
            target_rate = min(known_rate * 2.0, 100.0)
            model_inp = outputs
            if self._acil_active() and observed_mask is not None and not self._acil_prior_only():
                model_inp = self._apply_acil_prior(outputs, observed_mask, true_x=true_x, record=True)
            outputs = self.model(model_inp, None, known_rate, target_rate, observed_mask)
            if observed_mask is not None and true_x is not None:
                outputs = outputs * (1.0 - observed_mask) + true_x * observed_mask
            known_rate = target_rate
        return outputs

    def _prediction_seed_for_flag(self, flag):
        offsets = {'train': 3100000, 'val': 3200000, 'test': 3300000}
        return int(getattr(self.args, 'seed', 2021)) + offsets.get(flag, 3400000)

    def _mask_seed_offset(self, flag, batch_index):
        if flag == 'val':
            return batch_index
        if flag == 'test':
            return batch_index + 100000
        return batch_index + 200000

    def _collect_predictions(self, flag):
        data, loader = self._get_data(flag=flag)
        seed = self._prediction_seed_for_flag(flag)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)
        preds = []
        trues = []
        masks = []
        baselines = []
        acil_priors = []
        acil_untrained_priors = []
        acil_detail_buffers = {
            "acil_delta_r": [],
            "acil_r_hat": [],
            "acil_offset": [],
            "acil_edge_offset": [],
            "acil_uncertainty_q": [],
            "acil_gap_length": [],
            "acil_gap_type": [],
            "acil_relative_position": [],
            "acil_distance_to_nearest_anchor": [],
            "acil_is_edge_gap": [],
        }
        geo_hidden_samples = []
        geo_hidden_sample_budget = (
            int(getattr(self.args, 'geoanchor_hidden_sample_tokens', 2048))
            if int(getattr(self.args, 'geoanchor_save_hidden_sample', 0))
            else 0
        )
        geo_hidden_sample_count = 0
        self.model.eval()
        with torch.no_grad():
            print(f'len_{flag}:{len(loader)}')
            for i, batch_x in tqdm(enumerate(loader)):
                batch_x = self.prepare_batch(batch_x)
                observed_mask = self._build_observed_mask(
                    batch_x.shape,
                    self.args.mask_rate,
                    batch_x.device,
                    seed_offset=self._mask_seed_offset(flag, i),
                )
                inp = self._initial_fill_from_mask(batch_x, observed_mask)
                known_rate = float(observed_mask.mean().item()) * 100.0
                if self._acil_active():
                    acil_inp, acil_details = self._apply_acil_prior(
                        inp,
                        observed_mask,
                        true_x=batch_x,
                        record=False,
                        return_details=True,
                    )
                    acil_untrained = self._apply_acil_prior(
                        inp,
                        observed_mask,
                        true_x=batch_x,
                        record=False,
                        use_initial=True,
                    )
                else:
                    acil_inp = inp
                    acil_untrained = inp
                    acil_details = {}
                if self._acil_prior_only():
                    outputs = acil_inp
                else:
                    outputs = self._progressive_impute(inp, observed_mask, batch_x, known_rate)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_x.detach().cpu().numpy())
                masks.append(observed_mask.detach().cpu().numpy())
                baselines.append(inp.detach().cpu().numpy())
                acil_priors.append(acil_inp.detach().cpu().numpy())
                acil_untrained_priors.append(acil_untrained.detach().cpu().numpy())
                for key in acil_detail_buffers:
                    detail_key = key.replace("acil_", "")
                    value = acil_details.get(detail_key)
                    if value is None:
                        value = torch.zeros_like(inp)
                    acil_detail_buffers[key].append(value.detach().cpu().numpy())
                if geo_hidden_sample_count < geo_hidden_sample_budget:
                    model_obj = self.model.module if hasattr(self.model, 'module') else self.model
                    hidden = getattr(model_obj, 'last_geoanchor_hidden', None)
                    if hidden is not None:
                        hidden = hidden.detach().reshape(-1, hidden.shape[-1])
                        take = min(geo_hidden_sample_budget - geo_hidden_sample_count, hidden.shape[0])
                        if take > 0:
                            geo_hidden_samples.append(hidden[:take].cpu().numpy())
                            geo_hidden_sample_count += take
        arrays = {
            "preds": np.concatenate(preds, 0),
            "trues": np.concatenate(trues, 0),
            "masks": np.concatenate(masks, 0),
            "baselines": np.concatenate(baselines, 0),
            "acil_priors": np.concatenate(acil_priors, 0),
            "acil_untrained_priors": np.concatenate(acil_untrained_priors, 0),
        }
        for key, values in acil_detail_buffers.items():
            arrays[key] = np.concatenate(values, 0)
        if geo_hidden_samples:
            arrays["geo_hidden_sample"] = np.concatenate(geo_hidden_samples, 0)
        return arrays

    def _save_care_prediction_cache(self, setting, split, arrays):
        pred_dir = getattr(self.args, 'care_prediction_dir', '') or os.path.join(
            os.path.dirname(getattr(self.args, 'result_file', 'result_imputation.txt')) or '.',
            'prediction_cache',
        )
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"{self.args.model_id}_{split}.npz")
        metadata = {
            "setting": setting,
            "model_id": self.args.model_id,
            "split": split,
            "variant": getattr(self.args, 'variant', ''),
            "data": getattr(self.args, 'data', ''),
            "dataset": getattr(self.args, 'data_path', ''),
            "mask_rate": float(getattr(self.args, 'mask_rate', 0.0)),
            "mask_type": getattr(self.args, 'mask_type', 'random'),
            "structured_anchor_centric": int(getattr(self.args, 'structured_anchor_centric', 0)),
            "structured_block_len": int(getattr(self.args, 'structured_block_len', 8)),
            "structured_block_len_frac": float(getattr(self.args, 'structured_block_len_frac', 0.25)),
            "structured_num_bursts": int(getattr(self.args, 'structured_num_bursts', 3)),
            "structured_burst_min_len": int(getattr(self.args, 'structured_burst_min_len', 3)),
            "structured_burst_max_len": int(getattr(self.args, 'structured_burst_max_len', 8)),
            "structured_edge_len_frac": float(getattr(self.args, 'structured_edge_len_frac', 0.25)),
            "structured_preserve_mask_rate": int(getattr(self.args, 'structured_preserve_mask_rate', 1)),
            "structured_mask_seed": (
                int(getattr(self.args, 'structured_mask_seed'))
                if getattr(self.args, 'structured_mask_seed', None) is not None
                else int(getattr(self.args, 'mask_seed', 0))
            ),
            "mask_rate_semantics": "observed_rate",
            "seed": int(getattr(self.args, 'seed', 0)),
            "mask_seed": int(getattr(self.args, 'mask_seed', 0)),
            "fill_strategy": getattr(self.args, 'fill_strategy', ''),
            "train_protocol": getattr(self.args, 'train_protocol', ''),
            "stage_loss_scope": getattr(self.args, 'stage_loss_scope', ''),
        }
        np.savez_compressed(
            path,
            preds=arrays["preds"],
            trues=arrays["trues"],
            masks=arrays["masks"],
            baselines=arrays["baselines"],
            metadata=json.dumps(metadata, sort_keys=True),
        )
        print(f"care_prediction_saved:{path}")
        return path

    def _save_geoanchor_prediction_cache(self, setting, split, arrays):
        pred_dir = getattr(self.args, 'geoanchor_prediction_dir', '') or os.path.join(
            os.path.dirname(getattr(self.args, 'result_file', 'result_imputation.txt')) or '.',
            'prediction_cache',
        )
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"{self.args.model_id}_{split}.npz")
        metadata = {
            "setting": setting,
            "model_id": self.args.model_id,
            "split": split,
            "variant": getattr(self.args, 'variant', ''),
            "geoanchor_feature_set": getattr(self.args, 'geoanchor_feature_set', ''),
            "geoanchor_shuffle_geometry": int(getattr(self.args, 'geoanchor_shuffle_geometry', 0)),
            "geoanchor_feature_corruption": getattr(self.args, 'geoanchor_feature_corruption', 'none'),
            "data": getattr(self.args, 'data', ''),
            "dataset": getattr(self.args, 'data_path', ''),
            "mask_rate": float(getattr(self.args, 'mask_rate', 0.0)),
            "mask_type": getattr(self.args, 'mask_type', 'random'),
            "structured_anchor_centric": int(getattr(self.args, 'structured_anchor_centric', 0)),
            "structured_block_len": int(getattr(self.args, 'structured_block_len', 8)),
            "structured_block_len_frac": float(getattr(self.args, 'structured_block_len_frac', 0.25)),
            "structured_num_bursts": int(getattr(self.args, 'structured_num_bursts', 3)),
            "structured_burst_min_len": int(getattr(self.args, 'structured_burst_min_len', 3)),
            "structured_burst_max_len": int(getattr(self.args, 'structured_burst_max_len', 8)),
            "structured_edge_len_frac": float(getattr(self.args, 'structured_edge_len_frac', 0.25)),
            "structured_preserve_mask_rate": int(getattr(self.args, 'structured_preserve_mask_rate', 1)),
            "structured_mask_seed": (
                int(getattr(self.args, 'structured_mask_seed'))
                if getattr(self.args, 'structured_mask_seed', None) is not None
                else int(getattr(self.args, 'mask_seed', 0))
            ),
            "mask_rate_semantics": "observed_rate",
            "seed": int(getattr(self.args, 'seed', 0)),
            "mask_seed": int(getattr(self.args, 'mask_seed', 0)),
            "fill_strategy": getattr(self.args, 'fill_strategy', ''),
            "train_protocol": getattr(self.args, 'train_protocol', ''),
            "stage_loss_scope": getattr(self.args, 'stage_loss_scope', ''),
            "geo_hidden_sample_tokens": int(arrays.get("geo_hidden_sample", np.empty((0, 0))).shape[0]),
        }
        payload = {
            "preds": arrays["preds"],
            "trues": arrays["trues"],
            "masks": arrays["masks"],
            "baselines": arrays["baselines"],
            "metadata": json.dumps(metadata, sort_keys=True),
        }
        if "geo_hidden_sample" in arrays:
            payload["geo_hidden_sample"] = arrays["geo_hidden_sample"]
        np.savez_compressed(path, **payload)
        print(f"geoanchor_prediction_saved:{path}")
        return path

    def _save_acil_prediction_cache(self, setting, split, arrays):
        pred_dir = getattr(self.args, 'acil_prediction_dir', '') or os.path.join(
            os.path.dirname(getattr(self.args, 'result_file', 'result_imputation.txt')) or '.',
            'prediction_cache',
        )
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"{self.args.model_id}_{split}.npz")
        metadata = {
            "setting": setting,
            "model_id": self.args.model_id,
            "split": split,
            "variant": getattr(self.args, 'variant', ''),
            "feature_set": getattr(self.args, 'acil_feature_set', ''),
            "acil_shuffle_gap_geometry": int(getattr(self.args, 'acil_shuffle_gap_geometry', 0)),
            "acil_shuffle_anchor_values": int(getattr(self.args, 'acil_shuffle_anchor_values', 0)),
            "acil_freeze_to_linear": int(getattr(self.args, 'acil_freeze_to_linear', 0)),
            "acil_prior_only": int(self._acil_prior_only()),
            "data": getattr(self.args, 'data', ''),
            "dataset": getattr(self.args, 'data_path', ''),
            "mask_rate": float(getattr(self.args, 'mask_rate', 0.0)),
            "mask_type": getattr(self.args, 'mask_type', 'random'),
            "structured_anchor_centric": int(getattr(self.args, 'structured_anchor_centric', 0)),
            "structured_block_len": int(getattr(self.args, 'structured_block_len', 8)),
            "structured_block_len_frac": float(getattr(self.args, 'structured_block_len_frac', 0.25)),
            "structured_num_bursts": int(getattr(self.args, 'structured_num_bursts', 3)),
            "structured_burst_min_len": int(getattr(self.args, 'structured_burst_min_len', 3)),
            "structured_burst_max_len": int(getattr(self.args, 'structured_burst_max_len', 8)),
            "structured_edge_len_frac": float(getattr(self.args, 'structured_edge_len_frac', 0.25)),
            "structured_preserve_mask_rate": int(getattr(self.args, 'structured_preserve_mask_rate', 1)),
            "structured_mask_seed": (
                int(getattr(self.args, 'structured_mask_seed'))
                if getattr(self.args, 'structured_mask_seed', None) is not None
                else int(getattr(self.args, 'mask_seed', 0))
            ),
            "mask_rate_semantics": "observed_rate",
            "seed": int(getattr(self.args, 'seed', 0)),
            "mask_seed": int(getattr(self.args, 'mask_seed', 0)),
            "fill_strategy": getattr(self.args, 'fill_strategy', ''),
            "train_protocol": getattr(self.args, 'train_protocol', ''),
            "stage_loss_scope": getattr(self.args, 'stage_loss_scope', ''),
            "acil_diag": self._acil_diagnostics(),
        }
        np.savez_compressed(
            path,
            preds=arrays["preds"],
            trues=arrays["trues"],
            masks=arrays["masks"],
            baselines=arrays["baselines"],
            acil_priors=arrays["acil_priors"],
            acil_untrained_priors=arrays["acil_untrained_priors"],
            acil_delta_r=arrays["acil_delta_r"],
            acil_r_hat=arrays["acil_r_hat"],
            acil_offset=arrays["acil_offset"],
            acil_edge_offset=arrays["acil_edge_offset"],
            acil_uncertainty_q=arrays["acil_uncertainty_q"],
            acil_gap_length=arrays["acil_gap_length"],
            acil_gap_type=arrays["acil_gap_type"],
            acil_relative_position=arrays["acil_relative_position"],
            acil_distance_to_nearest_anchor=arrays["acil_distance_to_nearest_anchor"],
            acil_is_edge_gap=arrays["acil_is_edge_gap"],
            metadata=json.dumps(metadata, sort_keys=True),
        )
        print(f"acil_prediction_saved:{path}")
        return path

    def _eval_loop(self, loader, criterion, flag):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            print(f'len_{flag}:{len(loader)}')
            for i, batch_x in tqdm(enumerate(loader)):
                batch_x = self.prepare_batch(batch_x)
                observed_mask = self._build_observed_mask(
                    batch_x.shape,
                    self.args.mask_rate,
                    batch_x.device,
                    seed_offset=i + (0 if flag == 'val' else 100000),
                )
                eval_mask = 1.0 - observed_mask
                inp = self._initial_fill_from_mask(batch_x, observed_mask)
                known_rate = float(observed_mask.mean().item()) * 100.0
                if self._acil_prior_only():
                    pred = self._apply_acil_prior(inp, observed_mask, true_x=batch_x, record=True)
                else:
                    pred = self._progressive_impute(inp, observed_mask, batch_x, known_rate)
                loss = criterion(pred[eval_mask == 1], batch_x[eval_mask == 1])
                total_loss.append(float(loss.item()))
        self.model.train()
        return float(np.average(total_loss)) if total_loss else 0.0

    def vali(self, vali_data, vali_loader, criterion):
        return self._eval_loop(vali_loader, criterion, 'val')

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self._prepare_memory_bank(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            print(f'len_train:{len(train_loader)}')

            for i, batch_x in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = self.prepare_batch(batch_x)
                if self._acil_prior_only():
                    observed_mask = self._build_observed_mask(
                        batch_x.shape,
                        self.args.mask_rate,
                        batch_x.device,
                        seed_offset=epoch * 100000 + i,
                    )
                    eval_mask = 1.0 - observed_mask
                    inp = self._initial_fill_from_mask(batch_x.detach(), observed_mask)
                    outputs = self._apply_acil_prior(inp, observed_mask, true_x=batch_x, record=True)
                    all_loss = self._training_objective(
                        criterion,
                        outputs[eval_mask == 1],
                        batch_x[eval_mask == 1],
                    )
                elif getattr(self.args, 'train_protocol', 'stage') == 'mask':
                    observed_rate = (
                        self.args.mask_rate
                        if getattr(self.args, 'train_mask_rate', None) is None
                        else float(self.args.train_mask_rate)
                    )
                    observed_mask = self._build_observed_mask(
                        batch_x.shape,
                        observed_rate,
                        batch_x.device,
                        seed_offset=epoch * 100000 + i,
                    )
                    eval_mask = 1.0 - observed_mask
                    observed_values = batch_x.detach()
                    inp = self._initial_fill_from_mask(observed_values, observed_mask)
                    known_rate = float(observed_mask.mean().item()) * 100.0
                    outputs = self._progressive_impute(inp, observed_mask, observed_values, known_rate)
                    all_loss = self._training_objective(
                        criterion,
                        outputs[eval_mask == 1],
                        batch_x[eval_mask == 1],
                    )
                    all_loss = all_loss + self._salt_auxiliary_loss(criterion, batch_x)
                else:
                    stages = self._build_training_stages(batch_x)
                    rates = self._stage_rates()
                    stage_weights = self._stage_loss_weights(len(stages) - 1, self.device)
                    all_loss = torch.tensor(0.0, device=self.device)

                    for step in range(len(stages) - 1):
                        known_rate = rates[step]
                        target_rate = rates[step + 1]
                        if getattr(self.args, 'mask_aware', 0) or self._geoanchor_active() or self._acil_active():
                            model_mask = self._stage_observed_mask_for_rate(
                                batch_x.shape,
                                known_rate,
                                batch_x.device,
                            ).float()
                        else:
                            model_mask = None
                        model_inp = stages[step]
                        if self._acil_active() and model_mask is not None:
                            model_inp = self._apply_acil_prior(
                                stages[step],
                                model_mask,
                                true_x=batch_x,
                                record=True,
                            )
                        outputs = self.model(model_inp, None, known_rate, target_rate, model_mask)
                        loss_mask = self._stage_loss_mask(batch_x.shape, known_rate, target_rate, batch_x.device)
                        if loss_mask is None:
                            step_loss = self._training_objective(criterion, outputs, stages[step + 1])
                        else:
                            step_loss = self._training_objective(
                                criterion,
                                outputs[loss_mask],
                                stages[step + 1][loss_mask],
                            )
                        step_loss = step_loss + self._salt_auxiliary_loss(criterion, stages[step + 1])
                        all_loss = all_loss + stage_weights[step] * step_loss

                train_loss.append(float(all_loss.item()))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, all_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if all_loss.requires_grad:
                    all_loss.backward()
                    self._record_acil_gradient_norms()
                    self._record_geoattn_gradient_norms()
                    model_optim.step()
                else:
                    print("skip_backward_no_trainable_graph:1")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss)) if train_loss else 0.0
            vali_loss = self._eval_loop(vali_loader, criterion, 'val')
            test_loss = self._eval_loop(test_loader, criterion, 'test')
            torch.cuda.empty_cache()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print("train_time_sec:{:.6f}".format(time.time() - train_started))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if self.memory_bank is None and getattr(self.args, 'fill_strategy', 'linear') == 'memory':
            train_data, _ = self._get_data(flag='train')
            self._prepare_memory_bank(train_data)
        checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        print('loading model')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        self.model.eval()
        inference_started = time.time()
        test_arrays = self._collect_predictions('test')
        preds = test_arrays["preds"]
        trues = test_arrays["trues"]
        masks = test_arrays["masks"]

        print('test shape:', preds.shape, trues.shape)
        nmae, nrmse, kl = metric(preds[masks == 0], trues[masks == 0])
        print('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))
        print("inference_time_sec:{:.6f}".format(time.time() - inference_started))
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            print("gpu_memory_peak_mb:{:.3f}".format(peak_mb))
        parameter_count = sum(param.numel() for param in self.model.parameters())
        print("parameter_count:{}".format(parameter_count))
        salt_diag = self._salt_diagnostics()
        if salt_diag:
            print("salt_diag:{}".format(json.dumps(salt_diag, sort_keys=True)))
        care_diag = self._care_diagnostics()
        if care_diag:
            print("care_diag:{}".format(json.dumps(care_diag, sort_keys=True)))
        acil_diag = self._acil_diagnostics()
        if acil_diag:
            print("acil_diag:{}".format(json.dumps(acil_diag, sort_keys=True)))
        geoattn_diag = self._geoattn_diagnostics()
        if geoattn_diag:
            print("geoattn_diag:{}".format(json.dumps(geoattn_diag, sort_keys=True)))
        if int(getattr(self.args, 'care_save_predictions', 0)):
            splits = [
                item.strip()
                for item in str(getattr(self.args, 'care_prediction_splits', 'test')).split(',')
                if item.strip()
            ]
            cache = {'test': test_arrays}
            for split in splits:
                if split not in {'train', 'val', 'test'}:
                    print(f"care_prediction_skip_unknown_split:{split}")
                    continue
                arrays = cache.get(split)
                if arrays is None:
                    arrays = self._collect_predictions(split)
                    cache[split] = arrays
                self._save_care_prediction_cache(setting, split, arrays)
        if int(getattr(self.args, 'geoanchor_save_diagnostics', 0)):
            splits = [
                item.strip()
                for item in str(getattr(self.args, 'geoanchor_prediction_splits', 'test')).split(',')
                if item.strip()
            ]
            cache = {'test': test_arrays}
            for split in splits:
                if split not in {'train', 'val', 'test'}:
                    print(f"geoanchor_prediction_skip_unknown_split:{split}")
                    continue
                arrays = cache.get(split)
                if arrays is None:
                    arrays = self._collect_predictions(split)
                    cache[split] = arrays
                self._save_geoanchor_prediction_cache(setting, split, arrays)
        if int(getattr(self.args, 'acil_save_diagnostics', 0)):
            splits = [
                item.strip()
                for item in str(getattr(self.args, 'acil_prediction_splits', 'test')).split(',')
                if item.strip()
            ]
            cache = {'test': test_arrays}
            for split in splits:
                if split not in {'train', 'val', 'test'}:
                    print(f"acil_prediction_skip_unknown_split:{split}")
                    continue
                arrays = cache.get(split)
                if arrays is None:
                    arrays = self._collect_predictions(split)
                    cache[split] = arrays
                self._save_acil_prediction_cache(setting, split, arrays)
        with open(getattr(self.args, 'result_file', "result_imputation.txt"), 'a') as f:
            f.write(setting + "  \n")
            f.write('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))
            f.write('\n\n')
