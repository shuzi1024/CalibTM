import argparse
import os
import torch
from exp.exp_abilene import Exp_Imputation as Exp_Abilene
from exp.exp_geant import Exp_Imputation as Exp_GEANT
from exp.exp_imputation import Exp_Imputation as Exp_Default
from exp.exp_wsdream import Exp_Imputation as Exp_WSDREAM
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TM-LLM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/Abilene/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='abilene_tm.csv', help='data file')
parser.add_argument('--real_data_path', type=str, default='abilene.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--result_file', type=str, default='result_imputation.txt',
                    help='file used for appended test metrics')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset')

# imputation task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
parser.add_argument('--mask_type', type=str, default='random',
                    choices=['random', 'internal_block', 'burst', 'edge', 'mixed_structured'],
                    help='observation mask geometry; random preserves the original behavior')
parser.add_argument('--structured_anchor_centric', type=int, default=0,
                    help='enable anchor-centric structured mask generation when mask_type is not random')
parser.add_argument('--structured_block_len', type=int, default=8,
                    help='target controlled internal missing block length')
parser.add_argument('--structured_block_len_frac', type=float, default=0.25,
                    help='fallback internal block length as a fraction of seq_len')
parser.add_argument('--structured_num_bursts', type=int, default=3,
                    help='target number of controlled burst gaps')
parser.add_argument('--structured_burst_min_len', type=int, default=3,
                    help='minimum controlled burst gap length')
parser.add_argument('--structured_burst_max_len', type=int, default=8,
                    help='maximum controlled burst gap length')
parser.add_argument('--structured_edge_len_frac', type=float, default=0.25,
                    help='prefix/suffix edge gap length as a fraction of seq_len')
parser.add_argument('--structured_preserve_mask_rate', type=int, default=1,
                    help='preserve the original observed-budget semantics of mask_rate')
parser.add_argument('--structured_mask_seed', type=int, default=None,
                    help='seed for structured mask geometry; defaults to mask_seed when unset')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--train_loss', type=str, default='smape',
                    choices=['smape', 'mae', 'mse', 'huber', 'nmae'],
                    help='training loss for progressive imputation; default preserves the current implementation')
parser.add_argument('--distribution_loss_weight', type=float, default=0.0,
                    help='optional KL-style distribution-preservation auxiliary loss weight')
parser.add_argument('--mass_loss_weight', type=float, default=0.0,
                    help='optional total-traffic mass preservation auxiliary loss weight')
parser.add_argument('--distribution_loss_eps', type=float, default=1e-8,
                    help='epsilon for distribution-preservation auxiliary losses')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--seed', type=int, default=2021, help='global random seed')
parser.add_argument('--mask_seed', type=int, default=2024, help='evaluation mask seed')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# LLM Configuration
parser.add_argument('--gpt_layers', '--gpt_layer', dest='gpt_layers', type=int, default=6)
parser.add_argument('--depth_router', type=int, default=0,
                    help='enable observation-conditioned fusion over GPT hidden depths')
parser.add_argument('--depth_router_layers', type=str, default='1,2,3,6',
                    help='comma-separated GPT layer indices fused by the depth router')
parser.add_argument('--depth_router_hidden', type=int, default=64,
                    help='hidden dimension for the depth-router MLP')
parser.add_argument('--depth_router_dropout', type=float, default=0.1,
                    help='dropout inside the depth-router MLP')
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--llm_dim', type=int, default=768)
parser.add_argument('--llm_model', type=str, default="gpt2")
parser.add_argument('--initial_observed_rate', type=float, default=0.02,
                    help='initial observation ratio for training stages; values <=1 are fractions')
parser.add_argument('--stage_loss_gamma', type=float, default=1.0,
                    help='multiplicative weighting across progressive training stages; >1 emphasizes later finer stages')
parser.add_argument('--stage_loss_scope', type=str, default='full',
                    choices=['full', 'target_observed', 'new_observed'],
                    help='which entries supervise stage-to-stage training')
parser.add_argument('--fill_strategy', type=str, default='linear',
                    choices=['linear', 'nearest', 'mean', 'zero', 'memory'],
                    help='initial fill strategy for missing entries during validation/testing')
parser.add_argument('--memory_fill_k', type=int, default=128,
                    help='number of training windows used by memory fill; active when fill_strategy=memory')
parser.add_argument('--memory_fill_topk', type=int, default=4,
                    help='top-k nearest memory windows blended for memory fill')
parser.add_argument('--memory_fill_temperature', type=float, default=0.03,
                    help='softmax temperature for memory fill retrieval')
parser.add_argument('--memory_fill_blend', type=float, default=1.0,
                    help='blend weight for memory fill prior versus linear interpolation')
parser.add_argument('--mask_aware', type=int, default=0,
                    help='add an observation-mask token embedding to ARI-LLM Flow2Vec inputs')
parser.add_argument('--flow_encoder', type=str, default='lstm',
                    choices=['lstm', 'ssm', 'ssm_lite', 'tcn', 'conv', 'patch', 'patch_transformer'],
                    help='Flow2Vec token encoder: default LSTM, per-flow SSM-lite, temporal conv, or patch Transformer')
parser.add_argument('--flow_encoder_layers', type=int, default=1,
                    help='number of recurrent SSM-lite layers for Flow2Vec')
parser.add_argument('--flow_encoder_heads', type=int, default=4,
                    help='attention heads for patch Transformer Flow2Vec')
parser.add_argument('--flow_patch_len', type=int, default=10,
                    help='temporal patch length for patch Transformer Flow2Vec')
parser.add_argument('--flow_patch_stride', type=int, default=5,
                    help='temporal patch stride for patch Transformer Flow2Vec')
parser.add_argument('--flow_kernel_size', type=int, default=5,
                    help='temporal convolution kernel size for conv Flow2Vec')
parser.add_argument('--mask_embed_scale', type=float, default=0.05,
                    help='initial residual scale for observation-mask token embedding')
parser.add_argument('--mask_fusion', type=str, default='add',
                    choices=['add', 'concat', 'gate'],
                    help='mask-aware fusion type: add residual, concatenate value/mask channels, or gate value tokens')
parser.add_argument('--mask_prompt_tokens', type=int, default=0,
                    help='number of observation-mask soft prompt tokens inserted before Flow2Vec tokens')
parser.add_argument('--mask_prompt_mode', type=str, default='mask',
                    choices=['mask', 'learned'],
                    help='mask: encode the observed mask into soft prompts; learned: mask-free prompt-token control')
parser.add_argument('--mask_prompt_scale', type=float, default=0.05,
                    help='initial scale for observation-mask soft prompt tokens')
parser.add_argument('--mask_prompt_dropout', type=float, default=0.1,
                    help='dropout in the observation-mask prompt encoder')
parser.add_argument('--train_protocol', type=str, default='stage',
                    choices=['stage', 'mask'],
                    help='stage: original stage-to-stage supervision; mask: train on simulated missing entries')
parser.add_argument('--train_mask_rate', type=float, default=None,
                    help='observed rate for mask training protocol; defaults to --mask_rate')
parser.add_argument('--variant', type=str, default='default',
                    help='experiment variant label; SALT-ARI variants enable salt modules when set to salt_* or same_param_mlp_control')
parser.add_argument('--geoanchor_enable', type=int, default=0,
                    help='enable GeoAnchor observation-geometry-aware Flow2Vec tokenization')
parser.add_argument('--geoanchor_feature_set', type=str, default='full',
                    choices=['full', 'value_extra', 'mask_only', 'no_anchor_values'],
                    help='GeoAnchor feature set for AGT and same-parameter controls')
parser.add_argument('--geoanchor_shuffle_geometry', type=int, default=0,
                    help='shuffle non-value geometry features across batch/flow tokens')
parser.add_argument('--geoanchor_feature_corruption', type=str, default='none',
                    choices=['none', 'zero_geometry', 'shuffled_geometry', 'random_geometry', 'scaled_geometry_x10'],
                    help='runtime GeoAnchor feature corruption for diagnostics; default preserves old behavior')
parser.add_argument('--geoanchor_uncertainty', type=str, default='equal_weight',
                    choices=['equal_weight'],
                    help='heuristic interpolation uncertainty formula')
parser.add_argument('--geoanchor_hidden', type=int, default=64,
                    help='hidden width for the per-step GeoFlow2Vec channel projector')
parser.add_argument('--geoanchor_report_buckets', type=int, default=0,
                    help='reserved flag for GeoAnchor bucket reporting in experiment scripts')
parser.add_argument('--geoanchor_save_diagnostics', type=int, default=0,
                    help='save prediction, target, mask, and linear baseline arrays for GeoAnchor diagnostics')
parser.add_argument('--geoanchor_prediction_dir', type=str, default='',
                    help='directory for GeoAnchor prediction caches')
parser.add_argument('--geoanchor_prediction_splits', type=str, default='test',
                    help='comma-separated splits to save when geoanchor_save_diagnostics=1')
parser.add_argument('--geoanchor_save_hidden_sample', type=int, default=0,
                    help='save a bounded GeoFlow2Vec hidden-token sample for feature-use audits')
parser.add_argument('--geoanchor_hidden_sample_tokens', type=int, default=2048,
                    help='maximum number of flattened GeoFlow2Vec hidden tokens to save per prediction cache')
parser.add_argument('--geoanchor_v2_enable', type=int, default=0,
                    help='enable GeoAnchor-ARI v2 ACIL prior replacement')
parser.add_argument('--acil_enable', type=int, default=0,
                    help='enable Anchor-Conditioned Interpolation Layer before ARI/stagegate')
parser.add_argument('--acil_feature_set', type=str, default='full',
                    choices=['full', 'value_only', 'no_anchor_values'],
                    help='ACIL MLP input feature set')
parser.add_argument('--acil_shuffle_gap_geometry', type=int, default=0,
                    help='shuffle gap geometry scalar features while keeping anchors and values fixed')
parser.add_argument('--acil_shuffle_anchor_values', type=int, default=0,
                    help='shuffle anchor-value features for optional ACIL corruption control')
parser.add_argument('--acil_freeze_to_linear', type=int, default=0,
                    help='force ACIL to return the LinearInterp baseline exactly')
parser.add_argument('--acil_prior_only', type=int, default=0,
                    help='train/evaluate only ACIL prior without Flow2Vec, LLM body, or stagegate decoder')
parser.add_argument('--acil_hidden', type=int, default=64,
                    help='hidden width for ACIL geometry MLP')
parser.add_argument('--acil_beta_r', type=float, default=0.25,
                    help='maximum interpolation weight adjustment scale')
parser.add_argument('--acil_beta_o', type=float, default=0.10,
                    help='offset scale for internal gaps')
parser.add_argument('--acil_beta_e', type=float, default=0.10,
                    help='edge extrapolation offset scale')
parser.add_argument('--acil_use_edge_extrapolation', type=int, default=1,
                    help='use one-sided anchor extrapolation for left/right edge gaps')
parser.add_argument('--acil_save_diagnostics', type=int, default=0,
                    help='save ACIL prediction/prior/diagnostic arrays')
parser.add_argument('--acil_report_buckets', type=int, default=0,
                    help='reserved flag for ACIL bucket reporting in experiment scripts')
parser.add_argument('--acil_load_prior_ckpt', type=str, default='',
                    help='load ACIL layer weights from a prior-only checkpoint before downstream training')
parser.add_argument('--acil_freeze_prior', type=int, default=0,
                    help='freeze ACIL layer weights after loading or initialization')
parser.add_argument('--bucket_min_count', type=int, default=500,
                    help='minimum missing-entry count for bucket validity')
parser.add_argument('--bucket_min_fraction', type=float, default=0.01,
                    help='minimum missing-entry fraction for bucket validity')
parser.add_argument('--acil_prediction_dir', type=str, default='',
                    help='directory for ACIL prediction caches')
parser.add_argument('--acil_prediction_splits', type=str, default='test',
                    help='comma-separated splits to save when acil_save_diagnostics=1')
parser.add_argument('--test_setting_override', type=str, default='',
                    help='optional checkpoint setting name for test-only runs')
parser.add_argument('--topology_embed', type=int, default=0,
                    help='add topology-aware source/destination/pair embeddings to Flow2Vec tokens')
parser.add_argument('--topology_init', type=float, default=0.1,
                    help='initial scale for topology embeddings')
parser.add_argument('--flow_refine_layers', type=int, default=0,
                    help='number of bidirectional flow refinement Transformer layers after the LLM')
parser.add_argument('--flow_refine_heads', type=int, default=4,
                    help='number of heads for bidirectional flow refinement')
parser.add_argument('--flow_refine_dropout', type=float, default=0.1,
                    help='dropout for bidirectional flow refinement')
parser.add_argument('--flow_refine_init', type=float, default=0.1,
                    help='initial residual scale for bidirectional flow refinement')
parser.add_argument('--residual_head', type=int, default=0,
                    help='add a zero-initialized residual reconstruction head after the original ARI head')
parser.add_argument('--residual_scale', type=float, default=1.0,
                    help='scale applied to the residual reconstruction head')
parser.add_argument('--low_rank_head', type=int, default=0,
                    help='add a rank-constrained mixture-of-experts reconstruction head')
parser.add_argument('--low_rank_dim', type=int, default=16,
                    help='rank per expert for the low-rank reconstruction head')
parser.add_argument('--low_rank_experts', type=int, default=1,
                    help='number of low-rank decoder experts')
parser.add_argument('--low_rank_hidden', type=int, default=256,
                    help='hidden dimension inside the low-rank reconstruction head')
parser.add_argument('--low_rank_dropout', type=float, default=0.1,
                    help='dropout inside the low-rank reconstruction head')
parser.add_argument('--low_rank_scale', type=float, default=1.0,
                    help='initial residual scale for the low-rank reconstruction head')
parser.add_argument('--low_rank_replace', type=int, default=0,
                    help='replace the original FlattenHead with the low-rank head instead of adding a residual')
parser.add_argument('--residual_refine', type=int, default=0,
                    help='predict a coarse-to-fine residual added to the current normalized fill')
parser.add_argument('--residual_refine_scale', type=float, default=1.0,
                    help='initial scale for residual coarse-to-fine refinement')
parser.add_argument('--residual_refine_block', type=int, default=0,
                    help='use a stage-aware uncertainty/topology-conditioned residual refinement block')
parser.add_argument('--residual_refine_stage', type=int, default=1,
                    help='condition the residual refinement block on known/target stage rates')
parser.add_argument('--residual_refine_gate', type=int, default=1,
                    help='use a per-flow temporal gate inside the residual refinement block')
parser.add_argument('--residual_refine_anchor_observed', type=int, default=1,
                    help='zero residual updates on observed entries when an observation mask is available')
parser.add_argument('--residual_refine_topology_layers', type=int, default=0,
                    help='number of topology-biased attention layers inside the residual refinement block')
parser.add_argument('--residual_refine_topology_heads', type=int, default=4,
                    help='number of heads for residual refinement topology attention')
parser.add_argument('--residual_refine_topology_dropout', type=float, default=0.1,
                    help='dropout for residual refinement topology attention')
parser.add_argument('--residual_refine_topology_init', type=float, default=0.05,
                    help='initial residual scale for residual refinement topology attention')
parser.add_argument('--residual_refine_topology_bias_init', type=float, default=0.03,
                    help='initial relation bias for residual refinement topology attention')
parser.add_argument('--residual_refine_topology_zero_init', type=int, default=1,
                    help='zero-initialize residual refinement topology projections for identity startup')
parser.add_argument('--residual_refine_topology_stage_bias', type=int, default=1,
                    help='add stage-specific relation bias in residual refinement topology attention')
parser.add_argument('--residual_refine_topology_local_only', type=int, default=1,
                    help='mask unrelated OD-flow pairs in residual refinement topology attention')
parser.add_argument('--residual_refine_topology_ffn', type=int, default=1,
                    help='whether to use FFN sublayers in residual refinement topology attention')
parser.add_argument('--residual_refine_blend', type=int, default=0,
                    help='blend vanilla direct reconstruction with residual refinement instead of replacing the head')
parser.add_argument('--residual_refine_blend_init', type=float, default=0.5,
                    help='initial residual-refinement mixture weight for the blend gate')
parser.add_argument('--residual_refine_blend_stage', type=int, default=1,
                    help='condition the residual/direct blend gate on known/target stage rates')
parser.add_argument('--head_zero_init', type=int, default=0,
                    help='zero-initialize the main reconstruction head final layer')
parser.add_argument('--salt_latent_slots', type=int, default=16,
                    help='SALT latent traffic slot count')
parser.add_argument('--salt_num_heads', type=int, default=4,
                    help='SALT cross-attention head count')
parser.add_argument('--salt_latent_mixer', type=str, default='tiny_l1',
                    choices=['none', 'tiny_l1', 'tiny_l2', 'gpt2_l1'],
                    help='SALT latent slot mixer type')
parser.add_argument('--salt_alpha_init', type=float, default=0.05,
                    help='initial learnable SALT global residual scale')
parser.add_argument('--salt_gate_bias', type=float, default=-2.0,
                    help='initial SALT uncertainty-gate bias')
parser.add_argument('--salt_flow_drop_rate', type=float, default=0.15,
                    help='flow sampling rate for SALT counterfactual local-drop auxiliary loss')
parser.add_argument('--salt_cf_loss_weight', type=float, default=0.03,
                    help='weight for SALT counterfactual local-drop auxiliary loss')
parser.add_argument('--salt_gate_loss_weight', type=float, default=0.01,
                    help='weight for SALT gate weak-supervision loss')
parser.add_argument('--salt_disable_gate', type=int, default=0,
                    help='disable SALT uncertainty gate and use a fixed gate of one')
parser.add_argument('--salt_disable_cf_loss', type=int, default=0,
                    help='disable SALT counterfactual local-drop auxiliary loss')
parser.add_argument('--salt_disable_latent_mixer', type=int, default=0,
                    help='disable the latent slot mixer while keeping flow-latent cross attention')
parser.add_argument('--salt_same_param_control', type=int, default=0,
                    help='use the SALT same-parameter MLP control branch instead of latent cross attention')
parser.add_argument('--salt_dropout', type=float, default=0.1,
                    help='dropout used by SALT coupler, mixer, and residual head')
parser.add_argument('--care_latent_slots', type=int, default=16,
                    help='CARE latent traffic slot count')
parser.add_argument('--care_num_heads', type=int, default=4,
                    help='CARE latent cross-attention head count')
parser.add_argument('--care_latent_mixer', type=str, default='tiny_l1',
                    choices=['none', 'tiny_l1', 'tiny_l2', 'gpt2_l1'],
                    help='CARE latent slot mixer type')
parser.add_argument('--care_global_flow_drop_rate', type=float, default=0.15,
                    help='flow token dropout rate for standalone CARE global expert pretraining')
parser.add_argument('--care_router_loss_weight', type=float, default=0.1,
                    help='offline CARE router BCE loss weight')
parser.add_argument('--care_usage_loss_weight', type=float, default=0.01,
                    help='offline CARE router usage regularization weight')
parser.add_argument('--care_usage_min', type=float, default=0.03,
                    help='minimum target router usage for offline usage regularization')
parser.add_argument('--care_router_margin', type=float, default=0.0,
                    help='margin for router target y=1[e_global < e_local - margin]')
parser.add_argument('--care_router_type', type=str, default='per_flow',
                    choices=['per_flow', 'per_flow_time'],
                    help='CARE router granularity; per_flow_time is handled by offline analysis if implemented')
parser.add_argument('--care_freeze_experts', type=int, default=1,
                    help='freeze experts during offline CARE router training')
parser.add_argument('--care_high_uncertainty_sampling', type=int, default=0,
                    help='reserved flag for high-uncertainty global expert sampling')
parser.add_argument('--care_save_predictions', type=int, default=0,
                    help='save prediction caches for CARE oracle/router analysis')
parser.add_argument('--care_prediction_dir', type=str, default='',
                    help='directory for CARE prediction caches')
parser.add_argument('--care_prediction_splits', type=str, default='test',
                    help='comma-separated splits to save when care_save_predictions=1')
parser.add_argument('--care_load_local_ckpt', type=str, default='',
                    help='reserved path for loading a local CARE expert checkpoint')
parser.add_argument('--care_load_global_ckpt', type=str, default='',
                    help='reserved path for loading a global CARE expert checkpoint')
parser.add_argument('--care_anchor_observed', type=int, default=1,
                    help='zero CARE global residual on observed entries')
parser.add_argument('--care_dropout', type=float, default=0.1,
                    help='dropout used by CARE coupler, mixer, residual head, and offline router')
parser.add_argument('--temporal_adapter', type=int, default=0,
                    help='add a zero-initialized multi-scale temporal residual adapter before Flow2Vec')
parser.add_argument('--temporal_adapter_hidden', type=int, default=8,
                    help='hidden channels per temporal adapter kernel')
parser.add_argument('--temporal_adapter_kernels', type=str, default='3,5,7',
                    help='comma-separated positive odd kernel sizes for the temporal adapter')
parser.add_argument('--temporal_adapter_dropout', type=float, default=0.1,
                    help='dropout inside the temporal adapter')
parser.add_argument('--temporal_adapter_scale', type=float, default=1.0,
                    help='residual scale for the temporal adapter')
parser.add_argument('--flow_attention_layers', type=int, default=0,
                    help='number of topology-biased flow attention adapter layers before the LLM')
parser.add_argument('--flow_attention_heads', type=int, default=4,
                    help='number of heads for topology-biased flow attention')
parser.add_argument('--flow_attention_dropout', type=float, default=0.1,
                    help='dropout in topology-biased flow attention')
parser.add_argument('--flow_attention_init', type=float, default=0.1,
                    help='initial residual scale for topology-biased flow attention')
parser.add_argument('--flow_attention_bias_init', type=float, default=0.05,
                    help='initial structural attention-logit bias for related OD flows')
parser.add_argument('--flow_attention_stage_bias', type=int, default=1,
                    help='add stage-specific relation bias in topology-biased flow attention')
parser.add_argument('--flow_attention_zero_init', type=int, default=0,
                    help='zero-initialize flow attention residual projections for identity startup')
parser.add_argument('--flow_attention_local_only', type=int, default=0,
                    help='mask unrelated OD-flow pairs in topology-biased flow attention')
parser.add_argument('--flow_attention_ffn', type=int, default=1,
                    help='whether to use the FFN sublayer in topology-biased flow attention')
parser.add_argument('--reliability_attention_layers', type=int, default=0,
                    help='number of observation-reliability-biased flow attention layers before the LLM')
parser.add_argument('--reliability_attention_heads', type=int, default=4,
                    help='number of heads for observation-reliability-biased flow attention')
parser.add_argument('--reliability_attention_dropout', type=float, default=0.1,
                    help='dropout in observation-reliability-biased flow attention')
parser.add_argument('--reliability_attention_init', type=float, default=0.1,
                    help='initial residual scale for observation-reliability-biased flow attention')
parser.add_argument('--reliability_attention_key_bias_init', type=float, default=0.05,
                    help='initial attention-logit bias from uncertain query flows to reliable key flows')
parser.add_argument('--reliability_attention_coobs', type=int, default=1,
                    help='add co-observation similarity bias in observation-reliability-biased flow attention')
parser.add_argument('--reliability_attention_coobs_bias_init', type=float, default=0.02,
                    help='initial attention-logit bias for co-observed flow pairs')
parser.add_argument('--reliability_attention_zero_init', type=int, default=0,
                    help='zero-initialize reliability attention residual projections for identity startup')
parser.add_argument('--reliability_attention_ffn', type=int, default=1,
                    help='whether to use the FFN sublayer in reliability-biased flow attention')
parser.add_argument('--geoattn_enable', type=int, default=0,
                    help='enable GeoAttn-lite observation-geometry-biased flow-token attention')
parser.add_argument('--geoattn_bias_type', type=str, default='true',
                    choices=['true', 'shuffled', 'random', 'zero'],
                    help='pairwise observation-geometry bias type for GeoAttn-lite')
parser.add_argument('--geoattn_gamma_init', type=float, default=0.1,
                    help='initial additive attention-logit geometry-bias scale')
parser.add_argument('--geoattn_gamma_learnable', type=int, default=1,
                    help='learn the GeoAttn geometry-bias scale gamma')
parser.add_argument('--geoattn_freeze_to_zero', type=int, default=0,
                    help='force GeoAttn-lite to identity for baseline reproduction')
parser.add_argument('--geoattn_identity_when_gamma_zero', type=int, default=1,
                    help='make non-learnable gamma=0 an identity path for smoke/baseline checks')
parser.add_argument('--geoattn_descriptor_set', type=str, default='full',
                    choices=['basic', 'full'],
                    help='observation-geometry descriptor feature set')
parser.add_argument('--geoattn_descriptor_normalize', type=str, default='batch',
                    choices=['none', 'batch'],
                    help='normalization for per-flow geometry descriptors')
parser.add_argument('--geoattn_standardize_bias', type=int, default=1,
                    help='standardize pairwise geometry similarity before adding it to attention logits')
parser.add_argument('--geoattn_insert_location', type=str, default='pre_decoder',
                    choices=['llm_attention', 'pre_decoder', 'post_flow2vec'],
                    help='GeoAttn-lite insertion point; llm_attention falls back to pre_decoder in this implementation')
parser.add_argument('--geoattn_num_layers', type=int, default=1,
                    help='number of GeoAttn-lite flow-token attention layers')
parser.add_argument('--geoattn_num_heads', type=int, default=4,
                    help='number of GeoAttn-lite attention heads')
parser.add_argument('--geoattn_dropout', type=float, default=0.1,
                    help='dropout inside GeoAttn-lite attention/FFN')
parser.add_argument('--geoattn_residual_init', type=float, default=0.1,
                    help='initial residual scale for GeoAttn-lite layer outputs')
parser.add_argument('--geoattn_zero_init', type=int, default=0,
                    help='zero-initialize GeoAttn-lite output projections for identity startup')
parser.add_argument('--geoattn_ffn', type=int, default=1,
                    help='use FFN sublayers inside GeoAttn-lite')
parser.add_argument('--geoattn_same_param_control', type=int, default=0,
                    help='keep GeoAttn-lite parameters but replace geometry bias with zero bias')
parser.add_argument('--geoattn_save_diagnostics', type=int, default=0,
                    help='collect GeoAttn-lite attention/geometry diagnostics')
parser.add_argument('--geoattn_diag_topk', type=int, default=5,
                    help='top-k geometry-similar key flows used by GeoAttn diagnostics')

# Adversarial learning
parser.add_argument('--Lambda', type=int, default=2)

# the number of samples
parser.add_argument('--sample_num', type=int, default=1000)
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

exp_map = {
    'net_traffic_abilene': Exp_Abilene,
    'net_traffic_geant': Exp_GEANT,
    'net_traffic_trans': Exp_WSDREAM,
    'net_traffic_wsdream': Exp_WSDREAM,
}
Exp = exp_map.get(args.data, Exp_Default)

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.features,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.data,
        args.features,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)
    if getattr(args, 'test_setting_override', ''):
        setting = args.test_setting_override

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
