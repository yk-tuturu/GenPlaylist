"""
main.py - DISCO项目的主入口文件
功能：用于训练、评估离散扩散模型来生成产品bundle
支持四种模式：train(训练)、rec_eval(推荐评估)、ppl_eval(困惑度评估)、sample_eval(采样评估)
"""

# ============ 标准库导入 ============
import os  # 操作系统接口，用于文件路径等操作
import hashlib
import numpy as np  # 数值计算库，用于数组操作
import time  # 时间相关操作
import json  # JSON数据处理
import warnings  # 警告控制
from collections import defaultdict, OrderedDict  # 特殊字典类型，用于结果统计
from datetime import datetime, timezone
from pathlib import Path

# ============ 第三方库导入 ============
import fsspec  # 文件系统规范库，用于跨平台文件系统操作
import hydra  # 配置管理框架，用于管理实验配置参数
import lightning as L  # PyTorch Lightning，简化深度学习训练流程的框架
import omegaconf  # 配置管理库，与hydra配合使用
import rich.syntax  # 终端美化库（用于彩色输出）
import rich.tree  # 树状结构显示
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # PyTorch数据加载器
from tqdm import tqdm  # 进度条显示库
from safetensors.torch import load_file  # 安全的模型权重加载库

# ============ 项目内部模块导入 ============
import dataloader  # 数据加载模块，负责数据预处理和批量加载
import diffusion  # 扩散模型核心模块，定义了扩散过程和模型结构
from evaluator import Evaluator  # 评估器，计算推荐指标（recall、precision等）
from genplaylist_tokenizer import GenPlaylistTokenizer
import utils  # 工具函数集合
from dataset import AbstractDataset  # 抽象数据集类，负责加载原始数据
from warmstart import apply_ddbc_warmstart
from prepared_data import load_prepared_tokenized_dataset
from evaluation_protocol import OFFICIAL_EVALUATION_PROTOCOL


# ============ HuggingFace Dataset包装器 ============
class TorchDatasetWrapper(torch.utils.data.Dataset):
    """
    将HuggingFace Dataset包装为PyTorch Dataset
    这样Lightning可以正确应用DistributedSampler
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

# 忽略警告信息
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存


# ============ OmegaConf自定义解析器注册 ============
# OmegaConf是配置管理库，这里注册自定义解析器，允许在配置文件中使用动态值

# 注册'cwd'解析器：在配置中可以使用${cwd:}获取当前工作目录
omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)

# 注册'device_count'解析器：在配置中可以使用${device_count:}获取可用GPU数量
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)

# 注册'eval'解析器：在配置中可以使用${eval:}执行Python表达式
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)

# 注册'div_up'解析器：向上取整除法，例如${div_up:10,3}返回4
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  """
  从检查点加载预训练模型

  Args:
      config: 配置对象，包含模型架构和检查点路径等信息
      tokenizer: 分词器对象，用于处理token

  Returns:
      加载好的扩散模型实例
  """
  # 如果使用HuggingFace模型，直接创建新模型实例并移到CUDA设备
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  # 否则从指定的检查点路径加载已训练的模型
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


def _checkpoint_history_condition(checkpoint_path):
  """Return the history_condition a checkpoint was trained with.

  load_from_checkpoint(config=...) replaces the saved hyperparameters with the
  evaluation config, so the trained value is read from the file itself.
  Checkpoints trained before the conditioning ablation are ``full``.
  """
  checkpoint = torch.load(
    checkpoint_path, map_location='cpu', weights_only=False)
  trained_config = checkpoint.get('hyper_parameters', {}).get('config', {}) or {}
  return str(trained_config.get('history_condition', 'full'))


@L.pytorch.utilities.rank_zero_only  # 装饰器：仅在主进程（rank 0）执行，用于分布式训练
def _print_batch(train_ds, test_ds, tokenizer, k=64):
  """
  打印训练和验证数据批次的样例（调试用）

  Args:
      train_ds: 训练数据加载器
      test_ds: 测试数据加载器
      tokenizer: 分词器
      k: 打印前k个和后k个token
  """
  for dl_type, dl in [
    ('train', train_ds), ('test', test_ds)]:
    print(f'Printing {dl_type} dataloader batch.')

    batch = next(iter(dl))  # 获取一个批次的数据
    first = batch['input_ids'][0, :k]  # 第一个样本的前k个token
    last = batch['input_ids'][0, -k:]  # 第一个样本的后k个token
    print('ids:', first)
    print('ids:', last)


    # 词汇表结构示例（当使用RQ-VAE时，codebook_size=128, n_codebooks=3）：
    #     0: bos (序列开始标记)
    #     1-128: RQ第1层数字
    #     129-256: RQ第2层数字
    #     257-384: RQ第3层数字
    #     385-512: RQ第4层数字（避免冲突用）
    #     513: boi (物品开始标记)
    #     514: eos (序列结束标记)




def generate_samples(config, logger, tokenizer, tokenized_datasets):
  """
  生成样本评估模式（sample_eval）
  使用训练好的扩散模型生成bundle样本，并计算生成困惑度

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_datasets: 已分词的数据集

  Returns:
      text_samples: 生成的文本样本列表
  """
  logger.info('Generating samples.')

  # 加载训练好的模型
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()  # 重置生成困惑度指标

  # 如果配置禁用EMA（指数移动平均），则将模型的EMA设为None
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  # 获取采样配置参数
  stride_length = config.sampling.stride_length  # 每次生成的步长
  num_strides = config.sampling.num_strides  # 步数

  # 循环生成多个批次的样本
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      # 半自回归采样方法：分步生成序列
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)  # dt是时间步长
      text_samples = intermediate_samples[-1]  # 获取最后一步的生成结果
      # 注意：使用半自回归方法生成的样本包含大量<|endoftext|>标记，
      # 在计算生成困惑度前需要预处理，
      # 因为diffusion.compute_generative_perplexity()会丢弃第一个EOS后的所有文本
    else:
      # 标准DDPM采样方法：一次性生成完整序列
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)  # 将token解码为文本
      model.compute_generative_perplexity(text_samples)  # 计算困惑度

  # 打印生成的样本和困惑度
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _rec_eval(config, logger, tokenizer, tokenized_dataset):
  """
  推荐系统评估模式（rec_eval）
  使用模型生成bundle推荐，并计算推荐指标（Recall、Precision、Hit Rate、Jaccard、OAS等）

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集

  Returns:
      output_results: 包含各项评估指标的有序字典
  """
  logger.info('Starting RecSys Evaluation.')
  allow_protocol_override = bool(config.eval.get('allow_protocol_override', False))
  OFFICIAL_EVALUATION_PROTOCOL.validate_config(
      config, allow_override=allow_protocol_override)
  data_history_condition = str(config.get('history_condition', 'full'))
  checkpoint_history_condition = _checkpoint_history_condition(
      config.eval.checkpoint_path)
  history_condition_mismatch = (
      checkpoint_history_condition != data_history_condition)
  if history_condition_mismatch and not bool(
      config.eval.get('allow_history_condition_mismatch', False)):
    raise ValueError(
        f"Checkpoint was trained with history_condition="
        f"{checkpoint_history_condition!r} but the test contexts use "
        f"{data_history_condition!r}. Set "
        "eval.allow_history_condition_mismatch=true for a diagnostic run.")

  # 加载训练好的模型和评估器
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  evaluator = Evaluator(config['evaluator'], tokenizer)

  # 如果禁用EMA
  if config.eval.disable_ema:
      logger.info('Disabling EMA.')
      model.ema = None

  # 创建测试数据加载器
  test_ds = DataLoader(
      tokenized_dataset['test'],
      batch_size=config['eval_batch_size'],
      shuffle=False,  # 不打乱顺序，保证结果可复现
      collate_fn=tokenizer.collate_fn['test']
  )

  model.eval()  # 设置模型为评估模式（关闭dropout等）
  all_results = defaultdict(list)  # 存储所有批次的评估结果
  # Inference cost for the cue-budget trade-off: generation time excludes
  # data loading and metric computation.
  generation_seconds = 0.0
  loop_started = time.perf_counter()

  # 不计算梯度
  with torch.no_grad():
    for batch in tqdm(test_ds, desc="Evaluating", ncols=100):
      input_ids = batch['input_ids']  # 输入的bundle前半部分
      labels = batch.get('labels')  # 标签是bundle的后半部分
      if labels is None:
        raise ValueError("rec_eval requires tokenizer-provided ground-truth labels")

      # CFG: extract context_emb if guidance is enabled
      cfg_enabled = getattr(config.sampling, 'cfg_enabled', False)
      context_emb = None
      if cfg_enabled:
        context_emb = batch.get('context_emb', None)
        if context_emb is not None:
          context_emb = context_emb.to(next(model.parameters()).device).float()
      structure_conditioning = getattr(
          config.sampling, 'structure_conditioning', False)
      mu_c = batch.get('mu_c', None) if structure_conditioning else None
      sigma_c2 = batch.get('sigma_c2', None) if structure_conditioning else None
      if mu_c is not None:
        mu_c = mu_c.to(next(model.parameters()).device).float()
      if sigma_c2 is not None:
        sigma_c2 = sigma_c2.to(next(model.parameters()).device).float()

      # Evaluation performs one joint full-MASK completion of the five songs
      # following the fixed 15-song context.
      eval_num_samples = int(config.protocol.eval_num_samples)
      eval_target_items = int(config.protocol.eval_target_items)
      eval_generated_items = int(config.protocol.eval_generated_items)
      if labels.ndim != 3 or labels.shape[1] != eval_target_items:
        raise ValueError(
            f"rec_eval expects {eval_target_items} future-item labels, "
            f"got {tuple(labels.shape)}")
      if eval_num_samples != 1 or eval_generated_items != eval_target_items:
        raise ValueError(
            "Joint 15->5 evaluation requires one sample containing five items")
      num_items = eval_generated_items
      tokens_per_item = tokenizer.tokens_per_item
      stride_length = 2 + num_items * tokens_per_item

      # DEBUG: 只在第一个batch打印配置信息
      if len(all_results) == 0:
        print(f"\n[Rec Eval Config]")
        print("  Prediction mode: 15 references -> one joint 5-item completion")
        print(f"  labels.shape: {labels.shape}")
        print(f"  labels[0, :]: {labels[0, :].tolist()}")
        print(f"  input_ids.shape: {input_ids.shape}")
        print(f"  input_ids[0, :]: {input_ids[0, :].tolist()}")
        print(f"  items per draw: {num_items}")
        print(f"  joint draws: {eval_num_samples}")
        print(f"  tokens_per_item: {tokens_per_item} (BOI + {tokenizer.n_digit} RVQ digits + 1 conflict)")
        print(f"  calculated stride_length: {stride_length}")

      # One sample contains all five jointly denoised continuation items.
      text_samples = torch.zeros(
          (input_ids.shape[0], eval_num_samples, stride_length), dtype=torch.long)

      if torch.cuda.is_available():
        torch.cuda.synchronize()
      generation_started = time.perf_counter()
      generated_items = model.restore_model_and_sample_items(
          input_ids=input_ids,
          num_items=num_items,
          num_steps=config.sampling.steps,
          context_emb=context_emb,  # CFG context embedding (None if cfg_enabled=False)
          mu_c=mu_c,
          sigma_c2=sigma_c2,
          sequence_mask=batch.get('sequence_mask'),
      )
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      generation_seconds += time.perf_counter() - generation_started
      text_samples[:, 0, :] = generated_items.detach().cpu()

      # 计算该批次的推荐指标
      result = evaluator.calculate_metrics(text_samples, labels)
      for key, value in result.items():
          all_results[key].append(value)

  loop_seconds = time.perf_counter() - loop_started
  test_examples = len(tokenized_dataset['test'])
  timing = {
      'generation_seconds': round(generation_seconds, 3),
      'generation_seconds_per_history': round(
          generation_seconds / max(test_examples, 1), 6),
      'loop_seconds': round(loop_seconds, 3),
      'eval_batch_size': int(config['eval_batch_size']),
      'device': (torch.cuda.get_device_name()
                 if torch.cuda.is_available() else 'cpu'),
  }

  # 汇总所有批次的结果
  output_results = OrderedDict()

  for key, values in all_results.items():
    if isinstance(values, list):
        vals = []
        # 将所有批次的值展平
        for v in values:
            if torch.is_tensor(v):
                if v.numel() == 1:
                    vals.append(v.item())
                else:
                    vals.extend(v.detach().cpu().numpy().tolist())
            else:
                vals.append(float(v))
        # 计算平均值
        mean_val = float(torch.tensor(vals).mean().item())
        output_results[key] = round(mean_val, 4)

    elif torch.is_tensor(values):
        if values.numel() == 1:
            output_results[key] = round(values.item(), 4)
        else:
            output_results[key] = round(values.detach().cpu().numpy().mean().item(), 4)

    else:
        output_results[key] = round(float(values), 4)

  print("output_results", output_results)

  # Persist both metrics and the exact evaluation provenance. Official results
  # must not exist only in a terminal scrollback.
  results_path_value = config.eval.get('results_path', None)
  results_path = (
      Path(results_path_value).expanduser().resolve()
      if results_path_value
      else Path.cwd() / 'rec_eval_results.json')

  def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
      for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()

  checkpoint_path = Path(config.eval.checkpoint_path).expanduser().resolve()
  prepared_path_value = config.get('prepared_dataset_path', None)
  prepared_manifest_path = (
      Path(prepared_path_value).expanduser().resolve() / 'prepared_manifest.json'
      if prepared_path_value else None)
  if (len(evaluator.evaluated_prediction_ids) != len(tokenized_dataset['test'])
          or len(evaluator.evaluated_target_ids) != len(tokenized_dataset['test'])
          or len(evaluator.evaluated_prediction_semantic_token_ids)
          != len(tokenized_dataset['test'])
          or len(evaluator.evaluated_prediction_cue_ids)
          != len(tokenized_dataset['test'])):
      raise ValueError(
          "Evaluator did not retain one complete prediction/target plan per test example")
  payload = {
      'result_schema': 'genplaylist-wp-c-joint-15to5-eval-v4',
      'created_utc': datetime.now(timezone.utc).isoformat(),
      'git_commit': config.eval.get('git_commit', None),
      'checkpoint': {
          'path': str(checkpoint_path),
          'sha256': file_sha256(checkpoint_path),
      },
      'prepared_data': {
          'path': str(Path(prepared_path_value).expanduser().resolve())
          if prepared_path_value else None,
          'manifest_sha256': (
              file_sha256(prepared_manifest_path)
              if prepared_manifest_path and prepared_manifest_path.is_file() else None),
      },
      'protocol': omegaconf.OmegaConf.to_container(
          config.protocol, resolve=True),
      'official_evaluation_contract': OFFICIAL_EVALUATION_PROTOCOL.as_dict(),
      'evaluation': {
          'test_examples': len(tokenized_dataset['test']),
          'catalog_items': len(tokenizer.item_id_to_row),
          'seed': int(config.seed),
          'sampling_steps': int(config.sampling.steps),
          'ema_enabled': not bool(config.eval.disable_ema),
          'sampler': str(config.sampling.predictor),
          'full_catalog_retrieval': True,
          'generation': 'one_joint_full_mask_five_item_completion',
          'matching': 'hungarian_clhe_5x5',
          'history_condition': {
              'test_contexts': data_history_condition,
              'checkpoint': checkpoint_history_condition,
          },
          'active_cue_tokens': int(tokenizer.active_cues),
          'tokens_per_sequence': int(config.model.length),
          'timing': timing,
          'official_protocol': (
              not allow_protocol_override and not history_condition_mismatch),
      },
      'metrics': dict(output_results),
      'predictions': {
          'item_ids': evaluator.evaluated_prediction_ids,
          'target_item_ids': evaluator.evaluated_target_ids,
          'semantic_token_ids': evaluator.evaluated_prediction_semantic_token_ids,
          'cue_ids': evaluator.evaluated_prediction_cue_ids,
          'shape': [len(tokenized_dataset['test']), eval_generated_items],
      },
  }
  results_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = results_path.with_name(results_path.name + '.tmp')
  temporary_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  os.replace(temporary_path, results_path)
  print(f"Saved rec_eval results -> {results_path}")

  # 打印RVQ直接命中统计
  evaluator.print_rvq_hit_statistics()

  # 保存候选集缓存（仅在第一次运行时保存）
  if evaluator.predict_num_items is not None:
    evaluator.save_candidate_cache()

  return output_results

def _ppl_eval(config, logger, tokenizer, tokenized_dataset):
  """
  困惑度评估模式（ppl_eval）
  计算模型在测试集上的困惑度（perplexity），用于衡量模型生成质量

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集
  """
  logger.info('Starting Zero Shot Eval.')

  # 加载训练好的模型
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  # 配置日志记录器列表（支持多个logger并行运行）
  loggers = []

  # 添加 TensorBoard logger（本地实时可视化）
  if config.get('use_tensorboard', False):
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
      save_dir=os.getcwd(),
      name='tensorboard',
      version=config.get('run_name', 'default'),
      log_graph=False,  # 禁用计算图记录
      default_hp_metric=False  # 禁用 hyperparameter 指标
    )
    loggers.append(tb_logger)
    logger.info('TensorBoard logger initialized')

  # 实例化Lightning回调（如checkpoint保存、early stopping等）
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # 创建Lightning Trainer（负责管理训练/验证循环）
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),  # 分布式训练策略
    logger=loggers if len(loggers) > 0 else None)

  # 创建测试数据加载器
  test_ds = DataLoader(
    tokenized_dataset['test'],
    batch_size=config['eval_batch_size'],
    shuffle=False,
    collate_fn=tokenizer.collate_fn['test'])

  # _, valid_ds = dataloader.get_dataloaders(
  #   config, tokenizer, skip_train=True, valid_seed=config.seed)

  # 在测试集上验证模型，计算困惑度
  trainer.validate(model, test_ds)  # valid_ds


def _train(config, logger, tokenizer, tokenized_dataset, trainer=None):
  """
  训练模式（train）
  使用PyTorch Lightning训练离散扩散模型

  Args:
      config: 配置对象
      logger: 日志记录器
      tokenizer: 分词器
      tokenized_dataset: 已分词的数据集
      trainer: 可选的自定义trainer
  """
  logger.info('Starting Training.')

  # 配置日志记录器列表（支持多个logger并行运行）
  loggers = []

  # 添加 TensorBoard logger（本地实时可视化）
  if config.get('use_tensorboard', False):
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
      save_dir=os.getcwd(),
      name='tensorboard',
      version=config.get('run_name', 'default'),
      log_graph=False,  # 禁用计算图记录
      default_hp_metric=False  # 禁用 hyperparameter 指标
    )
    loggers.append(tb_logger)
    logger.info('TensorBoard logger initialized')

  # 检查是否需要从检查点恢复训练
  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None  # 从头开始训练

  # 实例化Lightning回调（checkpoint保存、early stopping、学习率监控等）
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # print(callbacks)

  # 创建训练数据加载器
  # 将HuggingFace Dataset包装为torch Dataset，使Lightning可以正确应用DistributedSampler
  train_batch_size = config.get('loader', {}).get('batch_size', config.get('train_batch_size', 64))
  world_size = int(config.trainer.devices) * int(config.trainer.num_nodes)
  accumulation = int(config.trainer.accumulate_grad_batches)
  configured_global_batch = int(config.loader.global_batch_size)
  effective_global_batch = train_batch_size * world_size * accumulation
  if effective_global_batch != configured_global_batch:
    raise ValueError(
      'Effective global batch size drifted: '
      f'per_device={train_batch_size} * world_size={world_size} * '
      f'accumulation={accumulation} = {effective_global_batch}, '
      f'configured={configured_global_batch}. Choose a per-device batch size '
      'that divides the configured global batch exactly.')
  logger.info(f'Creating train DataLoader: train_batch_size={train_batch_size}')
  logger.info(
    'Effective global batch size: %s x %s devices x %s accumulation = %s',
    train_batch_size, world_size, accumulation, effective_global_batch)
  logger.info(f'Train dataset type: {type(tokenized_dataset["train"])}, length: {len(tokenized_dataset["train"])}')

  # 包装HuggingFace Dataset为torch Dataset
  train_dataset_wrapped = TorchDatasetWrapper(tokenized_dataset['train'])

  logger.info(f'Wrapped train dataset type: {type(train_dataset_wrapped)}')

  train_ds = DataLoader(
      train_dataset_wrapped,  # 使用包装后的dataset
      batch_size=train_batch_size,
      shuffle=True,  # Lightning DDP会自动替换为DistributedSampler
      collate_fn=tokenizer.collate_fn['train'],
      num_workers=0,
      pin_memory=True,
      persistent_workers=False
  )

  # train_ds, valid_ds = dataloader.get_dataloaders(
    # config, tokenizer)
  # _print_batch(train_ds, valid_ds, tokenizer)

  # 创建扩散模型实例
  # model = _load_from_checkpoint(config, tokenizer)
  model = diffusion.Diffusion(
    config, tokenizer)  # tokenizer=valid_ds

  warmstart_path = config.checkpointing.get('warmstart_path', None)
  if warmstart_path:
    if ckpt_path is not None:
      raise ValueError(
        'checkpointing.warmstart_path cannot be combined with an existing '
        'resume checkpoint; warm-start begins a new optimizer/step history')
    warmstart_report = apply_ddbc_warmstart(model, warmstart_path)
    logger.info('Applied DDBC warm-start: %s', warmstart_report)

  # 创建Lightning Trainer
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),  # 分布式训练策略（DDP等）
    logger=loggers if len(loggers) > 0 else None)

  # 开始训练（自动处理训练循环、验证、checkpoint保存等）
  # 如果ckpt_path不为None，会自动从checkpoint恢复
  # There is deliberately no validation loader. The original val/test sources
  # form one final test set whose five answers are hidden in labels and therefore
  # cannot define a training-time target_mask loss. Checkpoints are step-based.
  trainer.fit(model, train_ds, ckpt_path=ckpt_path)

  # 以下是旧版本的trainer代码（已注释）
  # Trainer
  # if trainer is not None:
  #   trainer = trainer
  # else:
  #   trainer = get_trainer("MDLM")(config, model, tokenizer) #
  # trainer.fit(train_dataloader, val_dataloader)
    



@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """
  主入口函数
  使用Hydra装饰器管理配置，自动从configs/目录加载config.yaml

  整体流程：
  1. 设置随机种子
  2. 加载原始数据集（train/test；test 合并原 val/test 来源）
  3. 使用RQ-VAE将物品编码为离散token
  4. 根据mode参数执行不同任务：
     - train: 训练扩散模型
     - rec_eval: 评估推荐指标
     - ppl_eval: 评估困惑度
     - sample_eval: 生成样本

  Args:
      config: Hydra配置对象，包含所有实验参数
  """
  # 设置随机种子，确保实验可复现
  L.seed_everything(config.seed)

  # 获取日志记录器
  logger = utils.get_logger(__name__)

  # ============ 第1步：加载数据集 ============
  # 从 split 文件加载 train，以及合并后的统一 test
  dataset = AbstractDataset(config)
  split_datasets = dataset.split()  # 返回包含 train/test 的字典

  # ============ 第2步：初始化分词器 ============
  # 分词器负责将bundle转换为token序列
  # 使用RQ-VAE将物品嵌入向量量化为离散码本索引
  tokenizer = dataloader.get_tokenizer(config, dataset)
  if isinstance(tokenizer, GenPlaylistTokenizer):
    expected_length = tokenizer.max_token_seq_len
    if int(config.model.length) != expected_length:
      raise ValueError(
          f"Frozen GenPlaylist model.length is {expected_length}, "
          f"got {config.model.length}")

  # ============ 第3步：对数据集进行分词 ============
  # 根据cir（components-to-items ratio）参数选择不同的分词策略：
  # if config['cir'] == 'none': #不使用RQ-VAE，直接使用物品ID
  #   tokenized_datasets = tokenizer.raw_tokenize(split_datasets)

  # elif config['cir'] == 1: # 不将物品转换为组件，一个物品对应一个token序列
  prepared_path = config.get('prepared_dataset_path', None)
  if prepared_path:
    tokenized_datasets, prepared_manifest = load_prepared_tokenized_dataset(
        prepared_path, config, dataset, tokenizer)
    logger.info(
        f"Loaded prepared dataset {prepared_manifest['prepared_data_version']} "
        f"from {prepared_path}")
  else:
    tokenized_datasets = tokenizer.tokenize(split_datasets)

  # else: # cir为其他值（如3、5、10、15），将多个物品组合为一个组件
    # tokenized_datasets = tokenizer.transfor_tokenzie(split_datasets)
    # 例如：对于spotify数据集，将物品序列转换为组件序列以压缩表示

  # ============ 第4步：根据mode执行相应任务 ============
  if config.mode == 'sample_eval':
    # 生成样本模式：生成新的bundle并计算生成困惑度
    generate_samples(config, logger, tokenizer, tokenized_datasets)

  elif config.mode == 'ppl_eval':
    # 困惑度评估模式：计算模型在测试集上的困惑度
    _ppl_eval(config, logger, tokenizer, tokenized_datasets)

  elif config.mode == 'rec_eval':
    # 推荐评估模式：给定bundle前半部分，生成后半部分，计算推荐指标
    _rec_eval(config, logger, tokenizer, tokenized_datasets)

  else:
    # 默认为训练模式：训练离散扩散模型
    _train(config, logger, tokenizer, tokenized_datasets)


if __name__ == '__main__':
  # Python脚本入口点
  # 执行main函数，Hydra会自动解析命令行参数和配置文件
  main()
