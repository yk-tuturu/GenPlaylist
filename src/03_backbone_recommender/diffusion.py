import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

from dataset import AbstractDataset
from tokenizer import AbstractTokenizer


LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor
  unweighted_nlls: torch.FloatTensor = None  # 未加权的原始 per-token loss，用于各层 NLL 统计


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: AbstractTokenizer):
    super().__init__()
    # Convert OmegaConf to dict for proper YAML serialization in TensorBoard
    # Ignore tokenizer as it's not serializable and not needed in hparams
    import omegaconf
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    self.save_hyperparameters({'config': config_dict}, ignore=['tokenizer'])
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.bos_index = self.tokenizer.bos_token
      self.eos_index = self.tokenizer.eos_token
      self.boi_index = self.tokenizer.boi_token
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config,vocab_size=self.vocab_size) #, raw_vocab_size = 50258
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()

    self.train_metrics = torchmetrics.MetricCollection({
        'nll': NLL(),
        'bpd': BPD(),
        'ppl': Perplexity(),
    }).clone(prefix='train/')

    self.valid_metrics = torchmetrics.MetricCollection({
        'nll': NLL(),
        'bpd': BPD(),
        'ppl': Perplexity(),
    }).clone(prefix='val/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = None
    if self.config.eval.compute_generative_perplexity:
      self.eval_model_tokenizer = transformers.AutoTokenizer.\
        from_pretrained(self.gen_ppl_eval_model_name_or_path)
      if self.eval_model_tokenizer.pad_token is None:
        self.eval_model_tokenizer.pad_token =\
            self.eval_model_tokenizer.eos_token
        self.eval_model_tokenizer.pad_token_id =\
            self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._type_mask_cache = {}
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=dl.batch_size,
          num_workers=dl.num_workers,
          pin_memory=dl.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          collate_fn=dl.collate_fn,
          drop_last=dl.drop_last,
          persistent_workers=(dl.persistent_workers if dl.num_workers > 0 else False)))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def on_before_optimizer_step(self, optimizer):
    """Log gradient norms before clipping (training.log_grad_norm only).

    Lightning calls this hook after gradients are accumulated and before they
    are clipped, so ``train/grad_norm`` is the unclipped L2 norm and
    ``train/grad_clipped`` records whether clipping will act on this step.
    """
    if not bool(getattr(self.config.training, 'log_grad_norm', False)):
      return
    every = max(int(getattr(self.trainer, 'log_every_n_steps', 1) or 1), 1)
    if int(self.global_step) % every != 0:
      return
    grads = [param.grad.detach().float()
             for param in self.parameters() if param.grad is not None]
    if not grads:
      return
    total = torch.linalg.vector_norm(
      torch.stack([torch.linalg.vector_norm(grad) for grad in grads]))
    clip_value = getattr(self.trainer, 'gradient_clip_val', None)
    logs = {
      'train/grad_norm': total,
      'train/grad_clipped': torch.tensor(
        float(clip_value is not None and float(total) > float(clip_value))),
    }
    embedding = getattr(
      getattr(self.backbone, 'vocab_embed', None), 'embedding', None)
    if embedding is not None and embedding.grad is not None:
      from shared.schema import TOKEN_LAYOUT
      embedding_grad = embedding.grad.detach().float()
      logs['train/grad_norm_embeddings'] = torch.linalg.vector_norm(
        embedding_grad)
      logs['train/grad_norm_cue_embeddings'] = torch.linalg.vector_norm(
        embedding_grad[TOKEN_LAYOUT.cue_token_start:TOKEN_LAYOUT.mask_token])
    self.log_dict(logs, on_step=True, on_epoch=False, sync_dist=True)

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits



  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(
      self, x, sigma, context_emb=None, mu_c=None, sigma_c2=None,
      sequence_mask=None):
    """Return log scores conditioned on visible tokens and playlist statistics."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32): # la
      logits = self.backbone(
        x, sigma, context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2,
        sequence_mask=sequence_mask)
    if hasattr(self.tokenizer, 'make_type_mask'):
      cache_key = (x.shape[1], logits.device.type, logits.device.index)
      legal = self._type_mask_cache.get(cache_key)
      if legal is None:
        legal = torch.as_tensor(
          self.tokenizer.make_type_mask(x.shape[1]),
          dtype=torch.bool, device=logits.device)
        self._type_mask_cache[cache_key] = legal
      if legal.shape != logits.shape[1:]:
        raise ValueError(
          f'Type mask/logit mismatch: mask={tuple(legal.shape)}, '
          f'logits={tuple(logits.shape)}')
      logits = logits.masked_fill(~legal.unsqueeze(0), self.neg_infinity)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    # CFG: extract context_emb from batch, apply dropout during training
    cfg_enabled = getattr(self.config.sampling, 'cfg_enabled', False)
    context_emb = None
    if cfg_enabled:
      context_emb = batch.get('context_emb', None)
      if context_emb is not None:
        context_emb = context_emb.to(self.device).float()
    target_mask = batch.get('target_mask', batch['attention_mask'])
    sequence_mask = batch.get(
      'sequence_mask', torch.ones_like(batch['input_ids'], dtype=torch.bool))
    mu_c = batch.get('mu_c', None)
    sigma_c2 = batch.get('sigma_c2', None)
    if not getattr(self.config.sampling, 'structure_conditioning', False):
      mu_c = None
      sigma_c2 = None
    if mu_c is not None:
      mu_c = mu_c.to(self.device).float()
    if sigma_c2 is not None:
      sigma_c2 = sigma_c2.to(self.device).float()
    losses, preds = self._loss(
      batch['input_ids'], batch['attention_mask'], target_mask=target_mask,
      context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2,
      sequence_mask=sequence_mask)
    loss = losses.loss


    if prefix == 'train':
      self.train_metrics.update(losses.nlls.detach(), losses.token_mask.detach()) #la
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls.detach(), losses.token_mask.detach())
      metrics = self.valid_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)

    # 记录各 RVQ 层的未加权 NLL，帮助判断各层学习进度
    if losses.unweighted_nlls is not None:
      layer_stats = self._compute_layer_nll_stats(
        batch['input_ids'], losses.unweighted_nlls, target_mask=target_mask)
      self.log_dict(
        {f'{prefix}/{k}': v for k, v in layer_stats.items()},
        on_step=(prefix == 'train'),
        on_epoch=True,
        sync_dist=True)

    layer_weights_cfg = getattr(self.config.training, 'layer_loss_weights', None)
    if (prefix == 'train' and layer_weights_cfg is not None
        and layer_weights_cfg.get('enabled', False)):
      rvq_weights, conflict_weight, cue_weight = (
        self._current_layer_loss_weights())
      weight_logs = {
        f'train/loss_weight_d{index}': value
        for index, value in enumerate(rvq_weights)
      }
      weight_logs.update({
        'train/loss_weight_conflict': conflict_weight,
        'train/loss_weight_cues': cue_weight,
      })
      self.log_dict(
        weight_logs, on_step=True, on_epoch=False, sync_dist=True)

    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  @torch.no_grad()
  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
  
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples =  samples.tolist()
        # text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]

        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))


    # @torch.no_grad()
    # def on_validation_epoch_end(self):
    #     # Sample only from a subset of the validation set for efficiency
    #     num_batches = min(
    #         self.config.eval.num_sample_batches,
    #         len(self.val_dataloader())
    #     )
    #     sampled_batches = random.sample(
    #         list(self.val_dataloader()), num_batches
    #     )
    #     all_gen_samples = []
    #     all_targets = []

    #     for batch in sampled_batches:
    #         input_ids, cent_emb, target = batch["input_ids"], batch["cent_emb"], batch["target"]
    #         # Generate samples conditioned on input_ids and cent_emb
    #         gen_samples = self._sample_with_condition(input_ids, cent_emb)
    #         all_gen_samples.extend(gen_samples.tolist())
    #         all_targets.extend(target.tolist())

    #     # Optionally log some samples for inspection
    #     if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
    #         self.trainer.logger.log_table(
    #             key=f"samples@global_step{self.global_step}",
    #             columns=["Generated Samples", "Targets"],
    #             data=[[gen, tgt] for gen, tgt in zip(all_gen_samples, all_targets)]
    #         )

    #     # Restore EMA weights
    #     if self.ema:
    #         self.ema.restore(
    #             itertools.chain(
    #                 self.backbone.parameters(),
    #                 self.noise.parameters()
    #             )
    #         )

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    total_len = batch_dims[-1]
    result = self.mask_index * torch.ones(
        *batch_dims, dtype=torch.int64, device=self.device
    )
    # result[:,0] = self.bos_index
    # result[:,-1] = self.eos_index

    k = getattr(self.tokenizer, 'tokens_per_item', self.tokenizer.n_digit + 2)
    n = total_len //k
    ni = total_len % k
    for b in range(result.shape[0]):
        token_list = []

        if ni == 2:
            # [BOS] ... [EOS]
            token_list.append(self.bos_index)
            for i in range(n):
              token_list.extend([self.boi_index]+[self.mask_index] * (k-1)) 
            token_list.append(self.eos_index)

        elif not hasattr(self.tokenizer, 'make_type_mask') and (ni==0 or ni==4):
            # [BOS] ... [EOS] [BOS] ... [EOS]
            half_n = math.ceil(n // 2) #  for input/output=1:1
            token_list.append(self.bos_index)
            for i in range(half_n):
              token_list.extend([self.boi_index]+[self.mask_index] * (k-1))#
            token_list.append(self.eos_index)
            
            token_list.append(self.bos_index)
            for i in range(n-half_n):
              token_list.extend([self.boi_index]+[self.mask_index] * (k-1))
            token_list.append(self.eos_index)
        else:
            raise ValueError(f"Unsupported remainder ni = {ni} for sequence length {total_len}")

        if len(token_list) != total_len:
            raise ValueError(f"Expected sequence length {total_len}, got {len(token_list)}")
        result[b, :] = torch.tensor(token_list, dtype=torch.int64, device=self.device)

    return result

  def _apply_illegal_mask(self, values, tokenizer, current_x, is_logits=True, force_all_positions=False):
    """
    values: (batch, seq_len, vocab_size) - can be either logits or probabilities
    current_x: (batch, seq_len) - current token sequence, used to determine active positions
    is_logits: True if values are logits, False if values are probabilities
    force_all_positions: If True, apply mask to all positions regardless of whether they're masked

    Apply masking to filter out invalid tokens based on `pos % 5`.
    The first position (bos), the last position (eos), and the first token of every 5-token block (boi)
    should remain unchanged.

    If `cir` is None, it means each item corresponds to a single token（no RQ-VAE） and `boi` does not exist.
    In this case, only ensure that neither `bos` nor `eos` is generated.

    Constraints are applied only to the positions that are actually being unmasked at the current step
    to avoid affecting the probability distribution of other masked positions.
   """

    masked_values = values.clone()
    if hasattr(tokenizer, 'make_type_mask'):
      legal_by_position = torch.as_tensor(
        tokenizer.make_type_mask(masked_values.size(1)),
        dtype=torch.bool, device=values.device)
      active_positions = (
        torch.ones_like(current_x, dtype=torch.bool)
        if force_all_positions else current_x == self.mask_index)
      illegal = (~legal_by_position).unsqueeze(0) & active_positions.unsqueeze(-1)
      if is_logits:
        return masked_values.masked_fill(illegal, float('-inf'))
      masked_values = masked_values.masked_fill(illegal, 0.0)
      denominator = masked_values.sum(dim=-1, keepdim=True).clamp_min(1e-10)
      return masked_values / denominator

    codebook_size = tokenizer.config['rq_codebook_size']
    seq_len = masked_values.size(1)
    vocab_size = masked_values.size(-1)

    mask_index = self.mask_index

    # Active positions are those that currently have mask_index
    # These are the positions that will be resampled in this diffusion step
    # OR if force_all_positions=True, apply to all positions
    if force_all_positions:
      active_positions = torch.ones_like(current_x, dtype=torch.bool)
    else:
      active_positions = (current_x == mask_index)

    if not bool(active_positions.any()):
      return masked_values

    legal_mask = torch.ones_like(masked_values, dtype=torch.bool)
    # Apply mask to entire sequence, not just last 7 positions
    start_index = 0

    if self.config['cir'] == 'none':
      legal_indices = torch.arange(1, vocab_size - 2, device=values.device)
      for pos in range(start_index, seq_len):
        active_batches = active_positions[:, pos]
        if not bool(active_batches.any()):
          continue
        if pos == start_index or pos == seq_len - 1:
          continue
        legal_mask[active_batches, pos, :] = False
        legal_mask[active_batches, pos, legal_indices] = True
    else:
      # k = stride per item = BOI + n_digit个RVQ codes + 1个conflict
      # 3 codebooks: k=5, 4 codebooks: k=6
      k = tokenizer.n_digit + 2
      for pos in range(start_index, seq_len):
        active_batches = active_positions[:, pos]
        if not bool(active_batches.any()):
          continue

        # Skip BOS and EOS positions
        if pos == start_index or pos == seq_len - 1:
          continue

        if pos % k == 1:
          # BOI position: only allow BOI token ((n_digit+1) * codebook_size + 1)
          # 3 codebooks: 4*256+1=1025, 4 codebooks: 5*256+1=1281
          boi_token = (tokenizer.n_digit + 1) * codebook_size + 1
          temp_mask = torch.zeros(vocab_size, dtype=torch.bool, device=values.device)
          temp_mask[boi_token] = True
          legal_mask[active_batches, pos, :] = temp_mask
          continue

        # Determine digit position: d0, d1, d2, ... or conflict
        # pos % k == 2: digit0, pos % k == 3: digit1, ..., pos % k == 0: conflict
        pos_type = pos % k

        if 2 <= pos_type <= (tokenizer.n_digit + 1):
          # RVQ digit positions (d0, d1, d2, ...)
          # digit_idx: 0 for d0, 1 for d1, 2 for d2, etc.
          digit_idx = pos_type - 2
          # digit0 (idx=0): [1, codebook_size]
          # digit1 (idx=1): [codebook_size+1, 2*codebook_size]
          # digit2 (idx=2): [2*codebook_size+1, 3*codebook_size]
          # digit3 (idx=3): [3*codebook_size+1, 4*codebook_size]
          legal_range_start = digit_idx * codebook_size + 1
          legal_range_end = (digit_idx + 1) * codebook_size + 1
        else:  # pos % k == 0 (conflict)
          # conflict digit: [n_digit*codebook_size+1, (n_digit+1)*codebook_size]
          # 3 codebooks: [769, 1024], 4 codebooks: [1025, 1280]
          legal_range_start = tokenizer.n_digit * codebook_size + 1
          legal_range_end = (tokenizer.n_digit + 1) * codebook_size + 1


        temp_mask = torch.zeros(vocab_size, dtype=torch.bool, device=values.device)

        temp_mask[legal_range_start:legal_range_end] = True

        # Do NOT allow BOS/EOS in digit positions - only specific digit ranges are allowed
        legal_mask[active_batches, pos, :] = temp_mask

    # For probabilities, set illegal tokens to 0.0 and renormalize
    # For logits, set illegal tokens to -inf
    if is_logits:
      masked_values[~legal_mask] = float('-inf')
    else:
      masked_values[~legal_mask] = 0.0
      # Renormalize probabilities so they sum to 1.0
      # Add epsilon to avoid division by zero
      prob_sum = masked_values.sum(dim=-1, keepdim=True)
      masked_values = masked_values / (prob_sum + 1e-10)

    return masked_values

  def _ddpm_caching_update(self, x, t, dt, p_x0=None): #, cent_emb=None
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape

    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
      
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    # Apply illegal mask only during inference (not training)
    # q_xs is in probability space, so set is_logits=False
    # Pass current sequence x to determine which positions are masked
    # if not self.training:
    #   q_xs = self._apply_illegal_mask(q_xs, self.tokenizer, current_x=x, is_logits=False)

    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

    # Apply illegal mask only during inference (not training)
    # q_xs is in probability space, so set is_logits=False
    # Pass current sequence x to determine which positions are masked
    # if not self.training:
    #   q_xs = self._apply_illegal_mask(q_xs, self.tokenizer, current_x=x, is_logits=False)

    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        # Apply illegal mask during inference
        # logits are in logit space, so is_logits=True (default)
        # Pass current sequence x to determine which positions are masked
        if not self.training:
          logits = self._apply_illegal_mask(logits, self.tokenizer, current_x=x, is_logits=True, force_all_positions=True)
        x = logits.argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]

    # if seqlen > self.config.model.length:
    #   assert seqlen == 2 * self.config.model.length
    #   # cropping is needed for text8-crop dataset
    #   # try the same starting point for now
    #   start = np.random.choice(self.config.model.length)
    #   end = start + self.config.model.length
    #   input_tokens = x0[:, start: end]
    #   output_tokens = x0[:, start + 1: end + 1]
    #   new_attention_mask = attention_mask[:, start: end]

    #   # Helps with validation PPL, since the val
    #   # examples will all start and end with BOS/EOS
    #   input_tokens[:, 0] = self.tokenizer.bos_token_id
    #   output_tokens[:, -1] = self.tokenizer.eos_token_id
    # elif self.parameterization == 'ar':
    #   input_tokens = x0[:, :-1]
    #   output_tokens = x0[:, 1:]
    #   new_attention_mask = attention_mask[:, 1:]
    # else:
    input_tokens = x0
    output_tokens = None
    new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _current_layer_loss_weights(self, global_step=None):
    """Resolve RVQ/conflict/cue weights at one curriculum step."""
    weight_cfg = self.config.training.layer_loss_weights
    n_codebooks = self.tokenizer.n_digit

    def expanded(values, name):
      output = [float(value) for value in values]
      if not output:
        raise ValueError(f'{name} must contain at least one weight')
      while len(output) < n_codebooks:
        output.append(output[-1])
      output = output[:n_codebooks]
      if any(value < 0 for value in output):
        raise ValueError(f'{name} must be non-negative, got {output}')
      return output

    final_rvq = expanded(weight_cfg.rvq_weights, 'rvq_weights')
    final_conflict = float(weight_cfg.conflict_weight)
    final_cue = float(weight_cfg.get('cue_weight', 1.0))
    if final_conflict < 0 or final_cue < 0:
      raise ValueError('conflict_weight and cue_weight must be non-negative')
    if sum(final_rvq) + final_conflict + final_cue <= 0:
      raise ValueError('Final layer-loss weights cannot all be zero')

    warmup = weight_cfg.get('warmup', None)
    if warmup is None or not warmup.get('enabled', False):
      return final_rvq, final_conflict, final_cue

    start_step = int(warmup.start_step)
    end_step = int(warmup.end_step)
    if start_step < 0 or end_step <= start_step:
      raise ValueError(
        f'Invalid layer-loss warmup interval: {start_step}..{end_step}')
    initial_rvq = expanded(
      warmup.initial_rvq_weights, 'warmup.initial_rvq_weights')
    initial_conflict = float(warmup.initial_conflict_weight)
    initial_cue = float(warmup.initial_cue_weight)
    if initial_conflict < 0 or initial_cue < 0:
      raise ValueError('Initial conflict/cue weights must be non-negative')
    if sum(initial_rvq) + initial_conflict + initial_cue <= 0:
      raise ValueError('Initial layer-loss weights cannot all be zero')

    if global_step is None:
      try:
        global_step = int(self.global_step)
      except (AttributeError, RuntimeError):
        global_step = 0
    step = int(global_step)
    progress = min(
      max((step - start_step) / (end_step - start_step), 0.0), 1.0)

    def interpolate(initial, final):
      return initial + progress * (final - initial)

    rvq = [
      interpolate(initial, final)
      for initial, final in zip(initial_rvq, final_rvq)
    ]
    return (
      rvq,
      interpolate(initial_conflict, final_conflict),
      interpolate(initial_cue, final_cue),
    )

  def _compute_position_weights(self, x0, global_step=None):
    """
    为序列每个 token 位置计算 loss 权重，根据其在 item 内的角色。

    Token 结构（GenPlaylist v3）：
      [BOS, BOI, d0, d1, d2, conflict, c0, c1, c2, c3, c4, c5, c6, c7, ..., EOS]
      period = tokenizer.tokens_per_item = 13

    offset 与角色对应：
      offset 0           → BOI（special token，不会被 mask，权重为 1.0 但不影响 loss）
      offset 1..n        → RVQ Layer 0..n-1（按 rvq_weights 分配）
      offset n+1         → conflict digit（按 conflict_weight 分配）
      later offsets      → creative cues（按 cue_weight 分配）
      pos 0              → BOS，special token，权重 1.0
      pos seq_len-1      → EOS，special token，权重 1.0

    返回 shape: [batch, seq_len]，数值类型 float，设备与 x0 一致。
    """
    n_codebooks = self.tokenizer.n_digit
    period = self.tokenizer.tokens_per_item
    rvq_weights, conflict_weight, cue_weight = (
      self._current_layer_loss_weights(global_step=global_step))

    weights = torch.ones(x0.shape, dtype=torch.float, device=x0.device)
    seq_len = x0.shape[1]

    for pos in range(1, seq_len):  # pos 0 是 BOS，保持默认 1.0
      offset = (pos - 1) % period
      if offset == 0:
        pass  # BOI，保持 1.0
      elif 1 <= offset <= n_codebooks:
        weights[:, pos] = rvq_weights[offset - 1]
      elif offset == n_codebooks + 1:
        weights[:, pos] = conflict_weight
      else:
        weights[:, pos] = cue_weight

    return weights

  @staticmethod
  def _normalize_active_position_weights(pos_weights, effective_mask):
    """Keep the mean active target weight at one throughout curriculum."""
    active_weight_sum = (
      pos_weights * effective_mask).sum().clamp_min(1e-12)
    active_token_count = effective_mask.sum().clamp_min(1)
    return pos_weights * (active_token_count / active_weight_sum)

  def _compute_layer_nll_stats(self, x0, unweighted_nlls, target_mask=None):
    """
    按 token 类型（d0/d1/.../conflict）分别统计平均未加权 NLL。

    参数：
      x0: [batch, seq_len]  原始 token 序列（用于定位各层 token 位置）
      unweighted_nlls: [batch, seq_len]  未加权的 per-token loss
      target_mask: [batch, seq_len]  只统计 target payload；None 仅用于兼容

    返回：
      dict，key 为 'layer_nll/d{i}' 和 'layer_nll/conflict'，value 为标量 tensor
    """
    n_codebooks = self.tokenizer.n_digit
    period = self.tokenizer.tokens_per_item
    seq_len = x0.shape[1]
    stats = {}

    # 用向量化方式构建位置掩码，避免逐样本循环
    positions = torch.arange(1, seq_len, device=x0.device)  # 跳过 BOS（pos 0）
    offsets = (positions - 1) % period  # 每个位置在 item 内的偏移

    if target_mask is not None:
      target_mask = target_mask.to(device=x0.device, dtype=torch.bool)
      if target_mask.shape != x0.shape:
        raise ValueError(
          f'target_mask shape {tuple(target_mask.shape)} does not match '
          f'x0 {tuple(x0.shape)}')

    def masked_position_mean(column_mask):
      values = unweighted_nlls[:, 1:][:, column_mask]
      if target_mask is None:
        return values.mean() if values.numel() > 0 else None
      selected = target_mask[:, 1:][:, column_mask]
      return values[selected].mean() if bool(selected.any()) else None

    for layer_idx in range(n_codebooks):
      target_offset = layer_idx + 1  # d{layer_idx} 的偏移
      col_mask = (offsets == target_offset)  # shape: [seq_len-1]
      layer_mean = masked_position_mean(col_mask)
      if layer_mean is not None:
        stats[f'layer_nll/d{layer_idx}'] = layer_mean

    # conflict digit：offset = n_codebooks + 1
    conflict_offset = n_codebooks + 1
    col_mask = (offsets == conflict_offset)
    conflict_mean = masked_position_mean(col_mask)
    if conflict_mean is not None:
      stats['layer_nll/conflict'] = conflict_mean

    cue_mask = offsets > conflict_offset
    cue_mean = masked_position_mean(cue_mask)
    if cue_mean is not None:
      stats['layer_nll/cues'] = cue_mean

    return stats

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(
      self, x0, attention_mask, target_mask=None, context_emb=None,
      mu_c=None, sigma_c2=None, sequence_mask=None):
    
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])
    special_ids = torch.tensor([self.eos_index,  self.bos_index,self.boi_index], device=x0.device) #
    special_mask = torch.isin(x0, special_ids)
    move_chance = move_chance.masked_fill(special_mask, 0.0)
    if target_mask is not None:
      move_chance = move_chance.masked_fill(~target_mask.bool(), 0.0)
    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(
      xt, unet_conditioning, context_emb=context_emb,
      mu_c=mu_c, sigma_c2=sigma_c2, sequence_mask=sequence_mask)
    utils.print_nans(model_output,'model_output')

    # if self.parameterization == 'sedd':
    #   return dsigma[:, None] * self._score_entropy(
    #     model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)

      # if self.parameterization == 'd3pm':
      #   reconstruction_loss = self._reconstruction_loss(x0)
      # elif self.parameterization == 'subs':
      reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(
      self, x0, attention_mask, target_mask=None, context_emb=None,
      mu_c=None, sigma_c2=None, sequence_mask=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(
        input_tokens, attention_mask, target_mask=target_mask,
        context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2,
        sequence_mask=sequence_mask)
      
    # print("loss",loss,loss.shape)
    effective_mask = attention_mask.bool() if target_mask is None else target_mask.bool()
    nlls = loss * effective_mask

    # 应用 RVQ 层级 loss 权重（如果启用）
    layer_weights_cfg = getattr(self.config.training, 'layer_loss_weights', None)
    if layer_weights_cfg is not None and layer_weights_cfg.get('enabled', False):
      pos_weights = self._compute_position_weights(input_tokens)
      if layer_weights_cfg.get('normalize', True):
        pos_weights = self._normalize_active_position_weights(
          pos_weights, effective_mask)
      nlls = nlls * pos_weights

    count = effective_mask.sum().clamp_min(1)
    token_nll = nlls.sum() / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=effective_mask,
                unweighted_nlls=loss.detach()), output_tokens

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  def build_item_completion(self, context_ids, num_items, sequence_mask=None):
    """Append explicit joint full-MASK item slots to a reference batch.

    Input rows are right-padded ``[BOS, reference items..., EOS]`` sequences.
    Output rows are ``[BOS, references..., (BOI, MASK x payload) x N, EOS]``.
    Only payload positions are marked in ``completion_mask``; references, BOI,
    EOS, and padding are immutable during reverse diffusion.
    """
    if context_ids.ndim != 2:
      raise ValueError(
        f'context_ids must be [batch, length], got {tuple(context_ids.shape)}')
    if sequence_mask is None:
      sequence_mask = torch.zeros_like(context_ids, dtype=torch.bool)
      for row in range(context_ids.shape[0]):
        eos_positions = torch.nonzero(
          context_ids[row] == self.tokenizer.eos_token, as_tuple=False).flatten()
        if eos_positions.numel() != 1:
          raise ValueError(
            f'Context row {row} must contain exactly one EOS, got '
            f'{eos_positions.numel()}')
        sequence_mask[row, :int(eos_positions[0].item()) + 1] = True
    else:
      sequence_mask = sequence_mask.to(
        device=context_ids.device, dtype=torch.bool)
      if sequence_mask.shape != context_ids.shape:
        raise ValueError(
          f'sequence_mask shape {tuple(sequence_mask.shape)} does not match '
          f'context_ids {tuple(context_ids.shape)}')

    lengths = sequence_mask.sum(dim=1)
    tokens_per_item = self.tokenizer.tokens_per_item
    prepared_rows = []
    for row, length_tensor in enumerate(lengths):
      length = int(length_tensor.item())
      if length < 2 or not bool(sequence_mask[row, :length].all()) or bool(
          sequence_mask[row, length:].any()):
        raise ValueError(f'Context row {row} must use contiguous right padding')
      completed_row, completion_row = self.tokenizer.build_item_completion(
        context_ids[row, :length].detach().cpu().numpy(), num_items=num_items)
      prepared_rows.append((completed_row, completion_row))

    output_length = max(len(completed_row) for completed_row, _ in prepared_rows)
    model_length = int(getattr(self.config.model, 'length', output_length))
    if output_length > model_length:
      raise ValueError(
        f'Completed sequence length {output_length} exceeds model length {model_length}')
    padding_token = self.tokenizer.padding_token
    completed = torch.full(
      (context_ids.shape[0], output_length), padding_token,
      dtype=context_ids.dtype, device=context_ids.device)
    completion_mask = torch.zeros_like(completed, dtype=torch.bool)
    completed_sequence_mask = torch.zeros_like(completed, dtype=torch.bool)

    for row, (completed_row, completion_row) in enumerate(prepared_rows):
      row_length = len(completed_row)
      completed[row, :row_length] = torch.as_tensor(
        completed_row, dtype=context_ids.dtype, device=context_ids.device)
      completion_mask[row, :row_length] = torch.as_tensor(
        completion_row, dtype=torch.bool, device=context_ids.device)
      completed_sequence_mask[row, :row_length] = True

    return completed, completion_mask, completed_sequence_mask

  def build_next_item_completion(self, context_ids, sequence_mask=None):
    """Backward-compatible one-item completion used by the WP-D runtime."""
    return self.build_item_completion(
      context_ids, num_items=1, sequence_mask=sequence_mask)


  @torch.no_grad()
  def sample_masked_completion(
      self, masked_ids, completion_mask, num_steps, eps=1e-5,
      context_emb=None, mu_c=None, sigma_c2=None, sequence_mask=None):
    """Reverse diffuse only explicitly marked MASK positions in a full sequence."""
    if self.parameterization != 'subs':
      raise ValueError(
        'Full-mask next-item completion currently requires SUBS parameterization')
    if num_steps <= 0:
      raise ValueError(f'num_steps must be positive, got {num_steps}')
    if not 0 < eps < 1:
      raise ValueError(f'eps must be between 0 and 1, got {eps}')
    if masked_ids.ndim != 2:
      raise ValueError(
        f'masked_ids must be [batch, length], got {tuple(masked_ids.shape)}')
    if completion_mask.shape != masked_ids.shape:
      raise ValueError('completion_mask must have the same shape as masked_ids')

    x = masked_ids.to(self.device).clone()
    fixed_ids = x.clone()
    completion_mask = completion_mask.to(self.device).bool()
    if sequence_mask is None:
      sequence_mask = torch.ones_like(x, dtype=torch.bool)
    else:
      sequence_mask = sequence_mask.to(self.device).bool()
      if sequence_mask.shape != x.shape:
        raise ValueError('sequence_mask must have the same shape as masked_ids')
    if not bool(completion_mask.any(dim=1).all()):
      raise ValueError('Every row needs at least one completion position')
    if not bool((x[completion_mask] == self.mask_index).all()):
      raise ValueError('Every completion position must initially contain MASK')
    if bool((x[~completion_mask] == self.mask_index).any()):
      raise ValueError('MASK tokens outside completion_mask are not allowed')
    if bool((completion_mask & ~sequence_mask).any()):
      raise ValueError('Completion positions cannot be padding')

    n_samples = x.shape[0]
    ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

    def prepare_condition(value):
      if value is None:
        return None
      value = value.to(self.device).float()
      if value.shape[0] != n_samples:
        raise ValueError(
          f'Condition batch size {value.shape[0]} does not match {n_samples}')
      return value

    context_emb = prepare_condition(context_emb)
    mu_c = prepare_condition(mu_c)
    sigma_c2 = prepare_condition(sigma_c2)
    cfg_enabled = getattr(self.config.sampling, 'cfg_enabled', False)
    if not cfg_enabled:
      context_emb = None
    if not getattr(self.config.sampling, 'structure_conditioning', False):
      mu_c = None
      sigma_c2 = None
    cfg_w = getattr(self.config.sampling, 'cfg_w', 1.0)
    use_cfg = cfg_enabled and context_emb is not None and cfg_w != 1.0

    def clean_distribution(current_x, timestep):
      sigma_t, _ = self.noise(timestep)
      if use_cfg:
        log_p_cond = self.forward(
          current_x, sigma_t, context_emb=context_emb,
          mu_c=mu_c, sigma_c2=sigma_c2, sequence_mask=sequence_mask)
        log_p_uncond = self.forward(
          current_x, sigma_t, context_emb=None,
          mu_c=mu_c, sigma_c2=sigma_c2, sequence_mask=sequence_mask)
        guided = log_p_uncond + cfg_w * (log_p_cond - log_p_uncond)
        return (guided - guided.logsumexp(dim=-1, keepdim=True)).exp()
      return self.forward(
        current_x, sigma_t, context_emb=context_emb,
        mu_c=mu_c, sigma_c2=sigma_c2, sequence_mask=sequence_mask).exp()

    timesteps = torch.linspace(
      1.0, eps, num_steps + 1, device=self.device, dtype=self.dtype)
    p_x0_cache = None
    for step in range(num_steps):
      t = timesteps[step] * ones
      s = timesteps[step + 1] * ones
      if p_x0_cache is None:
        p_x0_cache = clean_distribution(x, t)
      sigma_t, _ = self.noise(t)
      sigma_s, _ = self.noise(s)
      move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
      move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
      q_xs = p_x0_cache * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      proposed = _sample_categorical(q_xs)
      x_next = torch.where(x == self.mask_index, proposed, x)
      # This explicit assignment is the inpainting contract. It protects known
      # references even if another parameterization changes copy semantics.
      x_next = torch.where(completion_mask, x_next, fixed_ids)
      if not torch.equal(x_next, x) or self.time_conditioning:
        p_x0_cache = None
      x = x_next

    remaining = completion_mask & (x == self.mask_index)
    if bool(remaining.any()):
      final_t = eps * ones
      final_p_x0 = clean_distribution(x, final_t)
      final_tokens = final_p_x0.argmax(dim=-1)
      x = torch.where(remaining, final_tokens, x)
    x = torch.where(completion_mask, x, fixed_ids)
    if bool((x[completion_mask] == self.mask_index).any()):
      raise RuntimeError('Masked completion ended with unresolved MASK tokens')
    return x


  def extract_item_completion(self, completed_ids, completion_mask, num_items):
    """Extract ``[BOS, (BOI, payload) x N, EOS]`` from completed sequences."""
    if completed_ids.shape != completion_mask.shape:
      raise ValueError('completed_ids and completion_mask shapes differ')
    rows = []
    payload_width = self.tokenizer.tokens_per_item - 1
    expected_payload = num_items * payload_width
    for row in range(completed_ids.shape[0]):
      positions = torch.nonzero(completion_mask[row], as_tuple=False).flatten()
      if positions.numel() != expected_payload:
        raise ValueError(
          f'Row {row} has {positions.numel()} payload positions; '
          f'expected {expected_payload}')
      start = int(positions[0].item())
      expected_groups = []
      for item_index in range(num_items):
        group_start = start + item_index * self.tokenizer.tokens_per_item
        expected_groups.append(torch.arange(
          group_start, group_start + payload_width, device=positions.device))
      expected = torch.cat(expected_groups)
      if not torch.equal(positions, expected):
        raise ValueError(f'Row {row} completion positions do not form {num_items} items')
      sequence_start = start - 1
      sequence_end = sequence_start + num_items * self.tokenizer.tokens_per_item + 1
      items_with_boundaries = completed_ids[row, sequence_start:sequence_end]
      bos = torch.as_tensor(
        [self.tokenizer.bos_token], dtype=completed_ids.dtype,
        device=completed_ids.device)
      rows.append(torch.cat((bos, items_with_boundaries)))
    return torch.stack(rows)

  def extract_next_item_completion(self, completed_ids, completion_mask):
    """Backward-compatible one-item extraction used by the WP-D runtime."""
    return self.extract_item_completion(
      completed_ids, completion_mask, num_items=1)


  @torch.no_grad
  def sample_subs_guidance(
    self, input_ids, stride_length, num_strides, dt=0.001,
    context_emb=None, mu_c=None, sigma_c2=None):
    n_samples = input_ids.shape[0]
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    def prepare_condition(value):
      if value is None:
        return None
      value = value.to(self.device).float()
      if value.shape[0] != n_samples:
        raise ValueError(
          f'Condition batch size {value.shape[0]} does not match {n_samples} samples')
      return value

    context_emb = prepare_condition(context_emb)
    mu_c = prepare_condition(mu_c)
    sigma_c2 = prepare_condition(sigma_c2)
    if not getattr(self.config.sampling, 'structure_conditioning', False):
      mu_c = None
      sigma_c2 = None

    cfg_enabled = getattr(self.config.sampling, 'cfg_enabled', False)
    cfg_w = getattr(self.config.sampling, 'cfg_w', 1.0)
    # Only do two-pass CFG when enabled, context provided, and w != 1
    use_cfg = cfg_enabled and context_emb is not None and cfg_w != 1.0

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []

    target = input_ids.to(self.device)
    # Save the original context length (will be used to split context and generation)
    original_context_length = target.shape[1] - 1  # target length minus EOS

    for _ in range(num_strides): #+1
      test_list = []
      p_x0_cache = None
      x_tmp = self._sample_prior(
        n_samples,
        stride_length).to(self.device)

      # Remove EOS from target and BOS from x_tmp to create a continuous sequence
      # This makes the prediction task identical to training (single continuous sequence)
      # Before: [BOS, ..., EOS] + [BOS, ..., EOS] (two separate sequences)
      # After:  [BOS, ..., (no EOS)] + [(no BOS), ..., EOS] (single continuous sequence)
      x = torch.cat((target[:, :-1], x_tmp[:, 1:]), dim=1)

      for i in range(num_steps + 1):
        # When cfg_enabled, pre-compute p_x0 here so _ddpm_caching_update
        # never calls self.forward internally (it skips forward when p_x0 is given)
        if p_x0_cache is None and (
            cfg_enabled or context_emb is not None or mu_c is not None or sigma_c2 is not None):
          t_now = (1 - i * dt) * ones
          sigma_t, _ = self.noise(t_now)
          if use_cfg:
            # Two-pass CFG: geometric interpolation in log-prob space
            log_p_cond = self.forward(
              x, sigma_t, context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2)
            # CFG drops only the optional prefix encoder. Any explicitly enabled
            # legacy structure condition is shared by both passes.
            log_p_uncond = self.forward(
              x, sigma_t, context_emb=None, mu_c=mu_c, sigma_c2=sigma_c2)
            log_p_guided = log_p_uncond + cfg_w * (log_p_cond - log_p_uncond)
            log_p_guided = log_p_guided - log_p_guided.logsumexp(dim=-1, keepdim=True)
            p_x0_cache = log_p_guided.exp()
          else:
            # cfg_w == 1.0: conditioned-only, single forward pass
            p_x0_cache = self.forward(
              x, sigma_t, context_emb=context_emb,
              mu_c=mu_c, sigma_c2=sigma_c2).exp()

        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)

        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next

      # Apply final denoising step with illegal mask
      if use_cfg:
        log_p_cond = self.forward(
          x, 0 * ones, context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2)
        log_p_uncond = self.forward(
          x, 0 * ones, context_emb=None, mu_c=mu_c, sigma_c2=sigma_c2)
        log_p_guided = log_p_uncond + cfg_w * (log_p_cond - log_p_uncond)
        logits = log_p_guided
      else:
        logits = self.forward(
          x, 0 * ones,
          context_emb=context_emb if cfg_enabled else None,
          mu_c=mu_c, sigma_c2=sigma_c2)

      # Apply illegal mask to ensure valid token generation
      # Now x is a continuous sequence: [BOS, context_items..., masked_items..., EOS]
      # The position alignment is correct: pos%5 determines token type
      if not self.training:
        # Apply illegal mask to the entire sequence
        # Since the sequence is now continuous (no intermediate EOS-BOS),
        # pos%5 correctly identifies token positions:
        #   pos%5==0: BOS or conflict digit
        #   pos%5==1: BOI
        #   pos%5==2: digit0
        #   pos%5==3: digit1
        #   pos%5==4: digit2
        logits = self._apply_illegal_mask(logits, self.tokenizer, current_x=x, is_logits=True, force_all_positions=True)

      # Update the entire sequence
      x_new = logits.argmax(dim=-1)

      # Calculate the length of the generation part (stride_length - 2 because we removed BOS and EOS)
      gen_length = stride_length - 2

      # Keep the context part unchanged, only update the generation part
      # Context part: x[:, :original_context_length] (original target without EOS)
      # Generation part: last gen_length tokens + EOS
      x[:, original_context_length:] = x_new[:, original_context_length:]

      # For intermediate_tokens, we only save the context part (without generation)
      # This is used for multi-stride generation
      intermediate_tokens.append(
        x[:, :original_context_length].cpu().numpy())

      # For next iteration, use the entire current sequence as target
      # This maintains the full context for better generation
      target = x

    # After all iterations, append only the generation part (not including context)
    # The generation part starts from original_context_length (where we removed original target's EOS)
    # Format should be: [BOS, BOI, d0, d1, d2, d3, BOI, d0, d1, d2, d3, ..., EOS]
    # But we removed the BOS during concatenation, so we need to add it back
    generation_part_no_bos = target[:, original_context_length:]  # [BOI, items..., EOS], length: stride_length - 1

    # Add BOS token to match expected stride_length
    bos_token = torch.full((generation_part_no_bos.shape[0], 1),
                          self.tokenizer.bos_token,
                          dtype=generation_part_no_bos.dtype,
                          device=generation_part_no_bos.device)
    generation_part = torch.cat((bos_token, generation_part_no_bos), dim=1)  # [BOS, BOI, items..., EOS], length: stride_length
    intermediate_tokens.append(generation_part.cpu().numpy())
    # intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token).cumsum(-1) == 0).sum(-1)

    # for i in range(2, len(intermediate_tokens) + 1):
    #   intermediate_text_samples.append(
    #     self.tokenizer.batch_decode(
    #       np.concatenate(intermediate_tokens[:i], axis=1)))
    
    return (sampling_steps, intermediate_tokens, #intermediate_text_samples
            sequence_lengths)


  def restore_model_and_sample_next_item(
      self, input_ids, num_steps, eps=1e-5, context_emb=None,
      mu_c=None, sigma_c2=None, sequence_mask=None):
    """Generate one item by full-MASK completion for the WP-D demo."""
    return self.restore_model_and_sample_items(
      input_ids=input_ids, num_items=1, num_steps=num_steps, eps=eps,
      context_emb=context_emb, mu_c=mu_c, sigma_c2=sigma_c2,
      sequence_mask=sequence_mask)

  def restore_model_and_sample_items(
      self, input_ids, num_items, num_steps, eps=1e-5, context_emb=None,
      mu_c=None, sigma_c2=None, sequence_mask=None):
    """Jointly generate ``num_items`` full-MASK item slots."""
    backbone_was_training = self.backbone.training
    noise_was_training = self.noise.training
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(), self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(), self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    try:
      masked_ids, completion_mask, completed_sequence_mask = (
        self.build_item_completion(
          input_ids, num_items=num_items, sequence_mask=sequence_mask))
      completed_ids = self.sample_masked_completion(
        masked_ids=masked_ids,
        completion_mask=completion_mask,
        num_steps=num_steps,
        eps=eps,
        context_emb=context_emb,
        mu_c=mu_c,
        sigma_c2=sigma_c2,
        sequence_mask=completed_sequence_mask)
      return self.extract_item_completion(
        completed_ids, completion_mask, num_items=num_items)
    finally:
      if self.ema:
        self.ema.restore(itertools.chain(
          self.backbone.parameters(), self.noise.parameters()))
      self.backbone.train(backbone_was_training)
      self.noise.train(noise_was_training)


  def restore_model_and_semi_ar_sample(
      self, input_ids, stride_length, num_strides, dt=0.001,
      context_emb=None, mu_c=None, sigma_c2=None):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
       input_ids=input_ids,
       stride_length=stride_length,
       num_strides=num_strides,
       dt=dt,
       context_emb=context_emb,
       mu_c=mu_c,
       sigma_c2=sigma_c2)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
