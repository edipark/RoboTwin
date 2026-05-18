"""SoftVLAPytorch: wraps PI0Pytorch with LoRA, DTW-NCE, and Soft Prompts.

Phase 1 uses:
  - DTW-guided soft InfoNCE contrastive alignment (l_soft_nce).
  - Full fine-tuning of PaliGemma LM by default (use_lora=False).
  - Optional LoRA injection at layer 9 (use_lora=True, ~102K params).

Phase 2 is unchanged: soft prompts + flow matching.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from softvla.models.softvla_config import SoftVLAConfig
from softvla.models.soft_prompt import SoftPromptHub


class SoftVLAPytorch(nn.Module):
    """Soft-VLA model built on top of PI0Pytorch.

    Phase 1 (encoder DTW-NCE alignment):
        - Full fine-tuning of PaliGemma LM (use_lora=False, default).
        - Optional LoRA injection at layer 9 (use_lora=True).
        - DTW-guided soft InfoNCE (l_soft_nce) computed by the caller.

    Phase 2 (decoder training):
        - Encoder frozen; soft prompts prepended to suffix.
        - Full prefix+suffix forward through PaliGemmaWithExpert.
        - Flow matching MSE loss on action predictions.
    """

    def __init__(self, config: SoftVLAConfig, pi0_config=None):
        super().__init__()
        self.config = config
        self._lora_applied = False

        # Build PI0Pytorch with a compatible config object
        if pi0_config is None:
            pi0_config = _make_pi0_config(config)
        self.pi0 = PI0Pytorch(pi0_config)

        # Phase 2 modules
        self.soft_prompt_hub = SoftPromptHub(
            num_robots=config.num_robots,
            prompt_length=config.soft_prompt_length,
            hidden_dim=config.expert_hidden_dim,
        )

    # ──────────────────────────────────────────────
    # LoRA injection
    # ──────────────────────────────────────────────

    def apply_lora(self, target_layer_idx: "int | list[int]") -> None:
        """Inject PEFT LoRA adapters into one or more VLM transformer layers.

        Targets the ``q_proj`` and ``v_proj`` linear modules inside
        ``paligemma_with_expert.paligemma.language_model.layers[idx]``
        for every index in *target_layer_idx*.

        After this call, the model has two modes toggled by the PEFT API:
          - adapters disabled → teacher forward (identical to pretrained VLM)
          - adapters enabled  → student forward (LoRA modified)

        Args:
            target_layer_idx: 0-based index or list of indices of the transformer
                layers to adapt.  Negative indices are supported (Python convention).
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ImportError("peft is required for LoRA injection. Install with: pip install peft") from exc

        if self._lora_applied:
            logging.warning("[LoRA] apply_lora() called more than once; skipping.")
            return

        # Normalise to list
        if isinstance(target_layer_idx, int):
            layer_indices = [target_layer_idx]
        else:
            layer_indices = list(target_layer_idx)

        language_model = self.pi0.paligemma_with_expert.paligemma.language_model
        # openpi's GemmaModel exposes .layers directly (not .model.layers)
        n_layers = len(language_model.layers)

        # Resolve negative indices and validate
        resolved = []
        for idx in layer_indices:
            if idx < 0:
                idx = n_layers + idx
            if not (0 <= idx < n_layers):
                raise ValueError(
                    f"target_layer_idx={idx} is out of range [0, {n_layers - 1}]."
                )
            resolved.append(idx)
        layer_indices = resolved

        # Build target module names for each selected layer.
        # Paths are relative to language_model (openpi GemmaModel), which exposes
        # .layers directly — NOT .model.layers.
        target_modules = []
        for idx in layer_indices:
            target_modules.extend([
                f"layers.{idx}.self_attn.q_proj",
                f"layers.{idx}.self_attn.v_proj",
            ])

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )

        # Wrap only the language model sub-module so Phase 2 is unaffected.
        # IMPORTANT: paligemma.language_model is a read-only @property that returns
        # self.model.language_model — assigning to it via nn.Module.__setattr__
        # only stores in _modules, but the property always shadows that.
        # Must assign to the underlying PaliGemmaModel attribute directly.
        peft_model = get_peft_model(language_model, lora_cfg)
        self.pi0.paligemma_with_expert.paligemma.model.language_model = peft_model
        self._lora_applied = True

        trainable = sum(
            p.numel()
            for n, p in self.pi0.paligemma_with_expert.paligemma.model.language_model.named_parameters()
            if "lora_" in n
        )
        logging.info(
            f"[LoRA] Injected rank-{self.config.lora_rank} adapters into layer(s) {layer_indices}. "
            f"Trainable adapter params: {trainable:,}"
        )

    def get_lora_params(self) -> list[nn.Parameter]:
        """Return LoRA adapter parameters identified by name (contain 'lora_').

        Using name-based detection is required because freeze_module() may have
        zeroed requires_grad before this is called.
        """
        if not self._lora_applied:
            return []
        return [
            p
            for n, p in self.pi0.paligemma_with_expert.paligemma.model.language_model.named_parameters()
            if "lora_" in n
        ]

    # ──────────────────────────────────────────────
    # Z extraction (prefix-only forward)
    # ──────────────────────────────────────────────

    def extract_z(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefix through PaliGemma LM only and mean-pool hidden states → Z [B, D]."""
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d)
        att_4d = att_4d.to(dtype=prefix_embs.dtype)  # Match query dtype for SDPA

        (prefix_output, _), _ = self.pi0.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        # prefix_output: [B, N_prefix, vlm_hidden_dim]
        # Mask-aware mean pooling
        mask = prefix_pad_masks.unsqueeze(-1).float()  # [B, N, 1]
        z = (prefix_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return z  # [B, vlm_hidden_dim]

    # ──────────────────────────────────────────────
    # Phase 1 forward: student embedding
    # ──────────────────────────────────────────────

    def forward_phase1(
        self,
        observation,
    ) -> torch.Tensor:
        """Phase 1 forward: produce student embedding for DTW-NCE loss.

        L_soft_nce (DTW-guided soft InfoNCE) is computed by the caller.

        Args:
            observation: OpenPI observation object.

        Returns:
            z_student [B, vlm_hidden_dim] with gradients.
        """
        images, img_masks, lang_tokens, lang_masks, _state = self.pi0._preprocess_observation(
            observation, train=True
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # prefix_embs is the output of frozen embedding layers → requires_grad=False.
        # HF gradient checkpointing with use_reentrant=False requires at least one
        # grad-requiring *input tensor* to build a grad_fn chain.  If none exist,
        # checkpoint() skips grad tracking → z_student has no grad_fn → backward fails.
        # Fix: detach (break link to frozen upstream) then requires_grad_(True) so that
        # checkpoint sees a grad-requiring input.  Gradients still flow to LoRA params
        # inside the layers via the normal autograd chain through the matmuls.
        prefix_embs_s = prefix_embs.detach().requires_grad_(True)
        z_student = self.extract_z(prefix_embs_s, prefix_pad_masks, prefix_att_masks)

        return z_student

    # ──────────────────────────────────────────────
    # Phase 2 forward: action generation with soft prompts
    # ──────────────────────────────────────────────

    def forward_phase2(
        self,
        observation,
        actions: torch.Tensor,
        domain_ids: torch.LongTensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Phase 2 forward: flow matching loss with soft prompt injection.

        Args:
            observation: OpenPI observation object.
            actions: Ground truth actions [B, action_horizon, action_dim].
            domain_ids: Robot domain IDs [B].
            noise: Optional pre-sampled noise.
            time: Optional pre-sampled timesteps.

        Returns:
            MSE loss tensor [B, action_horizon, action_dim] (unreduced).
        """
        images, img_masks, lang_tokens, lang_masks, state = self.pi0._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.pi0.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.pi0.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (encoder is frozen in phase 2 — caller must freeze params)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        # Detach prefix to prevent gradients flowing into encoder
        prefix_embs = prefix_embs.detach()

        # Embed suffix (action tokens)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.pi0.embed_suffix(
            state, x_t, time
        )

        # Inject soft prompts at the beginning of suffix
        soft_prompts = self.soft_prompt_hub(domain_ids)  # [B, P, expert_dim]
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_soft_prompts(
            soft_prompts, suffix_embs, suffix_pad_masks, suffix_att_masks
        )

        # Dtype alignment — language_model is GemmaModel (has .layers directly).
        # PEFT-wrapped models also support .layers via attribute delegation.
        lm = self.pi0.paligemma_with_expert.paligemma.language_model
        base_lm = lm.base_model if hasattr(lm, "base_model") else lm
        if (
            base_lm.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Combined forward
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(att_2d_masks)
        att_2d_masks_4d = att_2d_masks_4d.to(dtype=prefix_embs.dtype)  # Match query dtype for SDPA

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.pi0.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self.pi0._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # Take last action_horizon tokens (skip soft prompt outputs)
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.pi0.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # ──────────────────────────────────────────────
    # Unified forward dispatch
    # ──────────────────────────────────────────────

    def forward(self, phase: int | None = None, **kwargs):
        """Dispatch to phase1 or phase2 forward based on config or explicit phase arg."""
        phase = phase or self.config.training_phase
        if phase == 1:
            return self.forward_phase1(**kwargs)
        elif phase == 2:
            return self.forward_phase2(**kwargs)
        else:
            raise ValueError(f"Unknown training phase: {phase}")

    # ──────────────────────────────────────────────
    # Inference (wraps PI0Pytorch.sample_actions with soft prompt injection)
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        domain_ids: torch.LongTensor,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Inference: iterative denoising with soft prompt injection.

        Uses prefix KV cache for efficiency, then injects soft prompts into
        each denoise step's suffix.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.pi0.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self.pi0._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute prefix KV cache
        prefix_att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.pi0.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Soft prompts for this domain
        soft_prompts = self.soft_prompt_hub(domain_ids)  # [B, P, expert_dim]

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step_with_prompt(
                state, prefix_pad_masks, past_key_values,
                x_t, expanded_time, soft_prompts,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def _denoise_step_with_prompt(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        soft_prompts,
    ) -> torch.Tensor:
        """Single denoise step with soft prompt injection into suffix."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.pi0.embed_suffix(
            state, x_t, timestep
        )

        # Inject soft prompts
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_soft_prompts(
            soft_prompts, suffix_embs, suffix_pad_masks, suffix_att_masks
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(full_att_2d_masks)
        self.pi0.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.pi0.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.pi0.action_out_proj(suffix_out)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _prepend_soft_prompts(
        soft_prompts: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepend soft prompt tokens to suffix embeddings and extend masks accordingly."""
        B, P, _ = soft_prompts.shape
        device = suffix_embs.device

        # Match dtype
        soft_prompts = soft_prompts.to(dtype=suffix_embs.dtype, device=device)

        # Concatenate: [soft_prompts | suffix_embs]
        suffix_embs = torch.cat([soft_prompts, suffix_embs], dim=1)

        # Extend pad masks: soft prompts are always valid
        prompt_pad = torch.ones(B, P, dtype=suffix_pad_masks.dtype, device=device)
        suffix_pad_masks = torch.cat([prompt_pad, suffix_pad_masks], dim=1)

        # Extend attention masks: soft prompts use causal-style (1s) so prefix doesn't attend to them
        prompt_att = torch.ones(B, P, dtype=suffix_att_masks.dtype, device=device)
        suffix_att_masks = torch.cat([prompt_att, suffix_att_masks], dim=1)

        return suffix_embs, suffix_pad_masks, suffix_att_masks


def _make_pi0_config(config: SoftVLAConfig):
    """Create a simple namespace config compatible with PI0Pytorch constructor."""

    class _Pi0RuntimeConfig:
        pass

    c = _Pi0RuntimeConfig()
    c.pi05 = config.pi05
    c.dtype = config.dtype
    c.paligemma_variant = config.paligemma_variant
    c.action_expert_variant = config.action_expert_variant
    c.action_dim = config.action_dim
    c.action_horizon = config.action_horizon
    c.max_token_len = config.max_token_len
    c.pytorch_compile_mode = config.pytorch_compile_mode
    return c
