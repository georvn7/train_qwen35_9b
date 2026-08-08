#!/usr/bin/env python3
"""Memory-bounded, mathematically equivalent split-backward DPO support."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import Any

import torch
from trl.trainer.utils import flush_left, flush_right, selective_log_softmax


EXECUTION_MODES = {"batched", "split_backward"}


def release_branch_memory() -> None:
    """Release inactive branch blocks before the next graph or optimizer allocation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_split_configuration(trainer: Any) -> None:
    """Fail rather than silently changing an unsupported DPO objective."""
    loss_types = trainer.loss_type if isinstance(trainer.loss_type, list) else [trainer.loss_type]
    unsupported: list[str] = []
    if loss_types != ["sigmoid"]:
        unsupported.append(f"loss_type={loss_types!r}")
    if getattr(trainer.args, "rpo_alpha", None) is not None:
        unsupported.append("rpo_alpha")
    if getattr(trainer, "use_weighting", False):
        unsupported.append("policy_weighting")
    if getattr(trainer, "aux_loss_enabled", False):
        unsupported.append("auxiliary_router_loss")
    if getattr(trainer, "is_encoder_decoder", False):
        unsupported.append("encoder_decoder")
    if getattr(trainer, "padding_free", False):
        unsupported.append("padding_free")
    if getattr(trainer.args, "use_liger_loss", False):
        unsupported.append("liger_loss")
    if unsupported:
        raise ValueError(
            "split_backward does not support this DPO configuration: "
            + ", ".join(unsupported)
        )


def single_completion_forward(
    trainer: Any,
    model: torch.nn.Module,
    batch: dict[str, Any],
    completion: str,
) -> dict[str, torch.Tensor]:
    """Run one prompt/completion branch with TRL-compatible completion log-probs."""
    if completion not in {"chosen", "rejected"}:
        raise ValueError(f"unknown completion branch: {completion!r}")
    _validate_split_configuration(trainer)

    unsupported_keys = {"pixel_values", "pixel_attention_mask", "image_sizes"} & set(batch)
    if unsupported_keys:
        raise ValueError(
            "split_backward currently supports text-only causal models; got "
            + ", ".join(sorted(unsupported_keys))
        )

    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    completion_input_ids = batch[f"{completion}_input_ids"]
    completion_attention_mask = batch[f"{completion}_attention_mask"]

    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
        dim=1,
    )
    token_type_ids = None
    if "token_type_ids" in batch:
        prompt_token_type_ids = batch["token_type_ids"]
        pad_width = input_ids.shape[1] - prompt_token_type_ids.shape[1]
        token_type_ids = torch.nn.functional.pad(prompt_token_type_ids, (0, pad_width), value=0)

    if trainer.max_length is not None and trainer.max_length < attention_mask.size(1):
        if trainer.truncation_mode == "keep_start":
            if token_type_ids is None:
                attention_mask, input_ids, loss_mask = flush_left(
                    attention_mask, input_ids, loss_mask
                )
            else:
                attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                    attention_mask, input_ids, loss_mask, token_type_ids
                )
            attention_mask = attention_mask[:, : trainer.max_length]
            input_ids = input_ids[:, : trainer.max_length]
            loss_mask = loss_mask[:, : trainer.max_length]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, : trainer.max_length]
        elif trainer.truncation_mode == "keep_end":
            if token_type_ids is None:
                attention_mask, input_ids, loss_mask = flush_right(
                    attention_mask, input_ids, loss_mask
                )
            else:
                attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                    attention_mask, input_ids, loss_mask, token_type_ids
                )
                token_type_ids = token_type_ids[:, -trainer.max_length :]
            input_ids = input_ids[:, -trainer.max_length :]
            attention_mask = attention_mask[:, -trainer.max_length :]
            loss_mask = loss_mask[:, -trainer.max_length :]
            if token_type_ids is None:
                attention_mask, input_ids, loss_mask = flush_left(
                    attention_mask, input_ids, loss_mask
                )
            else:
                attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                    attention_mask, input_ids, loss_mask, token_type_ids
                )
        else:
            raise ValueError(f"unknown truncation mode: {trainer.truncation_mode!r}")
    else:
        if token_type_ids is None:
            attention_mask, input_ids, loss_mask = flush_left(
                attention_mask, input_ids, loss_mask
            )
        else:
            attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                attention_mask, input_ids, loss_mask, token_type_ids
            )

    if not loss_mask.any():
        raise ValueError(f"{completion} completion has no trainable tokens after truncation")

    model_kwargs: dict[str, Any] = {
        "use_cache": False,
        "attention_mask": attention_mask,
    }
    if token_type_ids is not None:
        model_kwargs["token_type_ids"] = token_type_ids

    logits_to_keep = None
    if trainer.use_logits_to_keep:
        first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
        logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
        model_kwargs["logits_to_keep"] = logits_to_keep

    outputs = model(input_ids, **model_kwargs)
    logits = outputs.logits
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    shifted_loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
    if logits_to_keep is not None:
        labels = labels[:, -logits_to_keep:]
        shifted_loss_mask = shifted_loss_mask[:, -logits_to_keep:]
    if logits.shape[:2] != labels.shape[:2]:
        logits = logits[:, -labels.shape[1] :]

    labels = labels.clone()
    labels[~shifted_loss_mask] = 0
    per_token_logps = selective_log_softmax(logits, labels)
    per_token_logps[~shifted_loss_mask] = 0
    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    logps = per_token_logps[:, 1:].sum(-1)
    mean_logits = logits[shifted_loss_mask].mean()
    return {"logps": logps, "mean_logits": mean_logits}


class SplitBackwardDPOTrainerMixin:
    """Serialize chosen/rejected policy graphs while preserving the DPO gradient."""

    dpo_execution_mode = "batched"

    def compute_ref_log_probs(self, batch):
        if self.dpo_execution_mode != "split_backward":
            return super().compute_ref_log_probs(batch)
        _validate_split_configuration(self)
        autocast_context = (
            torch.autocast(device_type=self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            if self.ref_model is None:
                with self.null_ref_context():
                    chosen = single_completion_forward(self, self.model, batch, "chosen")
                    rejected = single_completion_forward(self, self.model, batch, "rejected")
            else:
                chosen = single_completion_forward(self, self.ref_model, batch, "chosen")
                rejected = single_completion_forward(self, self.ref_model, batch, "rejected")
        return chosen["logps"], rejected["logps"]

    def _split_metrics(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        mean_chosen_logits: torch.Tensor,
        mean_rejected_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        chosen_leaf = chosen_logps.detach().requires_grad_(True)
        rejected_leaf = rejected_logps.detach().requires_grad_(True)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_leaf,
            rejected_leaf,
            ref_chosen_logps,
            ref_rejected_logps,
            "sigmoid",
            None,
        )
        loss = losses.mean()
        chosen_coefficient, rejected_coefficient = torch.autograd.grad(
            loss, (chosen_leaf, rejected_leaf)
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics = {
            "rewards/chosen": self.accelerator.gather_for_metrics(chosen_rewards).mean().item(),
            "rewards/rejected": self.accelerator.gather_for_metrics(rejected_rewards).mean().item(),
            "rewards/accuracies": self.accelerator.gather_for_metrics(reward_accuracies).mean().item(),
            "rewards/margins": self.accelerator.gather_for_metrics(
                chosen_rewards - rejected_rewards
            ).mean().item(),
            "logps/chosen": self.accelerator.gather_for_metrics(chosen_logps).mean().item(),
            "logps/rejected": self.accelerator.gather_for_metrics(rejected_logps).mean().item(),
            "logits/chosen": self.accelerator.gather_for_metrics(mean_chosen_logits).mean().item(),
            "logits/rejected": self.accelerator.gather_for_metrics(mean_rejected_logits).mean().item(),
        }
        return loss, metrics, chosen_coefficient, rejected_coefficient

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.dpo_execution_mode != "split_backward":
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        _validate_split_configuration(self)
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            if "ref_chosen_logps" not in inputs or "ref_rejected_logps" not in inputs:
                raise ValueError("split_backward requires precomputed reference log-probabilities")

            with self.compute_loss_context_manager(), torch.no_grad():
                chosen_probe = single_completion_forward(self, model, inputs, "chosen")
                rejected_probe = single_completion_forward(self, model, inputs, "rejected")

            ref_chosen = inputs["ref_chosen_logps"]
            ref_rejected = inputs["ref_rejected_logps"]
            loss, metrics, chosen_coefficient, rejected_coefficient = self._split_metrics(
                chosen_probe["logps"],
                rejected_probe["logps"],
                ref_chosen,
                ref_rejected,
                chosen_probe["mean_logits"],
                rejected_probe["mean_logits"],
            )
            del chosen_probe, rejected_probe
            release_branch_memory()

            normalize = (
                (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
                and self.compute_loss_func is None
            )
            divisor = self.current_gradient_accumulation_steps if normalize else 1

            with self.compute_loss_context_manager():
                chosen_train = single_completion_forward(self, model, inputs, "chosen")
                chosen_surrogate = (
                    chosen_coefficient.detach() * chosen_train["logps"]
                ).sum() / divisor
            self.accelerator.backward(chosen_surrogate)
            del chosen_train, chosen_surrogate
            release_branch_memory()

            with self.compute_loss_context_manager():
                rejected_train = single_completion_forward(self, model, inputs, "rejected")
                rejected_surrogate = (
                    rejected_coefficient.detach() * rejected_train["logps"]
                ).sum() / divisor
            self.accelerator.backward(rejected_surrogate)
            reported_loss = (loss / divisor).detach().to(self.args.device)
            del rejected_train, rejected_surrogate, inputs
            release_branch_memory()

            self.store_metrics(metrics, train_eval="train")
            return reported_loss
