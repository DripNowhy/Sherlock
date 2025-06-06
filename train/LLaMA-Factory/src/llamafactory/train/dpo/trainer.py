# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach

import os


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.refine_beta = finetuning_args.refine_beta
        self.loss_type = finetuning_args.pref_loss
        self.pref_loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def split_batch(self, batch, split_size1, split_size2, split_size3):
        """
        Split a batch into three parts with specified sizes.
        
        Args:
            batch: Input batch dictionary
            split_size1: Size of first split
            split_size2: Size of second split
            split_size3: Size of third split
            
        Returns:
            Tuple of (split batches, weights)
        """
        splitted_batches = []
        new_batch_1 = {}
        new_batch_2 = {}
        new_batch_3 = {}
        
        for k, v in batch.items():
            if k != "weights":
                if k not in new_batch_1:
                    new_batch_1[k] = None
                    new_batch_2[k] = None
                new_batch_1[k] = (v[0:split_size1])
                new_batch_2[k] = (v[split_size1:split_size1+split_size2])
                new_batch_3[k] = (v[split_size1+split_size2:split_size1+split_size2+split_size3])
            else:
                weights = v[0].item()

        splitted_batches.append(new_batch_1)
        splitted_batches.append(new_batch_2)
        splitted_batches.append(new_batch_3)
        return splitted_batches, weights

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def refine_loss(
        self,
        policy_chosen_refined_w_logps: "torch.Tensor",
        policy_chosen_refined_l_logps: "torch.Tensor",
        policy_rejected_refined_w_logps: "torch.Tensor",
        policy_rejected_refined_l_logps: "torch.Tensor",
        reference_chosen_refined_w_logps: Optional["torch.Tensor"],
        reference_chosen_refined_l_logps: Optional["torch.Tensor"],
        reference_rejected_refined_w_logps: Optional["torch.Tensor"],
        reference_rejected_refined_l_logps: Optional["torch.Tensor"],
        weight=1
    ) -> "torch.Tensor":
        """
        Compute refine loss for batched log probabilities.
        
        Args:
            policy_chosen_refined_w_logps: Policy model log probabilities for chosen refined win responses
            policy_chosen_refined_l_logps: Policy model log probabilities for chosen refined loss responses
            policy_rejected_refined_w_logps: Policy model log probabilities for rejected refined win responses
            policy_rejected_refined_l_logps: Policy model log probabilities for rejected refined loss responses
            reference_chosen_refined_w_logps: Reference model log probabilities for chosen refined win responses
            reference_chosen_refined_l_logps: Reference model log probabilities for chosen refined loss responses
            reference_rejected_refined_w_logps: Reference model log probabilities for rejected refined win responses
            reference_rejected_refined_l_logps: Reference model log probabilities for rejected refined loss responses
            weight: Weight factor for loss computation
            
        Returns:
            Tuple of (refine loss, rewards for different response types)
        """
        def get_logits(
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios
            return weight * self.refine_beta * logits

        refine_chosen_loss = 1 - get_logits(
            policy_chosen_refined_w_logps,
            policy_chosen_refined_l_logps,
            reference_chosen_refined_w_logps,
            reference_chosen_refined_l_logps
        ) - get_logits(
            policy_rejected_refined_w_logps,
            policy_rejected_refined_l_logps,
            reference_rejected_refined_w_logps,
            reference_rejected_refined_l_logps
        )

        # Compute rewards for different response types
        chosen_refined_w_rewards = (
            self.refine_beta
            * (
                policy_chosen_refined_w_logps.to(self.accelerator.device) - reference_chosen_refined_w_logps.to(self.accelerator.device)
            ).detach()
        )
        chosen_refined_l_rewards = (
            self.refine_beta
            * (
                policy_chosen_refined_l_logps.to(self.accelerator.device) - reference_chosen_refined_l_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_refined_w_rewards = (
            self.refine_beta
            * (
                policy_rejected_refined_w_logps.to(self.accelerator.device) - reference_rejected_refined_w_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_refined_l_rewards = (
            self.refine_beta
            * (
                policy_rejected_refined_l_logps.to(self.accelerator.device) - reference_rejected_refined_l_logps.to(self.accelerator.device)
            ).detach()
        )

        chosen_loss = torch.square(refine_chosen_loss)
        refine_loss = chosen_loss

        return refine_loss, chosen_refined_w_rewards, chosen_refined_l_rewards, rejected_refined_w_rewards, rejected_refined_l_rewards

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        weight=1,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, weight
            )

        return losses, chosen_rewards, rejected_rewards, None, None

    def compute_refine_loss(
        self,
        policy_chosen_refined_w_logps: "torch.Tensor"=None,
        policy_chosen_refined_l_logps: "torch.Tensor"=None,
        policy_rejected_refined_w_logps: "torch.Tensor"=None,
        policy_rejected_refined_l_logps: "torch.Tensor"=None,
        reference_chosen_refined_w_logps: Optional["torch.Tensor"]=None,
        reference_chosen_refined_l_logps: Optional["torch.Tensor"]=None,
        reference_rejected_refined_w_logps: Optional["torch.Tensor"]=None,
        reference_rejected_refined_l_logps: Optional["torch.Tensor"]=None,
        weight=None
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """

        refine_loss, chosen_refined_w_rewards, chosen_refined_l_rewards, rejected_refined_w_rewards, rejected_refined_l_rewards = self.refine_loss(
            policy_chosen_refined_w_logps, policy_chosen_refined_l_logps, policy_rejected_refined_w_logps, policy_rejected_refined_l_logps, reference_chosen_refined_w_logps, reference_chosen_refined_l_logps, reference_rejected_refined_w_logps, reference_rejected_refined_l_logps,
            weight
        )
        losses = 0.8 * refine_loss
        # del refine_loss
        return losses, chosen_refined_w_rewards, chosen_refined_l_rewards, rejected_refined_w_rewards, rejected_refined_l_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length, per_token_logp = get_batch_logps(logits=all_logits, labels=batch["labels"])

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        if self.loss_type in ["sr"]:
            if batch["input_ids"].size(0) == 2:
                batch_size = batch["input_ids"].size(0) // 2
                chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
                chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
                chosen_length, _ = valid_length.split(batch_size, dim=0)
                return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
            else:
                batch_size = batch["input_ids"].size(0) // 4
                chosen_refined_w_logps, chosen_refined_l_logps, rejected_refined_w_logps, rejected_refined_l_logps = all_logps.split(batch_size, dim=0)
                chosen_refined_w_logits, chosen_refined_l_logits, rejected_refined_w_logits, rejected_refined_l_logits = all_logits.split(batch_size, dim=0)
                chosen_refined_w_length,_, rejected_refined_w_length, _ = valid_length.split(batch_size, dim=0)
                return chosen_refined_w_logps, chosen_refined_l_logps, rejected_refined_w_logps, rejected_refined_l_logps, chosen_refined_w_logits, chosen_refined_l_logits, rejected_refined_w_logits, rejected_refined_l_logits, chosen_refined_w_logps / chosen_refined_w_length, rejected_refined_w_logps / rejected_refined_w_length

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    def concatenated_forward_refine(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length, per_token_logp = get_batch_logps(logits=all_logits, labels=batch["labels"])

        if self.loss_type in ["sr"]:
            batch_size = batch["input_ids"].size(0) // 4
            chosen_refined_w_logps, chosen_refined_l_logps, rejected_refined_w_logps, rejected_refined_l_logps = all_logps.split(batch_size, dim=0)
            chosen_refined_w_logits, chosen_refined_l_logits, rejected_refined_w_logits, rejected_refined_l_logits = all_logits.split(batch_size, dim=0)
            chosen_refined_w_length,_, rejected_refined_w_length, _ = valid_length.split(batch_size, dim=0)
            return chosen_refined_w_logps, chosen_refined_l_logps, rejected_refined_w_logps, rejected_refined_l_logps, chosen_refined_w_logits, chosen_refined_l_logits, rejected_refined_w_logits, rejected_refined_l_logits, chosen_refined_w_logps / chosen_refined_w_length, rejected_refined_w_logps / rejected_refined_w_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            if self.loss_type == "sr":
                reference_chosen_logps, reference_rejected_logps,*_ = self.concatenated_forward(ref_model, batch)
                return reference_chosen_logps, reference_rejected_logps
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    def compute_reference_log_probs_refine(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """
        Compute refined log probabilities of the reference model.
        
        Args:
            model: The model to compute log probabilities with
            batch: Input batch dictionary
            
        Returns:
            Tuple of reference model refined log probabilities for chosen and rejected responses
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            if self.loss_type == "sr":
                reference_chosen_refined_w_logps, reference_chosen_refined_l_logps, reference_rejected_refined_w_logps ,reference_rejected_refined_l_logps,_,_,_,_,_,_ = self.concatenated_forward_refine(ref_model, batch)
                return reference_chosen_refined_w_logps, reference_chosen_refined_l_logps, reference_rejected_refined_w_logps, reference_rejected_refined_l_logps

            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}

        if self.loss_type == "sr":
            # Split batch for different loss computations
            new_bacth, weight = self.split_batch(batch, 2, 4, 4)
            dpo_batch = new_bacth[0]
            refine_w_batch = new_bacth[1]
            refine_l_batch = new_bacth[2]

            # Compute refined log probabilities and loss
            (
                policy_chosen_refined_w_logps,
                policy_chosen_refined_l_logps,
                policy_rejected_refined_w_logps,
                policy_rejected_refined_l_logps,
                policy_chosen_refined_w_logits,
                policy_chosen_refined_l_logits,
                policy_rejected_refined_w_logits,
                policy_rejected_refined_l_logits,
                policy_chosen_refined_logps_avg,
                policy_rejected_refined_logps_avg,
            ) = self.concatenated_forward_refine(model, refine_w_batch)

            # Compute SFT loss if needed
            if self.ftx_gamma > 1e-6:
                refine_sft_loss = -policy_chosen_refined_logps_avg - policy_rejected_refined_logps_avg

            # Get reference model log probabilities
            reference_chosen_refined_w_logps, reference_chosen_refined_l_logps, reference_rejected_refined_w_logps, reference_rejected_refined_l_logps = self.compute_reference_log_probs_refine(model, refine_w_batch)

            # Compute refine loss and rewards
            refine_loss, chosen_refined_w_rewards, chosen_refined_l_rewards, rejected_refined_w_rewards, rejected_refined_l_rewards = self.compute_refine_loss(
                policy_chosen_refined_w_logps,
                policy_chosen_refined_l_logps,
                policy_rejected_refined_w_logps,
                policy_rejected_refined_l_logps,
                reference_chosen_refined_w_logps,
                reference_chosen_refined_l_logps,
                reference_rejected_refined_w_logps,
                reference_rejected_refined_l_logps,
                weight
            )

            # Add SFT loss if needed
            if self.ftx_gamma > 1e-6:
                refine_loss += self.ftx_gamma * refine_sft_loss

            refine_loss = refine_loss.mean()
            self.accelerator.backward(refine_loss)

            # Compute second set of refined log probabilities and loss
            (
                policy_chosen_refined_w_logps_2,
                policy_chosen_refined_l_logps_2,
                policy_rejected_refined_w_logps_2,
                policy_rejected_refined_l_logps_2,
                policy_chosen_refined_w_logits_2,
                policy_chosen_refined_l_logits_2,
                policy_rejected_refined_w_logits_2,
                policy_rejected_refined_l_logits_2,
                policy_chosen_refined_logps_avg_2,
                policy_rejected_refined_logps_avg_2,
            ) = self.concatenated_forward_refine(model, refine_l_batch)

            if self.ftx_gamma > 1e-6:
                refine_sft_loss_2 = -policy_chosen_refined_logps_avg_2 - policy_rejected_refined_logps_avg_2

            reference_chosen_refined_w_logps_2, reference_chosen_refined_l_logps_2, reference_rejected_refined_w_logps_2, reference_rejected_refined_l_logps_2 = self.compute_reference_log_probs_refine(model, refine_l_batch)
            refine_loss_2, chosen_refined_w_rewards_2, chosen_refined_l_rewards_2, rejected_refined_w_rewards_2, rejected_refined_l_rewards_2 = self.compute_refine_loss(
                policy_chosen_refined_w_logps_2,
                policy_chosen_refined_l_logps_2,
                policy_rejected_refined_w_logps_2,
                policy_rejected_refined_l_logps_2,
                reference_chosen_refined_w_logps_2,
                reference_chosen_refined_l_logps_2,
                reference_rejected_refined_w_logps_2,
                reference_rejected_refined_l_logps_2,
                weight
            )

            if self.ftx_gamma > 1e-6:
                refine_loss_2 += self.ftx_gamma * refine_sft_loss_2

            refine_loss_2 = refine_loss_2.mean()
            self.accelerator.backward(refine_loss_2)

            # Compute DPO loss
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg
            ) = self.concatenated_forward(model, dpo_batch)
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, dpo_batch)
            losses, chosen_rewards, rejected_rewards,_,_ = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                weight
            )
            losses = 0.2 * losses

            # Add SFT loss if needed
            sft_loss = -policy_chosen_logps_avg
            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss
                metrics["losses/sft"] = self.ftx_gamma * (sft_loss.mean().item()+refine_sft_loss.mean().item())

            # Record metrics
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}losses/dpo"] = 5*losses.mean().item()
            metrics[f"{prefix}losses/refine"] = 1.25*(refine_loss+refine_loss_2.mean()).item()
            metrics[f"{prefix}losses/total"] = losses.mean().item() + refine_loss.item()
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
            metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
            metrics[f"{prefix}rewards/chosen_refined_w_1"] = chosen_refined_w_rewards.mean().item()
            metrics[f"{prefix}rewards/chosen_refined_l_1"] = chosen_refined_l_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected_refined_w_1"] = rejected_refined_w_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected_refined_l_1"] = rejected_refined_l_rewards.mean().item()
            metrics[f"{prefix}rewards/accuracies_chosen_1"] = (chosen_refined_w_rewards > chosen_refined_l_rewards).float().mean().item()
            metrics[f"{prefix}rewards/margins_chosen_1"] = (chosen_refined_w_rewards - chosen_refined_l_rewards).mean().item()
            metrics[f"{prefix}rewards/accuracies_rejected_1"] = (rejected_refined_w_rewards > rejected_refined_l_rewards).float().mean().item()
            metrics[f"{prefix}rewards/margins_rejected_1"] = (rejected_refined_w_rewards - rejected_refined_l_rewards).mean().item()
            metrics[f"{prefix}rewards/chosen_refined_w_2"] = chosen_refined_w_rewards_2.mean().item()
            metrics[f"{prefix}rewards/chosen_refined_l_2"] = chosen_refined_l_rewards_2.mean().item()
            metrics[f"{prefix}rewards/rejected_refined_w_2"] = rejected_refined_w_rewards_2.mean().item()
            metrics[f"{prefix}rewards/rejected_refined_l_2"] = rejected_refined_l_rewards_2.mean().item()
            metrics[f"{prefix}rewards/accuracies_chosen_2"] = (chosen_refined_w_rewards_2 > chosen_refined_l_rewards_2).float().mean().item()
            metrics[f"{prefix}rewards/margins_chosen_2"] = (chosen_refined_w_rewards_2 - chosen_refined_l_rewards_2).mean().item()
            metrics[f"{prefix}rewards/accuracies_rejected_2"] = (rejected_refined_w_rewards_2 > rejected_refined_l_rewards_2).float().mean().item()
            metrics[f"{prefix}rewards/margins_rejected_2"] = (rejected_refined_w_rewards_2 - rejected_refined_l_rewards_2).mean().item()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
            metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
            metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()

            return losses.mean(), metrics

        # Standard DPO loss computation
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # Add SFT loss if needed
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # Record metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()

        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Subclass and override to accept extra kwargs.
        """
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        """
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
