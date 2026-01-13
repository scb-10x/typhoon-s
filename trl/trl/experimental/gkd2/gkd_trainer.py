# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 Typhoon Team. All rights reserved.
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
import time
import os
import random
import textwrap
import warnings
import logging
import json
import math
from collections import defaultdict, deque
from contextlib import nullcontext, contextmanager
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, SaveStrategy
from transformers.utils import (
    is_liger_kernel_available,
    is_peft_available,
    is_flash_attn_2_available
)

from termcolor import colored

from ...data_utils import is_conversational, is_conversational_from_value, maybe_convert_to_chatml, pack_dataset, prepare_multimodal_messages, truncate_dataset
from ...extras.profiling import profiling_decorator
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed, prepare_fsdp
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import (
    DataCollatorForChatML,
    create_model_from_path,
    disable_dropout_in_model,
    ensure_master_addr_port,
    remove_none_values,
    entropy_from_logits,
)
from .gkd_config import GKDConfig


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

from .fsdp_utils import (
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
    set_expandable_segments,
    aggressive_empty_cache,
    get_device_id,
    get_torch_device
)
if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))

def get_dataset_column_names(dataset: Union[Dataset, IterableDataset]) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names

def compute_position_id_with_mask(mask):
    """Compute position IDs from attention mask.
    
    Args:
        mask: Attention mask tensor of shape (batch_size, seq_len)
        
    Returns:
        Position IDs tensor of shape (batch_size, seq_len)
    """
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)

class GKDTrainer(SFTTrainer):
    _tag_names = ["trl", "gkd"]
    _name = "GKD"
    _paper = {}

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        self.teacher_tokenizer = None
        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # Apply monkey patch for use_remove_padding if enabled (similar to verl)
        if args.use_remove_padding:
            try:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                # Apply to student model
                apply_monkey_patch(
                    model=self.model,
                    use_remove_padding=True,
                    ulysses_sp_size=1,  # Not using Ulysses SP in TRL
                    use_fused_kernels=False,
                    fused_kernels_backend=None,
                )
                logger.info("Applied monkey patch for use_remove_padding on student model")
            except ImportError:
                logger.warning(
                    "use_remove_padding is enabled but verl monkey_patch is not available. "
                    "Flash Attention varlen support may not work correctly."
                )

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.distill_top_k = args.distill_top_k
        self.teacher_prompt_prefix = args.teacher_prompt_prefix
        
        # Tokenize teacher prompt prefix if provided
        self.teacher_prefix_ids = None
        self.teacher_prefix_length = 0
        if self.teacher_prompt_prefix is not None and len(self.teacher_prompt_prefix) > 0:
            # Tokenize the prefix without adding special tokens
            prefix_tokens = processing_class(
                self.teacher_prompt_prefix,
                add_special_tokens=False,
                return_tensors="pt"
            )
            self.teacher_prefix_ids = prefix_tokens.input_ids
            self.teacher_prefix_length = self.teacher_prefix_ids.size(1)
            logger.info(f"Teacher prompt prefix tokenized: '{self.teacher_prompt_prefix}' -> {self.teacher_prefix_length} tokens")

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # Hybrid ULD matched/unmatched accumulators (logged every step when ULD hybrid is used)
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.crossentropy_weight = args.crossentropy_weight

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.enable_hybrid_mode = args.enable_hybrid_mode
            if self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.enable_hybrid_mode,
                )

                if self.enable_hybrid_mode:
                    self.vllm_engine.reset_prefix_cache()
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex

            if isinstance(teacher_model, str):
                init_kwargs = dict(teacher_model_init_kwargs)
                if "torch_dtype" in init_kwargs and "dtype" not in init_kwargs:
                    init_kwargs["dtype"] = init_kwargs.pop("torch_dtype")
                
                teacher_model = create_model_from_path(teacher_model, **init_kwargs)
                teacher_model.resize_token_embeddings(self.model.config.vocab_size)
            teacher_model.requires_grad_(False)
            self.teacher_model = teacher_model
            self.teacher_model.eval()
            
            # Apply monkey patch to teacher model if use_remove_padding is enabled
            if args.use_remove_padding:
                try:
                    from verl.models.transformers.monkey_patch import apply_monkey_patch
                    apply_monkey_patch(
                        model=self.teacher_model,
                        use_remove_padding=True,
                        ulysses_sp_size=1,
                        use_fused_kernels=False,
                        fused_kernels_backend=None,
                    )
                    logger.info("Applied monkey patch for use_remove_padding on teacher model")
                except ImportError:
                    logger.warning(
                        "use_remove_padding is enabled but verl monkey_patch is not available for teacher model."
                    )
            
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "prompts",
            "prompt_attention_mask",
            "messages",
            "chat_template_kwargs",
            "tools",
            "original_prompt_text",
            "original_completion_text",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = 48

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(example, processing_class, dataset_text_field, assistant_only_loss):
                    # Parse tools field if it's a JSON string (for datasets with string-encoded tools)
                    if 'tools' in example and isinstance(example['tools'], str):
                        example['tools'] = json.loads(example['tools'])
                    
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            if self._is_vlm:
                                prompt = prepare_multimodal_messages(example["prompt"], images=[])
                                completion = prepare_multimodal_messages(example["completion"], images=[])
                            else:
                                prompt = example["prompt"]
                                completion = example["completion"]
                            prompt_ids = processing_class.apply_chat_template(
                                prompt,
                                tokenize=True,
                                add_generation_prompt=True,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                            prompt_completion_processed = processing_class.apply_chat_template(
                                prompt + completion,
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_completion_processed = {
                                k: v[0] if isinstance(v[0], list) else v
                                for k, v in prompt_completion_processed.items()
                            }
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        # Parse messages field if it's a JSON string (for datasets with string-encoded messages)
                        if 'messages' in example and isinstance(example['messages'], str):
                            example['messages'] = json.loads(example['messages'])
                        # Note: tools field is already parsed at the start of tokenize_fn
                        if is_conversational(example):
                            if self._is_vlm:
                                messages = prepare_multimodal_messages(example["messages"], images=[])
                            else:
                                messages = example["messages"]
                            processed = processing_class.apply_chat_template(
                                messages,
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}

                    if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                        raise RuntimeError(
                            "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                            "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                            "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                            "ensure it's correctly configured to support assistant masking."
                        )
                    return output

                dataset = dataset.map(
                    tokenize_fn,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in get_dataset_column_names(dataset):
                    columns.append("completion_mask")
                if "assistant_masks" in get_dataset_column_names(dataset):
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        return dataset

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=0,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If > 0, only use top-k logits for distillation (default: 0, use all logits)

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature
            
            # Top-k distillation: compute JSD on reduced k-dim support (no -inf masking)
            if top_k > 0:
                k = min(top_k, teacher_logits.size(-1))
                topk_indices = torch.topk(teacher_logits, k=k, dim=-1).indices
                # Gather both teacher and student logits on teacher's top-k indices
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
                student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            else:
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logaddexp(
                student_log_probs + torch.log1p(-beta_tensor),
                teacher_log_probs + torch.log(beta_tensor),
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student
            del mixture_log_probs, kl_teacher, kl_student

        jsd = jsd.sum(dim=-1)

        if labels is not None:
            mask = (labels != -100).float()
            if reduction == "batchmean" or reduction == "mean":
                num_valid = mask.sum().clamp(min=1.0)
                out = (jsd * mask).sum() / num_valid
            elif reduction == "sum":
                out = (jsd * mask).sum()
            else:
                out = jsd * mask
        else:
            if reduction == "batchmean" or reduction == "mean":
                out = jsd.mean()
            elif reduction == "sum":
                out = jsd.sum()
            else:
                out = jsd

        del student_log_probs, teacher_log_probs, jsd
        aggressive_empty_cache(force_sync=False, max_retries=1)
        return out

    def _debug_print_alignment(
        self,
        shifted_student_logits: torch.Tensor,
        shifted_teacher_logits: torch.Tensor,
        shifted_labels: torch.Tensor,
        shifted_attention_mask: torch.Tensor | None = None,
        shifted_input_ids: torch.Tensor | None = None,
    ) -> None:
        """Print alignment between labels and per-position KD JSD with token strings.

        Prints for the first sample in batch. Gated by env GKD_DEBUG_ALIGN=="1" and main process only.
        """
        try:
            if os.getenv("GKD_DEBUG_ALIGN", "0") != "1":
                return
            if not getattr(self.accelerator, "is_main_process", True):
                return
            with torch.no_grad():
                temp = max(1e-6, float(self.temperature))
                beta = float(self.beta)
                s_logits = shifted_student_logits.to(dtype=torch.float32)
                t_logits = shifted_teacher_logits.to(dtype=torch.float32)
                student_log_probs = F.log_softmax(s_logits / temp, dim=-1)
                teacher_log_probs = F.log_softmax(t_logits / temp, dim=-1)
                b = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
                mixture_log_probs = torch.logsumexp(
                    torch.stack([student_log_probs + torch.log1p(-b), teacher_log_probs + torch.log(b)]),
                    dim=0,
                )
                kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
                kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
                jsd_pos = beta * kl_teacher + (1 - beta) * kl_student  # (B, S)

                # Also compute JSD in probability domain for verification (numerically robust for debug)
                student_probs = F.softmax(s_logits / temp, dim=-1)
                teacher_probs = F.softmax(t_logits / temp, dim=-1)
                mixture_probs = (1.0 - beta) * student_probs + beta * teacher_probs
                eps = 1e-12
                jsd_prob_teacher = (teacher_probs * (torch.log(teacher_probs + eps) - torch.log(mixture_probs + eps))).sum(-1)
                jsd_prob_student = (student_probs * (torch.log(student_probs + eps) - torch.log(mixture_probs + eps))).sum(-1)
                jsd_prob = beta * jsd_prob_teacher + (1 - beta) * jsd_prob_student

                # KL divergences between student and teacher (prob-domain)
                kl_s_t_full = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)
                kl_t_s_full = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)

                i = 0
                seq_len = shifted_labels.size(1)
                max_print = seq_len  # print whole sequence
                labels_row = shifted_labels[i]
                jsd_row = jsd_pos[i]
                jsd_prob_row = jsd_prob[i]
                kl_s_row = kl_s_t_full[i]
                kl_t_row = kl_t_s_full[i]
                stud_logits_row = shifted_student_logits[i]
                teach_logits_row = shifted_teacher_logits[i]
                attn_row = shifted_attention_mask[i] if shifted_attention_mask is not None else None

                colored_tokens = []
                text_only = False
                for t in range(seq_len):
                    lbl_id = int(labels_row[t].item())
                    attn = int(attn_row[t].item()) if attn_row is not None else 1
                    # Choose base token to display: actual input token if provided, else student argmax
                    if shifted_input_ids is not None:
                        base_id = int(shifted_input_ids[i, t].item())
                        base_tok = self.processing_class.decode([base_id], skip_special_tokens=False)
                    else:
                        base_id = int(stud_logits_row[t].argmax(-1).item())
                        base_tok = self.processing_class.decode([base_id], skip_special_tokens=False)

                    # Color scheme similar to the example
                    if attn == 0:
                        color = "red"  # pad treated as ignored
                    elif lbl_id == -100:
                        color = "red"
                    elif lbl_id == 0:
                        color = "yellow"
                    else:
                        color = "green"

                    kl_s_val = float(kl_s_row[t].item())
                    kl_t_val = float(kl_t_row[t].item())

                    token_text = colored(base_tok, color)
                    if not text_only:
                        # Append diagnostics in white
                        diag = f"(lbl={lbl_id}, id={base_id}, kl_s={kl_s_val:.2e}, kl_t={kl_t_val:.2e})"
                        token_text += colored(diag, "white")
                    colored_tokens.append(token_text)

                    max_print -= 1
                    if max_print <= 0:
                        break

                delimiter = " " if not text_only else ""
                logger.warn(delimiter.join(colored_tokens))
                logger.warn("\n\n\n")
                target_labels_count = int((labels_row != -100).sum().item())
                total_len = int(seq_len)
                logger.warn(f"Total input len: {total_len}")
                logger.warn(f"Count of labels: {target_labels_count}")
        except Exception as e:
            # Avoid crashing training due to debug printing
            warnings.warn(f"Alignment debug printing failed: {e}")

    def _prepare_inputs_for_remove_padding(self, input_ids, attention_mask):
        """
        Prepare inputs for remove_padding mode exactly like VERL (dp_actor/dp_critic).
        
        - Unpads inputs with flash-attn utilities to get packed sequences of shape (1, total_nnz).
        - Computes position_ids from the attention mask and unpads them with the same indices.
        - Returns indices and original shape for later padding back via pad_input.
        """
        # Compute position IDs from attention mask (2D)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Use flash-attn unpad to pack along non-padding positions
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # Unpad the position_ids to align rotary/positional embedding
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        batch_size, seqlen = input_ids.shape
        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

        return {
            "input_ids_rmpad": input_ids_rmpad,
            "position_ids_rmpad": position_ids_rmpad,
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "indices": indices,
            "batch_size": batch_size,
            "seqlen": seqlen,
        }
    
    def _prepare_outputs_from_packed_format(self, logits_rmpad, indices, batch_size, seqlen):
        """
        Convert packed logits back to padded tensor via flash-attn pad_input.

        Returns a dense tensor of shape (batch_size, seqlen, vocab_size).
        """
        # Lazy import to avoid dependency when not using remove_padding
        if logits_rmpad.dim() == 3:
            logits_rmpad = logits_rmpad.squeeze(0)  # (total_nnz, vocab_size)
        logits_padded = pad_input(logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
        return logits_padded

    @contextmanager
    def _fsdp_swap_for_teacher(self, student_model):
        if not self.is_fsdp_enabled:
            yield
            return
        student_device = get_device_id()
        try:
            if self.enable_hybrid_mode:
                offload_fsdp_model_to_cpu(student_model)
                if getattr(self, "_fsdp_offload_optimizer", False) and getattr(self, "optimizer", None) is not None:
                    offload_fsdp_optimizer(self.optimizer)
                aggressive_empty_cache(force_sync=True)
                load_fsdp_model_to_gpu(self.teacher_model)
                aggressive_empty_cache(force_sync=True)
            yield
        finally:
            if self.enable_hybrid_mode:
                offload_fsdp_model_to_cpu(self.teacher_model)
                aggressive_empty_cache(force_sync=True)
                load_fsdp_model_to_gpu(student_model)
                if getattr(self, "_fsdp_offload_optimizer", False) and getattr(self, "optimizer", None) is not None:
                    load_fsdp_optimizer(self.optimizer, student_device)
                aggressive_empty_cache(force_sync=True)

    @contextmanager
    def _fsdp_swap_for_rollout(self):
        is_fsdp = self.is_fsdp_enabled
        if not is_fsdp:
            if self.vllm_mode == "colocate":
                self._move_model_to_vllm()
                if self.enable_hybrid_mode:
                    aggressive_empty_cache(force_sync=True)
                    self.vllm_engine.wake_up(tags=["kv_cache"])
            try:
                yield
            finally:
                if self.vllm_mode == "colocate" and self.enable_hybrid_mode:
                    self.vllm_engine.reset_prefix_cache()
                    self.vllm_engine.sleep(level=2)
            return

        student_device = get_device_id()
        try:
            if self.enable_hybrid_mode:
                aggressive_empty_cache(force_sync=True)
                load_fsdp_model_to_gpu(self.model)
            self._move_model_to_vllm()
            if self.enable_hybrid_mode:
                offload_fsdp_model_to_cpu(self.model)
                if getattr(self, "_fsdp_offload_optimizer", False) and getattr(self, "optimizer", None) is not None:
                    offload_fsdp_optimizer(self.optimizer)
            set_expandable_segments(False)
            if self.enable_hybrid_mode:
                aggressive_empty_cache(force_sync=True)
                self.vllm_engine.wake_up(tags=["kv_cache"])
            yield
        finally:
            if self.enable_hybrid_mode:
                self.vllm_engine.reset_prefix_cache()
                self.vllm_engine.sleep(level=2)
                aggressive_empty_cache(force_sync=True)
                set_expandable_segments(True)
            load_fsdp_model_to_gpu(self.model)
            if getattr(self, "_fsdp_offload_optimizer", False) and getattr(self, "optimizer", None) is not None:
                load_fsdp_optimizer(self.optimizer, student_device)

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        device = get_device_id()

        # Decode prompts for vLLM (without special tokens - vLLM expects clean text)
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=False # Keep this commented unless specific issues arise
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm]

        # Also decode prompts WITH special tokens for ULD loss computation
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=False,
        )

        max_completion_length = self.args.max_completion_length
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0

        if self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.vllm_guided_decoding_regex)
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full((padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Mask prompt tokens in labels
        prompt_lengths = prompt_ids.shape[1]
        new_labels[:, :prompt_lengths] = -100

        # Free temporaries no longer needed before building text outputs
        del padded_completion_ids, completion_ids_tensors, padded_completion_ids_list, prompt_tokenized

        # IMPORTANT: Preserve original text for cross-tokenizer ULD loss
        # Use prompts_text_with_special (with special tokens) for ULD loss computation
        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "colocate":
                        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.enable_hybrid_mode:
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        if mode == "train":
            device = get_device_id()
            # include matched/unmatched accumulators for distributed reduction
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                    self._matched_sum,
                    self._unmatched_sum,
                    self._matched_step_eq,
                    self._unmatched_step_eq,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
                matched_sum,
                unmatched_sum,
                matched_eq,
                unmatched_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # matched/unmatched averaged over same logging window (if present)
            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def train(self):
        """
        Override train() to implement efficient batching following PPO trainer pattern exactly.
        
        New efficient pattern following PPO trainer:
        - Collect grad_accum_steps worth of micro-batches from dataloader
        - Concatenate micro-batches into full batch for rollout/teacher forwarding
        - Process teacher forward in teacher_micro_batch_size chunks
        - Student training: process each micro-batch individually with gradient accumulation
        
        NOTE: This implementation maintains exact same results as original while improving efficiency
        by reducing model wake/release cycles. On-policy sampling granularity is preserved per-micro-batch.
        """
        args = self.args
        # We'll set `model` to the accelerator-prepared (wrapped) model after preparing the dataloader,
        # mirroring Trainer/SFTTrainer behavior so FSDP utils receive an actual FSDP instance.
        model = self.model
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        
        # Initialize training state following PPO trainer pattern
        self.state.global_step = 0
        raw_train_dataloader = self.get_train_dataloader()
        steps_per_epoch = math.ceil(len(raw_train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = getattr(args, "max_train_steps", None)
        if max_train_steps is not None:
            self.state.max_steps = int(max_train_steps)
        else:
            self.state.max_steps = int(steps_per_epoch * int(args.num_train_epochs))
        self.state.epoch = 0
        
        # Add teacher_micro_batch_size parameter if not present
        if not hasattr(args, 'teacher_micro_batch_size'):
            args.teacher_micro_batch_size = args.per_device_train_batch_size
        
        # Calculate batch sizes following PPO trainer pattern
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * self.accelerator.num_processes)
        args.batch_size = int(args.local_batch_size * self.accelerator.num_processes)
        
        # Set dataloader batch size to per_device_train_batch_size (not local_batch_size)
        # This allows us to collect micro-batches and concatenate them
        self.local_dataloader_batch_size = args.per_device_train_batch_size
        
        # Create custom dataloader with per_device_batch_size
        train_dataloader = raw_train_dataloader
        
        # Prepare dataloader with accelerator
        train_dataloader = self.accelerator.prepare(train_dataloader)

        # Prepare/wrap the model with accelerator exactly like Trainer does so that under FSDP
        # `self.model`/`model` is an actual FSDP instance (not unwrapped), which is required by fsdp_utils.
        self.model = self._wrap_model(self.model, training=True, dataloader=train_dataloader)
        self.create_optimizer_and_scheduler(self.state.max_steps)
        if hasattr(self.lr_scheduler, "step"):
            # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
            if self.is_tp_enabled:
                self.optimizer = self.accelerator.prepare(self.optimizer)
            else:
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        self.model.train()
        # re-access the model after accelerator.prepare
        model = self.model
        # Call training begin callbacks
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        
        # Main training loop following PPO trainer pattern
        for epoch in range(int(args.num_train_epochs)):
            self.state.epoch = epoch
            dataloader_iter = iter(train_dataloader)
            
            while True:
                # Collect 1 batch from dataloader (after accelerator.prepare)
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                
                # Split the batch into gradient accumulation micro-batches
                micro_batches = []
                on_policy_decisions = []
                
                batch_size = len(batch["input_ids"])
                per_device_batch_size = args.per_device_train_batch_size
                
                # Verify batch size matches per_device_batch_size (after accelerator.prepare)
                # In distributed settings, each device gets per_device_batch_size samples
                assert batch_size == per_device_batch_size, f"Expected batch size {per_device_batch_size}, got {batch_size}"
                
                # In distributed settings, we need to collect grad_accum_steps batches
                # to achieve the effective local_batch_size
                collected_batches = [batch]
                collected_on_policy_decisions = [random.random() <= self.lmbda]
                
                # Collect additional batches if needed for gradient accumulation
                for _ in range(args.gradient_accumulation_steps - 1):
                    try:
                        additional_batch = next(dataloader_iter)
                        collected_batches.append(additional_batch)
                        collected_on_policy_decisions.append(random.random() <= self.lmbda)
                    except StopIteration:
                        break
                
                # Create micro-batches from collected batches
                for i, current_batch in enumerate(collected_batches):
                    # Verify prompts field exists
                    if "prompts" not in current_batch:
                        raise ValueError("Batch must contain 'prompts' field. Check dataloader/collator implementation.")
                    
                    micro_batch = {
                        "input_ids": current_batch["input_ids"],
                        "attention_mask": current_batch["attention_mask"],
                        "labels": current_batch["labels"],
                        "prompts": current_batch["prompts"],
                    }
                    
                    micro_batches.append(micro_batch)
                    # Use existing on_policy decision from collected list
                    on_policy_decisions.append(collected_on_policy_decisions[i])
                
                # Skip if we've reached max steps
                if self.state.global_step >= self.state.max_steps:
                    break
                
                # Call step begin callbacks
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                # Timing accumulators for this step
                step_start_time = time.perf_counter()
                rollout_time_s = 0.0
                teacher_time_s = 0.0
                student_time_s = 0.0
                student_fwd_time_s = 0.0
                student_bwd_time_s = 0.0
                optim_time_s = 0.0
                
                # Process rollout for on-policy micro-batches (batched once per accumulation window)
                processed_micro_batches = []
                base_batches = []
                on_policy_indices: list[int] = []
                
                for i, (batch, on_policy) in enumerate(zip(micro_batches, on_policy_decisions)):
                    base_batches.append(batch)
                    if on_policy:
                        # Placeholder; will be filled after batched generation
                        processed_micro_batches.append(None)
                        on_policy_indices.append(i)
                    else:
                        # Retain original micro-batch for off-policy
                        processed_micro_batches.append(
                            {
                                "input_ids": batch["input_ids"],
                                "attention_mask": batch["attention_mask"],
                                "labels": batch["labels"],
                                "on_policy": False,
                                "prompts": batch["prompts"],
                            }
                        )

                # Single full-batch on-policy generation via vLLM (pad prompts across micro-batches)
                if len(on_policy_indices) > 0:
                    # Collect prompts and pad to longest along seq dim
                    pad_token_id = self.processing_class.pad_token_id
                    prompts_list = [base_batches[idx]["prompts"] for idx in on_policy_indices]
                    max_prompt_len = max(p.size(1) for p in prompts_list)
                    padded_prompts_list = [
                        (p if p.size(1) == max_prompt_len else F.pad(p, (max_prompt_len - p.size(1), 0), value=pad_token_id))
                        for p in prompts_list
                    ]
                    combined_prompts = torch.cat(padded_prompts_list, dim=0)
                    gen_inputs = {"prompts": combined_prompts}

                    # Perform generation once for the combined on-policy batch
                    rollout_t0 = time.perf_counter()
                    with self._fsdp_swap_for_rollout():
                        (
                            new_input_ids_all,
                            new_attention_mask_all,
                            new_labels_all,
                            prompt_texts_all,
                            completion_texts_all,
                        ) = self._generate_on_policy_outputs_vllm(
                            gen_inputs, self.generation_config, pad_token_id
                        )
                        aggressive_empty_cache(force_sync=True)
                    rollout_time_s += time.perf_counter() - rollout_t0
                        
                    # Split outputs back per on-policy micro-batch order
                    row_cursor = 0
                    for idx in on_policy_indices:
                        mb = base_batches[idx]
                        mb_rows = mb["prompts"].size(0)
                        slice_rows = slice(row_cursor, row_cursor + mb_rows)
                        processed_micro_batches[idx] = {
                            "input_ids": new_input_ids_all[slice_rows],
                            "attention_mask": new_attention_mask_all[slice_rows],
                            "labels": new_labels_all[slice_rows],
                            "original_prompt_text": prompt_texts_all[row_cursor: row_cursor + mb_rows],
                            "original_completion_text": completion_texts_all[row_cursor: row_cursor + mb_rows],
                            "on_policy": True,
                            "prompts": mb["prompts"],
                        }
                        row_cursor += mb_rows

                    # Free combined on-policy tensors after splitting back
                    del new_input_ids_all, new_attention_mask_all, new_labels_all
                    # Also drop prompt padding temporaries
                    del padded_prompts_list, combined_prompts, gen_inputs, prompts_list

                    aggressive_empty_cache(force_sync=True)
                
                # Teacher forward once over the concatenated full batch in chunks of teacher_micro_batch_size
                # Build padded full-batch tensors
                pad_token_id = self.processing_class.pad_token_id
                seq_max = max(pb["input_ids"].size(1) for pb in processed_micro_batches)
                padded_ids = []
                padded_mask = []
                mb_sizes = []
                for pb in processed_micro_batches:
                    ids = pb["input_ids"]
                    mask = pb["attention_mask"]
                    mb_sizes.append(ids.size(0))
                    if ids.size(1) != seq_max:
                        left = seq_max - ids.size(1)
                        ids = F.pad(ids, (left, 0), value=pad_token_id)
                        mask = F.pad(mask, (left, 0), value=0)
                    padded_ids.append(ids)
                    padded_mask.append(mask)

                full_ids = torch.cat(padded_ids, dim=0)
                full_mask = torch.cat(padded_mask, dim=0)
                
                # Prepend teacher prompt prefix if provided
                teacher_prefix_len = 0
                if self.teacher_prefix_ids is not None:
                    teacher_prefix_len = self.teacher_prefix_length
                    # Expand prefix to batch size
                    batch_size = full_ids.size(0)
                    prefix_ids_batch = self.teacher_prefix_ids.expand(batch_size, -1).to(full_ids.device)
                    prefix_mask_batch = torch.ones_like(prefix_ids_batch)
                    
                    # Concatenate prefix to the beginning of inputs
                    full_ids = torch.cat([prefix_ids_batch, full_ids], dim=1)
                    full_mask = torch.cat([prefix_mask_batch, full_mask], dim=1)

                teacher_logits_list = []
                logits_chunks = []
                total_rows = full_ids.size(0)
                teacher_t0 = time.perf_counter()
                
                # Prepare packed inputs if use_remove_padding is enabled
                use_remove_padding = self.args.use_remove_padding
                if use_remove_padding:
                    packed_data = self._prepare_inputs_for_remove_padding(full_ids, full_mask)
                    teacher_input_ids = packed_data["input_ids_rmpad"]
                    teacher_position_ids = packed_data["position_ids_rmpad"]
                    teacher_attention_mask = None  # Flash attention varlen doesn't need attention mask
                    indices_teacher = packed_data["indices"]
                else:
                    teacher_input_ids = full_ids
                    teacher_attention_mask = full_mask
                    teacher_position_ids = None
                    indices_teacher = None
                
                with self._fsdp_swap_for_teacher(model):
                    self.teacher_model.eval()
                    all_teacher_logits = None
                    with torch.no_grad(), self.accelerator.autocast():
                        if use_remove_padding:
                            # Process full packed batch at once (remove_padding path)
                            teacher_batch = {
                                "input_ids": teacher_input_ids,
                                "attention_mask": teacher_attention_mask,
                                "position_ids": teacher_position_ids,
                            }
                            out = self.teacher_model(**teacher_batch)
                            all_teacher_logits = out.logits  # (1, total_nnz, vocab_size)
                        else:
                            # Process in chunks (padded path)
                            for j in range(0, total_rows, args.teacher_micro_batch_size):
                                end_idx = min(j + args.teacher_micro_batch_size, total_rows)
                                teacher_batch = {"input_ids": full_ids[j:end_idx], "attention_mask": full_mask[j:end_idx]}
                                out = self.teacher_model(**teacher_batch)
                                if all_teacher_logits is None:
                                    out_shape = list(out.logits.shape)
                                    out_shape[0] = total_rows
                                    desired_dtype = out.logits.dtype
                                    if getattr(self.accelerator.state, "mixed_precision", None) == "bf16":
                                        desired_dtype = torch.bfloat16
                                    all_teacher_logits = torch.empty(
                                        out_shape, dtype=desired_dtype, device='cpu' if self.args.teacher_logits_cpu else out.logits.device
                                    )
                                if out.logits.dtype != all_teacher_logits.dtype:
                                    out_logits = out.logits.to(all_teacher_logits.dtype)
                                else:
                                    out_logits = out.logits
                                n = out.logits.shape[0]
                                all_teacher_logits[j : j + n].copy_(out_logits)
                                del out.logits, out, out_logits
                        get_torch_device().synchronize()
                        aggressive_empty_cache(force_sync=True)

                teacher_time_s += time.perf_counter() - teacher_t0

                # Split logits back per micro-batch and trim seq dim to each micro-batch length
                if use_remove_padding:
                    # Pad back to dense tensor for the full concatenated batch
                    # Note: seq_max here includes the prefix length if prefix was added
                    actual_seq_max = seq_max + teacher_prefix_len
                    all_teacher_logits_padded = self._prepare_outputs_from_packed_format(
                        all_teacher_logits, indices_teacher, total_rows, actual_seq_max
                    )
                else:
                    all_teacher_logits_padded = all_teacher_logits
                
                # Remove teacher prefix from logits if it was added
                if teacher_prefix_len > 0:
                    # Skip the prefix tokens in the sequence dimension
                    all_teacher_logits_padded = all_teacher_logits_padded[:, teacher_prefix_len:, :]

                row_cursor = 0
                for pb in processed_micro_batches:
                    k = pb["input_ids"].size(0)
                    L = pb["input_ids"].size(1)
                    mb_logits = all_teacher_logits_padded[row_cursor:row_cursor + k, -L:, :]
                    teacher_logits_list.append(mb_logits)
                    row_cursor += k

                # Free teacher full-batch tensors
                del full_ids, full_mask
                del logits_chunks, all_teacher_logits
                del padded_ids, padded_mask, mb_sizes
                if use_remove_padding:
                    del packed_data, teacher_input_ids, teacher_position_ids, indices_teacher
                if teacher_prefix_len > 0:
                    del prefix_ids_batch, prefix_mask_batch
                aggressive_empty_cache(force_sync=True)
                
                # Student training: process each micro-batch individually with gradient accumulation
                model.train()
                
                total_loss = 0.0
                entropy_vals = []
                
                # Use accelerator.accumulate for proper gradient accumulation
                student_full_t0 = time.perf_counter()
                with self.accelerator.accumulate(model):
                    for i, (processed_batch, teacher_logits) in enumerate(zip(processed_micro_batches, teacher_logits_list)):
                        student_kwargs = {}
                        if self.crossentropy_weight > 0.0 and not processed_batch["on_policy"]:
                            student_kwargs["labels"] = processed_batch["labels"]
                        
                        fwd_t0 = time.perf_counter()
                        
                        # Prepare packed inputs for student if use_remove_padding is enabled
                        if use_remove_padding:
                            student_packed_data = self._prepare_inputs_for_remove_padding(
                                processed_batch["input_ids"], processed_batch["attention_mask"]
                            )
                            student_input_ids = student_packed_data["input_ids_rmpad"]
                            student_position_ids = student_packed_data["position_ids_rmpad"]
                            student_attention_mask = None  # Flash attention varlen doesn't need attention mask
                        else:
                            student_input_ids = processed_batch["input_ids"]
                            student_attention_mask = processed_batch["attention_mask"]
                            student_position_ids = None
                        
                        with self.accelerator.autocast():
                            student_outputs = model(
                                input_ids=student_input_ids,
                                attention_mask=student_attention_mask,
                                position_ids=student_position_ids,
                                use_cache=False,
                                **student_kwargs,
                            )
                  
                        # Unpack/pad-back student logits if using remove_padding
                        if use_remove_padding:
                            L = processed_batch["input_ids"].size(1)
                            B = processed_batch["input_ids"].size(0)
                            student_logits_padded = self._prepare_outputs_from_packed_format(
                                student_outputs.logits, student_packed_data["indices"], B, L
                            )
                        else:
                            student_logits_padded = student_outputs.logits
                        
                        # Compute entropy like SFTTrainer (before freeing logits)
                        with torch.no_grad():
                            per_token_entropy = entropy_from_logits(student_logits_padded)
                            attention_mask = processed_batch["attention_mask"]
                            entropy_val = (per_token_entropy * attention_mask).sum() / attention_mask.sum()
                            entropy_vals.append(entropy_val)
                        
                        # Compute loss using teacher logits with standard next-token shift.
                        # Rely on label masking (-100) to ignore prompt tokens.
                        shifted_student_logits = student_logits_padded[:, :-1, :]
                        shifted_teacher_logits = teacher_logits[:, :-1, :].to(shifted_student_logits.device)
                        shifted_labels = processed_batch["labels"][:, 1:]
                        
                        # Clean up packed data
                        if use_remove_padding:
                            del student_packed_data, student_input_ids, student_position_ids

                        loss = self.generalized_jsd_loss(
                            student_logits=shifted_student_logits,
                            teacher_logits=shifted_teacher_logits,
                            labels=shifted_labels,
                            beta=self.beta,
                            temperature=self.temperature,
                            top_k=self.distill_top_k,
                        )
                        student_fwd_time_s += time.perf_counter() - fwd_t0
                        # Optional debug: print alignment for the first micro-batch
                        if i == 0:
                            shifted_attention_mask = processed_batch["attention_mask"][:, 1:]
                            shifted_input_ids = processed_batch["input_ids"][:, 1:]
                            self._debug_print_alignment(
                                shifted_student_logits.detach(),
                                shifted_teacher_logits.detach(),
                                shifted_labels.detach(),
                                shifted_attention_mask.detach(),
                                shifted_input_ids.detach(),
                            )
                        
                        # Add CE loss for off-policy steps
                        if self.crossentropy_weight > 0.0 and not processed_batch["on_policy"]:
                            loss = loss + (self.crossentropy_weight * student_outputs.loss)
                        
                        # Track loss statistics (use microbatch counts for correct averaging)
                        loss_scalar = float(loss.detach())
                        if processed_batch["on_policy"]:
                            self._on_policy_loss_total += loss_scalar
                            self._on_policy_step_equiv += 1.0
                        else:
                            self._off_policy_loss_total += loss_scalar
                            self._off_policy_step_equiv += 1.0
                        
                        total_loss += loss_scalar
                        
                        # Backward pass with proper scaling for gradient accumulation
                        bwd_t0 = time.perf_counter()
                        if args.gradient_accumulation_steps and args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        self.accelerator.backward(loss)
                        student_bwd_time_s += time.perf_counter() - bwd_t0

                        # Free per-iteration intermediates
                        del shifted_student_logits, shifted_teacher_logits, shifted_labels
                        del student_outputs, loss
                        # Drop references inside containers to allow early GC
                        teacher_logits_list[i] = None
                        processed_micro_batches[i] = None
                    aggressive_empty_cache(force_sync=True)
                    # Measure overall student time (forward+loss+backward across micro-batches)
                    student_time_s = time.perf_counter() - student_full_t0
                    
                    # Optimizer and scheduler step only when gradients are synchronized
                    if self.accelerator.sync_gradients:
                        # Gradient clipping before optimizer step
                        grad_norm = None
                        max_gn = getattr(self.args, "max_grad_norm", None)
                        if max_gn is not None and max_gn > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), max_gn)
                        optim_t0 = time.perf_counter()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        optim_time_s += time.perf_counter() - optim_t0
                        self.state.global_step += 1
                        
                        # Call step end callback (only when optimizer steps)
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        
                        # Log metrics (only when optimizer steps) — follow Trainer gating
                        if self.state.global_step % self.args.logging_steps == 0:
                            # Compute loss scalar for this logging event
                            tr_loss_scalar = float(total_loss / args.gradient_accumulation_steps)
                            logs = {"loss": tr_loss_scalar}
                            # Average entropy across micro-batches and processes
                        
                            if len(entropy_vals) > 0:
                                entropy_tensor = torch.stack(entropy_vals)
                                avg_entropy = self.accelerator.gather_for_metrics(entropy_tensor).mean().item()
                                logs["entropy"] = avg_entropy
                            # grad norm if available
                            if grad_norm is not None:
                                logs["grad_norm"] = float(grad_norm)
                            # learning rate
                            if hasattr(self, "_get_learning_rate"):
                                logs["learning_rate"] = float(self._get_learning_rate())
                            else:
                                lrs = self.lr_scheduler.get_last_lr()
                                logs["learning_rate"] = float(lrs[0] if isinstance(lrs, (list, tuple)) else lrs)
                            # Add per-step timing
                            logs["time/rollout_s"] = float(rollout_time_s)
                            logs["time/teacher_s"] = float(teacher_time_s)
                            logs["time/student_s"] = float(student_time_s)
                            logs["time/student_fwd_s"] = float(student_fwd_time_s)
                            logs["time/student_bwd_s"] = float(student_bwd_time_s)
                            logs["time/optim_s"] = float(optim_time_s)
                            logs["time/step_total_s"] = float(time.perf_counter() - step_start_time)
                            self.log(logs)
                            self.store_flos()
                    
                        # Save checkpoint if needed (Trainer-style by steps)
                        if (
                            args.save_strategy in ("steps", SaveStrategy.STEPS)
                            and args.save_steps
                            and self.state.global_step % args.save_steps == 0
                        ):
                            print('Saving checkpoint at step', self.state.global_step)
                            self._save_checkpoint(model, trial=None)
                            print('Saved')
                            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                # Free per-step collections
                del teacher_logits_list, processed_micro_batches
                aggressive_empty_cache(force_sync=True)
                
                # Check if training should stop
                if self.control.should_training_stop:
                    break
            
            # Call epoch end callbacks
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            
            # Save checkpoint at epoch end if configured
            if (
                args.save_strategy in ("epoch", SaveStrategy.EPOCH)
            ):
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
            if self.state.global_step >= self.state.max_steps:
                break
            self.accelerator.wait_for_everyone()
            if self.control.should_training_stop:
                break
        

        # Save at last step
        self._save_checkpoint(model, trial=None)
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        # Call training end callbacks
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        try:
            if getattr(self, "use_vllm", False) and hasattr(self, "vllm_engine"):
                self.vllm_engine.shutdown()
        except Exception:
            pass
        try:
            self.accelerator.end_training()
        except Exception:
            pass
        return self.state
