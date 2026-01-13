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

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments

from ...trainer.sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    r"""
    Configuration class for [`GKDTrainer`].

    This class includes only the parameters that are specific to GKD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_completion_length (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_init_kwargs (`dict[str, Any]]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        teacher_micro_batch_size (`int`, *optional*, defaults to `2`):
            Micro batch size for teacher model forward pass. Used to process large batches in chunks to avoid OOM.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode for student vLLM integration. Either `"server"` (connect to a running TRL vLLM server) or `"colocate"`
            (run vLLM in the same process).
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server (if `vllm_mode="server"`).
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            GPU memory utilization for the colocated student vLLM engine (if `vllm_mode="colocate"`). It is recommended
            to set this to a low value if the student and teacher models share the same GPU.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine (if `vllm_mode="colocate"`).
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding for the student model.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for the student vLLM engine. If set to `True`, the engine will enter sleep
            mode after each training step to save resources.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-7,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # GKD-specific parameters
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to "
            "`top_p` or higher are kept for generation."
        },
    )
    top_k: int = field(
        default=0,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy "
            "student-generated outputs)."
        },
    )
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence "
            "loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL "
            "Divergence."
        },
    )
    max_completion_length: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    student_model_revision: str = field(
        default="main",
        metadata={
            "help": "Revision of the student model to use. If not specified, the default revision of the model will be used."
        },
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model name or path of the teacher model. If `None`, the teacher model will be the same as the "
            "model being trained."
        },
    )
    teacher_model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={
            "help": "Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised "
            "FT on teacher-generated output)."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps."
        },
    )
    crossentropy_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the cross-entropy loss."
        },
    )
    teacher_micro_batch_size: int = field(
        default=2,
        metadata={
            "help": "Micro batch size for teacher model forward pass. Used to process large batches in chunks to avoid OOM."
        },
    )

    # vLLM parameters
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generating completions. Requires `vllm` to be installed."},
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": 'Mode for vLLM integration. "colocate" (run vLLM in the same process).'
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": 'GPU memory utilization for the colocated vLLM engine when `vllm_mode="colocate"`. Lower values reduce contention when sharing a device with the student/teacher models.'
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": 'Tensor parallel size for the colocated vLLM engine when `vllm_mode="colocate"`.'},
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex pattern used for vLLM guided decoding (optional)."},
    )
    enable_hybrid_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable hybrid mode. (offload vllm and teacher and student when possible)"
        },
    )
    use_remove_padding: bool = field(
        default=False,
        metadata={
            "help": "Whether to use remove padding (packing) for more efficient forward passes. "
            "This removes padding tokens and uses Flash Attention's variable-length sequence support."
        },
    )
    teacher_logits_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to move teacher logits to CPU."
        },
    )
    distill_top_k: int = field(
        default=0,
        metadata={
            "help": "If > 0, use top-k distillation instead of full logits distillation. "
            "Only the top-k logits from teacher and student are used for computing the loss. "
            "This reduces memory usage and can improve training by focusing on the most relevant tokens."
        },
    )
    teacher_prompt_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional constant prompt prefix to add to the teacher model input (e.g., 'You always answer in Thai'). "
            "This prefix is concatenated to the teacher's input but omitted during distillation loss computation. "
            "The student model does not see this prefix."
        },
    )
    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    log_completions_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps between logging (prompt, completion) pairs. Only used if `log_completions` is "
            "set to `True`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.")
        },
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=5, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    trl_project: str = field(
        default="smollm3",
        metadata={
            "help": "The TRL project to use for evaluation. This is used to determine the path to the evaluation script."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")

        # Validate that max_length is sufficient for max_completion_length
        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError(
                f"max_completion_length ({self.max_completion_length}) must be smaller than max_length ({self.max_length}) "
                f"to leave room for the prompt. Consider increasing max_length or reducing max_completion_length."
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
