#!/bin/bash
# === PARAMETERS ===
ROOT_DIR=... # Set your root directory here
MODEL=Qwen/Qwen3-4B-Instruct-2507 # Pretrained model to use
TRAIN_SET=$ROOT_DIR/dataset/verl/wangchan_nitibench/rl/train.parquet
SFT_SET=$ROOT_DIR/dataset/verl/wangchan_nitibench/sft/train.parquet
EXPERIMENT_NAME=wangchan_nitibench_pretrain
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32
SFT_BATCH_SIZE=64 # Batch size for SFT data
SFT_COEFF=0.1 # Coefficient for SFT loss
SFT_BETA=0.6 # Probability of running SFT step
LLM_JUDGE_API_KEY=...  # Set your OpenAI API key
LLM_JUDGE_BASE_URL="https://api.openai.com/v1"
LLM_JUDGE_MODEL="gpt-5-nano"
MAX_CONCURRENT=64
FORMAT_REWARD_WEIGHT=0.1
GUIDELINE_REWARD_WEIGHT=0.9
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
MAX_NUM_BATCHED_TOKENS=8192
MAX_LENGTH=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH))
MAX_TOKEN_LEN_PER_GPU=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH))
EPOCHS=2
TEMPERATURE=0.7

ENABLE_OVERLONG_BUFFER=True
OVERLONG_BUFFER_LEN=$((MAX_RESPONSE_LENGTH - 1024 * 3))
OVERLONG_PENALTY_FACTOR=1.0
cd $ROOT_DIR/verl

echo "⚡️ Training started..."
echo "Dataset: $TRAIN_SET"
echo "SFT Dataset: $SFT_SET"
echo "Model: $MODEL"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=$TRAIN_SET \
 data.val_files=$TRAIN_SET \
 +data.sft_files=$SFT_SET \
 data.train_batch_size=$TRAIN_BATCH_SIZE \
 +data.sft_batch_size=$SFT_BATCH_SIZE \
 data.max_prompt_length=$MAX_PROMPT_LENGTH \
 data.max_response_length=$MAX_RESPONSE_LENGTH \
 +data.max_length=$MAX_LENGTH \
 data.filter_overlong_prompts=True \
 data.truncation=left \
 data.return_raw_chat=True \
 +data.multiturn.raw_text=True \
 actor_rollout_ref.model.path=$MODEL \
 actor_rollout_ref.model.use_fused_kernels=True \
 actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.clip_ratio_high=0.24 \
 actor_rollout_ref.actor.clip_ratio_low=0.2 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
 +actor_rollout_ref.actor.sft_coeff=$SFT_COEFF \
 +actor_rollout_ref.actor.sft_beta=$SFT_BETA \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.ref.fsdp_config.param_offload=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.mode=async \
 actor_rollout_ref.rollout.temperature=$TEMPERATURE \
 actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.ref.strategy=fsdp2 \
 critic.strategy=fsdp2 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
 algorithm.use_kl_in_reward=False \
 reward_model.reward_manager=llm_judge \
 +reward_model.reward_kwargs.llm_judge_api_key=$LLM_JUDGE_API_KEY \
 +reward_model.reward_kwargs.llm_judge_base_url=$LLM_JUDGE_BASE_URL \
 +reward_model.reward_kwargs.llm_judge_model=$LLM_JUDGE_MODEL \
 +reward_model.reward_kwargs.max_concurrent=$MAX_CONCURRENT \
 +reward_model.reward_kwargs.format_reward_weight=$FORMAT_REWARD_WEIGHT \
 +reward_model.reward_kwargs.guideline_reward_weight=$GUIDELINE_REWARD_WEIGHT \
 +reward_model.reward_kwargs.overlong_buffer_cfg.enable=$ENABLE_OVERLONG_BUFFER \
 +reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_BUFFER_LEN \
 +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=$OVERLONG_PENALTY_FACTOR \
 +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
 +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=typhoon-s \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=200 \
 trainer.test_freq=1000000 \
 trainer.total_epochs=$EPOCHS 2>&1
