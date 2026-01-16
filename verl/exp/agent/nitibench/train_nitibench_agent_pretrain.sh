#!/bin/bash
set -e

# bash verl/exp/agent/nitibench/run_nitibench_agent.sh
# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$REPO_DIR"

# Configuration
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
EXPERIMENT_NAME="wangchan_nitibench_agent_pretrain"

# SFT PARAMETERS
ROOT_DIR=... # TODO: Set your root directory here
SFT_SET=$ROOT_DIR/dataset/verl/wangchan_nitibench/sft/train.parquet
SFT_BATCH_SIZE=16 # Batch size for SFT data
SFT_COEFF=0.1 # Coefficient for SFT loss
SFT_BETA=0.6 # Probability of running SFT step

DATA_DIR="$ROOT_DIR/dataset/verl/nitibench-agent"
TOOL_CONFIG="exp/agent/nitibench/nitibench_tool_config.yaml"

cleanup() {
    echo "Stopping server..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID || true
    fi
    exit
}

trap cleanup EXIT INT TERM TSTP

# 1. Start the tool server
echo "Starting Nitibench Tool Server..."
python exp/agent/nitibench/nitibench_server.py > nitibench_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 30

# 2. Preprocess data
echo "Preprocessing Nitibench dataset..."
python3 exp/agent/nitibench/nitibench_agent_loop.py --local_dir $DATA_DIR

# 3. Run Training
echo "Starting PPO Training..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +data.sft_files=$SFT_SET \
    +data.sft_batch_size=$SFT_BATCH_SIZE \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.multiturn.raw_text=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.clip_ratio_high=0.24 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.sft_coeff=$SFT_COEFF \
    +actor_rollout_ref.actor.sft_beta=$SFT_BETA \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=llm_judge \
    reward_manager.name=llm_judge \
    +reward_model.reward_kwargs.llm_judge_model="gpt-5-nano" \
    trainer.log_val_generations=10 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.save_freq=200 \
    trainer.critic_warmup=0 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.project_name=typhoon-s \
    trainer.experiment_name=$EXPERIMENT_NAME
