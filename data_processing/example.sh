python 000_autoif_pipeline_step1-5.py \
    --seed_path sample_data/seed_if_thai.txt \
    --output_path sample_data/back_trans_filter.jsonl \
    --openai_model "gpt-4o-2024-08-06" \
    --count 2 \
    --k_generations 1 \
    --n_choices 10 \
    --num_workers 16

python 001_autoif_pipeline_step6-8.py \
    --back_trans_filter sample_data/back_trans_filter.jsonl \
    --question_prompt sample_data/question_prompt.jsonl \
    --output sample_data/final_sft.jsonl \
    --openai_model "gpt-5" \
    --max_items 20 \
    --num_workers 16