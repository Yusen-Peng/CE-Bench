#!/bin/bash

# List of SAE patterns
sae_patterns=(
  #"sae_bench_gemma-2-2b_batch_top_k_width-2pow16_date-0107"
  #"sae_bench_gemma-2-2b_gated_width-2pow16_date-0107"
  #"sae_bench_gemma-2-2b_p_anneal_width-2pow16_date-0107"
  "sae_bench_gemma-2-2b_standard_new_width-2pow16_date-0107"
  "sae_bench_gemma-2-2b_top_k_width-2pow16_date-0107"
)

# Loop through each pattern and trainer ID
for pattern in "${sae_patterns[@]}"; do
  for trainer_id in {0..5}; do
    echo "Running pattern: $pattern | trainer: $trainer_id"
    CUDA_VISIBLE_DEVICES=0 nohup taskset -c 30-39 python ce_bench/CE_Bench.py \
      --sae_regex_pattern "$pattern" \
      --sae_block_pattern "blocks.12.hook_resid_post__trainer_${trainer_id}"
    
    echo "Sleeping for 20 seconds..."
    sleep 20
  done
done
