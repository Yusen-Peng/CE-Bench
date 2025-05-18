#!/bin/bash
# width analysis script
# List of SAE patterns
sae_patterns=(
  "gemma-scope-2b-pt-res"
)

# Loop through each pattern and trainer ID
for pattern in "${sae_patterns[@]}"; do
  for layer_id in 0 5 10 15 20 25; do
    echo "Running pattern: $pattern | layer: $layer_id"
    CUDA_VISIBLE_DEVICES=0 nohup taskset -c 0-10 python ce_bench/CE_Bench.py \
      --sae_regex_pattern "$pattern" \
      --sae_block_pattern "layer_$layer_id/width_16k/average_l0_.*"
    
    echo "Sleeping for 5 seconds..."
    sleep 5
  done
done
