#!/bin/bash
# width analysis script
# List of SAE patterns
sae_patterns=(
  "gemma-scope-2b-pt-att"
  "gemma-scope-2b-pt-mlp"
  "gemma-scope-2b-pt-res"
)

# Loop through each pattern and trainer ID
for pattern in "${sae_patterns[@]}"; do
    CUDA_VISIBLE_DEVICES=0 nohup taskset -c 0-10 python ce_bench/CE_Bench.py \
      --sae_regex_pattern "$pattern" \
      --sae_block_pattern "layer_12/width_16k/average_l0_.*"
    
    echo "Sleeping for 5 seconds..."
    sleep 5
  done
done
