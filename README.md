# KAN-LLaMA: An Interpretable Large Language Model With KAN-based Sparse Autoencoders

## CSE 5525 Final Project @ The Ohio State University

## Authors: Alex Gulko, Yusen Peng

## Project check-in task-list

### preliminary exploration

1. actually train an SAE on Llama 3.2 1B (prototype done)
    now we are able to train a SAE upon Llama 3.2 1B with the dataset "apollo-research/roneneldan-TinyStories-tokenizer-gpt2" using only 1M tokens;
    now we have "standard", "gated", "topk", "jumprelu" trained!
    
2. analyze the trained SAE for Llama 3.2 1B: L0 test (prototype done)
    now we are able to extract average L0 scores for every single batch and put them into a CSV fle (dataset is NeelNanda/pile-10k downsampling 400);
   
    A lower L0 score generally means a more interpretable model because it enforces sparsity, isolating distinct features, reducing redundancy, and improving tractability.

    now we have trained "standard", "gated", "topk", "jumprelu" L0 scores!

3. expressiveness evaluation: reconstruction test/zero ablation test (in progress)



4. expressiveness evaluation: specific capability test (in progress)


5. interpretability features: Logits Lens



6. interpretability features: SAE feature dashboard



7. interpretability evaluation: automated interpretability




## interpretability evaluation



## KAN autoencoder integration

1. KAN autoencoder code
2. 
