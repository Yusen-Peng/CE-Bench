# CE-Bench: A Contrastive Evaluation Benchmark of LLM Interpretability with Sparse Autoencoders

Authors: Alex Gulko, Yusen Peng; Advisor: Dr. Sachin Kumar

## Motivation
The two existing interpretability evaluation methods are based on LLM prompting, which can be inherently nondeterministic, unstable, and inconsistent, despite the fact that we can run the same prompt multiple times to slightly alleviate this problem. Instead of utilizing any LLM to evaluate or simulate neuron activations, we propose a contrastive evaluation framework, CE-Bench. Its architecture is illustrated below:

![alt text](docs/CE_Bench.png)

## Contrastive Dataset
we first constructed a contrastive dataset, consisting of entries each with 3 stories and a subject. Stories are generated synthetically using GPT-4o LLM based on the subject and two prefixes with the prompts specified below.

![alt text](docs/contrastive_dataset.png)

## Contrastive Score
We hypothesize that if neurons activate more differently between tokens with contrastive meanings between two contrastive paragraphs, the latent space is more interpretable. On the left side of the architecture, to implement this, for both input paragraphs, we compute the average activations of all tokens and jointly normalize them as well. We take the absolute element-wise difference of the average activations of two contrastive paragraphs, and we assign the maximum element-wise difference as the contrastive score.

## Independent Score
We also hypothesize that if neurons activate more differently between marked tokens and unmarked tokens regardless of in which paragraph they are, the latent space is more interpretable. On the left side of the architecture, to realize this, for both paragraphs, we compute the average activations of marked tokens and the average activations of unmarked tokens, then jointly normalize them. We take the absolute element-wise difference between the activations of marked tokens and unmarked tokens, and we assign the maximum element-wise difference as the independent score.

## Interpretability Score
we also hypothesize that the simple summation of them can be a naive yet reasonable indicator of the interpretability of sparse autoencoder probing: interpretable neurons, or interpretable sparse autoencoders as a whole, should demonstrate both strong contrastivity and independence.

## Results

### Depth of Layers

basic config: gemma-scope-2b-pt-res, width 16k, JumpReLU

| layer | contrastive score | independent score | interpretability score |
| ----- | ----------------- | ----------------- | ---------------------- |
| 0 | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD |
| 9 | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD |
| 11 | TBD | TBD | TBD |
| 12 | TBD | TBD | TBD |
| 13 | TBD | TBD | TBD |
| 14 | TBD | TBD | TBD |
| 15 | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD |
| 17 | TBD | TBD | TBD |
| 18 | TBD | TBD | TBD |
| 19 | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD |
| 21 | TBD | TBD | TBD |
| 22 | TBD | TBD | TBD |
| 23 | TBD | TBD | TBD |
| 24 | TBD | TBD | TBD |
| 25 | TBD | TBD | TBD |

### Type of Layers

basic config: gemma-scope-2b-pt

| layer | contrastive score | independent score | interpretability score |
| ----- | ----------------- | ----------------- | ---------------------- |
| residual stream | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD |
| MLP-canonical | TBD | TBD | TBD |
| Attnetion | TBD | TBD | TBD |
| Attention-canonical | TBD | TBD | TBD |

### Width of Latent Space

basic config: gemma-scope-2b-pt-res

| layer | contrastive score | independent score | interpretability score |
| ----- | ----------------- | ----------------- | ---------------------- |
| 16k | TBD | TBD | TBD |
| 32k | TBD | TBD | TBD |
| 65k | TBD | TBD | TBD |
| 131k | TBD | TBD | TBD |
| 262k | TBD | TBD | TBD |
| 524k | TBD | TBD | TBD |
| 1000k | TBD | TBD | TBD |

### Architecture of Sparse Autoencoders

| SAE | contrastive score | independent score | interpretability score |
| ----- | ----------------- | ----------------- | ---------------------- |
| standard | TBD | TBD | TBD |
| top-k | TBD | TBD | TBD |
| gated | TBD | TBD | TBD |
| jumprelu | TBD | TBD | TBD |
| batch-top-k | TBD | TBD | TBD |
| p-anneal | TBD | TBD | TBD |
| matryoshka | TBD | TBD | TBD |


## command

cores python ce_bench/CE_Bench.py --sae_regex_pattern "gemma-scope-2b-pt-res" --sae_block_pattern "layer_0/width_16k/average_l0_.*"
