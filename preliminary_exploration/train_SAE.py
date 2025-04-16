import torch
import os
from datasets import load_dataset
from transformer_lens import utils


from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

model_config = {
    "gpt2-small": {
        "model_name": "gpt2-small",
        "d_in": 768,
        "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "context_size": 512,
        "cached_activations_path": "GulkoA/TinyStories-gpt2-cache-100k",
        "wandb_project": "gpt2-small",
    },
    "tiny-stories-1L-21M": {
        "model_name": "tiny-stories-1L-21M",
        "d_in": 1024,
        "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "context_size": 512,
    },
    "Llama-3.2-1B": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "d_in": 2048,
        "dataset_path": "GulkoA/TinyStories-tokenized-Llama-3.2",
        "context_size": 128,
        "cached_activations_path": "GulkoA/TinyStories-Llama-3.2-1B-cache-layer-5",
        "wandb_project": "sae_llama_3_2_1B",
    },
}

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    total_training_steps = 1_000 # NOTE: let's do 1K for now
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = total_training_steps // 20 # 5% of training
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    n_checkpoints = 5
    cfg = LanguageModelSAERunnerConfig(
        **model_config["Llama-3.2-1B"],
        architecture="step",
        #activation_fn_kwargs={"kan_hidden_size": 2048 * 8, "kan_ae_type": "only_kan"},
        #model_name="meta-llama/Llama-3.2-1B",
        # model_name="tiny-stories-1L-21M",
        # cached_activations_path="GulkoA/TinyStories-Llama-3.2-1B-cache-100k", # for llama
        #cached_activations_path="GulkoA/TinyStories-gpt2-cache-100k", # for gpt2
        #cached_activations_path="./cache_activations/gpt2_tinystories", # for gpt2 locally
        use_cached_activations=True,
        #is_dataset_tokenized=True,
        hook_name=utils.get_act_name("mlp_out", 5),
        hook_layer=5,        # now we are probing layer $5 
        streaming=True,
        mse_loss_normalization="dense_batch",
        expansion_factor=8,
        b_dec_init_method="zeros",
        apply_b_dec_to_input=False,
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        lr=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
        lp_norm=1.0,
        train_batch_size_tokens=batch_size,
        n_batches_in_buffer=8,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=4,
        use_ghost_grads=False,
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        log_to_wandb=True,
        #wandb_project="gpt2-small",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=5,
        n_eval_batches=10,
        eval_batch_size_prompts=1,
        device=device,
        seed=42,
        n_checkpoints=n_checkpoints,
        checkpoint_path="checkpoints",
        dtype="float32", # type actually matters!
    )

    sparse_autoencoder = SAETrainingRunner(cfg).run()
    print("training done!")
    return sparse_autoencoder

if __name__ == "__main__":
    main()
