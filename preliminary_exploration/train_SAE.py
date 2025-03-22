import torch
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    total_training_steps = 150 # small
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    # different architectures experiment
    # standard, gated, topk, jumprelu
    # kan_ae_type = "kan_relu_dense"
    cfg = LanguageModelSAERunnerConfig(
        architecture="kan",
        activation_fn_kwargs={"kan_hidden_size": 2048 * 8, "kan_ae_type": "only_kan"},
        model_name="meta-llama/Llama-3.2-1B",
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=2048,
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        is_dataset_tokenized=True,
        streaming=True,
        mse_loss_normalization=None,
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
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
        lp_norm=1.0,
        train_batch_size_tokens=512,
        context_size=512,
        n_batches_in_buffer=8,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=4,
        use_ghost_grads=False,
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        log_to_wandb=False,
        wandb_project="sae_llama_3_2_1B",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="bfloat16",
    )
    
    # look at the next cell to see some instruction for what to do while this is running.
    sparse_autoencoder = SAETrainingRunner(cfg).run()
    print("training done!")
    return sparse_autoencoder



if __name__ == "__main__":
    main()
