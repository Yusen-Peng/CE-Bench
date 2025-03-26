from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig

cfg = CacheActivationsRunnerConfig(
    model_name="meta-llama/Llama-3.2-1B",
    hook_name="blocks.0.hook_mlp_out",
    hook_layer=0,
    d_in=2048,
    model_batch_size=1024,
    training_tokens=1_000_000,
    dataset_path="GulkoA/TinyStories-tokenized-Llama-3.2",
    # ...
    new_cached_activations_path="./tiny-stories-1L-21M-cache",
    hf_repo_id="GulkoA/TinyStories-Llama-3.2-1B-cache", # To push to hub
)

CacheActivationsRunner(cfg).run()