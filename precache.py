import torch
from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig
# from stw import Stopwatch
import uuid

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
id = uuid.uuid4()
print("UUID:", id)

# with Stopwatch(verbose=True) as watch:
cfg = CacheActivationsRunnerConfig(
    device=device,
    model_name="gpt2",
    hook_name="blocks.5.hook_mlp_out",
    hook_layer=5,
    d_in=768,
    model_batch_size=8,
    training_tokens=512 * 100_000,
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    # ...
    new_cached_activations_path=f"./cache_activations/{id}",
    hf_repo_id="GulkoA/TinyStories-gpt2-cache-100k", # To push to hub
)

CacheActivationsRunner(cfg).run()
