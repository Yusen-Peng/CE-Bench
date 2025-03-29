# %% [markdown]
# # Using transformer_lens to Get Activation Names from GPT-2
# 
# This notebook demonstrates how to use transformer_lens utilities to extract activation names from a GPT-2 model.

# %%
# Import necessary libraries
import transformer_lens
from transformer_lens import utils
from transformer_lens import HookedTransformer

# %%
# Load the GPT-2 model using HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")
print(f"Loaded model: {model.cfg.model_name}")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Number of heads: {model.cfg.n_heads}")
print(f"Number of features in layers:")
# Print the number of features in each layer
print(f"Embedding dimension: {model.cfg.d_model}")
print(f"MLP intermediate dimension: {model.cfg.d_mlp}")
print(f"Head dimension: {model.cfg.d_head}")
print(f"Key/Query/Value dimension: {model.cfg.d_head * model.cfg.n_heads}")


# %%
# Get all activation names from the model
activation_names = utils.get_act_name("gpt2")
print(f"Total number of activation names: {len(activation_names)}")
print("\nFirst 10 activation names:")
for name in activation_names[:10]:
    print(f"- {name}")

# %%
# Filtering activation names by type
attention_pattern_names = [name for name in activation_names if "pattern" in name]
mlp_names = [name for name in activation_names if "mlp" in name]
attn_output_names = [name for name in activation_names if "attn_out" in name]

print(f"\nAttention pattern activations: {len(attention_pattern_names)}")
print(f"MLP activations: {len(mlp_names)}")
print(f"Attention output activations: {len(attn_output_names)}")

# %%
# Example: Get activations for a specific layer
layer_num = 5  # Example: layer 5
layer_activations = [name for name in activation_names if f"blocks.{layer_num}." in name]

print(f"\nActivations for layer {layer_num}:")
for name in layer_activations:
    print(f"- {name}")

# %%
# Example: Get specific activation by using utils.get_act_name function with parameters
# Get the name for the attention output of layer 3, head 4
specific_act_name = utils.get_act_name("attn_out", 3, 4)
print(f"\nSpecific activation name: {specific_act_name}")

# Get the name for the MLP output of layer 2
mlp_out_name = utils.get_act_name("mlp_out", 2)
print(f"MLP output activation name: {mlp_out_name}")


