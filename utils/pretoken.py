from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="meta-llama/Llama-3.2-1B",
    dataset_path="roneneldan/TinyStories", # this is just a tiny test dataset
    shuffle=False,
    num_proc=10, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=1024,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    split="validation",

    # uncomment to upload to huggingface
    hf_repo_id="GulkoA/TinyStories-tokenized-Llama-3.2-1024-context",

    # uncomment to save the dataset locally
    # save_path="./TinyStories-tokenized-Llama-3.2-1B"
)

dataset = PretokenizeRunner(cfg).run()
 