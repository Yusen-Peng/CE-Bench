from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="llama3.2-1B",
    dataset_path="roneneldan/TinyStories", # this is just a tiny test dataset
    shuffle=True,
    num_proc=10, # increase this number depending on how many CPUs you have

    # tweak these settings depending on the model
    context_size=128,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",

    # uncomment to upload to huggingface
    hf_repo_id="GulkoA/TinyStories-tokenized-Llama-3.2-1B"

    # uncomment to save the dataset locally
    # save_path="./c4-10k-tokenized-gpt2"
)

dataset = PretokenizeRunner(cfg).run()