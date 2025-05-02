import argparse
import asyncio
import gc
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal, TypeAlias

import torch
from openai import OpenAI
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import numpy as np
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
from sae_lens import SAE, HookedSAETransformer
from collections import defaultdict
import matplotlib.pyplot as plt
import json

import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.eval_output import (
    EVAL_TYPE_ID_AUTOINTERP,
    AutoInterpEvalOutput,
    AutoInterpMetricCategories,
    AutoInterpMetrics,
)
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.indexing_utils import (
    get_iw_sample_indices,
    get_k_largest_indices,
    index_with_buffer,
)
from sae_bench.sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
)

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def display_messages(messages: Messages) -> str:
    return tabulate(
        [m.values() for m in messages], tablefmt="simple_grid", maxcolwidths=[None, 120]
    )


def str_bool(b: bool) -> str:
    return "Y" if b else ""


def escape_slash(s: str) -> str:
    return s.replace("/", "_")


class Example:
    """
    Data for a single example sequence.
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(
            self.toks_are_active
        )  # this is what we predict in the scoring phase

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)  # type: ignore
            )
            .replace("�", "")
            .replace("\n", "↵")
            # .replace(">><<", "")
        )


class Examples:
    """
    Data for multiple example sequences. Includes methods for shuffling seuqences, and displaying them.
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def display(self, predictions: list[int] | None = None) -> str:
        """
        Displays the list of sequences. If `predictions` is provided, then it'll include a column for both "is_active"
        and these predictions of whether it's active. If not, then neither of those columns will be included.
        """
        return tabulate(
            [
                (
                    [max(ex.acts), ex.to_str(mark_toks=True)]
                    if predictions is None
                    else [
                        max(ex.acts),
                        str_bool(ex.is_active),
                        str_bool(i + 1 in predictions),
                        ex.to_str(mark_toks=False),
                    ]
                )
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterp:
    """
    This is a start-to-end class for generating explanations and optionally scores. It's easiest to implement it as a
    single class for the time being because there's data we'll need to fetch that'll be used in both the generation and
    scoring phases.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: HookedTransformer,
        sae: SAE,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenized_dataset = tokenized_dataset
        self.device = device
        self.api_key = api_key
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            sparsity *= cfg.total_tokens
            alive_latents = (
                torch.nonzero(sparsity > self.cfg.dead_latent_threshold)
                .squeeze(1)
                .tolist()
            )
            if len(alive_latents) < self.cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\n\n\nWARNING: Found only {len(alive_latents)} alive latents, which is less than {self.cfg.n_latents}\n\n\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=self.cfg.n_latents)
        self.n_latents = len(self.latents)

    async def run(
        self, explanations_override: dict[int, str] = {}
    ) -> dict[int, dict[str, Any]]:
        """
        Runs both generation & scoring phases. Returns a dict where keys are latent indices, and values are dicts with:

            "explanation": str, the explanation generated for this latent
            "predictions": list[int], the predicted activating indices
            "correct seqs": list[int], the true activating indices
            "score": float, the fraction of correct predictions (including positive and negative)
            "logs": str, the logs for this latent
        """
        generation_examples, scoring_examples = self.gather_data()
        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead"
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = [
                self.run_single_feature(
                    executor,
                    latent,
                    generation_examples[latent],
                    scoring_examples[latent],
                    explanations_override.get(latent, None),
                )
                for latent in latents_with_data
            ]
            results = {}
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Calling API (for gen & scoring)",
            ):
                result = await future
                if result:
                    results[result["latent"]] = result

        return results

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: str | None = None,
    ) -> dict[str, Any] | None:
        # Generation phase
        gen_prompts = self.get_generation_prompts(generation_examples)
        (explanation_raw,), logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            gen_prompts,
            self.cfg.max_tokens_in_explanation,
        )
        explanation = self.parse_explanation(explanation_raw)
        results = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
        }

        # Scoring phase
        if self.cfg.scoring:
            scoring_prompts = self.get_scoring_prompts(
                explanation=explanation_override or explanation,
                scoring_examples=scoring_examples,
            )
            (predictions_raw,), logs = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.get_api_response,
                scoring_prompts,
                self.cfg.max_tokens_in_prediction,
            )
            predictions = self.parse_predictions(predictions_raw)
            if predictions is None:
                return None
            score = self.score_predictions(predictions, scoring_examples)
            results |= {
                "predictions": predictions,
                "correct seqs": [
                    i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active
                ],
                "score": score,
                "logs": results["logs"]
                + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }

        return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int] | None:
        predictions_split = (
            predictions.strip()
            .rstrip(".")
            .replace("and", ",")
            .replace("None", "")
            .split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == []:
            return []
        if not all(pred.strip().isdigit() for pred in predictions_list):
            return None
        predictions_ints = [int(pred.strip()) for pred in predictions_list]
        return predictions_ints

    def score_predictions(
        self, predictions: list[int], scoring_examples: Examples
    ) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum(
            [c == cc for c, cc in zip(classifications, correct_classifications)]
        ) / len(classifications)

    def get_api_response(
        self, messages: Messages, max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        """Generic API usage function for OpenAI"""
        for message in messages:
            assert message.keys() == {"content", "role"}
            assert message["role"] in ["system", "user", "assistant"]

        client = OpenAI(api_key=self.api_key)

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
        )
        response = [choice.message.content.strip() for choice in result.choices]

        logs = tabulate(
            [
                m.values()
                for m in messages + [{"role": "assistant", "content": response[0]}]
            ],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )

        return response, logs

    def get_generation_prompts(self, generation_examples: Examples) -> Messages:
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=True)}"
                for i, ex in enumerate(generation_examples)
            ]
        )

        SYSTEM_PROMPT = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""
        if self.cfg.use_demos_in_explanation:
            SYSTEM_PROMPT += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            SYSTEM_PROMPT += (
                """Your response should be in the form "This neuron activates on..."."""
            )
        USER_PROMPT = (
            f"""The activating documents are given below:\n\n{examples_as_str}"""
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: Examples
    ) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            [
                f"{i + 1}. {ex.to_str(mark_toks=False)}"
                for i, ex in enumerate(scoring_examples)
            ]
        )

        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join([str(i) for i in example_response])
        SYSTEM_PROMPT = f"""We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {self.cfg.n_ex_for_scoring} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "{example_response_str}". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important."""
        USER_PROMPT = f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def gather_data(self) -> tuple[dict[int, Examples], dict[int, Examples]]:
        """
        Stores top acts / random seqs data, which is used for generation & scoring respectively.
        """
        dataset_size, seq_len = self.tokenized_dataset.shape

        acts = activation_collection.collect_sae_activations(
            self.tokenized_dataset,
            self.model,
            self.sae,
            self.cfg.llm_batch_size,
            self.sae.cfg.hook_layer,
            self.sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
            selected_latents=self.latents,
            activation_dtype=torch.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
        )

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(
            enumerate(self.latents), desc="Collecting examples for LLM judge"
        ):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            # (2/3) Get top-scoring examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = index_with_buffer(
                acts[..., i], top_indices, buffer=self.cfg.buffer
            )
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = index_with_buffer(
                self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer
            )
            iw_values = index_with_buffer(
                acts[..., i], iw_indices, buffer=self.cfg.buffer
            )

            # Get random values to use for splitting
            rand_top_ex_split_indices = torch.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[
                : self.cfg.n_top_ex_for_generation
            ]
            top_scoring_indices = rand_top_ex_split_indices[
                self.cfg.n_top_ex_for_generation :
            ]
            rand_iw_split_indices = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[
                : self.cfg.n_iw_sampled_ex_for_generation
            ]
            iw_scoring_indices = rand_iw_split_indices[
                self.cfg.n_iw_sampled_ex_for_generation :
            ]

            def create_examples(
                all_toks: Tensor, all_acts: Tensor | None = None
            ) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=acts,
                        act_threshold=act_threshold,
                        model=self.model,
                    )
                    for (toks, acts) in zip(all_toks.tolist(), all_acts.tolist())
                ]

            # Get the generation & scoring examples
            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
                + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]),
            )
            scoring_examples[latent] = Examples(
                create_examples(
                    top_toks[top_scoring_indices], top_values[top_scoring_indices]
                )
                + create_examples(
                    iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices]
                )
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples


def run_eval_single_sae(
    config: AutoInterpEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sae_sparsity: torch.Tensor | None = None,
) -> dict[str, float]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = f"{escape_slash(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = torch.load(tokens_path).to(device)
    else:
        tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
            config.dataset_name,
            config.llm_context_size,
            config.total_tokens,
            model.tokenizer,  # type: ignore
        ).to(device)
        torch.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if sae_sparsity is None:
        sae_sparsity = activation_collection.get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            sae,
            config.llm_batch_size,
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )

    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokenized_dataset,
        sparsity=sae_sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp.run())
    return results  # type: ignore


def run_eval(
    config: AutoInterpEvalConfig,
    selected_saes: list[tuple[str, str]] | list[tuple[str, SAE]],
    device: str,
    api_key: str,
    output_path: str,
    force_rerun: bool = False,
    save_logs_path: str | None = None,
    artifacts_path: str = "artifacts",
) -> dict[str, Any]:
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    sae = None
    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        sae_id, sae, sparsity = general_utils.load_and_format_sae(
            sae_release, sae_object_or_id, device
        )  # type: ignore
        sae = sae.to(device=device, dtype=llm_dtype)

        # check type
        if not isinstance(sae, SAE):
            raise TypeError(
                f"Expected SAE object, but got {type(sae)}. Please provide a valid SAE object."
            )
        else:
            print(f"Success! Loaded SAE {sae_id} from {sae_release}")
        
    
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load LLaMA tokenizer
    model_name = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    experiment_name = "Gemma-2-2b_jumprelu"
    architecture = "Gemma-2-2b" # Gemma-2-2b
    steps = "244k"
    best_model = "width-2pow16_trainer_0" # only kan

    generate_histograms = True
    log_vectors = False

    logs_folder = f"interpretability_eval/{experiment_name}"
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(f"{logs_folder}/histograms", exist_ok=True)
    os.makedirs(f"{logs_folder}/raw", exist_ok=True)

    # Load the model using HookedSAETransformer
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        **sae.cfg.model_from_pretrained_kwargs
    )
    print("Model loaded!")

    # load the contrastive dataset from huggingface
    from datasets import load_dataset
    dataset = load_dataset("GulkoA/contrastive-stories-v1", split="train")
    import re

    contrastive_scores = []
    independent_scores = []
    interpretability_scores = []
    elementwise_interpretability_scores_per_subject = defaultdict(list)
    interpretability_scores_per_subject = defaultdict(list)

    neuron_interpretability_score_subject_pairs = {}

    total_rows = len(dataset)
    for pair_index in tqdm(range(total_rows)):

        # filter out marked tokens
        text_A_original = dataset[pair_index]["story1"]
        text_B_original = dataset[pair_index]["story2"]
        ground_truth_subject = dataset[pair_index]["subject"]

        if "relevance to " in ground_truth_subject:
            continue
    
        tokenizer.add_special_tokens({"additional_special_tokens": ["<subject>", "</subject>"]})
        subject_tokens = tokenizer.convert_tokens_to_ids(["<subject>", "</subject>"])
        # print(subject_tokens)
        tokens = [tokenizer(text_A_original).to(device)["input_ids"], tokenizer(text_B_original).to(device)["input_ids"]]
        clean_tokens = [[],[]]

        # find all marked tokens and record ids of all marked tokens
        marked_tokens_indices = [[], []]
        in_subject = False
        # print(tokens_A["input_ids"])
        for story_i in range(2):
            for token_index in range(len(tokens[story_i])):
                token_id = tokens[story_i][token_index]
                # string = tokenizer.decode(token_id)
                # print(f"token index: {token_id} {string}")
                if in_subject:
                    if token_id == subject_tokens[1]:
                        in_subject = False
                    else:
                        marked_tokens_indices[story_i].append(len(clean_tokens[story_i]))
                        clean_tokens[story_i].append(token_id)
                elif token_id == subject_tokens[0]:
                    in_subject = True
                else:
                    clean_tokens[story_i].append(token_id)
            
            tqdm.write(f"story {story_i} marked tokens: {len(marked_tokens_indices[story_i])}")

        # Extract activations from the correct layer
        clean_tokens_A = torch.tensor(clean_tokens[0]).to(device)
        clean_tokens_B = torch.tensor(clean_tokens[1]).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_A) 
        hidden_states_A = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            _, cache = model.run_with_cache(clean_tokens_B)
        hidden_states_B = cache["blocks.5.hook_mlp_out"]  # now we are poking layer 5

        with torch.no_grad():
            activations_A = sae.encode(hidden_states_A)
            activations_B = sae.encode(hidden_states_B)
            # activations_A = hidden_states_A
            # activations_B = hidden_states_B

        # Convert activations to NumPy
        activations_A = activations_A.to(dtype=torch.float32).detach().cpu().numpy()
        activations_B = activations_B.to(dtype=torch.float32).detach().cpu().numpy()

        # baseline
        # activations_A = np.random.rand(*activations_A.shape)
        # activations_B = np.random.rand(*activations_B.shape)

        # keep track of I1 and I2 for independent study
        I1 = np.zeros(activations_A.shape[2])
        I1_token_num = 0
        I2 = np.zeros(activations_B.shape[2])
        I2_token_num = 0
        # compute V1 and V2 only for the marked tokens
        V1 = np.zeros(activations_A.shape[2])
        V1_token_num = 0
        V2 = np.zeros(activations_B.shape[2])
        V2_token_num = 0

        for token_index, token_id in enumerate(clean_tokens[0]):
            if token_index in marked_tokens_indices[0]:
                # add the activations of this token to V1
                # tqdm.write(f"token: {token_index} {token_id} {tokenizer.decode(token_id)}")
                V1 += activations_A[0, token_index, :]
                V1_token_num += 1
                I1 += activations_A[0, token_index, :]
                I1_token_num += 1
            else:
                V1 += activations_A[0, token_index, :] # FIXME: if average over everything
                V1_token_num += 1
                I2 += activations_A[0, token_index, :]
                I2_token_num += 1
        
        for token_index, token_id in enumerate(clean_tokens[1]):
            # NOTE: a prefix space is added to match the marked tokens
            if token_index in marked_tokens_indices[1]:
                # add the activations of this token to V1
                V2 += activations_B[0, token_index, :]
                V2_token_num += 1
                I1 += activations_B[0, token_index, :]
                I1_token_num += 1
            else:
                V2 += activations_B[0, token_index, :] # FIXME: if average over everything
                V2_token_num += 1
                I2 += activations_B[0, token_index, :]
                I2_token_num += 1

        V1 = V1 / V1_token_num if V1_token_num > 0 else V1
        V2 = V2 / V2_token_num if V2_token_num > 0 else V2
        I1 = I1 / I1_token_num if I1_token_num > 0 else I1
        I2 = I2 / I2_token_num if I2_token_num > 0 else I2


        if log_vectors:
            df = pd.DataFrame({"V1": V1, "V2": V2, "delta": V1 - V2, "abs_delta": np.abs(V1 - V2)})
            df.to_csv(f"{logs_folder}/raw/V1_V2_{pair_index}.csv", index=True)


        elementwise_contrast_distance = np.abs(V1 - V2)
        elementwise_contrastive_score = elementwise_contrast_distance - np.average(elementwise_contrast_distance)
        st_dev = np.std(elementwise_contrastive_score) if np.std(elementwise_contrastive_score) != 0 else 1
        elementwise_contrastive_score /= st_dev
        contrastive_score = np.max(elementwise_contrastive_score)

        elementwise_independence_distance = np.abs(I1 - I2)
        elementwise_independence_score = elementwise_independence_distance - np.average(elementwise_independence_distance)
        st_dev = np.std(elementwise_independence_score) if np.std(elementwise_independence_score) != 0 else 1
        elementwise_independence_score /= st_dev
        independence_score = np.max(elementwise_independence_score)

        elementwise_interpretability_distance = elementwise_contrast_distance + elementwise_independence_distance
        elementwise_interpretability_score = elementwise_interpretability_distance - np.average(elementwise_interpretability_distance)
        st_dev = np.std(elementwise_interpretability_distance) if np.std(elementwise_interpretability_distance) != 0 else 1
        elementwise_interpretability_score /= st_dev
        interpretability_score = np.max(elementwise_interpretability_score)


        if generate_histograms:
            # Create a single row of plots with better title structure
            plt.figure(figsize=(20, 5))  # Wider figure for one row
            
            # Set up title and subtitle
            plt.suptitle(f"Interpretability Analysis - {ground_truth_subject}", fontsize=14, y=0.98)
            plt.figtext(0.5, 0.91, 
                    f"Contrastive: {contrastive_score:.4f} | Independent: {independence_score:.4f} | Interpretability: {interpretability_score:.4f} | Story1: {text_A_original[:100]}...", 
                    ha="center", fontsize=12)
            
            # Scatter plot
            plt.subplot(1, 4, 1)
            scatter = plt.scatter(elementwise_contrastive_score, elementwise_independence_score, 
                        c=elementwise_interpretability_score, cmap='viridis')
            plt.colorbar(scatter, label="Interpretability Score")
            plt.xlabel("Contrastive Score")
            plt.ylabel("Independent Score")
            plt.title("Feature Space")
            
            # Histograms in a row
            plt.subplot(1, 4, 2)
            plt.hist(elementwise_contrastive_score, bins=50)
            plt.title("Contrastive Distribution")
            plt.xlabel("z-score")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 4, 3)
            plt.hist(elementwise_independence_score, bins=50)
            plt.title("Independence Distribution")
            plt.xlabel("z-score")
            
            plt.subplot(1, 4, 4)
            plt.hist(elementwise_interpretability_score, bins=50)
            plt.title("Interpretability Distribution")
            plt.xlabel("z-score")
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make room for the titles
            
            plt.savefig(f"{logs_folder}/histograms/{pair_index}.png")
            plt.close()

        """
            Responsibility Clustering
        """
        # clustering neurons into different interpreter groups based on their highest interpretability score 
        for neuron_index in range(len(elementwise_interpretability_score)):
            # check if the neuron index is already in the dictionary
            if neuron_index not in neuron_interpretability_score_subject_pairs:
                neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score[neuron_index], ground_truth_subject]
            else:
                # if it is, check if the current interpretability score is higher than the previous one
                if elementwise_interpretability_score[neuron_index] > neuron_interpretability_score_subject_pairs[neuron_index][0]:
                    neuron_interpretability_score_subject_pairs[neuron_index] = [elementwise_interpretability_score[neuron_index], ground_truth_subject]
        

        # wirte them to a file
        # with open(f"interpretability_eval/{architecture}_interpretability_scores.csv", "a") as f:
        #     f.write(f"{pair_index},{interpretability_score:4f},{responsible_neuron},{ground_truth_subject}\n")
        tqdm.write(f"pair index: {pair_index} {ground_truth_subject}:\n contrastive score: {contrastive_score:4f}\n independent score: {independence_score:4f}\n interpretability score: {interpretability_score:4f}\n")
        # print(f"interpretability score: {(contrastive_score + independence_score):4f}\n")

        # append the scores to the lists
        contrastive_scores.append(contrastive_score)
        independent_scores.append(independence_score)
        interpretability_scores.append(interpretability_score)
        elementwise_interpretability_scores_per_subject[ground_truth_subject].append(elementwise_interpretability_score)
        interpretability_scores_per_subject[ground_truth_subject].append(interpretability_score)

    # compute the average for contrastive and independent scores, and overall interpretability score
    contrastive_scores = np.array(contrastive_scores)
    independent_scores = np.array(independent_scores)
    interpretability_scores = np.array(interpretability_scores)
    contrastive_score_mean = np.mean(contrastive_scores)
    independent_score_mean = np.mean(independent_scores)
    interpretability_score_mean = np.mean(interpretability_scores)
    tqdm.write(f"Contrastive score mean: {contrastive_score_mean:4f}")
    tqdm.write(f"Independent score mean: {independent_score_mean:4f}")
    tqdm.write(f"Interpretability score mean: {interpretability_score_mean:4f}")

    interpretability_scores_per_neuron_per_subject = {}
    for subject, scores in elementwise_interpretability_scores_per_subject.items():
        all_stories = np.stack(scores, axis=0)
        interpretability_scores_per_neuron_per_subject[subject] = np.mean(all_stories, axis=0).tolist()
        # tqdm.write(f"Interpretability score mean for {subject}: {average_interpretability_scores_per_subject[subject]:4f}")
    
    average_interpretability_scores_per_subject = {}
    for subject, scores in interpretability_scores_per_subject.items():
        average_interpretability_scores_per_subject[subject] = np.mean(np.array(scores))

    # save the interpretability scores per subject to a CSV file
    df = pd.DataFrame.from_dict(interpretability_scores_per_neuron_per_subject, orient='index').T
    df.to_csv(f"{logs_folder}/interpretability_scores_per_subject.csv", index=True, header=True) # we need to keep track of the indices
    
    df = pd.DataFrame.from_dict(average_interpretability_scores_per_subject, orient='index', columns=['average_interpretability_score'])
    df.sort_values(by='average_interpretability_score', ascending=False, inplace=True)
    df.to_csv(f"{logs_folder}/average_interpretability_scores_per_subject.csv", index=True) # we need to keep track of the indices

    # responsible neurons are regrouped based on subject and written to a CSV file
    # create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(neuron_interpretability_score_subject_pairs, orient='index', columns=['interpretability_score', 'subject'])
    # reorder by subject
    df = df.sort_values(by='subject')
    # save to csv
    df.to_csv(f"{logs_folder}/responsible_neurons.csv", index=True) # we need to keep track of the indices

    results = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "steps": steps,
        "best_model": best_model,
        "model_name": model_name,
    
        "contrastive_score_mean": contrastive_score_mean,
        "independent_score_mean": independent_score_mean,
        "interpretability_score_mean": interpretability_score_mean,

        "total_rows": total_rows,
    }

    with open(f"{logs_folder}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    return {}


def create_config_and_selected_saes(
    args,
) -> tuple[AutoInterpEvalConfig, list[tuple[str, str]]]:
    config = AutoInterpEvalConfig(
        model_name=args.model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run auto interp evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/autointerp",
        help="Output folder",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="artifacts",
        help="Path to save artifacts",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )

    return parser


if __name__ == "__main__":
    """
    python evals/autointerp/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped

    python evals/autointerp/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_20/width_16k/average_l0_139" \
    --model_name gemma-2-2b

    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    api_key = "HELLO WORLD"

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        api_key,
        args.output_folder,
        args.force_rerun,
        artifacts_path=args.artifacts_path,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")
