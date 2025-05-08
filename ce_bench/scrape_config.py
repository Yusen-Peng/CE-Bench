import re, json, yaml, os
from pathlib import Path
from typing import Dict
from huggingface_hub import list_repo_files, hf_hub_download

def pow_token_to_readable(tkn: str) -> str:
    m = re.fullmatch(r"2pow(\d+)", tkn)
    if not m:
        return tkn
    val = 2 ** int(m.group(1))
    return f"{val//1024}k" if val % 1024 == 0 else str(val)

def normalise_layer_folder(folder: str) -> str:
    m = re.fullmatch(r"resid_post_layer_(\d+)", folder)
    return f"blocks.{m.group(1)}.hook_resid_post" if m else folder

def extract_layer_idx(layer_hook: str) -> str:
    m = re.search(r"(\d+)", layer_hook)
    return m.group(1) if m else "unknown"

def sweep_meta(sweep_folder: str, model: str):
    base = sweep_folder.replace(f"{model}_", "")
    variant, rest = base.split("_width-", 1)
    width_token, date = rest.split("_date-")
    return variant, width_token, date

def build_yaml(repo_id: str,
               model_name: str = "gemma-2-2b",
               out_yaml: str = "all_sweeps.yaml"):

    repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
    cfg_files = [f for f in repo_files if f.endswith("config.json")]

    yaml_root: Dict[str, Dict] = {}

    for cfg_path in cfg_files:
        sweep_folder, layer_folder, trainer_folder, _ = Path(cfg_path).parts

        cfg_local = hf_hub_download(repo_id, cfg_path, repo_type="model")
        with open(cfg_local) as jf:
            meta = json.load(jf)

        layer_hook = normalise_layer_folder(layer_folder)
        layer_idx = extract_layer_idx(layer_hook)
        trainer_id = trainer_folder.split("_")[1]
        variant, width_token, date = sweep_meta(sweep_folder, model_name)

        width_tok = meta.get("width_token", width_token)
        width_readable = pow_token_to_readable(width_tok)

        
        top_key = f"sae_bench_{model_name}_{variant}_width-{width_token}_date-{date}"

        

        sweep_block = yaml_root.setdefault(
            top_key,
            {
                "conversion_func": "dictionary_learning_1",
                "links": {"model": f"https://huggingface.co/{model_name}"},
                "model": model_name,
                "repo_id": repo_id,
                "saes": []
            }
        )

        sweep_block["saes"].append({
            "id": f"{layer_hook}__trainer_{trainer_id}",
            "neuronpedia": (
                f"{model_name}/{layer_idx}-sae_bench-{variant}-res-"
                f"{width_readable}__trainer_{trainer_id}_step_final"
            ),
            "path": f"{sweep_folder}/{layer_folder}/trainer_{trainer_id}",
        })

    for blk in yaml_root.values():
        blk["saes"].sort(key=lambda x: x["id"])

    with open(out_yaml, "w") as f:
        yaml.dump(yaml_root, f, sort_keys=False)
    print(f"✓  wrote {out_yaml} with {len(yaml_root)} sweep blocks")

if __name__ == "__main__":
    build_yaml(
        repo_id="canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
        model_name="gemma-2-2b",
        out_yaml="ce_bench/all_sweeps.yaml",
    )
