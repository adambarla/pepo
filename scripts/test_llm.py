import random
from typing import Any

import hydra
import torch
from datasets import Dataset, load_dataset
from datasets.table import np
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(cfg: DictConfig) -> tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(cfg.llm.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.tokenizer)  # type: ignore[no-untyped-call]
    return model, tokenizer


def load_data(cfg: DictConfig) -> Any:
    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    return dataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def test_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    cfg: DictConfig,
) -> None:
    device = get_device()
    model.to(device)  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]
    for i in range(cfg.test.num_samples):
        batch = dataset[i]

        prompt = batch["prompt"]

        template = tokenizer.chat_template  # type: ignore[attr-defined]
        print(f"\nTemplate: {template}")

        inputs = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        print(f"\nInputs: {tokenizer.decode(inputs[0], skip_special_tokens=False)}")  # type: ignore[attr-defined]
        outputs = model.generate(  # type: ignore[attr-defined]
            inputs, max_new_tokens=cfg.test.max_length, do_sample=True, temperature=0.7
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)  # type: ignore[attr-defined]
        print(f"\nDecoded Output: {decoded}")
        break


@hydra.main(config_path="../configs/", config_name="test_llm.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    model, tokenizer = load_model(cfg)
    dataset = load_data(cfg)
    test_model(model, tokenizer, dataset, cfg)


if __name__ == "__main__":
    main()
