# script to train the pepo model, using a hydra config file,
# log the training process to wb and save the model to huggingface


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pepo import PEPOModel
from pepo.utils import constants, set_seed, setup_logging

OmegaConf.register_new_resolver(
    "pepo.constants",
    lambda name: getattr(constants, name),
)


def chat(model: PEPOModel) -> None:
    # prompts = ["Hello, how are you?", "What is the capital of France?"]
    prompts = ["Hello, how are you? I am Adam. What is your name?"]
    # model.generate_base_model(prompts, max_length=200)
    # model.generate_base_model(prompts, max_length=200, apply_chat_template=False)
    # model.generate(prompts, max_length=200)
    # model.generate(prompts, max_length=200, apply_chat_template=False)
    #
    # print("--------------------------------")
    # model.generate_base_model(prompts, max_length=1000)

    tokenizer = model.get_tokenizer()  # type: ignore[no-untyped-call]
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    tokenizer.padding_side = "left"
    tokenized = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    if model.generator is None:
        raise ValueError(
            "Generator not set on model. Cannot generate without generator."
        )
    input_ids, attention_mask = model.generator.generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
    )
    print(f"Input ids: {input_ids}")
    print(f"Attention mask: {attention_mask}")
    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Generated sequence idx=0:\n{output}")


@hydra.main(config_path="../configs", config_name="chat.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log_level_str = cfg.get("log_level", "INFO").upper()
    logger = setup_logging(level=log_level_str)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.info("PEPO Chat - Starting")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(resolved_cfg)}")

    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    device_manager = instantiate(cfg.device)

    hub_manager = instantiate(cfg.hub)

    model = instantiate(
        cfg.model,
        device_manager=device_manager,
        hub_manager=hub_manager,
    )

    generator = None
    if "generator" in cfg.model:
        generator = instantiate(cfg.model.generator)
        model.generator = generator

    chat(model)


if __name__ == "__main__":
    main()
