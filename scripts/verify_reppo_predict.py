import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Register custom resolvers
try:
    OmegaConf.register_new_resolver("pepo.constants", lambda x: x)
    OmegaConf.register_new_resolver("pepo.device", lambda x: x)
except Exception:
    pass

from pepo.utils import init_device_manager, init_hub_manager


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Initialize managers
    print("Initializing Managers...")
    init_device_manager()
    init_hub_manager()

    # Instantiate model
    print("Instantiating Model...")
    model = hydra.utils.instantiate(cfg.model, _recursive_=True)

    # Mock data for predict
    batch_size = 2
    seq_len = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    device_input_ids = [torch.randint(0, 1000, (batch_size, seq_len)).to(device)]
    device_attention_masks = [torch.ones((batch_size, seq_len)).to(device)]

    print("Loading model (skipping reward model)...")
    model.load(init_new=False)

    # Check if reward model is loaded (should NOT be)
    if model.reward_model._models is not None:
        print("FAIL: Reward model was loaded even though it shouldn't have been!")
    else:
        print("PASS: Reward model was NOT loaded by default.")

    print("Loading model (with reward model)...")
    model.unload()
    model.load(init_new=True)  # init_new=True should force reward model loading
    if model.reward_model._models is None:
        print("FAIL: Reward model was NOT loaded even though init_new=True!")
    else:
        print("PASS: Reward model was loaded with init_new=True.")

    print("Running predict...")
    try:
        log_probs = model.predict(device_input_ids, device_attention_masks)
        print(f"Predict successful! Shape: {log_probs.shape}")

        # Verify shape (B, V) where V is tokenizer vocab size
        vocab_size = model.tokenizer.vocab_size
        expected_shape = (batch_size, vocab_size)
        assert log_probs.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {log_probs.shape}"
        )

        # Verify it's log probs (should be negative)
        assert torch.all(log_probs <= 0), "Log probs should be non-positive"

    except NotImplementedError:
        print("Predict not implemented yet (Expected)")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        model.unload()


if __name__ == "__main__":
    main()
