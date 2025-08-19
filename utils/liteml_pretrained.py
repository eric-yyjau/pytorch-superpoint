import torch
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel


def from_pretrained(
        model,
        liteml_config_yaml: str,
        state_dict: str,
        device: torch.device,
        dummy_input: torch.Tensor,
        strict: bool = True,
        map_location=None,
) -> torch.nn.Module:
    """
    Update the state of a Lite ML model from a pretrained model.
    Args:
        model: base model instance.
        liteml_config_yaml (str): Path to the Lite ML configuration YAML file.
        state_dict (str): Path to the pretrained model's state dictionary file.
        device (torch.device): The device (CPU or GPU) to which the model should be moved.
        dummy_input(torch.Tensor): a dummy input to initial the shape of state_dict buffers
        strict(bool): setting strict=False enables loading state dict with missing keys or extra keys.
        map_location: a function, torch.device, string or a dict specifying how to remap storage locations
    Returns:
        model (RetrainerMode): The updated Lite ML model.
    """
    # Load the pretrained model's state dictionary
    retrained_state_dict = torch.load(state_dict, map_location=map_location)
    # Load Lite ML configuration
    retrain_config = RetrainerConfig(liteml_config_yaml)
    # Create a Lite ML model instance
    model = RetrainerModel(model, retrain_config, pretrained=True)
    model.to(device)
    model(dummy_input.to(device))
    # Load the pretrained state dictionary into the Lite ML model
    model.load_state_dict(retrained_state_dict['model_state_dict'], strict=strict)
    # Move the model to the specified device
    model.to(device)
    torch.cuda.empty_cache()
    return model