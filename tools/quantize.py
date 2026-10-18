import torch
import torch.nn as nn
import copy
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    quantize_fx,
    QConfigMapping,
    QuantWrapper,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import sys
import os

# Must set quantized engine before any quantization ops
torch.backends.quantized.engine = 'qnnpack'  # or 'fbgemm' for x86

sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_def import get_model


def fuse_model(model):
    """Fuse Conv+ReLU and Linear+ReLU modules for better quantization accuracy.
    Note: Our model uses functional F.relu, not nn.ReLU modules, so no fusion possible.
    Returns model unchanged."""
    fused_model = copy.deepcopy(model)
    fused_model.eval()
    return fused_model


def prepare_for_quantization(model, example_inputs):
    """Prepare model with observers for calibration."""
    model.eval()
    
    # QConfig for INT8 weights (per-channel) and UINT8 activations (per-tensor)
    qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.observer.MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
    )
    
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # Prepare with FX graph mode quantization
    model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    return model_prepared


def calibrate(model, data_loader, device, num_batches=16):
    """Run calibration data through model to collect activation statistics."""
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.cpu()  # Force CPU
            model(data)
    print(f"Calibrated on {num_batches} batches")


def quantize_model(model, data_loader, device):
    """Full PTQ pipeline: fuse -> prepare -> calibrate -> convert.
    Quantization must run on CPU (MPS doesn't support quantized ops)."""
    print("Fusing modules...")
    model_fused = fuse_model(model)
    model_fused.cpu()  # Force CPU for quantization
    
    example_inputs = next(iter(data_loader))[0][:1].cpu()
    
    print("Preparing for quantization...")
    model_prepared = prepare_for_quantization(model_fused, example_inputs)
    
    print("Calibrating...")
    calibrate(model_prepared, data_loader, device)
    
    print("Converting to quantized model...")
    model_quantized = convert_fx(model_prepared)
    
    return model_quantized


def evaluate_quantized(model, data_loader, device):
    """Evaluate quantized model accuracy (on CPU)."""
    model.eval()
    model.cpu()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cpu(), target.cpu()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f"Quantized model accuracy: {accuracy:.2f}%")
    return accuracy


def save_quantized_model(model, path):
    """Save quantized model state dict."""
    torch.save(model.state_dict(), path)
    print(f"Saved quantized model to {path}")


if __name__ == "__main__":
    from train import get_data_loaders
    
    model_name = "ultra_tiny_cnn"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(f"checkpoints/best_{model_name}.pt", map_location=device))
    print(f"Loaded float model: {model_name}")
    
    _, test_loader = get_data_loaders(64)
    
    # Evaluate float model
    float_acc = evaluate_quantized(model, test_loader, device)
    
    # Quantize
    model_quantized = quantize_model(model, test_loader, device)
    
    # Evaluate quantized
    quant_acc = evaluate_quantized(model_quantized, test_loader, device)
    
    print(f"Float: {float_acc:.2f}%, Quantized: {quant_acc:.2f}%, Drop: {float_acc - quant_acc:.2f}%")
    
    save_quantized_model(model_quantized, f"checkpoints/quantized_{model_name}.pt")