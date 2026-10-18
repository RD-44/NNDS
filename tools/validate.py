import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_def import get_model
from train import get_data_loaders


def get_layer_outputs(model, data, layer_names):
    """Hook to capture intermediate layer outputs."""
    outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = output.detach()
        return hook
    
    hooks = []
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        model(data)
    
    for h in hooks:
        h.remove()
    
    return outputs


def compare_models(float_model, quant_model, data_loader, device, num_samples=100):
    """Compare float vs quantized layer by layer."""
    float_model.eval()
    quant_model.eval()
    
    layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'fc2']
    
    max_diffs = {name: 0.0 for name in layer_names}
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            if total_samples >= num_samples:
                break
            data = data.to(device)
            
            float_out = get_layer_outputs(float_model, data, layer_names)
            quant_out = get_layer_outputs(quant_model, data, layer_names)
            
            for name in layer_names:
                f = float_out[name].cpu().numpy()
                q = quant_out[name].dequantize().cpu().numpy() if hasattr(quant_out[name], 'dequantize') else quant_out[name].cpu().numpy()
                diff = np.abs(f - q).max()
                max_diffs[name] = max(max_diffs[name], diff)
            
            total_samples += data.size(0)
    
    print("\nLayer-wise max absolute difference (float vs quantized):")
    for name, diff in max_diffs.items():
        print(f"  {name}: {diff:.6f}")
    
    return max_diffs


def export_test_vectors(float_model, quant_model, data_loader, device, num_samples=100, output_path="test_vectors.npz"):
    """Export input/output pairs for C-side validation."""
    float_model.eval()
    quant_model.eval()
    
    inputs = []
    float_logits = []
    quant_logits = []
    labels_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            if len(inputs) >= num_samples:
                break
            data = data.to(device)
            
            f_out = float_model(data)
            q_out = quant_model(data)
            
            # Dequantize quantized output if needed
            if hasattr(q_out, 'dequantize'):
                q_out = q_out.dequantize()
            
            inputs.append(data.cpu().numpy())
            float_logits.append(f_out.cpu().numpy())
            quant_logits.append(q_out.cpu().numpy())
            labels_list.append(target.numpy())
    
    inputs = np.concatenate(inputs)[:num_samples]
    float_logits = np.concatenate(float_logits)[:num_samples]
    quant_logits = np.concatenate(quant_logits)[:num_samples]
    labels_list = np.concatenate(labels_list)[:num_samples]
    
    # Quantize inputs to UINT8 (matching DS preprocessing)
    # Input is normalized: (x - 0.1307) / 0.3081
    # To get UINT8: x_uint8 = round((x_float * 0.3081 + 0.1307) * 255)
    inputs_uint8 = np.round((inputs * 0.3081 + 0.1307) * 255).astype(np.uint8)
    
    np.savez(output_path,
             inputs=inputs_uint8,
             float_logits=float_logits,
             quant_logits=quant_logits,
             labels=labels_list)
    
    print(f"\nExported {num_samples} test vectors to {output_path}")
    print(f"  inputs: {inputs_uint8.shape} (UINT8)")
    print(f"  float_logits: {float_logits.shape}")
    print(f"  quant_logits: {quant_logits.shape}")
    print(f"  labels: {labels_list.shape}")


def main():
    model_name = "ultra_tiny_cnn"
    device = torch.device("cpu")
    
    float_model = get_model(model_name).to(device)
    float_model.load_state_dict(torch.load(f"checkpoints/best_{model_name}.pt", map_location=device, weights_only=True))
    float_model.eval()
    
    # Re-quantize to get quantized model structure
    from quantize import quantize_model
    _, test_loader = get_data_loaders(64)
    
    quant_model = quantize_model(float_model, test_loader, device)
    quant_model.eval()
    
    print("Comparing float vs quantized layer outputs...")
    compare_models(float_model, quant_model, test_loader, device)
    
    print("\nExporting test vectors...")
    export_test_vectors(float_model, quant_model, test_loader, device, 100)


if __name__ == "__main__":
    main()