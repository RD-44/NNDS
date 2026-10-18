import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_def import get_model


def extract_bias(bias_tensor, n):
    """Return the float bias, or None if the tensor is already integer."""
    if bias_tensor is None:
        return None
    b = bias_tensor.detach().cpu()
    if b.dtype in (torch.int32, torch.int64, torch.qint32):
        return b.numpy().astype(np.int64)
    return b.numpy().astype(np.float64)


def quantize_bias(bias_float_or_int, weight_scale, input_scale):
    """Compute int32 bias the way PyTorch does: round(b / (w_scale * in_scale))."""
    if bias_float_or_int is None:
        return None
    if np.issubdtype(np.asarray(bias_float_or_int).dtype, np.integer):
        return np.asarray(bias_float_or_int).astype(np.int32)
    return np.round(bias_float_or_int / (weight_scale.astype(np.float64) * input_scale)).astype(np.int32)


def extract_quantized_conv2d(layer, prefix):
    """Extract quantized Conv2D layer weights and quantization params."""
    # Weight: [out_ch, in_ch, kH, kW] INT8
    weight = layer.weight().int_repr().detach().cpu().numpy().astype(np.int8)
    
    # Weight scales per channel
    w_scale = layer.weight().q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    w_zp = layer.weight().q_per_channel_zero_points().detach().cpu().numpy().astype(np.int32)
    
    # Bias: qnnpack stores it as float; int32 is derived at export time via quantize_bias()
    bias_float = extract_bias(layer.bias(), weight.shape[0])
    
    # Output scale/zp (this layer's output quantization)
    out_scale = float(layer.scale)
    out_zp = int(layer.zero_point)
    
    # Input scale/zp will be set by caller (from previous layer's output)
    in_scale = 1.0
    in_zp = 0
    
    # Precompute requantization multiplier (Q31) - will be corrected after in_scale is known
    combined_scale = (w_scale * in_scale) / out_scale
    output_mult = np.round(combined_scale * (1 << 31)).astype(np.int32)
    output_shift = np.full_like(output_mult, 31, dtype=np.int32)
    
    return {
        'weight': weight,
        'bias_float': bias_float,
        'bias': np.zeros(weight.shape[0], dtype=np.int32),
        'weight_scale': w_scale,
        'weight_zp': w_zp,
        'input_scale': np.array([in_scale], dtype=np.float32),
        'input_zp': np.array([in_zp], dtype=np.int32),
        'output_scale': np.array([out_scale], dtype=np.float32),
        'output_zp': np.array([out_zp], dtype=np.int32),
        'output_mult': output_mult,
        'output_shift': output_shift,
        'shape': {
            'out_channels': weight.shape[0],
            'in_channels': weight.shape[1],
            'kernel_h': weight.shape[2],
            'kernel_w': weight.shape[3],
            'stride': layer.stride[0] if hasattr(layer, 'stride') else 1,
            'padding': layer.padding[0] if hasattr(layer, 'padding') else 0,
        }
    }


def extract_quantized_linear(layer, prefix):
    """Extract quantized Linear layer weights and quantization params."""
    weight = layer.weight().int_repr().detach().cpu().numpy().astype(np.int8)  # [out, in]
    
    # Weight scale (per-channel or per-tensor)
    qscheme = layer.weight().qscheme()
    if qscheme == torch.per_channel_affine or qscheme == torch.per_channel_symmetric:
        w_scale = layer.weight().q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
        w_zp = layer.weight().q_per_channel_zero_points().detach().cpu().numpy().astype(np.int32)
    else:
        w_scale = np.array([float(layer.weight().q_scale())], dtype=np.float32)
        w_zp = np.array([int(layer.weight().q_zero_point())], dtype=np.int32)
    
    bias_float = extract_bias(layer.bias(), weight.shape[0])
    
    # Output scale/zp
    out_scale = float(layer.scale)
    out_zp = int(layer.zero_point)
    
    # Input scale/zp will be set by caller
    in_scale = 1.0
    in_zp = 0
    
    combined_scale = (w_scale * in_scale) / out_scale
    output_mult = np.round(combined_scale * (1 << 31)).astype(np.int32)
    output_shift = np.full_like(output_mult, 31, dtype=np.int32)
    
    return {
        'weight': weight,
        'bias_float': bias_float,
        'bias': np.zeros(weight.shape[0], dtype=np.int32),
        'weight_scale': w_scale,
        'weight_zp': w_zp,
        'input_scale': np.array([in_scale], dtype=np.float32),
        'input_zp': np.array([in_zp], dtype=np.int32),
        'output_scale': np.array([out_scale], dtype=np.float32),
        'output_zp': np.array([out_zp], dtype=np.int32),
        'output_mult': output_mult,
        'output_shift': output_shift,
        'shape': {
            'out_features': weight.shape[0],
            'in_features': weight.shape[1],
        }
    }


def extract_maxpool(layer):
    """Extract MaxPool params."""
    return {
        'kernel_size': layer.kernel_size,
        'stride': layer.stride,
        'padding': layer.padding,
    }


def generate_c_array(name, data, dtype='int8_t'):
    """Generate C array declaration."""
    flat = data.flatten()
    lines = [f"const {dtype} {name}[{len(flat)}] = {{"]
    for i, val in enumerate(flat):
        if i % 12 == 0:
            lines.append("    ")
        lines[-1] += f"{val}, "
        if i % 12 == 11:
            lines.append("")
    lines.append("};")
    return "\n".join(lines)


def find_input_qparams(model):
    """Find the graph's input quantize node and return (scale, zero_point)."""
    for node in model.graph.nodes:
        if node.op == 'call_function' and 'quantize_per_tensor' in str(node.target):
            scale_node, zp_node = node.args[1], node.args[2]
            return float(getattr(model, scale_node.target)), int(getattr(model, zp_node.target))
    raise RuntimeError('no quantize_per_tensor node found in graph')


def generate_c_header(model, model_name, output_path, input_scale, input_zp):
    """Generate complete model.h for DS."""
    lines = [
        f"// Auto-generated from {model_name}",
        f"// DO NOT EDIT MANUALLY",
        "",
        "#ifndef MODEL_H",
        "#define MODEL_H",
        "",
        "#include <stdint.h>",
        "",
        "// Fixed-point helpers",
        "#define Q31_SHIFT 31",
        "#define SATURATE_U8(x) ((x) > 255 ? 255 : ((x) < 0 ? 0 : (x)))",
        "",
    ]
    
    layer_data = []
    
    # Extract layers first
    conv1 = extract_quantized_conv2d(model.conv1, "conv1")
    layer_data.append(('conv1', conv1))
    
    pool1 = extract_maxpool(model.pool1)
    layer_data.append(('pool1', pool1))
    
    conv2 = extract_quantized_conv2d(model.conv2, "conv2")
    layer_data.append(('conv2', conv2))
    
    pool2 = extract_maxpool(model.pool2)
    layer_data.append(('pool2', pool2))
    
    fc1 = extract_quantized_linear(model.fc1, "fc1")
    layer_data.append(('fc1', fc1))
    
    fc2 = extract_quantized_linear(model.fc2, "fc2")
    layer_data.append(('fc2', fc2))
    
    # Chain input/output scales: each layer's input = previous layer's output.
    # The first layer's input is the graph input quantizer (observed during calibration),
    # NOT a hardcoded 1/255 - hardcoding it silently breaks the requantization multipliers.
    prev_out_scale = input_scale
    prev_out_zp = input_zp
    
    for name, data in layer_data:
        if name.startswith('conv') or name.startswith('fc'):
            data['input_scale'] = np.array([prev_out_scale], dtype=np.float32)
            data['input_zp'] = np.array([prev_out_zp], dtype=np.int32)
            
            # Recompute output_mult with correct input_scale
            w_scale = data['weight_scale']
            out_scale = data['output_scale'][0]
            combined_scale = (w_scale * prev_out_scale) / out_scale
            data['output_mult'] = np.round(combined_scale * (1 << 31)).astype(np.int32)
            
            # Quantize bias like PyTorch does: round(b_float / (w_scale * in_scale))
            bias = quantize_bias(data['bias_float'], w_scale, prev_out_scale)
            if bias is not None:
                data['bias'] = bias
            
            prev_out_scale = data['output_scale'][0]
            prev_out_zp = data['output_zp'][0]
    
    # Input: 1x28x28 UINT8 (0-255) - pixel values map 1:1 onto the model's input quantizer
    lines.append("// Input quantization (touchscreen preprocessing feeds raw pixel values)")
    lines.append(f"const float model_input_scale = {input_scale}f;")
    lines.append(f"const int32_t model_input_zp = {input_zp};")
    lines.append("")
    
    # Generate arrays for each layer
    for name, data in layer_data:
        if name.startswith('conv'):
            lines.append(f"// {name}: Conv2D {data['shape']['in_channels']}x{data['shape']['kernel_h']}x{data['shape']['kernel_w']} -> {data['shape']['out_channels']}")
            lines.append(generate_c_array(f"{name}_weight", data['weight'], 'int8_t'))
            lines.append(generate_c_array(f"{name}_bias", data['bias'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_weight_scale", data['weight_scale'], 'float'))
            lines.append(generate_c_array(f"{name}_weight_zp", data['weight_zp'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_output_mult", data['output_mult'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_output_shift", data['output_shift'], 'int32_t'))
            lines.append(f"const float {name}_input_scale = {data['input_scale'][0]}f;")
            lines.append(f"const int32_t {name}_input_zp = {data['input_zp'][0]};")
            lines.append(f"const float {name}_output_scale = {data['output_scale'][0]}f;")
            lines.append(f"const int32_t {name}_output_zp = {data['output_zp'][0]};")
            lines.append("")
        
        elif name.startswith('fc'):
            lines.append(f"// {name}: Linear {data['shape']['in_features']} -> {data['shape']['out_features']}")
            lines.append(generate_c_array(f"{name}_weight", data['weight'], 'int8_t'))
            lines.append(generate_c_array(f"{name}_bias", data['bias'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_weight_scale", data['weight_scale'], 'float'))
            lines.append(generate_c_array(f"{name}_weight_zp", data['weight_zp'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_output_mult", data['output_mult'], 'int32_t'))
            lines.append(generate_c_array(f"{name}_output_shift", data['output_shift'], 'int32_t'))
            lines.append(f"const float {name}_input_scale = {data['input_scale'][0]}f;")
            lines.append(f"const int32_t {name}_input_zp = {data['input_zp'][0]};")
            lines.append(f"const float {name}_output_scale = {data['output_scale'][0]}f;")
            lines.append(f"const int32_t {name}_output_zp = {data['output_zp'][0]};")
            lines.append("")
        
        elif name.startswith('pool'):
            lines.append(f"// {name}: MaxPool kernel={data['kernel_size']} stride={data['stride']}")
            lines.append("")
    
    # Tensor arena size estimate
    lines.append("// Tensor arena size (bytes) - adjust based on actual usage")
    lines.append("#define TENSOR_ARENA_SIZE (24 * 1024)")
    lines.append("")
    
    lines.append("#endif // MODEL_H")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Generated {output_path}")


def main():
    model_name = "ultra_tiny_cnn"
    device = torch.device("cpu")
    
    # Load the float model and re-quantize to get quantized structure
    float_model = get_model(model_name).to(device)
    float_model.load_state_dict(torch.load(f"checkpoints/best_{model_name}.pt", map_location=device, weights_only=True))
    float_model.eval()
    
    # Import quantization functions
    from quantize import quantize_model
    from train import get_data_loaders
    
    _, test_loader = get_data_loaders(64)
    
    # Re-quantize to get the quantized model structure
    model = quantize_model(float_model, test_loader, device)
    model.eval()
    
    input_scale, input_zp = find_input_qparams(model)
    print(f"Input quantizer: scale={input_scale}, zp={input_zp}")
    
    import os
    output_path = os.path.join(os.path.dirname(__file__), "..", "source", "model.h")
    generate_c_header(model, model_name, output_path, input_scale, input_zp)


if __name__ == "__main__":
    main()