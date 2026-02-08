"""QuantizedLinear converter for ECO training.

This converter replaces nn.Linear layers with QuantizedLinear layers,
enabling FP8 storage with BF16 compute for ECO training.
"""

import torch
import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.components.quantized_linear import QuantizedLinear
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter


class QuantizedLinearConverter(ModelConverter):
    """Converter that replaces nn.Linear with QuantizedLinear.
    
    This enables:
    - FP8 E4M3 weight storage
    - BF16 computation (dequantized on-the-fly)
    - Compatibility with ECOAdamW optimizer
    - ECO error compensation
    """
    
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.config = job_config
    
    def convert(self, model: nn.Module):
        """Replace nn.Linear with QuantizedLinear."""
        replaced_count = 0
        
        for name, module in list(model.named_children()):
            # Recursively process child modules
            self.convert(module)
            
            # Replace nn.Linear with QuantizedLinear
            if isinstance(module, nn.Linear):
                # Get device and dtype from existing module
                device = module.weight.device
                dtype = module.weight.dtype
                
                # Create QuantizedLinear with same dimensions
                # Convert float32 to bfloat16 for compute
                compute_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
                
                quantized_linear = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                ).to(device)
                
                # Replace module
                setattr(model, name, quantized_linear)
                replaced_count += 1
        
        if replaced_count > 0:
            from torchtitan.tools.logging import logger
            logger.info(f"Replaced {replaced_count} nn.Linear layers with QuantizedLinear")
        
        return model


# Register the converter
register_model_converter(QuantizedLinearConverter, "quantize.linear.quantized")
