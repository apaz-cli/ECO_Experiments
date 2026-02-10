
Dtype Settings Reference

1. Model/Training Dtype (training.dtype)
- Purpose: Base dtype for model parameters, gradients, and activations during forward/backward
- Options: bfloat16, float32
- Default: float32
- Note: When using ECO with QuantizedLinear, weights are stored in FP8 but presented as this dtype

2. Weight Quantization Dtype (eco.quant_dtype)
- Purpose: Storage format for quantized weights
- Options: fp8 (FP8 E4M3), bf16 (bfloat16 baseline)
- Default: bf16
- Note: bf16 serves as the baseline (same code path, no quantization loss). fp8 matches "FP8 E4M3" from Table 1.

3. Activation Dtype (eco.activation_dtype) 
- Purpose: Quantization format for activations in QuantizedLinear.forward()
- Options: none, fp8_e4m3, fp8_e5m2
- Default: fp8_e4m3
- Paper: Matches "weights and activations to FP8 E4M3" from Section 4.2

4. Optimizer State Dtype (eco.optim_state_dtype)
- Purpose: Storage for Adam's first/second moments (m, v)
- Options: fp32, bf16, fp16
- Default: fp32 (paper default: 9 bytes/param = 1 FP8 + 4 m + 4 v)
- Experiment: Set to bf16 for 5 bytes/param (1 FP8 + 2 m + 2 v)

5. Optimizer Compute Dtype (eco.optim_compute_dtype)
- Purpose: Precision for computing θ̃ (pre-quantization weights) and error e = θ̃ − Q(θ̃)
- Options: fp32, bf16, fp16
- Default: fp32
- Critical: Paper recommends FP32 to avoid catastrophic cancellation when computing quantization error

Paper Configuration (Default)
training.dtype = float32
eco.quant_dtype = fp8                # 1 byte/param
eco.activation_dtype = fp8_e4m3     # Activations quantized
eco.optim_state_dtype = fp32        # 4+4 bytes/param
eco.optim_compute_dtype = fp32      # Error in FP32
Total: 9 bytes/param (vs 12 with master weights)

BF16 Experiment
eco.optim_state_dtype = bf16        # 2+2 bytes/param
Total: 5 bytes/param

Key Point: optim_compute_dtype should stay FP32 even when using BF16 states, because computing the error e = θ̃ − Q(θ̃) requires high precision to avoid losing the small quantization error signal.
