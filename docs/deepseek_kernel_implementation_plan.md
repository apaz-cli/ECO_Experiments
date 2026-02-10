# ECO Low-Precision Kernel Implementation Plan

## Executive Summary

This document outlines the implementation plan for low-precision kernels to enable true FP8 and NVFP4 training in the ECO (Error-Compensating Optimizer) framework. Currently, ECO uses `QuantizedTensor` for FP8 storage but dequantizes to BF16/FP16 for all computations via `__torch_dispatch__`, missing the performance benefits of low-precision hardware. This plan details the kernel requirements, implementation strategy, and priority order to achieve end-to-end low-precision training with error compensation.

## 1. Current State Analysis

### 1.1 QuantizedTensor Architecture
- **Storage**: FP8 E4M3/E5M2 with per-tensor/per-channel scaling
- **Compute Path**: `__torch_dispatch__` → dequantize to compute dtype (BF16/FP16) → compute → (re-quantize)
- **Limitation**: No actual low-precision kernels; all ops run in BF16/FP16
- **Models Supported**: Llama3/4, Qwen3, MoE variants via `QuantizedLinear`

### 1.2 QuantizedLinear Layer
```python
# Current implementation (dequantizes before matmul):
weight_fp = self._weight_data.to(self.compute_dtype) * self._weight_scale
output = F.linear(input, weight_fp, self.bias)  # BF16/FP16 matmul
```
- **Missing**: True FP8 matmul kernel integration
- **Activation Quantization**: Not implemented (paper mentions FP8 activations)

### 1.3 ECOAdamW Optimizer
- **Algorithm 3**: Error injection between Adam update and requantization
- **Compute Dtype**: FP32 for error computation (prevents catastrophic cancellation)
- **State Storage**: FP32 momentum/variance, FP8 weights
- **Limitation**: Requires dequantization for each optimizer step

### 1.4 Existing Kernel Infrastructure
- **`eco_kernels.py`**: STUB only, no real kernels
- **MoE kernels**: `_fill_indices_kernel` (data movement, not compute)
- **TorchAO Integration**: Available but not used in ECO path
- **PyTorch Native**: `torch._grouped_mm` (BF16), `torch._scaled_mm` (unused)

## 2. Kernel Requirements

### 2.1 Core Compute Operations

| Operation | Description | Current Precision | Target Precision |
|-----------|-------------|-------------------|------------------|
| **Linear Matmul** | `y = x @ W^T` | BF16/FP16 | FP8 (E4M3/E5M2), NVFP4 (E2M1) |
| **MoE Grouped GEMM** | Expert computation across tokens | BF16 | FP8, NVFP4 with scaling |
| **Attention QKV** | Projections for Q, K, V | BF16/FP16 | FP8/NVFP4 |
| **Attention Output** | Output projection | BF16/FP16 | FP8/NVFP4 |
| **FeedForward** | SwiGLU/SiGLU with linear layers | BF16/FP16 | FP8/NVFP4 |
| **Elementwise Ops** | Silu, multiplication, addition | BF16/FP16 | Can remain BF16/FP16 |

### 2.2 Quantization Operations

| Operation | Description | Kernel Type |
|-----------|-------------|-------------|
| **Dynamic Quantization** | Compute scale, quantize to FP8/NVFP4 | CUDA/Triton |
| **Dequantization** | Convert low-precision → high-precision | CUDA/Triton |
| **Stochastic Rounding** | Add noise in [-0.5 ulp, +0.5 ulp] before rounding | CUDA/Triton |
| **Scale Computation** | Per-tensor/per-channel `amax(abs(x))` | CUDA/Triton |

### 2.3 Optimizer Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| **Fused ECOAdamW** | Dequantize → Adam update → requantize → error injection | High |
| **Low-Precision Update** | Apply gradients to FP8/NVFP4 weights directly | Medium |
| **Momentum Injection** | Inject quantization error into momentum buffer | Low |

### 2.4 Distributed Operations

| Operation | Description | Communication Dtype |
|-----------|-------------|---------------------|
| **FSDP All-Gather** | Sharded weight gathering | FP8/NVFP4 + scales |
| **Tensor Parallel** | Row/column-wise sharded matmul | FP8/NVFP4 + scales |
| **Expert Parallel** | MoE expert distribution | FP8/NVFP4 + scales |

## 3. Implementation Strategy

### 3.1 Phase 1: FP8 Integration with Existing Infrastructure

#### 3.1.1 Leverage TorchAO for FP8 Matmul
- **Approach**: Integrate `torchao.float8.Float8Linear` with ECO error injection
- **Benefits**: Production-ready FP8 kernels via `torch._scaled_mm`
- **Changes Required**:
  1. Modify `QuantizedLinear.forward()` to use `torch._scaled_mm`
  2. Maintain ECO error injection compatibility
  3. Support both tensorwise and rowwise scaling

#### 3.1.2 Extend MoE Training with FP8
- **Approach**: Use `torchao.prototype.moe_training` for grouped GEMM
- **Benefits**: Existing FP8 grouped GEMM kernels
- **Changes Required**:
  1. Integrate `MoETrainingConfig` with ECO optimizer
  2. Ensure token group alignment (multiples of 16 for FP8)
  3. Handle scale communication across experts

#### 3.1.3 Activation Quantization
- **Approach**: Dynamic FP8 quantization in `QuantizedLinear.forward()`
- **Benefits**: Matches paper's "weights and activations to FP8 E4M3"
- **Implementation**:
  ```python
  if self.activation_quant_dtype:
      input_fp8, scale_input = dynamic_quantize_fp8(input)
      output_fp8 = torch._scaled_mm(input_fp8, weight_fp8, 
                                   scale_a=scale_input, scale_b=weight_scale)
      output = dequantize_fp8(output_fp8, scale_input * weight_scale)
  ```

### 3.2 Phase 2: Custom Kernel Development

#### 3.2.1 Fused ECOAdamW Kernel
- **Goal**: Single kernel for Algorithm 3 steps
- **Operations**:
  1. Load FP8 weight, dequantize locally
  2. Adam update with BF16 gradient
  3. Quantize updated weight to FP8
  4. Compute error e = θ̃ − θ̂
  5. Inject error into momentum buffer
  6. Store FP8 weight
- **Implementation**: Triton kernel with block-wise processing

#### 3.2.2 NVFP4 Matmul Kernels
- **Goal**: 4-bit matrix multiplication for NVFP4 format
- **Format**: E2M1 (2 exponent, 1 mantissa bit), packed into uint8 (2 values/byte)
- **Approach**: Use qutlass library or implement custom CUDA kernels
- **Integration**: Extend `QuantizedTensor` to support NVFP4 storage format

#### 3.2.3 Low-Precision Grouped GEMM for MoE
- **Goal**: FP8/NVFP4 grouped matrix multiplication for experts
- **Challenge**: Variable token counts per expert, alignment requirements
- **Implementation**: Extend `torch._grouped_mm` with low-precision support or custom kernel

### 3.3 Phase 3: Full Integration and Optimization

#### 3.3.1 Kernel Dispatch Architecture
- **Goal**: Transparent low-precision execution via `QuantizedTensor.__torch_dispatch__`
- **Design**:
  ```python
  def __torch_dispatch__(self, func, types, args, kwargs):
      if func == torch.ops.aten.mm or func == torch.ops.aten.linear:
          return low_precision_matmul(args[0], args[1], self._scale)
      else:
          return super().__torch_dispatch__(func, types, args, kwargs)
  ```

#### 3.3.2 Compilation Integration
- **Goal**: `torch.compile` compatibility for kernel fusion
- **Approach**: Register custom ops with PyTorch inductor
- **Benefits**: Automatic kernel fusion, memory optimization

#### 3.3.3 Distributed Training Integration
- **Goal**: Low-precision communication for FSDP, TP, EP
- **Implementation**:
  - FP8 all-gather for FSDP (partially in torchao)
  - Scale synchronization across devices
  - Low-precision gradient reduction

## 4. Dependencies and Prerequisites

### 4.1 Required Libraries

| Library | Purpose | Status |
|---------|---------|--------|
| **TorchAO** | Production FP8/MXFP8 kernels | Installed but unused in ECO |
| **Triton** | Custom kernel development | Installed |
| **Qutlass** | NVFP4 matmul kernels | Not installed (requires CUDA 12.x, SM 89+) |
| **PyTorch** | Base framework with `_scaled_mm` | Installed |

### 4.2 Hardware Requirements

| Hardware | FP8 Support | NVFP4 Support |
|----------|-------------|---------------|
| **H100** | Native (via tensor cores) | Via qutlass/CUDA kernels |
| **B200** | Native (via tensor cores) | Native NVFP4 tensor cores |
| **A100** | Emulation only | Emulation only |

### 4.3 Software Dependencies

| Component | Version Requirement | Purpose |
|-----------|---------------------|---------|
| **CUDA** | 12.x+ | Kernel compilation |
| **PyTorch** | 2.5.0+ | FP8 dtype support |
| **torchao** | 0.9.0+ | FP8 training infrastructure |
| **triton** | 3.0.0+ | Custom kernel development |

## 5. Priority and Timeline

### 5.1 Priority Order

1. **P0: FP8 Matmul Integration** - Leverage `torch._scaled_mm` for immediate benefit
2. **P1: Fused ECOAdamW Kernel** - Eliminate dequantization overhead in optimizer
3. **P2: NVFP4 Matmul Kernels** - Enable 4-bit training experiments
4. **P3: MoE Low-Precision GEMM** - Critical for MoE model scaling
5. **P4: Activation Quantization** - Full pipeline optimization
6. **P5: Distributed Low-Precision** - Scaling to multi-GPU

### 5.2 Implementation Timeline

#### Week 1-2: FP8 Foundation
- Integrate `torch._scaled_mm` into `QuantizedLinear`
- Add activation quantization support
- Validate correctness vs. dequantization path

#### Week 3-4: Fused Optimizer Kernel
- Implement fused ECOAdamW Triton kernel
- Benchmark vs. pure PyTorch implementation
- Integrate with existing `ECOAdamW` class

#### Week 5-6: NVFP4 Support
- Extend `QuantizedTensor` with NVFP4 storage format
- Integrate qutlass or implement basic NVFP4 matmul
- Add NVFP4 quantization/dequantization kernels

#### Week 7-8: MoE Optimization
- Implement low-precision grouped GEMM for MoE
- Integrate with existing MoE architecture
- Validate convergence vs. BF16 baseline

#### Week 9-10: Integration and Validation
- Full model training validation
- Performance benchmarking
- Documentation and examples

## 6. Testing and Validation

### 6.1 Correctness Tests

| Test Category | Description | Validation Method |
|--------------|-------------|-------------------|
| **Numerical Equivalence** | Low-precision vs. BF16 reference | Relative error < 1e-3 |
| **ECO Error Injection** | Error compensation correctness | Compare with master weights baseline |
| **Gradient Matching** | Backward pass correctness | Gradient comparison with autograd |
| **Convergence Validation** | End-to-end training | Match validation loss curves |

### 6.2 Performance Benchmarks

| Benchmark | Metric | Target Improvement |
|-----------|--------|-------------------|
| **Matmul Throughput** | TFLOPS | 2x over BF16 (FP8), 4x over BF16 (NVFP4) |
| **End-to-End Training** | Tokens/sec | 1.5x over BF16 baseline |
| **Memory Usage** | Bytes/parameter | 9B (FP8) → 8.5B (NVFP4) with FP32 optimizer |
| **Optimizer Overhead** | Step time reduction | 30% reduction with fused kernel |

### 6.3 Model Validation

| Model | Size | Test Scenario |
|-------|------|---------------|
| **Llama3** | 8B | FP8 training, compare validation loss |
| **Llama4 MoE** | 17B×16E | MoE FP8 training, expert computation |
| **DeepSeek-V3** | 16B | NVFP4 fine-tuning, compare accuracy |

## 7. Risk Mitigation

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Numerical instability** | Medium | High | Maintain FP32 error computation, gradual precision reduction |
| **Kernel correctness** | High | High | Extensive unit tests, compare with reference implementation |
| **Performance regressions** | Medium | Medium | A/B testing, gradual rollout |
| **Hardware compatibility** | Low | High | Feature detection, fallback to emulation |

### 7.2 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **PyTorch version compatibility** | Medium | Medium | Version pinning, conditional imports |
| **Distributed training issues** | High | High | Incremental integration, thorough testing |
| **Model conversion complexity** | Medium | Medium | Automated conversion tools, backward compatibility |

### 7.3 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **qutlass integration delays** | High | Medium | Alternative NVFP4 implementation path |
| **Kernel optimization time** | High | Medium | Prioritize correctness first, optimize later |
| **Testing complexity** | Medium | Medium | Automated testing pipeline, CI integration |

## 8. Success Criteria

### 8.1 Technical Success Criteria
1. **FP8 Training**: End-to-end FP8 training matches BF16 baseline accuracy (±0.5% validation loss)
2. **NVFP4 Training**: NVFP4 training converges within 5% of FP8 baseline
3. **Performance**: 1.5× training throughput improvement for FP8, 2× for NVFP4
4. **Memory**: Achieve 8.5B/param with NVFP4 + FP32 optimizer states

### 8.2 Implementation Success Criteria
1. **Integration**: All kernels integrate seamlessly with existing ECO codebase
2. **Usability**: Same CLI flags and configuration as current ECO implementation
3. **Compatibility**: Support for all model architectures (Llama, Qwen, MoE variants)
4. **Documentation**: Comprehensive examples and benchmarks

### 8.3 Research Success Criteria
1. **Paper Reproduction**: Match or exceed paper results with true low-precision kernels
2. **New Capabilities**: Enable NVFP4 training not covered in original paper
3. **Open Source**: Clean, maintainable code suitable for upstream contribution

## 9. Appendix

### 9.1 Kernel API Design

```python
# Proposed kernel interface
def low_precision_matmul(
    a: torch.Tensor,  # FP8/NVFP4
    b: torch.Tensor,  # FP8/NVFP4  
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Low-precision matrix multiplication with scaling."""
    pass

def fused_ecoadamw_step(
    weight: torch.Tensor,  # FP8/NVFP4 inplace
    grad: torch.Tensor,    # BF16 gradient
    momentum: torch.Tensor,
    variance: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    step: int,
    scale: torch.Tensor,   # Weight scale
) -> None:
    """Fused ECO AdamW update with error injection."""
    pass
```

### 9.2 File Structure Changes

```
torchtitan/kernels/
├── __init__.py
├── fp8_matmul.py          # FP8 matmul kernels
├── nvfp4_matmul.py        # NVFP4 matmul kernels  
├── quantize.py            # Quantization/dequantization kernels
├── ecoadamw.py            # Fused ECOAdamW kernel
├── moe_gemm.py            # MoE grouped GEMM kernels
└── utils.py               # Kernel utilities

torchtitan/components/
├── quantized_tensor.py    # Extended with kernel dispatch
├── quantized_linear.py    # Updated with true low-precision matmul
└── eco_optimizer.py       # Optional fused kernel path
```

### 9.3 Configuration Extensions

```python
# New config options
eco_config:
    kernel_backend: "triton" | "cuda" | "torchao"  # Kernel implementation
    fused_optimizer: true | false                  # Use fused ECOAdamW kernel
    activation_quant_dtype: "fp8_e4m3" | "none"    # Activation quantization
    quant_dtype: "fp8_e4m3" | "nvfp4_e2m1"         # Weight quantization format
```

---

*Last Updated: 2026-02-09*  
*Version: 1.0*  
*Author: ECO Kernel Implementation Team*