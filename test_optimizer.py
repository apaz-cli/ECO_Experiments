#!/usr/bin/env python3
"""Test ECOMuon instantiation with kwargs from build_optimizers."""

import sys
import torch
from torchtitan.components.eco_muon import ECOMuon

def test_ecomuon_kwargs():
    # Simulate kwargs that would be passed from build_optimizers for muon.toml
    kwargs = {
        "lr": 3e-4,
        "momentum": 0.95,
        "ns_steps": 5,
        "adam_betas": (0.9, 0.999),
        "adam_eps": 1e-8,
        "weight_decay": 0.1,
        "optim_state_dtype": torch.float32,
        "optim_compute_dtype": torch.float32,
        "eco_enabled": True,
        "eco_approach": "jacobian",
        "stochastic_rounding": False,
        "heuristic_log_freq": 1,
        "quantize_weights": True,
        "quant_dtype": "fp8",
        "include_weight_decay_in_injection": True,
        "master_weights_dtype": None,
    }
    
    # Create a dummy parameter (2D tensor to trigger Muon update)
    param = torch.randn(64, 128, requires_grad=True)
    
    print("Testing ECOMuon instantiation...")
    try:
        opt = ECOMuon([param], **kwargs)
        print("✓ ECOMuon instantiated successfully")
    except Exception as e:
        print(f"✗ ECOMuon instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Perform a dummy step (gradient needed)
    loss = param.sum()
    loss.backward()
    
    print("Testing step()...")
    try:
        opt.step()
        print("✓ step() succeeded")
    except Exception as e:
        print(f"✗ step() failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("All tests passed.")

if __name__ == "__main__":
    test_ecomuon_kwargs()