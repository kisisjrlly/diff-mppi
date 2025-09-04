#!/usr/bin/env python3
"""
Quick Verification Test
======================

Simple test to verify the experiments work correctly before running the full suite.
Tests basic functionality of each acceleration method on a simple system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from diff_mppi import DiffMPPI


def simple_dynamics(state, control):
    """Simple 1D dynamics: x_dot = u"""
    return state + 0.1 * control


def simple_cost(state, control):
    """Simple quadratic cost: (x-1)^2 + u^2"""
    return (state[:, 0] - 1.0)**2 + 0.1 * control[:, 0]**2


def test_method(method_name, config):
    """Test a single acceleration method."""
    print(f"Testing {method_name}...")
    
    controller = DiffMPPI(
        state_dim=1,
        control_dim=1,
        dynamics_fn=simple_dynamics,
        cost_fn=simple_cost,
        horizon=10,
        num_samples=100,
        temperature=1.0,
        device='cpu',  # Use CPU for quick test
        **config
    )
    
    initial_state = torch.tensor([0.0])
    
    try:
        # Run a few iterations
        control_seq = controller.solve(initial_state, num_iterations=5, verbose=False)
        
        # Check that we get a reasonable result
        final_state = controller.rollout(initial_state, control_seq)[-1]
        error = abs(final_state[0].item() - 1.0)  # Target is x=1
        
        print(f"  ✅ {method_name}: Final state = {final_state[0].item():.3f}, Error = {error:.3f}")
        return True
        
    except Exception as e:
        print(f"  ❌ {method_name}: Failed with error: {e}")
        return False


def main():
    """Run quick verification tests."""
    print("=" * 60)
    print("QUICK VERIFICATION TEST")
    print("Testing all acceleration methods on simple 1D system")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test configurations
    methods = {
        'Standard MPPI': {'acceleration': None},
        'Adam': {'acceleration': 'adam', 'lr': 0.1},
        'NAG': {'acceleration': 'nag', 'lr': 0.1, 'momentum': 0.9},
        'RMSprop': {'acceleration': 'rmsprop', 'lr': 0.1}
    }
    
    # Run tests
    results = {}
    for method_name, config in methods.items():
        results[method_name] = test_method(method_name, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    total_tests = len(methods)
    passed_tests = sum(results.values())
    
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All methods working correctly!")
        print("You can now run the full experiment suite with:")
        print("   python run_all_experiments.py")
    else:
        print("\n⚠️  Some methods failed. Check your installation.")
        
        failed_methods = [name for name, passed in results.items() if not passed]
        print("Failed methods:", failed_methods)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
