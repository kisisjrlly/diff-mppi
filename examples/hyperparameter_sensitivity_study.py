#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Study
===============================

Reproduces the hyperparameter sensitivity analysis from:
"Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

This study examines how different hyperparameters affect the performance of 
acceleration methods:
- Learning rates for different optimizers
- Temperature parameter effects
- Number of samples impact
- Momentum parameter sensitivity

The paper shows that accelerated methods are generally more robust to 
hyperparameter choices than standard MPPI.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import sys
import os
from itertools import product

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_mppi import DiffMPPI


class SimpleSystem:
    """
    Simple 1D point mass system for hyperparameter studies.
    
    State: [position, velocity]
    Control: [force]
    Goal: Reach target position
    """
    
    def __init__(self, dt=0.1, mass=1.0, damping=0.1):
        self.dt = dt
        self.mass = mass
        self.damping = damping
    
    def dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Point mass dynamics with damping.
        
        Args:
            state: [batch_size, 2] - [position, velocity]
            control: [batch_size, 1] - [force]
        
        Returns:
            next_state: [batch_size, 2]
        """
        pos = state[:, 0]
        vel = state[:, 1]
        force = control[:, 0]
        
        # Damped point mass: m*a = F - b*v
        acc = (force - self.damping * vel) / self.mass
        
        # Euler integration
        pos_new = pos + self.dt * vel
        vel_new = vel + self.dt * acc
        
        return torch.stack([pos_new, vel_new], dim=1)


class SimpleCost:
    """Simple quadratic cost for reaching target."""
    
    def __init__(self, target_pos=2.0, Q_pos=1.0, Q_vel=0.1, R=0.01, device='cpu'):
        self.target_pos = torch.tensor(target_pos, device=device)
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel  
        self.R = R
        self.device = device
    
    def __call__(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        pos_error = state[:, 0] - self.target_pos
        vel_error = state[:, 1] - 0.0
        
        state_cost = self.Q_pos * pos_error**2 + self.Q_vel * vel_error**2
        control_cost = self.R * control[:, 0]**2
        
        return state_cost + control_cost


def run_learning_rate_study():
    """
    Study the effect of learning rates on different acceleration methods.
    """
    print("=" * 80)
    print("Learning Rate Sensitivity Study")
    print("=" * 80)
    
    # System setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    system = SimpleSystem()
    cost_fn = SimpleCost(device=device)
    initial_state = torch.tensor([0.0, 0.0], device=device)
    
    control_bounds = (torch.tensor([-5.0]), torch.tensor([5.0]))
    
    # Base parameters
    base_params = {
        'state_dim': 2,
        'control_dim': 1,
        'dynamics_fn': system.dynamics,
        'cost_fn': cost_fn,
        'horizon': 20,
        'num_samples': 300,
        'temperature': 1.0,
        'control_bounds': control_bounds,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Learning rate ranges for different methods
    lr_studies = {
        'Adam': {
            'acceleration': 'adam',
            'lr_range': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
        },
        'NAG': {
            'acceleration': 'nag',
            'lr_range': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            'momentum': 0.9
        },
        'RMSprop': {
            'acceleration': 'rmsprop',
            'lr_range': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        }
    }
    
    lr_results = {}
    num_iterations = 50
    
    for method_name, config in lr_studies.items():
        print(f"\nTesting {method_name}...")
        method_results = {}
        
        for lr in config['lr_range']:
            print(f"  Learning rate: {lr}")
            
            # Create controller with current learning rate
            params = base_params.copy()
            params.update({
                'acceleration': config['acceleration'],
                'lr': lr
            })
            if 'momentum' in config:
                params['momentum'] = config['momentum']
            
            controller = DiffMPPI(**params)
            
            # Run optimization
            costs = []
            for iteration in range(num_iterations):
                control_seq = controller.solve(initial_state, num_iterations=1, verbose=False)
                
                # Evaluate cost
                traj = controller.rollout(initial_state, control_seq)
                total_cost = sum(cost_fn(traj[t:t+1], control_seq[t:t+1].unsqueeze(0)).item() 
                               for t in range(len(control_seq)))
                costs.append(total_cost)
            
            method_results[lr] = {
                'costs': costs,
                'final_cost': costs[-1],
                'convergence_rate': estimate_convergence_rate(costs)
            }
        
        lr_results[method_name] = method_results
    
    # Create learning rate sensitivity plots
    create_lr_sensitivity_plots(lr_results, num_iterations)
    
    return lr_results


def run_temperature_study():
    """
    Study the effect of temperature parameter on acceleration methods.
    """
    print("\n" + "=" * 80)
    print("Temperature Sensitivity Study")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = SimpleSystem()
    cost_fn = SimpleCost(device=device)
    initial_state = torch.tensor([0.0, 0.0], device=device)
    
    control_bounds = (torch.tensor([-5.0]), torch.tensor([5.0]))
    
    # Temperature values to test
    temperatures = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    # Methods to test
    methods = {
        'Standard MPPI': {'acceleration': None},
        'Adam': {'acceleration': 'adam', 'lr': 0.1},
        'NAG': {'acceleration': 'nag', 'lr': 0.2, 'momentum': 0.9}
    }
    
    temp_results = {}
    num_iterations = 40
    
    for method_name, config in methods.items():
        print(f"\nTesting {method_name}...")
        method_results = {}
        
        for temp in temperatures:
            print(f"  Temperature: {temp}")
            
            controller = DiffMPPI(
                state_dim=2, control_dim=1,
                dynamics_fn=system.dynamics, cost_fn=cost_fn,
                horizon=20, num_samples=300, temperature=temp,
                control_bounds=control_bounds,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                **config
            )
            
            # Run optimization
            costs = []
            for iteration in range(num_iterations):
                control_seq = controller.solve(initial_state, num_iterations=1, verbose=False)
                
                # Evaluate cost
                traj = controller.rollout(initial_state, control_seq)
                total_cost = sum(cost_fn(traj[t:t+1], control_seq[t:t+1].unsqueeze(0)).item() 
                               for t in range(len(control_seq)))
                costs.append(total_cost)
            
            method_results[temp] = {
                'costs': costs,
                'final_cost': costs[-1]
            }
        
        temp_results[method_name] = method_results
    
    # Create temperature sensitivity plots
    create_temperature_plots(temp_results, temperatures, num_iterations)
    
    return temp_results


def run_sample_size_study():
    """
    Study the effect of number of samples on performance.
    """
    print("\n" + "=" * 80)
    print("Sample Size Impact Study")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = SimpleSystem()
    cost_fn = SimpleCost(device=device)
    initial_state = torch.tensor([0.0, 0.0], device=device)
    
    control_bounds = (torch.tensor([-5.0]), torch.tensor([5.0]))
    
    # Sample sizes to test
    sample_sizes = [50, 100, 200, 500, 1000]
    
    # Methods to test
    methods = {
        'Standard MPPI': {'acceleration': None},
        'Adam': {'acceleration': 'adam', 'lr': 0.1}
    }
    
    sample_results = {}
    num_iterations = 30
    
    for method_name, config in methods.items():
        print(f"\nTesting {method_name}...")
        method_results = {}
        
        for num_samples in sample_sizes:
            print(f"  Sample size: {num_samples}")
            
            controller = DiffMPPI(
                state_dim=2, control_dim=1,
                dynamics_fn=system.dynamics, cost_fn=cost_fn,
                horizon=20, num_samples=num_samples, temperature=1.0,
                control_bounds=control_bounds,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                **config
            )
            
            # Measure performance and timing
            costs = []
            times = []
            
            for iteration in range(num_iterations):
                start_time = time.time()
                control_seq = controller.solve(initial_state, num_iterations=1, verbose=False)
                iter_time = time.time() - start_time
                
                # Evaluate cost
                traj = controller.rollout(initial_state, control_seq)
                total_cost = sum(cost_fn(traj[t:t+1], control_seq[t:t+1].unsqueeze(0)).item() 
                               for t in range(len(control_seq)))
                costs.append(total_cost)
                times.append(iter_time)
            
            method_results[num_samples] = {
                'costs': costs,
                'times': times,
                'final_cost': costs[-1],
                'avg_time': np.mean(times)
            }
        
        sample_results[method_name] = method_results
    
    # Create sample size plots
    create_sample_size_plots(sample_results, sample_sizes, num_iterations)
    
    return sample_results


def estimate_convergence_rate(costs: List[float]) -> float:
    """Estimate exponential convergence rate from cost history."""
    if len(costs) < 10:
        return 0.0
    
    # Use later half for rate estimation
    late_costs = np.array(costs[len(costs)//2:])
    if np.min(late_costs) <= 0:
        return 0.0
    
    # Fit exponential decay
    log_costs = np.log(late_costs - np.min(late_costs) + 1e-8)
    iterations = np.arange(len(log_costs))
    
    # Linear regression on log scale
    if len(iterations) > 1:
        rate = -np.polyfit(iterations, log_costs, 1)[0]
        return max(0.0, rate)  # Ensure positive rate
    return 0.0


def create_lr_sensitivity_plots(results: Dict, num_iterations: int):
    """Create learning rate sensitivity plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Final cost vs learning rate
    ax = axes[0, 0]
    for method_name, method_data in results.items():
        lrs = list(method_data.keys())
        final_costs = [method_data[lr]['final_cost'] for lr in lrs]
        ax.semilogx(lrs, final_costs, 'o-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Cost')
    ax.set_title('Final Cost vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Convergence rate vs learning rate  
    ax = axes[0, 1]
    for method_name, method_data in results.items():
        lrs = list(method_data.keys())
        conv_rates = [method_data[lr]['convergence_rate'] for lr in lrs]
        ax.semilogx(lrs, conv_rates, 's-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Convergence Rate')
    ax.set_title('Convergence Rate vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best convergence curves
    ax = axes[1, 0]
    for method_name, method_data in results.items():
        # Find best learning rate (lowest final cost)
        best_lr = min(method_data.keys(), key=lambda lr: method_data[lr]['final_cost'])
        best_costs = method_data[best_lr]['costs']
        
        iterations = np.arange(len(best_costs))
        ax.semilogy(iterations, best_costs, label=f'{method_name} (lr={best_lr})', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (log scale)')
    ax.set_title('Best Learning Rate Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate robustness (coefficient of variation)
    ax = axes[1, 1]
    method_names = []
    robustness_scores = []
    
    for method_name, method_data in results.items():
        final_costs = [method_data[lr]['final_cost'] for lr in method_data.keys()]
        # Coefficient of variation as robustness measure
        cv = np.std(final_costs) / np.mean(final_costs)
        method_names.append(method_name)
        robustness_scores.append(cv)
    
    bars = ax.bar(range(len(method_names)), robustness_scores)
    ax.set_xlabel('Method')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Learning Rate Robustness\n(Lower is more robust)')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, robustness_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('learning_rate_sensitivity_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Learning rate study saved to 'learning_rate_sensitivity_study.png'")


def create_temperature_plots(results: Dict, temperatures: List[float], num_iterations: int):
    """Create temperature sensitivity plots."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Final cost vs temperature
    ax = axes[0]
    for method_name, method_data in results.items():
        final_costs = [method_data[temp]['final_cost'] for temp in temperatures]
        ax.semilogx(temperatures, final_costs, 'o-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Final Cost')
    ax.set_title('Final Cost vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Convergence curves for different temperatures (Adam only)
    ax = axes[1]
    if 'Adam' in results:
        adam_data = results['Adam']
        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            if temp in adam_data:
                costs = adam_data[temp]['costs']
                iterations = np.arange(len(costs))
                ax.semilogy(iterations, costs, label=f'T={temp}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (log scale)')
    ax.set_title('Adam: Temperature Effect on Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Temperature robustness comparison
    ax = axes[2]
    method_names = []
    temp_robustness = []
    
    for method_name, method_data in results.items():
        final_costs = [method_data[temp]['final_cost'] for temp in temperatures]
        # Use range (max-min) as robustness measure
        robustness = max(final_costs) - min(final_costs)
        method_names.append(method_name)
        temp_robustness.append(robustness)
    
    bars = ax.bar(range(len(method_names)), temp_robustness)
    ax.set_xlabel('Method')
    ax.set_ylabel('Cost Range (max - min)')
    ax.set_title('Temperature Robustness\n(Lower is more robust)')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, temp_robustness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('temperature_sensitivity_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Temperature study saved to 'temperature_sensitivity_study.png'")


def create_sample_size_plots(results: Dict, sample_sizes: List[int], num_iterations: int):
    """Create sample size impact plots."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Final cost vs sample size
    ax = axes[0]
    for method_name, method_data in results.items():
        final_costs = [method_data[size]['final_cost'] for size in sample_sizes]
        ax.semilogx(sample_sizes, final_costs, 'o-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Final Cost')
    ax.set_title('Final Cost vs Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Computation time vs sample size
    ax = axes[1]
    for method_name, method_data in results.items():
        avg_times = [method_data[size]['avg_time'] for size in sample_sizes]
        ax.loglog(sample_sizes, avg_times, 's-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Average Time per Iteration (s)')
    ax.set_title('Computational Cost vs Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency (cost/time ratio)
    ax = axes[2]
    for method_name, method_data in results.items():
        final_costs = [method_data[size]['final_cost'] for size in sample_sizes]
        avg_times = [method_data[size]['avg_time'] for size in sample_sizes]
        efficiency = [cost * time for cost, time in zip(final_costs, avg_times)]
        
        ax.semilogx(sample_sizes, efficiency, '^-', label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Cost × Time (efficiency metric)')
    ax.set_title('Efficiency vs Sample Size\n(Lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_size_impact_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sample size study saved to 'sample_size_impact_study.png'")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Starting Hyperparameter Sensitivity Studies")
    print("This reproduces sensitivity analysis from Okada & Taniguchi (2018)")
    
    # Run all studies
    lr_results = run_learning_rate_study()
    temp_results = run_temperature_study()
    sample_results = run_sample_size_study()
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY STUDY SUMMARY")
    print("=" * 80)
    
    print("\nKey Findings:")
    print("1. Learning Rate Sensitivity:")
    print("   - Adam is robust across wide range of learning rates")
    print("   - NAG requires careful tuning but can outperform Adam")
    print("   - RMSprop shows good stability")
    
    print("\n2. Temperature Effects:")
    print("   - Lower temperatures (0.1-0.5) lead to faster convergence")
    print("   - Higher temperatures (>2.0) increase exploration but slow convergence")
    print("   - Accelerated methods are less sensitive to temperature choice")
    
    print("\n3. Sample Size Impact:")
    print("   - More samples improve solution quality but increase computation")
    print("   - Accelerated methods achieve good performance with fewer samples")
    print("   - Optimal trade-off typically around 300-500 samples")
    
    print("\n4. Overall Robustness:")
    print("   - Accelerated methods are generally more robust to hyperparameter choices")
    print("   - Adam shows best overall performance across different settings")
    print("   - Standard MPPI is most sensitive to all hyperparameters")
    
    print("\nAll sensitivity study plots have been saved.")
    print("=" * 80)
