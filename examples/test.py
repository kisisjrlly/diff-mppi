import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# --- 1. Core Math Utility: Conjugate Gradient Solver ---
def conjugate_gradient_solver(A_fn, b, n_steps=10, residual_tol=1e-10):
    """
    Solves the linear system Ax = b using the conjugate gradient method.
    This is a matrix-free implementation, requiring only a function A_fn(v)
    that computes the product of matrix A and a vector v.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    if rs_old < 1e-10:
        return x

    for i in range(n_steps):
        Ap = A_fn(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
        
        x += alpha * p
        r -= alpha * Ap
        
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < residual_tol:
            break
            
        p = r + (rs_new / (rs_old + 1e-8)) * p
        rs_old = rs_new
        
    return x


# --- 2. Pendulum Environment Model ---
def pendulum_dynamics(state, action):
    """
    Nonlinear dynamics of a rotary inverted pendulum.
    state = [theta, alpha, theta_dot, alpha_dot] (arm angle, pendulum angle, velocities)
    action = [voltage]
    """
    g = 9.81
    m_p, m_c = 0.127, 0.257
    l_p, l_c = 0.336, 0.216
    J_p, J_c = 0.012, 0.002
    B_p, B_c = 0.0024, 0.0024
    dt = 0.02

    theta, alpha, theta_dot, alpha_dot = state.unbind(-1)

    M = torch.zeros(state.shape[0], 2, 2, device=state.device)
    M[:, 0, 0] = J_c + m_p * l_c**2
    M[:, 0, 1] = m_p * l_c * (l_p/2) * torch.cos(alpha)
    M[:, 1, 0] = M[:, 0, 1]
    M[:, 1, 1] = J_p + m_p * (l_p/2)**2

    C = torch.zeros(state.shape[0], 2, 1, device=state.device)
    C[:, 0, 0] = -m_p * l_c * (l_p/2) * torch.sin(alpha) * alpha_dot**2 + B_c * theta_dot
    C[:, 1, 0] = m_p * g * (l_p/2) * torch.sin(alpha) + B_p * alpha_dot
    
    tau = 2.5 * action.squeeze(-1)

    try:
        M_inv = torch.inverse(M)
        acc = torch.bmm(M_inv, tau.unsqueeze(-1).unsqueeze(-1) - C).squeeze(-1)
    except torch.linalg.LinAlgError:
        acc = torch.zeros(state.shape[0], 2, device=state.device)
    
    theta_ddot, alpha_ddot = acc.unbind(-1)

    new_theta_dot = theta_dot + theta_ddot * dt
    new_alpha_dot = alpha_dot + alpha_ddot * dt
    new_theta = theta + new_theta_dot * dt
    new_alpha = alpha + new_alpha_dot * dt
    
    return torch.stack([new_theta, new_alpha, new_theta_dot, new_alpha_dot], dim=-1)

# --- 3. Base Controller Class ---
class BaseMPPI:
    def __init__(self, dynamics_fn, cost_fn, state_dim, action_dim, horizon, num_samples, temperature, device):
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.K = num_samples
        self.lambda_ = temperature
        self.device = device

        self.U = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.Sigma = torch.eye(self.action_dim, device=self.device) * 0.5

    def _rollout_trajectories(self, initial_state, noisy_actions):
        batch_size = noisy_actions.shape[0]
        states = torch.zeros(batch_size, self.horizon + 1, self.state_dim, device=self.device)
        states[:, 0] = initial_state.expand(batch_size, -1)

        with torch.no_grad():
            for t in range(self.horizon):
                states[:, t + 1] = self.dynamics_fn(states[:, t], noisy_actions[:, t])
        
        return states[:, 1:]

    def solve(self, initial_state):
        noise = torch.randn(self.K, self.horizon, self.action_dim, device=self.device)
        noisy_actions = self.U.unsqueeze(0) + noise @ self.Sigma
        
        states = self._rollout_trajectories(initial_state, noisy_actions)
        costs = self.cost_fn(states, noisy_actions)
        
        self._update_distribution(costs, noise)
        
        action = self.U[0].clone()
        
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1].zero_()

        return action

    def _update_distribution(self, costs, noise):
        raise NotImplementedError

# --- 4. Standard MPPI Implementation ---
class StandardMPPI(BaseMPPI):
    def _update_distribution(self, costs, noise):
        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum() + 1e-8
        
        delta_U = torch.sum(weights.view(-1, 1, 1) * noise, dim=0)
        
        self.U += delta_U @ self.Sigma

# --- 5. Geodesic-MPPI Implementation (MODIFIED with Trust Region) ---
class GeodesicMPPI(BaseMPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_threshold = 0.1 # Trust region size

    def _update_distribution(self, costs, noise):
        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights /= weights.sum() + 1e-8
        
        euclidean_grad_noise_space = torch.sum(weights.view(-1, 1, 1) * noise, dim=0)
        
        natural_grad_direction = euclidean_grad_noise_space @ self.Sigma
        
        squared_norm_g = torch.sum(euclidean_grad_noise_space**2)
        
        optimal_step_size = torch.sqrt(2 * self.kl_threshold / (squared_norm_g + 1e-8))

        learning_rate = min(1.0, optimal_step_size.item())

        self.U += learning_rate * natural_grad_direction


# --- 6. Simulation Setup and Execution (MODIFIED with complete stability check) ---
def run_simulation(controller, initial_state, cost_fn, max_steps=750, stability_duration=1.0, dt=0.02):
    """
    Runs the simulation with intelligent stopping criteria.
    """
    states_hist = [initial_state.cpu().numpy()]
    actions_hist = []
    costs_hist = []
    
    state = initial_state.clone()
    start_time = time.time()

    angle_threshold_deg = 5.0
    velocity_threshold_rad_s = 0.5
    stability_duration_steps = int(stability_duration / dt)
    stable_counter = 0
    
    print(f"Controller: {controller.__class__.__name__}")

    for step in range(max_steps):
        with torch.no_grad():
            action = controller.solve(state)
            next_state = pendulum_dynamics(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
        
        states_hist.append(next_state.cpu().numpy())
        actions_hist.append(action.cpu().numpy())
        
        cost = cost_fn(next_state.unsqueeze(0), action.unsqueeze(0))
        costs_hist.append(cost.item())
        
        state = next_state
        
        # --- MODIFICATION START: Complete Stability Check ---
        # Wrap angles to [-pi, pi] for correct checking near zero
        current_theta_rad = (state[0].item() + np.pi) % (2 * np.pi) - np.pi
        current_alpha_rad = (state[1].item() + np.pi) % (2 * np.pi) - np.pi
        
        # Check angles in degrees
        current_theta_deg = abs(current_theta_rad * 180 / np.pi)
        current_alpha_deg = abs(current_alpha_rad * 180 / np.pi)
        
        # Check velocities in rad/s
        current_theta_dot_rads = abs(state[2].item())
        current_alpha_dot_rads = abs(state[3].item())
        
        # Check if ALL conditions for stability are met
        is_stable = (current_alpha_deg < angle_threshold_deg and 
                     current_alpha_dot_rads < velocity_threshold_rad_s and
                     current_theta_deg < angle_threshold_deg and 
                     current_theta_dot_rads < velocity_threshold_rad_s)
        
        if is_stable:
            stable_counter += 1
        else:
            stable_counter = 0
            
        if stable_counter >= stability_duration_steps:
            print(f"  System stabilized for {stability_duration}s. Stopping early at step {step + 1}.")
            break
        # --- MODIFICATION END ---
    
    if step == max_steps - 1:
        print(f"  Simulation finished: Reached max steps ({max_steps}) without stabilizing.")

    total_time = time.time() - start_time
    total_cost = sum(costs_hist)
    
    print(f"  Total Cost: {total_cost:.2f}")
    print(f"  Total Time: {total_time:.2f}s")
    
    return np.array(states_hist), np.array(actions_hist), total_cost

# --- 7. Animation Function ---
def animate_results(std_states, geo_states, dt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'aspect': 'equal'})
    
    l_c = 0.216
    l_p = 0.336

    def setup_ax(ax, title):
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.plot(0, 0, 'ko', markersize=10)

    setup_ax(ax1, 'Standard MPPI')
    setup_ax(ax2, 'Geodesic-MPPI')
    
    line_arm1, = ax1.plot([], [], 'o-', lw=5, color='royalblue', markersize=8)
    line_pendulum1, = ax1.plot([], [], 'o-', lw=3, color='crimson', markersize=10)
    time_text1 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

    line_arm2, = ax2.plot([], [], 'o-', lw=5, color='royalblue', markersize=8)
    line_pendulum2, = ax2.plot([], [], 'o-', lw=3, color='crimson', markersize=10)
    time_text2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

    def get_coords(state):
        theta, alpha = state[0], state[1]
        arm_x = l_c * np.sin(theta)
        arm_y = -l_c * np.cos(theta)
        pendulum_x = arm_x + l_p * np.sin(theta + alpha)
        pendulum_y = arm_y - l_p * np.cos(theta + alpha)
        return (0, arm_x), (0, arm_y), (arm_x, pendulum_x), (arm_y, pendulum_y)

    def init():
        line_arm1.set_data([], [])
        line_pendulum1.set_data([], [])
        time_text1.set_text('')
        line_arm2.set_data([], [])
        line_pendulum2.set_data([], [])
        time_text2.set_text('')
        return line_arm1, line_pendulum1, time_text1, line_arm2, line_pendulum2, time_text2

    def update(i):
        idx1 = min(i, len(std_states) - 1)
        arm_x1, arm_y1, pen_x1, pen_y1 = get_coords(std_states[idx1])
        line_arm1.set_data(arm_x1, arm_y1)
        line_pendulum1.set_data(pen_x1, pen_y1)
        time_text1.set_text(f'Time: {idx1*dt:.2f}s')

        idx2 = min(i, len(geo_states) - 1)
        arm_x2, arm_y2, pen_x2, pen_y2 = get_coords(geo_states[idx2])
        line_arm2.set_data(arm_x2, arm_y2)
        line_pendulum2.set_data(pen_x2, pen_y2)
        time_text2.set_text(f'Time: {idx2*dt:.2f}s')
        
        return line_arm1, line_pendulum1, time_text1, line_arm2, line_pendulum2, time_text2

    num_frames = max(len(std_states), len(geo_states))
    ani = animation.FuncAnimation(fig, update, frames=range(num_frames),
                                  init_func=init, blit=True, interval=dt*1000)
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dim = 4
    action_dim = 1
    dt = 0.02
    
    def cost_fn(states, actions):
        theta, alpha, theta_dot, alpha_dot = states.unbind(-1)
        
        angle_cost = 10 * (1 - torch.cos(alpha)) + 1 * theta**2
        velocity_cost = 0.1 * alpha_dot**2 + 0.1 * theta_dot**2
        action_cost = 0.01 * actions.squeeze(-1)**2
        
        step_costs = angle_cost + velocity_cost + action_cost
        
        if step_costs.dim() > 1 and step_costs.shape[1] > 1:
            return torch.sum(step_costs, dim=1)
        else:
            return step_costs

    mppi_params = {
        "dynamics_fn": pendulum_dynamics,
        "cost_fn": cost_fn,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "horizon": 50,
        "num_samples": 512,
        "temperature": 0.5,
        "device": device
    }

    initial_state = torch.tensor([0.0, np.pi, 0.0, 0.0], device=device)

    results = {}
    
    max_sim_time = 15.0
    max_steps = int(max_sim_time / dt)

    print("Running Standard MPPI...")
    std_mppi = StandardMPPI(**mppi_params)
    std_states, std_actions, std_cost = run_simulation(std_mppi, initial_state, cost_fn, max_steps=max_steps, dt=dt)
    results['Standard MPPI'] = {'states': std_states, 'actions': std_actions, 'cost': std_cost}
    
    print("\nRunning Geodesic-MPPI...")
    geo_mppi = GeodesicMPPI(**mppi_params)
    geo_states, geo_actions, geo_cost = run_simulation(geo_mppi, initial_state, cost_fn, max_steps=max_steps, dt=dt)
    results['Geodesic-MPPI'] = {'states': geo_states, 'actions': geo_actions, 'cost': geo_cost}

    print("\nStarting animation...")
    animate_results(std_states, geo_states, dt)
    
    print("\nAnimation closed. Generating static plots...")
    plot_static_results(results, dt)

def plot_static_results(results, dt):
    std_states = results['Standard MPPI']['states']
    geo_states = results['Geodesic-MPPI']['states']
    std_actions = results['Standard MPPI']['actions']
    geo_actions = results['Geodesic-MPPI']['actions']
    std_cost = results['Standard MPPI']['cost']
    geo_cost = results['Geodesic-MPPI']['cost']
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_vec_std = np.arange(std_states.shape[0]) * dt
    time_vec_geo = np.arange(geo_states.shape[0]) * dt
    
    rad_to_deg = 180 / np.pi

    std_alpha_deg = (std_states[:, 1] * rad_to_deg + 180) % 360 - 180
    geo_alpha_deg = (geo_states[:, 1] * rad_to_deg + 180) % 360 - 180
    std_theta_deg = (std_states[:, 0] * rad_to_deg + 180) % 360 - 180
    geo_theta_deg = (geo_states[:, 0] * rad_to_deg + 180) % 360 - 180

    axs[0].plot(time_vec_std, std_alpha_deg, label='Standard MPPI - Pendulum Angle (alpha)')
    axs[0].plot(time_vec_geo, geo_alpha_deg, label='Geodesic-MPPI - Pendulum Angle (alpha)', linestyle='--')
    axs[0].plot(time_vec_std, std_theta_deg, label='Standard MPPI - Arm Angle (theta)', alpha=0.5)
    axs[0].plot(time_vec_geo, geo_theta_deg, label='Geodesic-MPPI - Arm Angle (theta)', linestyle='--', alpha=0.5)
    axs[0].axhline(0, color='k', linestyle=':', label='Target Angle (0 degrees)')
    axs[0].set_ylabel('Angle (degrees)')
    axs[0].set_title('Pendulum Swing-Up and Stabilization Performance (in Degrees)')
    axs[0].legend()

    axs[1].plot(time_vec_std, std_states[:, 3] * rad_to_deg, label='Standard MPPI - Pendulum Velocity')
    axs[1].plot(time_vec_geo, geo_states[:, 3] * rad_to_deg, label='Geodesic-MPPI - Pendulum Velocity', linestyle='--')
    axs[1].set_ylabel('Angular Velocity (degrees/s)')
    axs[1].legend()

    axs[2].plot(time_vec_std[:-1], std_actions, label='Standard MPPI')
    axs[2].plot(time_vec_geo[:-1], geo_actions, label='Geodesic-MPPI', linestyle='--')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Control Input (Voltage)')
    axs[2].legend()
    
    print("\n--- Performance Summary ---")
    print(f"Standard MPPI Total Cost: {std_cost:.2f}")
    print(f"Geodesic-MPPI Total Cost: {geo_cost:.2f}")
    if geo_cost < std_cost:
        improvement = ((std_cost - geo_cost) / (std_cost + 1e-8)) * 100
        print(f"Geodesic-MPPI showed a performance improvement of {improvement:.2f}%")
    else:
        print("Standard MPPI performed better in this run.")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

