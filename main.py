import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from pid import PIDController

rng = np.random.default_rng(123)


# Inflow functions for different weather conditions
def Q_normal(t):
    return 0.2 * (rng.random() * np.sin(t / 2) / 3 + np.sin(t / 5 + np.pi / 2) / 3 + np.sin(
        t / 10 + np.pi) / 3 + 1) + 0.05 * rng.random()


def Q_drought(t):
    return 0.1 * (rng.random() * np.sin(t / 2) / 3 + np.sin(t / 5 + np.pi / 2) / 3 + np.sin(
        t / 10 + np.pi) / 3 + 1) + 0.02 * rng.random()


def Q_rainy(t):
    return 0.3 * (rng.random() * np.sin(t / 2) / 3 + np.sin(t / 5 + np.pi / 2) / 3 + np.sin(
        t / 10 + np.pi) / 3 + 1) + 0.075 * rng.random()


# Additional inflow functions for different weather conditions
def r_normal(t, rt1):
    rt = 0.1 * (rng.random() - 0.5)
    rt = rt if rt > 0 else 0
    if rt1 == 0 and rng.random() > 0.2:
        rt = 0
    return rt


def r_drought(t, rt1):
    rt = 0.05 * (rng.random() - 0.5)
    rt = rt if rt > 0 else 0
    if rt1 == 0 and rng.random() > 0.2:
        rt = 0
    return rt


def r_rainy(t, rt1):
    rt = 0.15 * (rng.random() - 0.5)
    rt = rt if rt > 0 else 0
    if rt1 == 0 and rng.random() > 0.2:
        rt = 0
    return rt


def q2t(t, V):
    q2_value = np.sqrt(np.maximum(V, 0)) / 4
    if V < q2_value:
        q2_value = V
    return q2_value


def vvdt(t, q1t_value, rt_value, q2t_value, Qt_value):
    dv1dt = Qt_value - q1t_value
    dv2dt = q1t_value + rt_value - q2t_value
    return np.array([dv1dt, dv2dt])


def checkVolume(V, minallowed, maxallowed, verbose=True, reservoirName=""):
    if V < 0 or V > 1:
        if verbose:
            print(f"{reservoirName}: value {V:5.3f} is not valid")
        return -1  # invalid
    elif V < minallowed:
        if verbose:
            print(f"{reservoirName}: value {V:5.3f} is close to empty (lower than allowed)")
        return 0  # close to empty
    elif V > maxallowed:
        if verbose:
            print(f"{reservoirName}: value {V:5.3f} is close to overflow (higher than allowed)")
        return 2  # close to overflow
    else:
        if verbose:
            print(f"{reservoirName}: value is valid")
        return 1  # valid


def run_simulation(Qt_func, rt_func, scenario_title, params=None, optimize=False, use_rl=False, rl_model=None):
    T = 100
    dt = 0.1
    ttt = np.arange(0, T, dt)
    V = np.zeros((2, np.size(ttt)))
    V[:, 0] = [0.5, 0.5]  # initial values should be >=0, <=1
    minallowed = 0.1
    maxallowed = 0.9
    Q_values = np.zeros_like(ttt)
    Q_values[0] = Qt_func(0)
    q1_values = np.zeros_like(ttt)
    q2_values = np.zeros_like(ttt)
    r_values = np.zeros_like(ttt)
    r_values[0] = rt_func(0, 0)

    invalid_count = 0
    low_count = 0
    high_count = 0
    valid_count = 0

    if params is not None and not use_rl:
        kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, feedforward_gain = params
        pid_controller = PIDController(kp, ki, kd, setpoint1, setpoint2, weight_v1, weight_v2, feedforward_gain)
    else:
        pid_controller = PIDController(1.0, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 1.0)

    for i in range(1, np.size(ttt)):
        t = ttt[i]
        if use_rl:
            state = np.array([V[0, i - 1], V[1, i - 1], r_values[i - 1]])
            action = rl_model.select_action(state)
            q1t_value = max(0, min(1, action[0]))
        else:
            feedforward_input = Q_values[i - 1]  # Example feedforward input
            q1t_value = pid_controller.compute(V[0, i - 1], V[1, i - 1], feedforward_input)

        q1_values[i] = q1t_value
        rt_value = rt_func(t, r_values[i - 1])
        r_values[i] = rt_value
        q2t_value = q2t(t, V[1, i - 1])
        q2_values[i] = q2t_value
        Qt_value = Qt_func(t)
        Q_values[i] = Qt_value
        V[:, i] = V[:, i - 1] + dt * vvdt(ttt[i], q1t_value, rt_value, q2t_value, Qt_value)
        V[:, i] = np.clip(V[:, i], 0, 1)
        status1 = checkVolume(V[0, i], minallowed, maxallowed, verbose=False, reservoirName="1st reservoir")
        status2 = checkVolume(V[1, i], minallowed, maxallowed, verbose=False, reservoirName="2nd reservoir")

        for status in [status1, status2]:
            if status == 1:
                valid_count += 1
            elif status == 0:
                low_count += 1
            elif status == 2:
                high_count += 1
            elif status == -1:
                invalid_count += 1

    # Calculate stability as the inverse of variance (higher stability => lower variance)
    stability_q1 = 1 / (np.var(q1_values) + 1e-6)
    stability_V1 = 1 / (np.var(V[0, :]) + 1e-6)
    stability_V2 = 1 / (np.var(V[1, :]) + 1e-6)

    # Normalize stability values
    stability_q1_norm = stability_q1 / np.max([stability_q1, stability_V1, stability_V2])
    stability_V1_norm = stability_V1 / np.max([stability_q1, stability_V1, stability_V2])
    stability_V2_norm = stability_V2 / np.max([stability_q1, stability_V1, stability_V2])

    if optimize:
        return valid_count, low_count, high_count, invalid_count, stability_q1_norm, stability_V1_norm, stability_V2_norm

    else:
        print(f"\n{scenario_title} Scenario Results:")
        print(f"Valid volumes: {valid_count}")
        print(f"Close to empty volumes: {low_count}")
        print(f"Close to overflow volumes: {high_count}")
        print(f"Invalid volumes: {invalid_count}")
        print(f"Stability of q1: {stability_q1:.2f}")
        print(f"Stability of V1: {stability_V1:.2f}")
        print(f"Stability of V2: {stability_V2:.2f}")

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for reservoir volumes
        ax1.plot(ttt, V[0, :], 'b-', label="1st reservoir")
        ax1.plot(ttt, V[1, :], 'r-', label="2nd reservoir")
        ax1.add_patch(Rectangle((0, -0.5), T, 0.5 + minallowed, facecolor="red", alpha=0.5))
        ax1.add_patch(Rectangle((0, maxallowed), T, (1.5 - maxallowed), facecolor="red", alpha=0.5))
        ax1.set_xlim(0, T)
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid()
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Volume')
        ax1.legend()
        ax1.set_title('Reservoir Volumes')

        # Plot for flow rates
        ax2.plot(ttt, q1_values, label="q1 - outflow (controlled)")
        ax2.plot(ttt, Q_values, label="Q1 - inflow")
        ax2.plot(ttt, r_values, label="r - additional sources to 2nd")
        ax2.plot(ttt, q2_values, label="q2 - outflow from 2nd")
        ax2.set_xlim(0, T)
        ax2.grid()
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Units')
        ax2.legend()
        ax2.set_title('Flow Rates')

        plt.tight_layout()
        plt.show()


def main():
    state_dim = 3
    action_dim = 1
    max_action = 1.0

    mode = input(
        "Enter 'test' for testing, 'optimize' for genetic optimization or 'learn' for RL training: ").strip().lower()

    if mode == "optimize":
        from optimization import genetic_algorithm  # Import here to avoid circular dependency
        # best_params_drought = genetic_algorithm(num_generations=15, population_size=500, scenario="drought")
        # np.save("best_params_drought.npy", best_params_drought)
        best_params_normal = genetic_algorithm(num_generations=50, population_size=1000, scenario="normal")
        np.save("best_params_normal.npy", best_params_normal)
        best_params_rainy = genetic_algorithm(num_generations=50, population_size=1000, scenario="rainy")
        np.save("best_params_rainy.npy", best_params_rainy)
    elif mode == "test":
        test_mode = input("Enter 'pid' for PID controller testing or 'rl' for RL agent testing: ").strip().lower()
        if test_mode == "pid":
            best_params_normal = np.load("best_params_normal.npy")
            best_params_drought = np.load("best_params_drought.npy")
            best_params_rainy = np.load("best_params_rainy.npy")
            run_simulation(Q_normal, r_normal, "Normal Weather", params=best_params_normal)
            run_simulation(Q_drought, r_drought, "Drought Weather", params=best_params_drought)
            run_simulation(Q_rainy, r_rainy, "Rainy Weather", params=best_params_rainy)
        elif test_mode == "rl":
            from rl import ActorCriticAgent
            agent = ActorCriticAgent(state_dim, action_dim, max_action)
            agent.actor.load_state_dict(torch.load("actor_weights.pth"))
            agent.critic.load_state_dict(torch.load("critic_weights.pth"))
            run_simulation(Q_normal, r_normal, "Normal Weather", use_rl=True, rl_model=agent)
            run_simulation(Q_drought, r_drought, "Drought Weather", use_rl=True, rl_model=agent)
            run_simulation(Q_rainy, r_rainy, "Rainy Weather", use_rl=True, rl_model=agent)
    elif mode == "learn":
        from rl import ActorCriticAgent, train_actor_critic_agent
        agent = ActorCriticAgent(state_dim, action_dim, max_action)
        train_actor_critic_agent(agent, Q_normal, r_normal, episodes=700)
        torch.save(agent.actor.state_dict(), "actor_weights.pth")
        torch.save(agent.critic.state_dict(), "critic_weights.pth")
        run_simulation(Q_normal, r_normal, "Normal Weather with RL", use_rl=True, rl_model=agent)
        run_simulation(Q_drought, r_drought, "Drought Weather with RL", use_rl=True, rl_model=agent)
        run_simulation(Q_rainy, r_rainy, "Rainy Weather with RL", use_rl=True, rl_model=agent)
    else:
        print("Invalid mode selected. Please choose 'test', 'optimize' or 'learn'.")


if __name__ == "__main__":
    main()
