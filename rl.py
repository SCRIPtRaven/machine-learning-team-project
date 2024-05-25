import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from main import q2t, vvdt, checkVolume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state = self.normalize_state(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return np.clip(action, 0, self.max_action)

    def normalize_state(self, state):
        return (state - np.array([0.5, 0.5, 0])) / np.array([0.5, 0.5, 1])

    def train(self, batch_size=100):
        state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(1 - done).to(device)

        # Critic update
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * self.discount * target_Q).detach()

        current_Q = self.critic(state, action)

        target_Q = target_Q.view(-1, 1)
        current_Q = current_Q.view(-1, 1)

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train_actor_critic_agent(agent, Qt_funcs, rt_funcs, episodes=1000, T=100, dt=0.1):
    cumulative_rewards = []
    fitness_scores = []
    pt = tqdm(range(episodes), unit="episode")

    for episode in pt:
        scenario_index = 0
        Qt_func = Qt_funcs[scenario_index]
        rt_func = rt_funcs[scenario_index]

        V = np.array([[0.5], [0.5]])
        minallowed = 0.1
        maxallowed = 0.9
        ttt = np.arange(0, T, dt)
        r_values = np.zeros_like(ttt)
        r_values[0] = rt_func(0, 0)

        episode_reward = 0
        prev_q1t_value = 0
        V_prev = V.copy()

        valid_count = 0
        low_count = 0
        high_count = 0
        invalid_count = 0
        q1_values = []

        for i in range(1, np.size(ttt)):
            t = ttt[i]
            state = np.array([V[0, 0], V[1, 0], r_values[i - 1]])
            action = agent.select_action(state)
            q1t_value = max(0, action[0])
            q1_values.append(q1t_value)
            rt_value = rt_func(t, r_values[i - 1])
            r_values[i] = rt_value
            q2t_value = q2t(t, V[1, 0])
            Qt_value = Qt_func(t)
            V = V + dt * vvdt(t, q1t_value, rt_value, q2t_value, Qt_value).reshape(2, 1)
            V = np.clip(V, 0, 1)

            # Volume validity check
            status1 = checkVolume(V[0, 0], minallowed, maxallowed, verbose=False, reservoirName="1st reservoir")
            status2 = checkVolume(V[1, 0], minallowed, maxallowed, verbose=False, reservoirName="2nd reservoir")
            for status in [status1, status2]:
                if status == 1:
                    valid_count += 1
                elif status == 0:
                    low_count += 1
                elif status == 2:
                    high_count += 1
                elif status == -1:
                    invalid_count += 1

            # Reward for keeping both volumes within the allowed range
            if minallowed <= V[0, 0] <= maxallowed and minallowed <= V[1, 0] <= maxallowed:
                reward = 10.0
            else:
                reward = -5.0

            # Penalize large changes in q1
            reward -= 0.1 * abs(q1t_value - prev_q1t_value)

            # Penalize instability in volumes
            reward -= 0.05 * abs(V[0, 0] - V_prev[0, 0])
            reward -= 0.05 * abs(V[1, 0] - V_prev[1, 0])

            episode_reward += reward
            prev_q1t_value = q1t_value
            V_prev = V.copy()

            next_state = np.array([V[0, 0], V[1, 0], r_values[i]])
            done = 1 if i == np.size(ttt) - 1 else 0
            agent.replay_buffer.add((state, next_state, action, reward, done))

            if len(agent.replay_buffer.storage) > 1000:
                agent.train()

        fitness = (valid_count - (low_count + high_count + invalid_count))
        fitness_scores.append(fitness)

        cumulative_rewards.append(episode_reward)
        avg_fitness = sum(fitness_scores) / (episode + 1)
        pt.set_description(
            f"Cumulative reward: {round(sum(cumulative_rewards) / (episode + 1), 2)} | Current reward: {round(episode_reward, 2)} | Avg fitness: {round(avg_fitness, 2)}")

    # Plot cumulative reward trend
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Trend During Training')
    plt.grid(True)

    # Plot fitness score trend
    plt.subplot(1, 2, 2)
    plt.plot(fitness_scores)
    plt.xlabel('Episode')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Score Trend During Training')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
