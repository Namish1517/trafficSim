import numpy as np
from env.traffic_env import TrafficIntersectionEnv
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer

def train():
    env = TrafficIntersectionEnv(arrival_rate=0.4, max_queue=10, phase_duration=5)
    state_dim = 9  # 8 queue counts + 1 traffic light state
    action_dim = 2  # 0: NS_green, 1: EW_green

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(10000)

    num_episodes = 300
    batch_size = 64
    target_update_freq = 10
    max_steps_per_episode = 50

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            action_str = "NS_green" if action == 0 else "EW_green"
            next_state, reward, done, _ = env.step(action=action_str)
            replay_buffer.push(state, action, reward, next_state, done)

            agent.optimize_model(replay_buffer, batch_size)
            state = next_state
           
