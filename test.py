import numpy as np
from env.traffic_env import TrafficIntersectionEnv
from agent.dqn_agent import DQNAgent

def test():
    env = TrafficIntersectionEnv(arrival_rate=0.4, max_queue=10, phase_duration=5)
    state_dim = 9
    action_dim = 2

    agent = DQNAgent(state_dim, action_dim)
    agent.load("dqn_traffic.pth")

    num_episodes = 10
    max_steps_per_episode = 50

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        print(f"--- Episode {episode} ---")
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            action_str = "NS_green" if action == 0 else "EW_green"
            state, reward, done, _ = env.step(action=action_str)
            total_reward += reward
            env.render()
            if done:
                break
        print(f"Total reward this episode: {total_reward:.2f}")

if __name__ == "__main__":
    test()
