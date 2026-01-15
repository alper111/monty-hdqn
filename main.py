import torch
from gym_montezuma.envs.montezuma_env import MontezumasRevengeEnv

from dqn_agent import DQNAgent


if __name__ == "__main__":
    UPDATE_FREQ = 1
    MAX_ITER = 10000000

    env = MontezumasRevengeEnv(single_life=False,
                               single_screen=False,
                               render_mode="rgb_array",
                               observation_mode="privileged",
                               max_timesteps=1000)

    network = torch.nn.Sequential(
        torch.nn.Linear(50, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 27)
    )

    agent = DQNAgent(
        network=network,
        device="mps",
        lr=0.0001,
        batch_size=128,
        target_update_iter=10000,
        save_folder="save/agent1",
        resume=False
    )
    it = 0
    episode = 0

    while agent.iter < MAX_ITER:
        obs, info = env.reset()
        obs = obs / 255.0
        done = False
        total_reward = 0
        t = 0
        loss = 0.0
        while not done:
            init_vec = env.available_mask
            action = agent.action(obs, init_vec)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs / 255.0
            done = terminated or truncated
            next_init_vec = env.available_mask
            total_reward += reward
            t += 1

            agent.record(obs, action, reward, next_obs, terminated, init_vec, next_init_vec)
            it += 1

            if it % UPDATE_FREQ == 0:
                loss += agent.update()
                it = 0

            obs = next_obs

        episode += 1
        loss /= t
        if episode % 10 == 0:
            agent.save()
        print(f"Episode: {episode:,}, reward={total_reward:.2f}, loss={loss:.5f}, t={t}, "
              f"eps={agent.epsilon:.3f}, iter={agent.iter:,}, buffer_size={len(agent.buffer)}")

    env.close()
