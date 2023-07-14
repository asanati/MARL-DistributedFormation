import sys
import os
from stable_baselines3 import PPO

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from vectorized_env import FormationEnv

import hydra

def simulate_func(i):
    # run simulation
    global obs
    print('-'*10)
    print(f"Step {i}")
    actions, _states = model.predict(obs, deterministic=True)
    print(f"actions: {actions}")
    obs, rewards, dones, info = env.step(actions)
    print(f"obs: {obs}")
    print(f"rewards: {rewards}")
    print(f"dones: {dones}")

@hydra.main(config_path="cfg", config_name="config")
def run(cfg):
    # get checkpoint path from command line argument

    # load model from checkpoint
    # find the checkpoint with the largest number in the filename, named "rl_model_*_steps.zip"
    this_dir = hydra.utils.get_original_cwd()
    checkpoint_dir = this_dir + "/logs/" + cfg.name
    checkpoint_file = max([f for f in os.listdir(checkpoint_dir) if "rl_model" in f], key=lambda x: int(x.split("_")[-2].split(".")[0]))
    checkpoint_path = checkpoint_dir + "/" + checkpoint_file
    print(f"Loading model from {checkpoint_path}")
    global model, env, obs
    model = PPO.load(checkpoint_path)
    cfg.num_formation = 1  # override
    env = FormationEnv(cfg, visualize=True, log=False)
    first_env = env.formationsim_list[0]
    obs = env.reset()

    steps_to_simulate = 1000
    # Create an animation, updating every 100ms (adjust as needed)
    ani = animation.FuncAnimation(first_env.fig,
                                simulate_func,
                                frames=range(steps_to_simulate),
                                interval=200)

    plt.show()

if __name__ == "__main__":
    run()