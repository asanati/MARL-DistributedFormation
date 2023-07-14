from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import torch
import datetime
import os
import hydra
from omegaconf import DictConfig, OmegaConf


from simulate import FormationSimulator

class FormationEnv(VecEnv):
    """
    simulate multiple FormationSimulator instances in parallel, where each agent in each FormationSimulator is its separate RL agent
    https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    TODO: make it work on simple reward first (must implement reset?)
    """
    def __init__(self, cfg, visualize=False, log=True):
        """
        num_agents_per_field: number of agents in each FormationSimulator
        num_formation: number of FormationSimulator instances
        """
        self.num_agents_per_formation = cfg.num_agents_per_formation
        if cfg.goal_in_obs:
            self.obs_dim = 8
        else:
            self.obs_dim = 6
        num_envs = self.num_agents_per_formation * cfg.num_formation

        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim,), dtype=np.float32)
        super(FormationEnv, self).__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)

        self.formationsim_list = []
        for i in range(cfg.num_formation):
            num_obstacles = 0  # use very obstacleless environment for now
            visualize_env = visualize and i == 0
            log_env = log and i == 0
            self.formationsim_list.append(FormationSimulator(num_agents=self.num_agents_per_formation, visualize=visualize_env, goal_in_obs=cfg.goal_in_obs, num_obstacles=num_obstacles, log=log_env))
        
        # create buffers
        self.obs_buf = torch.zeros((num_envs, self.obs_dim), dtype=torch.float32)
        self.reward_buf = torch.zeros(num_envs, dtype=torch.float32)
        self.done_buf = torch.zeros(num_envs, dtype=torch.bool)
        self.infos = [{} for _ in range(self.num_envs)]
        self.log = log

    def reset(self):
        for formation_sim in self.formationsim_list:
            formation_sim.reset()
        return self.compute_observations()
    
    def compute_observations(self):
        """
        concatenate observations from all FormationSimulator instances
        """
        for i, formation_sim in enumerate(self.formationsim_list):
            formation_start_idx = i*self.num_agents_per_formation
            formation_end_idx = (i+1)*self.num_agents_per_formation
            obses = formation_sim.compute_obs()
            self.obs_buf[formation_start_idx:formation_end_idx] = obses
        return self.obs_buf.numpy()

    def step(self, actions):
        max_speed = 10
        self.actions = max_speed * torch.tensor(actions, dtype=torch.float32)  # scale before sending
        for i, formation_sim in enumerate(self.formationsim_list):
            formation_start_idx = i*self.num_agents_per_formation
            formation_end_idx = (i+1)*self.num_agents_per_formation
            # send slice of actions to each FormationSimulator instance
            # done and info have the same value for all agents in the same FormationSimulator instance
            obses, rewards, done, _ = formation_sim.step(self.actions[formation_start_idx:formation_end_idx])
            self.obs_buf[formation_start_idx:formation_end_idx] = obses
            self.reward_buf[formation_start_idx:formation_end_idx] = rewards
            self.done_buf[formation_start_idx:formation_end_idx] = done
            if self.log:
                wandb.log({'reward': rewards.mean().item()})
        return self.obs_buf.numpy(), self.reward_buf.numpy(), self.done_buf.numpy(), self.infos


    # define methods that won't be called but must be overriden when inheriting from VecEnv

    def close(self):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class):
        raise NotImplementedError
    
    def get_attr(self, attr_name, indices=None):
        raise AttributeError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError


@hydra.main(config_path="cfg", config_name="config")
def run(cfg: DictConfig):
    # simple test program to check that FormationEnv works
    from stable_baselines3 import PPO
    num_steps = 5000
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run = wandb.init(project="formation-rl", name=f"{cfg.name}-{timestamp}")#, config=cfg.to_dict())
    
    env = FormationEnv(cfg)
    this_dir = hydra.utils.get_original_cwd()

    # save a checkpoint every 100 steps to cfg.name folder
    checkpoint_callback = CheckpointCallback(save_freq=10, save_path=f'{this_dir}/logs/{cfg.name}/')

    model = PPO('MlpPolicy', env,
                verbose=1, 
                n_steps=10,
                tensorboard_log="./tensorboard/",
                learning_rate=1e-3,
                ent_coef=0.01)
    # use a smaller default standard deviation, e^(-2) = 0.1353
    model.policy.log_std_init = -2
    model.learn(total_timesteps=num_steps*cfg.num_formation,
                log_interval=4,
                progress_bar=True,
                callback=checkpoint_callback)

if __name__ == "__main__":
    run()