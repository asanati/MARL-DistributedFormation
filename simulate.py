import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import wandb

class FormationSimulator:
    """
    simulate the formation within a single 2D environment
    """
    def __init__(self, num_agents=10, num_obstacles=0, share_reward_ratio=0.25, goal_in_obs=True, visualize=True, log=True):
        self.visualize = visualize
        self.width = 400
        self.height = 600

        assert num_obstacles == 0, "obstacle collision not working"
        self.num_obstacles = num_obstacles
        self.num_agents = num_agents
        self.obstacle_size = 10
        self.max_steps = 1000

        self.log = log

        # agents should form a regular polygon
        self.desired_radius = 60
        self.desired_neighbor_dist = 2 * self.desired_radius * np.sin(np.pi / self.num_agents)
        # how much reward to share between neighbors
        assert 0 <= share_reward_ratio <= 0.5
        self.share_reward_ratio = share_reward_ratio

        self.goal_in_obs = goal_in_obs

        if self.visualize:
            # set up window
            self.fig = plt.figure(figsize=(self.width/100, self.height/100))
            self.ax = self.fig.add_subplot(111)
            visualize_margin = 10
            self.ax.set_xlim(-visualize_margin, self.width + visualize_margin)
            self.ax.set_ylim(-visualize_margin, self.height + visualize_margin)
            # draw edges of environment
            self.ax.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0], color='black')
            
            # add obstacles, agents, and goal to window with placeholder positions
            self.agents_viz = []
            self.agents_lines_viz = []
            self.obstacles_viz = []
            for i in range(self.num_agents):
                agent_viz = plt.Circle((0, 0), radius=2, color='blue')
                self.agents_viz.append(agent_viz)
                self.ax.add_artist(agent_viz)
                agent_line_viz = plt.Line2D([0, 0], [0, 0], color='blue', linewidth=0.2)
                self.agents_lines_viz.append(agent_line_viz)
                self.ax.add_artist(agent_line_viz)
            for i in range(self.num_obstacles):
                obstacle_viz = plt.Rectangle((0, 0), width=2*self.obstacle_size, height=2*self.obstacle_size, color='green')
                self.obstacles_viz.append(obstacle_viz)
                self.ax.add_artist(obstacle_viz)
            self.goal_viz = plt.Circle((0, 0), radius=10, color='red')
            self.ax.add_artist(self.goal_viz)

        self.reset()
    
    def visualize_agents(self):
        for agent, agent_viz in zip(self.agents, self.agents_viz):
            agent_viz.center = (agent[0], agent[1])
        for agent, next_agent, agent_line_viz in zip(self.agents, np.roll(self.agents, -1, axis=0), self.agents_lines_viz):
            agent_line_viz.set_data([agent[0], next_agent[0]], [agent[1], next_agent[1]])


    def step(self, input_velocity):
        """
        input the velocity for each agent
        returns:
            obs: observation for each agent
            reward: reward for each agent
            done: whether the entire formation is done
            info: additional info
        """
        assert input_velocity.shape == self.agents.shape

        # apply the action to each agent
        self.agents += input_velocity
        
        # use pytorch for detection and application of collision, for speed (versus using for loops)
        # Check if any agent is outside the allowed width and height, then reverse its velocity
        self.out_of_bounds = ((self.agents[:, 0] <= 0) | (self.agents[:, 1] <= 0) |
                         (self.agents[:, 0] >= self.width) | (self.agents[:, 1] >= self.height))
        # clip agents to be within bounds
        self.agents[:, 0] = torch.clip(self.agents[:, 0], 0, self.width)
        self.agents[:, 1] = torch.clip(self.agents[:, 1], 0, self.height)

        # TODO: implement better logic for when it bumps into obstacles
        # First let's reshape self.obstacles for easier computation later
        obstacles = self.obstacles.view(-1, 1, 2)
        # Now let's calculate if any agent is inside any obstacle
        is_in_obstacle_matrix = ((obstacles <= self.agents) & (self.agents <= obstacles + self.obstacle_size)).all(dim=-1)
        # row i, column j of is_in_obstacle_matrix is True if agent j is inside obstacle i
        self.is_in_obstacle = is_in_obstacle_matrix.any(dim=0)
        # We are checking if both the x and y coordinates of the agent are within the obstacle boundaries

        if self.visualize:
            for i, obstacle in enumerate(self.obstacles):
                if is_in_obstacle_matrix[i].any():
                    self.obstacles_viz[i].color = (255, 0, 0)
                else:
                    self.obstacles_viz[i].color = (0, 255, 0)
            self.visualize_agents()

        # calculate reward and check if done
        reward, done = self.compute_reward_and_done()
        self.steps_since_reset += 1

        if done:
            # as mentioned in https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
            # Thus, the observation returned when done is true will in fact be the first observation of the next episode
            self.reset()
        self.compute_metrics()
        return self.compute_obs(), reward, done, {}
    
    def reset(self):
        """
        reset the positions of agents, obstacles, and goal
        """
        # set obstacles - bottom 100 px and top 100 px should not have obstacles
        self.obstacles = torch.rand((self.num_obstacles, 2))
        self.obstacles[:, 0] = self.obstacles[:, 0] * (self.width - 2 * self.obstacle_size) + self.obstacle_size
        self.obstacles[:, 1] = self.obstacles[:, 1] * (self.height - 200 - 2 * self.obstacle_size) + 100 + self.obstacle_size
        if self.visualize:
            for obstacle, obstacle_viz in zip(self.obstacles, self.obstacles_viz):
                obstacle_viz.xy = (obstacle[0] - self.obstacle_size, obstacle[1] - self.obstacle_size)

        # set agents - place them in the bottom 100 px
        self.agents = torch.rand((self.num_agents, 2))
        self.agents[:, 0] = self.agents[:, 0] * self.width
        self.agents[:, 1] = self.agents[:, 1] * 100
        if self.visualize:
            self.visualize_agents()

        # set goal postion
        self.goal = torch.rand(2)
        # there should be at least self.desired_radius distance from the wall
        self.goal[0] = self.goal[0] * (self.width - 2 * self.desired_radius) + self.desired_radius
        self.goal[1] = self.goal[1] * (self.height - 2 * self.desired_radius) + self.desired_radius
        if self.visualize:
            self.goal_viz.center = (self.goal[0], self.goal[1])
        
        self.steps_since_reset = 0

    
    def compute_obs(self):
        """
        compute observation for each agent
        first elements are agent position and last elements are goal position
        """
        # Normalize agent positions
        normalized_agents = self.agents / torch.tensor([self.width, self.height])

        num_agents = self.num_agents
        obs = torch.zeros((num_agents, 6))

        # Compute modified observations taking into account the neighboring agents
        for i in range(num_agents):
            prev_agent = (i - 1) % num_agents
            next_agent = (i + 1) % num_agents
            obs[i, :2] = normalized_agents[i]
            obs[i, 2:4] = normalized_agents[prev_agent] - normalized_agents[i]
            obs[i, 4:6] = normalized_agents[next_agent] - normalized_agents[i]

        if self.goal_in_obs:
            # append relative goal position to every agent's observation
            # Normalize goal position
            normalized_relative_goal = (self.goal - self.agents) / torch.tensor([self.width, self.height])
            obs = torch.cat((obs, normalized_relative_goal), dim=1)
        return obs

    def compute_reward_and_done(self):
        num_agents = self.num_agents

        # distance of each agent to goal
        dist_to_goal = torch.linalg.norm(self.agents - self.goal, dim=1)

        # Compute conditions for each agent individually
        close_to_goal = dist_to_goal < 100
        reward_dist_scale = 0.1

        close_to_goal_bonus = 10.
        close_to_goal_reward = close_to_goal_bonus * close_to_goal
        if self.log:
            wandb.log({"close_to_goal_reward": close_to_goal_reward.mean().item()})
        
        reward_dist = -reward_dist_scale*dist_to_goal
        if self.log:
            wandb.log({"reward_dist": reward_dist.mean().item()})

        # the distance between neighbors in the formation, when they form a circle
        # compute distance of each agent to its neighbors
        dist_to_right_neighbor = torch.linalg.norm(self.agents - torch.roll(self.agents, -1, dims=0), dim=1)
        dist_to_left_neighbor = torch.linalg.norm(self.agents - torch.roll(self.agents, 1, dims=0), dim=1)

        # compute the potential
        neighbor_dist_penalty_scale = 0.01
        right_dist_diff = dist_to_right_neighbor - self.desired_neighbor_dist
        left_dist_diff = dist_to_left_neighbor - self.desired_neighbor_dist
        reward_right_neighbor = - neighbor_dist_penalty_scale * torch.where(right_dist_diff < 0, right_dist_diff ** 2, right_dist_diff)
        reward_left_neighbor = - neighbor_dist_penalty_scale * torch.where(left_dist_diff < 0, left_dist_diff ** 2, left_dist_diff)
        if self.log:
            wandb.log({"reward_right_neighbor": reward_right_neighbor.mean().item()})
            wandb.log({"reward_left_neighbor": reward_left_neighbor.mean().item()})

        # Compute individual rewards
        individual_rewards = reward_dist + close_to_goal_reward + reward_right_neighbor + reward_left_neighbor

        # Penalty for going out of bounds and hitting an obstacle
        out_of_bounds_penalty = -100.0 * self.out_of_bounds
        obstacle_penalty = -100.0 * self.is_in_obstacle

        individual_rewards += out_of_bounds_penalty + obstacle_penalty

        # Create a new tensor to hold the modified rewards
        modified_rewards = torch.zeros_like(individual_rewards)

        # Compute modified rewards taking into account the neighboring agents
        for i in range(num_agents):
            prev_agent = (i - 1) % num_agents
            next_agent = (i + 1) % num_agents

            # Include the penalties and bonuses from neighboring agents
            modified_rewards[i] = (1. - 2 * self.share_reward_ratio) * individual_rewards[i] \
                                  + self.share_reward_ratio * (individual_rewards[prev_agent] + individual_rewards[next_agent])

        timeout = self.steps_since_reset > self.max_steps

        # done = close_to_goal.all() or timeout
        done = timeout

        return modified_rewards, done
        
    def compute_metrics(self):
        """
        does not affect the environment, just calculate metrics for checking performance
        """
        # calculate average distance to goal
        dist_to_goal = torch.linalg.norm(self.agents - self.goal, dim=1)
        avg_dist_to_goal = dist_to_goal.mean().item()

        # calculate average distance to neighbors
        dist_to_right_neighbor = torch.linalg.norm(self.agents - torch.roll(self.agents, -1, dims=0), dim=1)
        ave_dist_to_right_neighbor = dist_to_right_neighbor.mean().item()
        std_dist_to_right_neighbor = dist_to_right_neighbor.std().item()

        if self.log:
            wandb.log({"avg_dist_to_goal": avg_dist_to_goal})
            wandb.log({"ave_dist_to_neighbor": ave_dist_to_right_neighbor})
            wandb.log({"std_dist_to_neighbor": std_dist_to_right_neighbor})

def control(i, env):
    # formation control
    num_agents = env.agents.shape[0]
    desired_radius = 40
    # compute info about neighbors on circle
    # array of all agent positions shifted by one on either direction
    agents_shiftA = torch.zeros_like(env.agents)
    agents_shiftA[:-1, :] = env.agents[1:, :]
    agents_shiftA[-1, :] = env.agents[0, :]
    agents_shiftB = torch.zeros_like(env.agents)
    agents_shiftB[1:, :] = env.agents[:-1, :]
    agents_shiftB[0, :] = env.agents[-1, :]

    # neighboring direction and distance on graph
    neighbor_distA = torch.linalg.norm(env.agents - agents_shiftA, dim=1)
    neighbor_dirA = agents_shiftA - env.agents
    neighbor_dirA /= neighbor_distA.unsqueeze(1)  # normalize
    neighbor_distB = torch.linalg.norm(env.agents - agents_shiftB, dim=1)
    neighbor_dirB = agents_shiftB - env.agents
    neighbor_dirB /= neighbor_distB.unsqueeze(1)  # normalize

    # compute info about agent that is opposite side of circle
    agents_opposite = torch.zeros_like(env.agents)
    assert num_agents % 2 == 0
    agents_opposite[:num_agents//2, :] = env.agents[num_agents//2:, :]
    agents_opposite[num_agents//2:, :] = env.agents[:num_agents//2, :]
    opposite_dist = torch.linalg.norm(env.agents - agents_opposite, dim=1)
    opposite_dir = agents_opposite - env.agents
    opposite_dir /= opposite_dist.unsqueeze(1)  # normalize

    desired_dist = np.pi * desired_radius / num_agents

    # first two terms try to keep desired_dist between neighbors
    # last term tries to keep the diameter distance between opposite agents
    f_formation = 0.02 * (neighbor_distA - desired_dist).unsqueeze(1) * neighbor_dirA \
        + 0.02 * (neighbor_distB - desired_dist).unsqueeze(1) * neighbor_dirB \
        + 0.02 * (opposite_dist - 2 * desired_radius).unsqueeze(1) * opposite_dir
    f_formation = torch.clip(f_formation, -1, 1)

    # obstacle avoidance
    f_obstacle = torch.zeros_like(f_formation)
    for obstacle in env.obstacles:
        obstacle_dir = env.agents - obstacle
        obstacle_dist = torch.linalg.norm(obstacle_dir, dim=1)
        obstacle_dir /= obstacle_dist.unsqueeze(1)

        avoid_dist = env.obstacle_size * 2  # when to start avoiding

        repel_magnitude = -0.3 * (obstacle_dist - avoid_dist)
        repel_magnitude = torch.maximum(repel_magnitude, torch.tensor(0))
        repel = repel_magnitude.unsqueeze(1) * obstacle_dir
        f_obstacle += repel

    # goal attraction
    f_goal = torch.zeros_like(f_formation)
    goal_dir = env.agents - env.goal
    goal_dist = torch.linalg.norm(goal_dir, dim=1)
    goal_dir /= goal_dist.unsqueeze(1)

    attract_magnitude = 0.01 * (goal_dist - desired_radius)
    f_goal = - attract_magnitude.unsqueeze(1) * goal_dir
    f_goal = torch.clip(f_goal, -1, 1)

    env.step(f_formation + f_obstacle + f_goal)

if __name__ == "__main__":
    steps = 1000

    env = FormationSimulator(num_agents=10, visualize=True)
    ani = animation.FuncAnimation(env.fig, control,
                                  fargs=(env,),
                                  frames=range(steps),
                                  interval=1)
    plt.show()