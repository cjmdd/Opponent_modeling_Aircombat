import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config):
        self.config = config
        # inner variables
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        self.pre_rewards = defaultdict(float)
        self.reward_trajectory = defaultdict(list)
        self.reward_item_names = [self.__class__.__name__]

    def reset(self, task, env):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        info = dict(current_step=0, rew_heading=0, rew_pitch=0, rew_roll=0, rew_dist=0, rew_airspeed=0,
                    rew_num_fire_misilles=0, rew_hit_missiles=0, rew_hitted_missiles=0, rew_crash=0, rew_allycol=0,
                    rew_oppocol=0,
                    rew_proximity=0)
        extra_info = dict(num_collisions=0)
        self.infos = [dict(rewards=info, episode_extra_stats=extra_info)] * self.config.num_agents
        if self.is_potential:
            self.pre_rewards.clear()
            for ind,agent_id in enumerate(env.agents.keys()):
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id,self.infos[ind])
        self.reward_trajectory.clear()

    @abstractmethod
    def get_reward(self, task, env, agent_id,info):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id, render_items=()):
        """Process reward and inner variables.

        Args:
            new_reward (float)
            agent_id (str)
            render_items (tuple, optional): Must set if `len(reward_item_names)>1`. Defaults to None.

        Returns:
            [type]: [description]
        """
        reward = new_reward * self.reward_scale
        if self.is_potential:
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward

    def get_reward_trajectory(self):
        """Get all the reward history of current episode.py

        Returns:
            (dict): {reward_name(str): reward_trajectory(np.array)}
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory.values()).transpose(2, 0, 1)))
