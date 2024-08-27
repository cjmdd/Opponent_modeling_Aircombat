import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class CollisionReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward=0
        allcol_reward=0
        oppocol_reward = 0
        num_collisions=0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = env.agents[agent_id].get_position()
        ego_x, ego_y, ego_z = ego_feature
        for ally in env.agents[agent_id].partners:
            ally_pos = ally.get_position()
            enm_x,enm_y,enm_z=ally_pos[0],ally_pos[1],ally_pos[2]
            delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
            R = np.linalg.norm([delta_x, delta_y, delta_z])/5
            if R<3:
                reward-=5
                allcol_reward-=5
                num_collisions+=1
        for oppo in env.agents[agent_id].enemies:
            oppo_pos = oppo.get_position()
            enm_x,enm_y,enm_z=oppo_pos[0],oppo_pos[1],oppo_pos[2]
            delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
            R = np.linalg.norm([delta_x, delta_y, delta_z])/10
            if R<3:
                reward-=5
                oppocol_reward-=5
                num_collisions += 1
        info["rewards"]["rew_allycol"] = allcol_reward
        info["rewards"]["rew_oppocol"] = oppocol_reward
        info["episode_extra_stats"]["num_collisions"]=num_collisions
        return self._process(reward, agent_id)

    def get_orientation_function(self, version):
        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
