import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R2


class PostureReward(BaseRewardFunction):
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
        self.previous_dist_ally = np.full((self.config.num_ally, self.config.num_oppo), 16630).astype(
            float)  # lon 120,lat 60, lon120, lat60.15
        self.previous_dist_oppo = np.full((self.config.num_oppo, self.config.num_ally), 16630).astype(float)

    def reset(self, task, env):
        self.previous_dist_ally = np.full((self.config.num_ally, self.config.num_oppo),
                                          16630).astype(float)  # lon 120,lat 60, lon120, lat60.15
        self.previous_dist_oppo = np.full((self.config.num_oppo, self.config.num_ally), 16630).astype(float)
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id, info):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        id_agent = np.where(np.array(env.total_ids) == agent_id)[0][0]  # or env.total_ids.index(agent_id)

        if id_agent < self.config.num_ally:
            # id_ally=np.where(agent_id in np.array(env.ego_ids))[0]
            id_ally = np.where(np.array(env.ego_ids) == agent_id)[0][0]
        else:
            # id_oppo=np.where(agent_id in np.array(env.enm_ids))[0]
            id_oppo = np.where(np.array(env.enm_ids) == agent_id)[0][0]
        new_reward = 0
        orientation_reward = 0
        range_reward = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            if id_agent < self.config.num_ally:
                id_enem = np.where(np.array(env.enm_ids) == enm.uid)[0][0]
            else:
                id_enem = np.where(np.array(env.ego_ids) == enm.uid)[0][0]
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                AO, TA, angle, R, ego_v, enm_v = get_AO_TA_R2(ego_feature, enm_feature)
                # orientation_reward = self.orientation_fn(AO, TA)
                range_reward = self.range_fn(R / 1000)
                # new_reward+=orientation_reward*range_reward
                a = (AO + TA) / (2 * np.pi)
                a1 = np.cos(angle)
                dd = (self.target_dist - R / 10000) / self.target_dist
                # if a < 0.55:
                if a < 0.55:
                    # new_reward += np.exp(-(0.7 - dd)) * (5 - a)
                    new_reward += np.exp(0.8 + dd) * (8 - 8*a)
                else:
                    new_reward += np.exp((0.8 - dd)) * (8 - 8*a)
                    # new_reward -= np.exp(-(0.7 - 0.1*dd)) * (5 - a)
                    # new_reward += np.exp((0.8 - dd)) * (1+a)
                # elif a1<0 and ego_v<enm_v:
                #     if id_agent<self.config.num_ally:
                #         # print('iddddd',id_ally)
                #         # print('pppppprially', self.previous_dist_ally)
                #         new_reward -= 10 * (R - self.previous_dist_ally[id_ally][id_enem])
                #     else:
                #         # print('idddddooop', id_oppo)
                #         # print('pppppooooppp', self.previous_dist_oppo)
                #         new_reward -= 10 * (R - self.previous_dist_oppo[id_oppo][id_enem])
                # else:
                #     new_reward += np.exp((0.5 - dd)) * (5 - a)

                if id_agent < self.config.num_ally:
                    self.previous_dist_ally[id_ally, id_enem] = R
                else:
                    self.previous_dist_ally[id_oppo, id_enem] = R

        info["rewards"]["rew_position_attack"] = new_reward  # 0.6,-0.22,15
        return self._process(new_reward, agent_id)

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
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (
                        1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R ** 2 + 0.284 * R + 0.38, 0, 1) + np.clip(
                np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
