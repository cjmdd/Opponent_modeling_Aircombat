import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R_V_angle2


class VelocityReward(BaseRewardFunction):
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
        self.angle_max = np.pi/4

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]
        self.previous_angle_ally = np.full((self.config.num_ally,self.config.num_oppo), np.pi).astype(float)
        self.previous_angle_oppo = np.full((self.config.num_oppo, self.config.num_ally), np.pi).astype(float)

    def reset(self, task, env):
        self.previous_angle_ally = np.full((self.config.num_ally, self.config.num_oppo), np.pi).astype(float)
        self.previous_angle_oppo = np.full((self.config.num_oppo, self.config.num_ally), np.pi).astype(float)
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id,info):
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
                Angle, R, delta_v, angle,AO,TA = get_AO_TA_R_V_angle2(ego_feature, enm_feature)

                a = (AO + TA) / (2 * np.pi)
                a1=np.cos(Angle)
                dd = (self.angle_max - abs(angle)) / self.angle_max
                
                if a1 > 0 and delta_v <= 1:
                    new_reward += np.exp((0.8 + dd)) * (2 - delta_v)
                elif a1 > 0 and delta_v > 1:
                    # new_reward -= np.exp(-(0.5 - dd)) * (0.5 + delta_v)
                    if (2 - delta_v) > 0:
                        new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                    else:
                        new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                elif a1 < 0 and delta_v > 1:
                    if a <= 0.25:
                        # if id_agent < self.config.num_ally:
                        #     new_reward +=  5*(angle - self.previous_angle_ally[id_ally][id_enem])
                        # else:
                        #     new_reward +=  5*(angle - self.previous_angle_oppo[id_oppo][id_enem])
                        if (2 - delta_v) > 0:
                            # new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                            new_reward += np.exp((0.8 + dd)) * (2 - delta_v)
                        else:
                            new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                    else:
                        new_reward += 5 * (1 - abs(AO) / self.angle_max)
                        # if (2 - delta_v) > 0:
                        #     new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                        # else:
                        #     new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                else:
                    if a > 0.75:
                        new_reward += np.exp((0.8 - dd)) * (2 - delta_v)

                    else:
                        # new_reward += 5*(1 - abs(angle) / self.angle_max)
                        new_reward += 5 * (1 - abs(AO) / self.angle_max)
                # if a1>0 and delta_v<=1:
                #     new_reward += np.exp((0.8 + dd)) * (2 - delta_v)
                # elif a1>0 and delta_v>1:
                #     # new_reward -= np.exp(-(0.5 - dd)) * (0.5 + delta_v)
                #     if (2 - delta_v) > 0:
                #         new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                #     else:
                #         new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                # elif a1<0 and delta_v>1:
                #     if a<=0.25:
                #         # if id_agent < self.config.num_ally:
                #         #     new_reward +=  5*(angle - self.previous_angle_ally[id_ally][id_enem])
                #         # else:
                #         #     new_reward +=  5*(angle - self.previous_angle_oppo[id_oppo][id_enem])
                #         if (2 - delta_v) > 0:
                #             # new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                #             new_reward += np.exp((0.8 + dd)) * (2 - delta_v)
                #         else:
                #             new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                #     else:
                #         new_reward+=5*(1-abs(AO)/self.angle_max)
                #         # if (2 - delta_v) > 0:
                #         #     new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                #         # else:
                #         #     new_reward += np.exp(-(0.8 - dd)) * (2 - delta_v)
                # else:
                #     if a>0.75:
                #         new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                #
                #     else:
                #         # new_reward += 5*(1 - abs(angle) / self.angle_max)
                #         new_reward += 5 * (1 - abs(AO) / self.angle_max)
                        # new_reward += np.exp((0.8 - dd)) * (2 - delta_v)
                    # if id_agent < self.config.num_ally:
                    #     new_reward += 10 * (angle - self.previous_angle_ally[id_ally, id_enem])
                    # else:
                    #     new_reward += 10 * (angle - self.previous_angle_oppo[id_oppo, id_enem])
                if id_agent < self.config.num_ally:
                    self.previous_angle_ally[id_ally, id_enem] = angle
                else:
                    self.previous_angle_oppo[id_oppo, id_enem] = angle
                # a=angle -self.previous_angle_oppo[0,0]
        # print(f"bbbbbbbbbb_{agent_id}", new_reward)
        info["rewards"]["rew_velocity_angle"] = new_reward #-0.18
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

