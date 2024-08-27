import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R_V_angle

class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None
        self.target_dist=3
    def reset(self, task, env):
        self.previous_missile_v = None
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): rewards
        """
        reward = 0
        missiles_sim = env.agents[agent_id].check_missile_warning2()
        # aircraft_v = env.agents[agent_id].get_velocity()
        aircraft_p = env.agents[agent_id].get_position()
        ego_feature = np.hstack([aircraft_p,
                                 env.agents[agent_id].get_velocity()])
        if len(missiles_sim)!=0:
            for missile_sim in missiles_sim:
                # missile_v = missile_sim.get_velocity()
                if missile_sim.is_alive:
                    missile_p = missile_sim.get_position()
                    enm_feature = np.hstack([missile_p,
                                             missile_sim.get_velocity()])
                    enm_x, enm_y, enm_z = missile_p[0], missile_p[1], missile_p[2]
                    ego_x, ego_y, ego_z = aircraft_p
                    # delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
                    # R = np.linalg.norm([delta_x, delta_y, delta_z])/1000
                    AO, TA, R, delta_v, angle = get_AO_TA_R_V_angle(ego_feature, enm_feature)
                    R = R / 10000 # 6000
                    a = (AO + TA) / (2 * np.pi)
                    # if a <= 0.75:
                    #     reward -= max(-10 / self.target_dist * R + 10, 0)
                    # elif a > 0.75:
                    #     reward += 0.1 * R
                    reward -= max(-10 / self.target_dist * R + 10, 0)
        info["rewards"]["rew_position_missile"] = reward
        self.reward_trajectory[agent_id].append([reward])
        return self._process(reward, agent_id)
