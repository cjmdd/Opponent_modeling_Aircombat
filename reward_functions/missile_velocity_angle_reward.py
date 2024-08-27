import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R_V_angle

class MissileVelocityReward(BaseRewardFunction):
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
        aircraft_v = env.agents[agent_id].get_velocity()
        aircraft_p = env.agents[agent_id].get_position()
        ego_feature = np.hstack([aircraft_p,
                                 env.agents[agent_id].get_velocity()])
        if len(missiles_sim)!=0:
            for missile_sim in missiles_sim:
                missile_v = missile_sim.get_velocity()
                if missile_sim.is_alive:
                    if self.previous_missile_v is None:
                        self.previous_missile_v=missile_v
                    v_decrease=(np.linalg.norm(self.previous_missile_v)-np.linalg.norm(missile_v))/340
                    angle=np.dot(missile_v,aircraft_v)/(np.linalg.norm(missile_v)*np.linalg.norm(aircraft_v))
                    if angle<0:
                        reward+=angle/(max(v_decrease,0)+1)
                    else:
                        reward+=angle*max(v_decrease,0)
        info["rewards"]["rew_position_missile"] = reward
        self.reward_trajectory[agent_id].append([reward])
        return self._process(reward, agent_id)