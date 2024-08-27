import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R_V_angle
from ..core.catalog import Catalog as c
class MissilePotentialReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None
        self.angle_max = np.pi / 4
        self.target_dist=3
        self.previous_v = np.full(self.config.num_agents, 242.67550364687295)
        self.previous_altitude = np.full((self.config.num_agents), 6096)
        self.previous_pitch = np.zeros(self.config.num_agents)


    def reset(self, task, env):
        self.previous_missile_v = None
        self.previous_v = np.full(self.config.num_agents, 242.67550364687295)
        self.previous_altitude = np.full((self.config.num_agents), 6096)
        self.previous_pitch = np.zeros(self.config.num_agents)
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
        dive_reward_pitch = 0
        dive_reward_v = 0
        dive_reward_alti = 0
        dive_reward_dist = 0
        id_agent = np.where(np.array(env.total_ids) == agent_id)[0][0]
        elevator = env.agents[agent_id].get_property_value(c.fcs_elevator_cmd_norm)
        h = env.agents[agent_id].get_position()[-1]
        # feature: (north, east, down, vn, ve, vd)
        pitch_angle = env.agents[agent_id].get_property_value(c.attitude_pitch_rad)


        ego_features = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        ego_feature = env.agents[agent_id].get_position()
        ego_v0=env.agents[agent_id].get_velocity()
        ego_v=np.linalg.norm(ego_v0)
        missiles_sim = env.agents[agent_id].check_missile_warning2()
        if len(missiles_sim)!=0:
            for missile_sim in missiles_sim:

                if missile_sim.is_alive:
                    enm_features = np.hstack([missile_sim.get_position(),
                                              missile_sim.get_velocity()])
                    enm_feature = missile_sim.get_position()
                    missile_v0 = missile_sim.get_velocity()
                    missile_v = np.linalg.norm(missile_v0)
                    delta_h = ego_feature[2] - enm_feature[2]
                    AO, TA, R, delta_v, angle = get_AO_TA_R_V_angle(ego_features, enm_features)
                    dd = (self.target_dist - delta_h/5000 ) / self.target_dist #6000
                    a = (AO + TA) / (2 * np.pi)
                    # if a <= 0.75:
                    #     reward -= np.exp(-(0.5 - dd))
                    # elif a > 0.75:
                    #     reward += np.exp((0.5 - dd)) * (2 - missile_v / ego_v)
                    if (2 - missile_v / ego_v) > 0:
                        reward += np.exp((0.7 - dd)) * (2 - missile_v / ego_v)
                    else:
                        reward += np.exp(-(0.7 - dd)) * (2 - missile_v / ego_v)
                    # if delta_h>0: # climb
                    #     if (2 - missile_v / ego_v) > 0:
                    #         reward += np.exp((0.7 - dd)) * (2 - missile_v / ego_v)
                    #     else:
                    #         # reward += np.exp(-(0.7 - dd)) * (2 - missile_v / ego_v)
                    #         reward += np.exp(-(0.7 - dd)) * (2 - missile_v / ego_v)
                    #     # print("delllllltah",delta_h)
                    # else: #dive
                    #     # if elevator>0:
                    #     #     reward-=0.05
                    #     # if elevator < 0 and pitch_angle < -5 / 180 * np.pi:  # dive
                    #     #     dive_reward_pitch += max(0, (self.previous_pitch[
                    #     #                                      id_agent] - pitch_angle))  # or abs(min(0,pitch_angle-self.previous_pitch[id_agent]))
                    #     # if elevator < 0:
                    #     #     dive_reward_v += max(0, (ego_v - self.previous_v[id_agent]) )
                    #     #     dive_reward_alti += max(0, (self.previous_altitude[id_agent] - h)/h )
                    #     #     if self.previous_altitude[id_agent] - h<0:
                    #     #         reward-=0.05
                    #     if elevator > 0:
                    #         reward-=0.1
                    #     if elevator<0 or pitch_angle<0:  # dive and pitch_angle < -5 / 180 * np.pi
                    #         if self.previous_pitch[id_agent] - pitch_angle > 0:
                    #             dive_reward_pitch += 0.2
                    #         else:
                    #             dive_reward_pitch-=0.2
                    #         # dive_reward_pitch += max(0, (self.previous_pitch[
                    #         #                                  id_agent] - pitch_angle))  # or abs(min(0,pitch_angle-self.previous_pitch[id_agent]))
                    #     # if elevator < 0:
                    #         if ego_v - self.previous_v[id_agent] > 0:
                    #             dive_reward_v += 0.2
                    #         else:
                    #             dive_reward_v-=0.2
                    #         # dive_reward_v+=max(0,(ego_v-self.previous_v[id_agent]))
                    #         if self.previous_altitude[id_agent] - h > 0:
                    #             dive_reward_alti += 0.2
                    #         else:
                    #             dive_reward_alti-=0.2




        else:
            self.previous_missile_v = None
            reward = 0
        # if dive_reward_v!=0 or dive_reward_pitch!=0 or dive_reward_alti!=0:
        #     reward = reward * 0.2 + dive_reward_v * 0.35 + dive_reward_alti * 0.25 + dive_reward_pitch * 0.2
        # reward = reward  + dive_reward_v + dive_reward_alti + dive_reward_pitch
        self.reward_trajectory[agent_id].append([reward]) #
        self.previous_v[id_agent] = ego_v
        self.previous_pitch[id_agent] = pitch_angle
        self.previous_altitude[id_agent] = h
        info["rewards"]["rew_potential_energy_missile"] = reward
        return self._process(reward, agent_id)
