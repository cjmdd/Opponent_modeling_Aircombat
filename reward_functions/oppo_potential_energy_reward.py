import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R_delta_h2
from ..core.catalog import Catalog as c

class PotentialReward(BaseRewardFunction):
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
        self.initial_altitude = 6096
        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]
        self.ideal_min_height_delta=500
        self.ideal_max_height_delta=4000
        self.previous_v = np.full(self.config.num_agents, 242.67550364687295)
        self.previous_altitude = np.full((self.config.num_agents), 6096)
        self.previous_pitch = np.zeros(self.config.num_agents)
        self.previous_dist_ally = np.full((self.config.num_ally, self.config.num_oppo), 16630).astype(
            float)  # lon 120,lat 60, lon120, lat60.15
        self.previous_dist_oppo = np.full((self.config.num_oppo, self.config.num_ally), 16630).astype(float)
        self.ego_v=0
        self.max_dive_speed=333
        self.max_dive_angle=45/180*np.pi

    def reset(self, task, env):
        self.previous_v = np.full(self.config.num_agents, 242.67550364687295)
        self.previous_altitude = np.full((self.config.num_agents), 6096)
        self.previous_pitch = np.zeros(self.config.num_agents)
        self.previous_dist_ally = np.full((self.config.num_ally, self.config.num_oppo), 16630).astype(
            float)  # lon 120,lat 60, lon120, lat60.15
        self.previous_dist_oppo = np.full((self.config.num_oppo, self.config.num_ally), 16630).astype(float)
        return super().reset(task, env)

    def cal_dive_score(self, speed, dive_angle,distance):
        speed_score=0
        angle_score=0
        if dive_angle<0:
            speed_score += abs(self.max_dive_speed - speed)
            angle_score+=min(1,abs(dive_angle)/self.max_dive_angle)

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        new_reward = 0

        dive_reward_pitch=0
        dive_reward_v=0
        dive_reward_alti=0
        dive_reward_dist=0
        id_agent = np.where(np.array(env.total_ids) == agent_id)[0][0]
        elevator = env.agents[agent_id].get_property_value(c.fcs_elevator_cmd_norm)
        h = env.agents[agent_id].get_position()[-1]
        # feature: (north, east, down, vn, ve, vd)
        pitch_angle = env.agents[agent_id].get_property_value(c.attitude_pitch_rad)
        if id_agent < self.config.num_ally:
            # id_ally=np.where(agent_id in np.array(env.ego_ids))[0]
            id_ally = np.where(np.array(env.ego_ids) == agent_id)[0][0]
        else:
            # id_oppo=np.where(agent_id in np.array(env.enm_ids))[0]
            id_oppo = np.where(np.array(env.enm_ids) == agent_id)[0][0]
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        h=env.agents[agent_id].get_position()[-1]
        # dive_reward_1 = max(0, (self.previous_pitch[id_agent] - pitch_angle))
        # dive_reward_2 = max(0, (self.ego_v - self.previous_v[id_agent]))
        # dive_reward_3 = max(0, (self.previous_altitude[id_agent] - h))
        for enm in env.agents[agent_id].enemies:
            if id_agent < self.config.num_ally:
                id_enem=np.where(np.array(env.enm_ids)==enm.uid)[0][0]
            else:
                id_enem = np.where(np.array(env.ego_ids)==enm.uid )[0][0]
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                # enm_vz=enm.get_velocity()[-1]
                AO, TA,angle, R, delta_h0,delta_v,ego_v = get_AO_TA_R_delta_h2(ego_feature, enm_feature)
                self.ego_v=ego_v
                delta_h = delta_h0 / 5000
                a = (AO + TA) / (2 * np.pi)
                a1 = np.cos(angle)
                orientation_reward = self.orientation_fn(AO, TA)
                height_range_reward = self.range_fn(abs(delta_h0) / 5000)
                new_reward += orientation_reward * height_range_reward




                if id_agent < self.config.num_ally:
                    self.previous_dist_ally[id_ally, id_enem] = R
                else:
                    self.previous_dist_ally[id_oppo, id_enem] = R
        # if dive_reward_v!=0 or dive_reward_pitch!=0 or dive_reward_alti!=0 or dive_reward_dist!=0:
        #     new_reward = new_reward * 0.15 + dive_reward_v * 0.25 + dive_reward_alti * 0.2 + dive_reward_pitch * 0.2 + dive_reward_dist * 0.2
        new_reward = new_reward  + dive_reward_v  + dive_reward_alti + dive_reward_pitch  + dive_reward_dist
        info["rewards"]["rew_potential_attack"] = new_reward #18,-17
        self.previous_v[id_agent]=self.ego_v
        self.previous_pitch[id_agent]=pitch_angle
        self.previous_altitude[id_agent]=h

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