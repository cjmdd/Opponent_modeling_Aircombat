import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh
        self.initial_altitude = 6096
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]
        self.previous_altidude = np.full((self.config.num_agents),6096)

    def reset(self, task, env):
        self.previous_altidude = np.full((self.config.num_agents), 6096)

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        id_agent = np.where(agent_id in env.total_ids)
        h= env.agents[agent_id].get_position()[-1]
        delta_h = h- self.initial_altitude
        altitude_change=h-self.previous_altidude[id_agent]
        self.previous_altidude[id_agent]=h
        elevator=env.agents[agent_id].get_property_value(c.fcs_elevator_cmd_norm)

        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340    # unit: mh
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        Delta_h=0
        if delta_h<0:
            Delta_h+= 10 * delta_h / self.initial_altitude
            if elevator<0:
                Delta_h-=0.5
            elif elevator>0 and altitude_change>0:
                Delta_h+=1
            else:
                Delta_h-=1
        else:
            Delta_h+=0.8 #0.5

        new_reward = Pv + PH + Delta_h
        return self._process(new_reward, agent_id, (Pv, PH))
