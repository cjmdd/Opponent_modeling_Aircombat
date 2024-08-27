import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class AOAReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if attack of plane out of range [-18, 30]
    """
    def __init__(self, config):
        super().__init__(config)
        # self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        # self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        # self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh
        #
        # self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward=0
        alpha=env.agents[agent_id].get_property_value(c.aero_alpha_deg) # degrees
        if alpha<0:
            reward+=alpha/30
        elif alpha>=30:
            reward-=(alpha-30)/30


        return self._process(reward, agent_id) #-0.014
