from .reward_function_base import BaseRewardFunction
import numpy as np

class EventDrivenReward2(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)



    def get_reward(self, task, env, agent_id, info):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        rew_hitted = 0
        rew_crash=0
        id_agent = np.where(np.array(env.total_ids) == agent_id)[0][0]
        if env.agents[agent_id].is_shotdown:
            reward -= 100
            rew_hitted -= 100           

            info["rewards"]["rew_hitted_missiles"] = rew_hitted
        if env.agents[agent_id].is_crash:
            reward -= 100
            rew_crash -= 100
            

            info["rewards"]["rew_crash"] = rew_crash


        return self._process(reward, agent_id)
