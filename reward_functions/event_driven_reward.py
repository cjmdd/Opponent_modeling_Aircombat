from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
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
        rew_hit = 0


        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                reward += 100
                rew_hit+=100
        info["rewards"]["rew_hit_missiles"] = rew_hit
        # if len(env.agents[agent_id].under_missiles)!=0 and env.agents[agent_id].bloods==100: # alive reward
        #     reward+=5

        return self._process(reward, agent_id)
