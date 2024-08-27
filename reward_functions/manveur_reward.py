import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np
import time
from sample_factory.utils.utils import log
from ..utils.utils import get_AO_TA_R_delta_h2
class MnveurReward(BaseRewardFunction):
    """
    Measure the difference manvuer to avoid missiles
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]
        self.time_invert_flight=np.zeros(self.config.num_agents)
        self.time_f_pol_flight=np.zeros(self.config.num_agents)
        self.previous_altitude=np.full((self.config.num_agents),6096)
        self.previous_pitch_rate=np.zeros(self.config.num_agents)
        self.previous_roll_rate=np.zeros(self.config.num_agents)
        self.previous_pitch = np.zeros(self.config.num_agents)
        self.previous_roll = np.zeros(self.config.num_agents)
        self.immelman_weight = 0.6  # 假设Immelmann机动权重为0.6
        self.f_pol_weight = 0.4  # 假设F-POL机动权重为0.4
        self.initial_altitude =6096
        self.initial_speed = 800
        self.flag = np.full(self.config.num_agents,False)

    def reset(self, task, env):
        self.time_invert_flight = np.zeros(self.config.num_agents)
        self.time_f_pol_flight = np.zeros(self.config.num_agents)
        self.previous_altitude = np.full((self.config.num_agents),6096)
        self.previous_pitch = np.zeros(self.config.num_agents)
        self.previous_roll = np.zeros(self.config.num_agents)
        self.previous_pitch_rate = np.zeros(self.config.num_agents)
        self.previous_roll_rate = np.zeros(self.config.num_agents)
        self.flag = np.full((self.config.num_agents),False)
        return super().reset(task, env)

    def immelman_score_function(self,agent_id,roll_angle, pitch_angle, altitude_loss,speed_loss):
        # 各参数含义：滚转角、俯仰角、倒飞持续时间、高度损失

        max_roll_weight = 1.0
        max_pitch_weight = 1.5
        inverted_time_weight = 2.0
        altitude_loss_weight = -0.5
        speed_loss_weight=-0.5
        max_roll_angle=math.pi
        max_pitch_angle=math.pi/2
        target_inverted_time=50 #150
        pitch_angle_threshold = math.pi/3 # 俯仰角度阈值，达到这个角度认为进入了倒飞
        acceptable_loss_ratio = 0.2
        acceptable_speed_loss_ratio=0.2
        inverted_pitch_reward = max(0, (pitch_angle - pitch_angle_threshold) / (
                    180 - pitch_angle_threshold))
        inverted_time_score=0
        reward=0
        if inverted_pitch_reward>0 and self.time_invert_flight[agent_id]==0:
            self.time_invert_flight[agent_id] = time.time()
            # print('pitch*****', pitch_angle)
            # print("%%%%%%%%%%% start immelman %%%%%%%%%%")
            log.debug(
                '%%%%%%%%%%% start immelman %%%%%%%%%%'
                )
        if self.time_invert_flight[agent_id] != 0:
            roll_score = 1-abs(max_roll_angle - roll_angle) / max_roll_angle * max_roll_weight
            pitch_score = 1-abs(max_pitch_angle - pitch_angle) / max_pitch_angle * max_pitch_weight
            altitude_loss_ratio = altitude_loss / self.previous_altitude[agent_id]
            altitude_loss_score = max(0, altitude_loss_ratio - acceptable_loss_ratio) * altitude_loss_weight
            loss_ratio = speed_loss / self.initial_speed
            speed_score = max(0, loss_ratio - acceptable_speed_loss_ratio) * speed_loss_weight
            reward = roll_score + pitch_score + inverted_pitch_reward+altitude_loss_score+speed_score
        # 如果正在倒飞，则累计倒飞时间
        if self.time_invert_flight[agent_id] != 0 and pitch_angle <= 80 / 180 * np.pi:
            time_invert_flight = time.time() - self.time_invert_flight[agent_id]
            if time_invert_flight > target_inverted_time:
                inverted_time_score = 1.0
            else:
                inverted_time_score = time_invert_flight / target_inverted_time * inverted_time_weight
            reward += inverted_time_score
            log.info(
                "%%%%%%%%%%% stop immelman %%%%%%%%%%"
                )
            print(f"time_invert_flight_{agent_id}",time.time() - self.time_invert_flight[agent_id])
            self.time_invert_flight[agent_id] = 0

        if self.time_invert_flight[agent_id]!=0 and time.time() - self.time_invert_flight[agent_id]> target_inverted_time:
            log.info(
                "%%%%%%%%%%% stop immelman %%%%%%%%%%"
                )
            print(f"time_invert_flight_{agent_id}",time.time() - self.time_invert_flight[agent_id])
            self.time_invert_flight[agent_id] = 0
        return reward

    def is_starting_f_pol_maneuver(self,agent_id,pitch_rate, roll_rate, threshold_pitch_rate, threshold_roll_rate):
        # state: 当前飞行状态（包含俯仰角、滚转角、俯仰角速率、滚转角速率等）
        # previous_state: 上一时刻的飞行状态

        current_pitch_rate = pitch_rate
        current_roll_rate = roll_rate

        # 判断是否开始F-POL机动的条件示例
        pitch_rate_increase = abs(current_pitch_rate - self.previous_pitch_rate[agent_id]) > threshold_pitch_rate
        roll_rate_increase = abs(current_roll_rate - self.previous_roll_rate[agent_id]) > threshold_roll_rate

        # 假设俯仰角速率和滚转角速率同时显著增加，则认为可能开始F-POL机动
        return pitch_rate_increase and roll_rate_increase

    def is_f_pol_ended(self,roll_angle, pitch_angle):
        # 假设俯仰角恢复到接近水平、滚转角度变化符合预期、倒飞持续时间达到阈值并且速度恢复到安全范围作为F-POL结束的标志

        # 俯仰角恢复接近水平
        pitch_end_threshold = 10  # 俯仰角阈值，可以根据实际情况调整
        pitch_completed= abs(pitch_angle) <= pitch_end_threshold
        # 判断滚转是否达到目标
        roll_completed = abs(roll_angle) >= 135/180*np.pi
        return pitch_completed and roll_completed

    def f_pol_score_function(self,agent_id,max_g_force, roll_rate, pitch_rate,roll, pitch):
        # 各参数含义：最大G力、速率、俯仰速率、完成机动所需时间
        max_g_force_weight = -2.0
        roll_rate_weight = 1.0
        pitch_rate_weight = 1.5
        time_to_complete_weight = 0.5
        g_force_score=0
        max_allowed_g=9
        max_pitch_rate=5/180*math.pi
        max_roll_rate=30/180*math.pi
        reward=0
        a=time.time()
        if self.is_starting_f_pol_maneuver(agent_id,pitch_rate, roll_rate, max_pitch_rate, max_roll_rate) and self.flag[agent_id]==False:
            self.time_f_pol_flight[agent_id]= time.time()
            self.flag[agent_id]=True
            log.debug('%%%%%%%%%%% start F-pole %%%%%%%%%%')
        if self.flag[agent_id]==True:
            if max_g_force > max_allowed_g:
                g_force_score = (max_g_force - max_allowed_g) / max_allowed_g * max_g_force_weight

            roll_score = abs(roll_rate) / max_roll_rate * roll_rate_weight
            pitch_score = abs(pitch_rate) / max_pitch_rate * pitch_rate_weight
            reward=g_force_score + roll_score + pitch_score
        if self.flag[agent_id]==True and self.is_f_pol_ended(roll, pitch):
            time_to_complete=time.time()-self.time_f_pol_flight[agent_id]
            time_score = 1 / (time_to_complete + 0.0001) * time_to_complete_weight  # 避免分母为零，epsilon是一个较小正数
            reward+=time_score
            self.flag[agent_id] =False
            log.info(
                "%%%%%%%%%%% stop F-pole %%%%%%%%%%"
                )
            print(f'time_f_pol_flight_{agent_id}:', time.time() - self.time_f_pol_flight[agent_id])
            self.time_f_pol_flight[agent_id] = 0

        if self.flag[agent_id] == True and time.time()-self.time_f_pol_flight[agent_id]>50:
            self.flag[agent_id] = False
            log.info(
                "%%%%%%%%%%% stop F-pole %%%%%%%%%%"
            )
            print(f'time_f_pol_flight_{agent_id}:', time.time() - self.time_f_pol_flight[agent_id])
            self.time_f_pol_flight[agent_id] = 0

        return reward

    def get_reward(self, task, env, agent_id,info):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        immelman_reward=0
        f_pol_reward=0
        roll_angle=env.agents[agent_id].get_property_value(c.attitude_roll_rad)
        pitch_angle=env.agents[agent_id].get_property_value(c.attitude_pitch_rad)
        roll_rate=env.agents[agent_id].get_property_value(c.velocities_p_rad_sec)
        pitch_rate = env.agents[agent_id].get_property_value(c.velocities_q_rad_sec)
        max_g_force = env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm)

        cur_altitude=env.agents[agent_id].get_position()[-1]
        speed_loss=self.initial_speed-env.agents[agent_id].get_property_value(c.velocities_u_fps)
        id_agent = np.where(np.array(env.total_ids) == agent_id)[0][0]

        altitude_loss = self.previous_altitude[id_agent] - cur_altitude
        altitude_change = cur_altitude - self.previous_altitude[id_agent]
        missiles_sim = env.agents[agent_id].check_missile_warning2()
        reward_pith_change=0
        reward_pith_rate_change=0
        reward_roll_rate_change=0
        punish=0
        if len(missiles_sim) != 0:
            for missile in missiles_sim:
                R=np.linalg.norm(missile.get_position())
                if R<6000 and missile.is_alive:
                    elevator = env.agents[agent_id].get_property_value(c.fcs_elevator_cmd_norm)
                    if elevator<0:
                        punish-=0.2
                    if elevator>0 or pitch_angle>0 and pitch_angle-self.previous_pitch[id_agent]>0:
                        reward_pith_change+=0.4
                    else:
                        reward_pith_change-=0.4
                    if elevator > 0 or pitch_angle>0 and altitude_change > 0:
                        reward_pith_change += 0.4
                    else:
                        reward_pith_change-=0.4
                    if pitch_rate - self.previous_pitch_rate[id_agent]>0:
                        reward_pith_rate_change+=0.4
                    else:
                        reward_pith_rate_change-=0.4
                    if roll_rate - self.previous_roll_rate[id_agent] > 0:
                        reward_roll_rate_change += 0.4
                    else:
                        reward_roll_rate_change-=0.4
            for missile in missiles_sim:
                R=np.linalg.norm(missile.get_position())
                if R<6000 and missile.is_alive:
                    immelman_reward = self.immelman_score_function(id_agent, roll_angle, pitch_angle, altitude_loss,
                                                                   speed_loss)
                    f_pol_reward = self.f_pol_score_function(id_agent, max_g_force, roll_rate, pitch_rate,
                                                             roll_angle,
                                                             pitch_angle)
                    if immelman_reward!=0 and f_pol_reward!=0:
                        immelman_reward=immelman_reward* self.immelman_weight
                        f_pol_reward =  f_pol_reward * self.f_pol_weight


                    break
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                AO, TA, angle, R, delta_h0, delta_v, ego_v = get_AO_TA_R_delta_h2(ego_feature, enm_feature)
                delta_h = delta_h0 / 5000
                a = (AO + TA) / (2 * np.pi)
                a1 = np.cos(angle)
                flag = a1 < 0 and delta_v > 1 and a < 0.25
                if a1 > 0 and delta_v <= 1 or flag:
                    reward=0 # pass
                else:
                    if delta_h<0 and R<3000:
                        elevator = env.agents[agent_id].get_property_value(c.fcs_elevator_cmd_norm)
                        if elevator<0:
                            punish-=0.4
                        if elevator > 0 or pitch_angle > 0 and pitch_angle - self.previous_pitch[id_agent] > 0:
                            reward_pith_change += 0.4
                        else:
                            reward_pith_change -= 0.4
                        if elevator > 0 or pitch_angle > 0 and altitude_change > 0:
                            reward_pith_change += 0.4
                        else:
                            reward_pith_change -= 0.4
                        if pitch_rate - self.previous_pitch_rate[id_agent] > 0:
                            reward_pith_rate_change += 0.4
                        else:
                            reward_pith_rate_change -= 0.4
                        # if roll_rate - self.previous_roll_rate[id_agent] > 0:
                        #     reward_roll_rate_change += 4
                        # else:
                        #     reward_roll_rate_change -= 4

                        immelman_reward += self.immelman_score_function(id_agent, roll_angle, pitch_angle, altitude_loss,
                                                                       speed_loss)
                        # f_pol_reward += self.f_pol_score_function(id_agent, max_g_force, roll_rate, pitch_rate,
                        #                                          roll_angle,
                        #                                          pitch_angle)
                        if immelman_reward != 0:
                            immelman_reward += immelman_reward * self.immelman_weight
                            # f_pol_reward += f_pol_reward * self.f_pol_weight

        self.previous_altitude[id_agent]=cur_altitude
        self.previous_pitch[id_agent] = pitch_angle
        self.previous_pitch_rate[id_agent] = pitch_rate
        self.previous_roll[id_agent] = roll_angle
        self.previous_roll_rate[id_agent] = roll_rate
        reward=immelman_reward+f_pol_reward+reward_pith_change+reward_pith_rate_change+reward_roll_rate_change
        info["rewards"]["rew_Immelan"]=immelman_reward
        info["rewards"]["rew_F_pole"]=f_pol_reward

        return self._process(reward, agent_id)

