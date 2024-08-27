import torch
import numpy as np
from torch.distributions import Normal
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
class Z_pred:
    def __init__(self, cfg, rssm=None, obs_model=None, reward_model=None,
                 pred_model=None,shared_buffers=None,device=None):
        self.cfg=cfg
        self.device=device
        self.rssm=rssm
        self.obs_model=obs_model
        self.reward_model=reward_model
        self.pred_model=pred_model
        self.z=shared_buffers._posterior_z_tensors.reshape(-1,self.cfg.rnn_hidden_dim)
        self.flag=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker*self.cfg.quads_num_agents,1)
        self.flag.share_memory_()
        self.delta_s=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker*self.cfg.quads_num_agents,self.cfg.rnn_hidden_dim)
        self.delta_s.share_memory_()
        self.flag.fill_(True)
        # self.flag[:8]=False
        # self.flag[16:24]=False
        # self.alpha=self.pred_model.alpha
        self.alpha =0.1
        # self.mu=self.pred_model.mu
        self.sigma=1e-3
        self.env_num_per_split = self.cfg.num_envs_per_worker // self.cfg.worker_num_splits
        # self.cov=self.pred_model.cov
        self.D=dict(obs=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker,10,self.cfg.quads_num_agents,54+10*cfg.num_obstacle_obs).share_memory_(),
                    z=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker,10,self.cfg.quads_num_agents,cfg.rnn_hidden_dim).share_memory_(),
                    delta_s=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker,10,self.cfg.quads_num_agents,cfg.rnn_hidden_dim).share_memory_(),
                    state_post=torch.zeros(self.cfg.num_workers*self.cfg.num_envs_per_worker,10,self.cfg.quads_num_agents,cfg.rnn_hidden_dim).share_memory_())#M=10
        # self.D['obs'][0]=torch.ones(10,self.cfg.quads_num_agents,54+10*cfg.num_obstacle_obs)
        # self.D['obs'][1] = torch.ones(10, self.cfg.quads_num_agents, 54 + 10 * cfg.num_obstacle_obs)
        # self.D['obs'][2]=torch.ones(10,self.cfg.quads_num_agents,54+10*cfg.num_obstacle_obs)
        self.all_params = (list(self.rssm.parameters()) +
                           list(self.obs_model.parameters())

                           # list(self.reward_goal_model.parameters())+
                           # list(self.reward_obstacle_model.parameters())
                           )
        self.optimizer3 = torch.optim.Adam(self.all_params, lr=cfg.lr2, eps=cfg.eps2)

    def init(self):
        self.rssm = self.rssm.to(self.device)
        self.obs_model = self.obs_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.pred_model = self.pred_model.to(self.device)
        self.z = self.z.to(self.device)
        self.flag = self.flag.to(self.device)
        self.delta_s=self.delta_s.to(self.device)
        self.D['obs'] = self.D['obs'].to(self.device)
        self.D['z'] = self.D['z'].to(self.device)
        self.D['delta_s']= self.D['delta_s'].to(self.device)
        self.D['state_post']= self.D['state_post'].to(self.device)


    def reset(self,env_id):
        self.flag.fill_(True)
        self.z[env_id*self.cfg.quads_num_agents:env_id*self.cfg.quads_num_agents+self.cfg.quads_num_agents-1]=0
        self.delta_s[env_id*self.cfg.quads_num_agents:env_id*self.cfg.quads_num_agents+self.cfg.quads_num_agents-1]=0
        self.D['obs'][env_id]=0
        self.D['z'][env_id] = 0
        self.D['delta_s'][env_id] = 0
        self.D['state_post'][env_id] = 0
        # self.D= dict(
        #     obs=torch.zeros(self.cfg.num_workers * self.cfg.num_envs_per_worker, 10, self.cfg.quads_num_agents,
        #                     54 + 10 * self.cfg.num_obstacle_obs),
        #     z=torch.zeros(self.cfg.num_workers * self.cfg.num_envs_per_worker, 10, self.cfg.quads_num_agents,
        #                   self.cfg.rnn_hidden_dim),
        #     delta_s=torch.zeros(self.cfg.num_workers * self.cfg.num_envs_per_worker, 10, self.cfg.quads_num_agents,
        #                         self.cfg.rnn_hidden_dim),
        #     state_post=torch.zeros(self.cfg.num_workers * self.cfg.num_envs_per_worker, 10, self.cfg.quads_num_agents,
        #                            self.cfg.rnn_hidden_dim))  # M=10

    def rollout(self,observation,indice,actor):
        obs=observation['obs'].reshape(-1,self.cfg.quads_num_agents,observation['obs'].shape[-1])
        state_post=self.rssm.posterior(obs)
        idx_global=indice[0]*self.cfg.num_envs_per_worker*self.cfg.quads_num_agents+indice[1]*self.cfg.quads_num_agents*self.env_num_per_split+indice[2]*self.cfg.quads_num_agents+indice[3]
        idx_global=idx_global.to(self.device)
        self.idx_global=idx_global
        # tt=torch.where(self.flag[idx_global] == True)
        t=torch.where(self.flag[idx_global] == True)[0]
        t2 = torch.where(self.flag[idx_global] == False)[0]
        idx_raw = idx_global[torch.where(self.flag[idx_global] == True)[0]]
        idx = idx_global[torch.where(self.flag[idx_global] == False)[0]]
        if any(idx_raw): #True: reset
            self.delta_s[idx_raw]=state_post.reshape(-1,state_post.shape[-1])[t]
            self.flag[idx_raw]=False
        if any(idx):
            self.delta_s[idx]=self.rssm.posterior_delta(observation['obs'][t2].reshape(-1,self.cfg.quads_num_agents,observation['obs'].shape[-1]),self.z[idx].reshape(-1,self.cfg.quads_num_agents,self.cfg.rnn_hidden_dim)).reshape(-1,self.cfg.rnn_hidden_dim)
        total_reward = []
        Z_C = torch.zeros(self.cfg.N_iterations,obs.shape[0], self.cfg.quads_num_agents, self.cfg.rnn_hidden_dim).to(self.device)
        for i in range(self.cfg.N_iterations):#10
            total_predicted_reward = torch.zeros(obs.shape[0], self.cfg.quads_num_agents, 1).to(self.device)
            Z_c= torch.zeros(obs.shape[0], self.cfg.quads_num_agents, self.cfg.rnn_hidden_dim).to(self.device)
            delta_s = self.delta_s[idx_global].reshape(-1, self.cfg.quads_num_agents, self.cfg.rnn_hidden_dim)
            pred_obst1,zc,_,_,_ = self.pred_model(obs[:, :, :54], delta_s)                       
            pred_obst2,_ = self.pred_model.post(pred_obst1)
            pred_obst3,_ = self.pred_model.post(pred_obst2)
            obs_all = torch.cat([obs, pred_obst1, pred_obst2, pred_obst3], dim=-1)
            actions= actor.forward2(obs_all.reshape(-1,obs_all.shape[-1]))[0].reshape(-1,self.cfg.quads_num_agents,4)
            for t in range(self.cfg.horizon):
                delta_s2, z,_,_,_ = self.rssm.prior(state_post, actions)
                state_post = state_post + delta_s2
                obs2,_ = self.obs_model(state_post, z)
                pred_obst21,zc1,mean,std,_ = self.pred_model(obs2[:, :, :54], delta_s2)                
                pred_obst22,_ = self.pred_model.post(pred_obst21)
                pred_obst23,_ = self.pred_model.post(pred_obst22)
                obs_all2 = torch.cat([obs2, pred_obst21, pred_obst22, pred_obst23], dim=-1)
                actions = actor.forward2(obs_all2.reshape(-1,obs_all2.shape[-1]))[0].reshape(-1,self.cfg.quads_num_agents,4)
                total_predicted_reward +=self.reward_model(state_post, z)
                Z_c+=zc1
            total_reward.append(total_predicted_reward)
            Z_C[i]=Z_c/self.cfg.horizon
        self.weights_local = np.array([np.exp(total_reward[i].cpu().numpy()) for i in range(len(total_reward))])
        self.weights_local /=self.weights_local.sum()

        self.mu=(torch.from_numpy(self.weights_local).to(self.device)*Z_C).sum(dim=0)

        delta_s = self.delta_s[idx_global].reshape(-1, self.cfg.quads_num_agents, self.cfg.rnn_hidden_dim)
        _, _, mean, std,_ = self.pred_model(obs[:, :, :54], delta_s)

        mean_new =  self.mu
        delt = mean_new - mean
        self.cov = (torch.from_numpy(self.weights_local).to(self.device) * (delt * delt)).sum(axis=0)
        self.cov = (self.sigma * self.cov / torch.linalg.norm(self.cov))
        std_new = self.cov

        z_c_distribution = Normal(mean_new, std_new)
        z_c=z_c_distribution.sample()
        # self.z[idx_global]=z_c

        obs_zc = torch.cat([obs[:, :, :54], z_c], dim=-1)
        pred_obstacle_obs_mean= self.pred_model.fc_pred_w(obs_zc)
        pred_obstacle_obs_std = self.pred_model.fc_pred_std(obs_zc) + 0.1
        pred_obstacle_distribution = Normal(pred_obstacle_obs_mean, pred_obstacle_obs_std)
        pred_obstacle1 = pred_obstacle_distribution.rsample()        
        pred_obstacle2,_ = self.pred_model.post(pred_obstacle1)
        pred_obstacle3,_ = self.pred_model.post(pred_obstacle2)
        obs_all = torch.cat([obs, pred_obstacle1, pred_obstacle2, pred_obstacle3], dim=-1)
        return obs_all.reshape(-1,obs_all.shape[-1])

    def update_zc(self,obs,action,indices):
        state_post=self.rssm.posterior(obs.reshape(-1,self.cfg.quads_num_agents,obs.shape[-1]))
        delta_s, z,_,_,_ = self.rssm.prior(state_post, action.reshape(-1,self.cfg.quads_num_agents,action.shape[-1]))
        self.z[self.idx_global]=z.reshape(-1,self.cfg.rnn_hidden_dim)


        vector_idx = indices[1] *self.env_num_per_split  + indices[2]  # 0*2+0/1； 1*2+0/1； local total:4 per cpu/actorworker

        # global env id within the entire system
        env_id = indices[0] * self.cfg.num_envs_per_worker + vector_idx
        # idx = indices[0] * self.cfg.num_envs_per_worker * self.cfg.quads_num_agents + indices[
        #     1] * self.cfg.quads_num_agents * self.cfg.num_envs_per_worker // self.cfg.worker_num_splits + indices[
        #                  2] * self.cfg.quads_num_agents + indices[3]
        step = indices[-1]%10
        idx=[env_id,step,indices[3]]
        idx=tuple(idx)

        # a=obs.reshape(obs.shape[0] // self.cfg.quads_num_agents, self.cfg.quads_num_agents, 1, obs.shape[-1])
        # b=self.D['obs'][idx][step]
        self.D['obs'][idx]=obs
        self.D['z'][idx]=self.z[self.idx_global]
        self.D['delta_s'][idx] = delta_s.reshape(-1,self.cfg.rnn_hidden_dim)
        self.D['state_post'][idx]=state_post.reshape(-1,self.cfg.rnn_hidden_dim)
        with torch.enable_grad():
            idx2 = torch.all(self.D['obs'][..., :,:,:] == 0.0,dim=-1)
            idx3=torch.any(idx2[...,:,:]== True,dim=-1)
            idx4 = torch.any(idx3[..., :] == True, dim=-1)
            # idx30 = torch.all(idx2[..., :, :] != 0, dim=-1)
            idx31 = torch.where(~idx4 == True)[0]  # tuple
            idx3 = torch.unique(idx31)
            # idx4 = ~torch.any(self.D['obs'][..., :, :, :] == 0,dim=1)
            if any(idx3):
                obs2 = self.D['obs'][idx3]
                z2 = self.D['z'][idx3]
                delta_s2 = self.D['delta_s'][idx3]
                state_post2 = self.D['state_post'][idx3]
                loss = 0
                for i in range(obs2.shape[0]):
                    delta_s_posterior = state_post2[i][1:] - state_post2[i][:-1]
                    delta_s_prior = delta_s2[i][:-1]
                    loss_delta_s_prior = 0.5 * mse_loss(
                        delta_s_prior, delta_s_posterior, reduction='none').mean([0, 1]).sum()
                    delta_s_posterior2 = self.rssm.posterior_delta(obs2[i][1:], z2[i][:-1])
                    loss_delta_s_posterior = 0.5 * mse_loss(
                        delta_s_posterior2, delta_s_posterior, reduction='none').mean([0, 1]).sum()
                    _,predict_obs_distribution= self.obs_model(state_post2[i][1:], z2[i][:-1])
                    obs_loss = -torch.mean(predict_obs_distribution.log_prob(obs2[i][1:]))
                    loss0 = loss_delta_s_prior + loss_delta_s_posterior + obs_loss
                    loss += loss0

                self.optimizer3.zero_grad()
                loss.backward()
                clip_grad_norm_(self.all_params, self.cfg.clip_grad_norm)
                self.optimizer3.step()
                print('loss_env', loss)


