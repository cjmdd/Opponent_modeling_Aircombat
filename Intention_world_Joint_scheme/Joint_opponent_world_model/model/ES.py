import numpy as np
import torch
import copy
import threading
from numpy import float32
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs
from torch.distributions import Normal
import random
class CEM_Mutate_Novelty:
    def __init__(self,cfg,num_params=0, action_parameterization=None, rssm=None,obs_model=None, reward_model=None,reward_goal_model=None,reward_obstacle_model=None,shared_buffers=None):
        self.cfg = cfg
        self.device = next(reward_model.parameters()).device
        self.num_params = num_params
        self.pop = []
        # self.pop_action=torch.zeros(self.pop_size,4)

        self.action_parameterization=action_parameterization
        self.rssm=rssm
        self.obs_model=obs_model
        self.reward_model=reward_model
        self.reward_goal_model = reward_goal_model
        self.reward_obstalce_model = reward_obstacle_model

        self.mu=torch.zeros(self.num_params).share_memory_()
        self.mu = self.mu.numpy()
        # print('mu',id(self.mu))
        # print('action',id(self.action_parameterization))
        
        self.pop_size = cfg.pop_size
        self.cov = cfg.cov_init * torch.ones(self.num_params)
        self.cov.share_memory_()
        self.cov= self.cov.numpy()
        self.cov_limit=cfg.cov_limit

        self.es_params=torch.zeros(self.pop_size,self.num_params)
        self.es_params.share_memory_()
        self.es_params = self.es_params.numpy()

        self.worst_index = torch.tensor([0]).share_memory_()
        self.flags = torch.tensor([False]).share_memory_()

        self.damp=cfg.cov_init
        self.alpha=cfg.cov_alpha
        self.alpha_e=torch.tensor([cfg.alpha_e],dtype=torch.float32).share_memory_()
        self.alpha_e=self.alpha_e.numpy()
        self.prob = torch.tensor([cfg.prob],dtype=torch.float32).share_memory_()
        self.prob=self.prob.numpy()

        self.elitism = cfg.elitism
        self.elitism_param = self.mu
        
        self.parents_local = cfg.quads_num_agents
        self.parents_global = cfg.pop_size //self.parents_local//4
        self.group = cfg.pop_size // cfg.quads_num_agents

        self.rnn_hidden = torch.zeros(self.group,self.cfg.quads_num_agents,self.rssm.rnn_hidden_dim,device=self.device).share_memory_()
        self.total_predicted_reward =torch.zeros(self.group,self.cfg.quads_num_agents,1,device=self.device).share_memory_()
        
        self.weights_local = np.array([np.log((self.parents_local + 1) / i) for i in range(1, self.parents_local + 1)])
        self.weights_global = np.array([np.log((self.parents_global + 1) / i) for i in range(1, self.parents_global + 1)])
        self.weights_local /= self.weights_local.sum()
        self.weights_global /= self.weights_global.sum()

        self.traj_tensors = shared_buffers.tensors


    def init_pop_params(self,params):        
        self.mu = np.array(params,dtype=float32)
        
        
        
    def update(self,es_params_goal,es_params_obstacle):
        es_params_goal = es_params_goal.reshape(len(es_params_goal),-1)
        es_params_obstacle = es_params_obstacle.reshape(len(es_params_obstacle), -1)
        z_goal = (es_params_goal - self.mu)
        z_obstacle = (es_params_obstacle - self.mu)
        self.damp = (1 - self.alpha) * self.damp + self.alpha * self.cov_limit
        cov_goal = 1 / self.parents_global * self.weights_global @ (z_goal * z_goal) + self.damp * np.ones(self.num_params)
        cov_obstacle = 1 / self.parents_global * self.weights_global @ (z_obstacle * z_obstacle) + self.damp * np.ones(
            self.num_params)
        self.mu = self.sigmoid(self.alpha_e[0])*self.weights_global @ es_params_goal+(1-self.sigmoid(self.alpha_e[0]))*self.weights_global @ es_params_obstacle
        self.cov=self.sigmoid(self.alpha_e[0])*cov_goal+(1-self.sigmoid(self.alpha_e[0]))*cov_obstacle

        
    def proximal_mutate(self,half_elite_pop_actions_goal,pop_actor_goal,half_elite_pop_actions_obstacle,pop_actor_obstacle):
        half_elite_pop_actions_goal = half_elite_pop_actions_goal.reshape(
            self.cfg.N_iterations * self.cfg.quads_num_agents, -1,4)
        half_elite_pop_actions_obstacle = half_elite_pop_actions_obstacle.reshape(
            self.cfg.N_iterations * self.cfg.quads_num_agents, -1, 4)
        pop_actor_goal = pop_actor_goal.reshape(-1)
        pop_actor_obstacle = pop_actor_obstacle.reshape(-1)

        tot_size = len(self.elitism_param)

        jacobin = np.zeros((self.cfg.N_iterations * self.cfg.quads_num_agents, half_elite_pop_actions_obstacle.shape[-1], tot_size))
        grad_output = torch.zeros(half_elite_pop_actions_goal[0].size()).to(self.device)

        for i in range(self.pop_size//4): # half of half of  pop_size 32
            self.es_params[i] = pop_actor_goal[i].get_params() #can change parameters of corresponding pop automatically
        for i in range(self.pop_size//4): # half of half of pop_size
            self.es_params[i+32] = pop_actor_obstacle[i].get_params()

        # for i in range(self.pop_size//2, self.pop_size//2+self.pop_size//4):
        for i in range(64,66):
            for j in range(4):  # action_space
                pop_actor_goal[i-64].zero_grad()
                grad_output.zero_()
                grad_output[:, j] = 1.0

                half_elite_pop_actions_goal[i-64].backward(grad_output, retain_graph=True)
                jacobin[i-64][j] = pop_actor_goal[i-64].get_grads()

            scaling = np.sqrt((jacobin[i-64] ** 2).sum(0))
            scaling[scaling == 0] = 1.0
            scaling[scaling < 0.01] = 0.01
            delta = np.random.normal(np.zeros_like(self.elitism_param), np.ones_like(self.elitism_param) * self.cfg.mut_mag)
            delta /= scaling
            self.es_params[i] = copy.deepcopy(self.es_params[i - 64]) + self.prob[0]*delta  # self.es_params<->pop
            # for i in range(self.pop_size//2+self.pop_size//4, self.pop_size):
            #     for j in range(4):  # action_space
            #         pop_actor_obstacle[i-96].zero_grad()
            #         grad_output.zero_()
            #     grad_output[:, j] = 1.0
            #
            #     half_elite_pop_actions_obstacle[i-96].backward(grad_output, retain_graph=True)
            #     jacobin[i-96][j] = pop_actor_obstacle[i-96].get_grads()
            #
            # scaling = np.sqrt((jacobin[i-96] ** 2).sum(0))
            # scaling[scaling == 0] = 1.0
            # scaling[scaling < 0.01] = 0.01
            # delta = np.random.normal(np.zeros_like(self.elitism_param), np.ones_like(self.elitism_param) * self.cfg.mut_mag)
            # delta /= scaling
            # print('mutate')
            # self.es_params[i] = copy.deepcopy(self.es_params[i - 64]) + self.prob*delta  # self.es_params<->pop
        
    def sampling(self,pop_size):
        epsilon = np.sqrt(self.cov) * np.random.randn(self.pop_size, self.num_params)
        for i in range(self.pop_size):
            self.es_params[i] = self.mu + epsilon[i]
        # self.es_params = np.array(self.es_params,dtype=float32) # will change id of es_params, removing it can guarantee id are all the same so that es_param of learner can updated by here automotoly

        # if self.elitism:
        #     self.es_params[0] = self.elitism_param

        return self.es_params
    
    def set_pop_params(self, actor):
        es_params=self.sampling(self.pop_size)
        
        for i in range(self.pop_size):
            self.pop.append(copy.deepcopy(actor))
            self.pop[i].set_params1(es_params[i]) #pop<-->es_params
        # self.pop = np.stack(self.pop)
        # # (np.hstack([es_params[:,].cpu().data.numpy().flatten()
        # self.pop2=ensemble_actor
        # self.p3 = list(ensemble_actor.get_params())

        # print('test')


    def update_pop_params(self): #after rollout steps(i.e.,cem)
        # es_params=self.sampling(self.pop_size) # change id of self.es_params, then cannot change automotoly when injecting updated rl into pop in learner.py
        # for i in range(self.pop_size):
        #     self.pop[i].set_params1(es_params[i])
        epsilon = np.sqrt(self.cov) * np.random.randn(self.pop_size, self.num_params)
        for i in range(self.pop_size):
            self.es_params[i] = self.mu + epsilon[i]  # qian copy can change parameters of pop correspondingly 
        # self.es_params = np.array(self.es_params, dtype=float32)

        

    def get_action(self,obs,pop):
        pop_action = []
        pop_action_log=torch.zeros(obs.shape[0], 4)
        for i in range(obs.shape[0]):
            action_mean = pop[i](obs[i].unsqueeze(dim=0), 'pop_actor')
            action_distribution_params, action_distribution = self.action_parameterization(action_mean)
            # for non-trivial action spaces it is faster to do these together
            actions, log_prob_actions = sample_actions_log_probs(action_distribution)
            pop_action.append(actions)
            pop_action_log[i]=log_prob_actions
        pop_action = torch.stack([actions for actions in pop_action], dim=0)
        # pop_action_log = torch.stack([actions for log_prob_actions in pop_action_log], dim=0)# error,this way will change requires_grad into false
        pop_action = pop_action.reshape(-1, self.cfg.quads_num_agents, 4)
        pop_action_log = pop_action_log.reshape(-1, self.cfg.quads_num_agents, 4)
        return pop_action,pop_action_log

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
        
    def evaluate(self,obs,rollout_step):
        """

        :param obs:
        :param rollout_step:
        :param action_parameterization:
        :return:
        """
        # print('evvvvvvvvvvvvv',id(self.action_parameterization))
        # obs = obs['obs'].unsqueeze(dim=0)        
        elite_params_goal =[]
        elite_params_obstacle = []
        elite_reward =[]
        elite_reward_goal = []
        elite_reward_obstacle = []
        worst_indices=[]
        half_elite_pop_actions=torch.zeros(self.cfg.N_iterations, self.cfg.quads_num_agents, (self.cfg.horizon+1)*4)
        half_elite_pop_actions_goal=torch.zeros(self.cfg.N_iterations, self.cfg.quads_num_agents, (self.cfg.horizon+1)*4)
        half_elite_pop_actions_obstacle=torch.zeros(self.cfg.N_iterations, self.cfg.quads_num_agents, (self.cfg.horizon+1)*4)
        es_params_goal = np.zeros((self.cfg.quads_num_agents, self.num_params))
        es_params_obstacle = np.zeros((self.cfg.quads_num_agents, self.num_params))
        pop_actor_goal =[]
        pop_actor_obstacle = []
        
        pop = copy.deepcopy(self.pop)
        es_params =copy.deepcopy(self.es_params)
        # print('obs', obs['obs'].shape)
        obs=obs['obs'].repeat(self.pop_size//obs['obs'].shape[0],1)
        # print('obs',obs.shape)
        obs2 = obs.reshape(self.group, self.cfg.quads_num_agents, -1)
        state_posterior = self.rssm.posterior(obs2, self.rnn_hidden)
        rnn_hidden = self.rnn_hidden

        for i in range(self.cfg.N_iterations): #8->4
            total_predicted_reward = torch.zeros(self.group, self.cfg.quads_num_agents, 1)
            total_predicted_reward_goal = torch.zeros(self.group, self.cfg.quads_num_agents, 1)
            total_predicted_reward_obstacle = torch.zeros(self.group, self.cfg.quads_num_agents, 1)
            pop_actions_log = []
            pop_action,pop_action_log=self.get_action(obs, pop)
            pop_actions_log.append(pop_action_log)
            state = state_posterior.sample()
            for t in range(self.cfg.horizon):
                next_state_prior, rnn_hidden = self.rssm.prior(state, pop_action, rnn_hidden)
                state = next_state_prior.sample()
                obs_t = self.obs_model(state, rnn_hidden)
                # print('obs',obs_t)
                pop_action,pop_action_log=self.get_action(obs_t.reshape(-1,obs.shape[1]), pop)
                pop_actions_log.append(pop_action_log)
                total_predicted_reward += self.reward_model(state, rnn_hidden)
                total_predicted_reward_goal += self.reward_goal_model(state, rnn_hidden)
                total_predicted_reward_obstacle += self.reward_obstalce_model(state, rnn_hidden)

            fitness = np.array(torch.sum(total_predicted_reward, dim=1).detach())
            index_global_rank = np.argsort(fitness.squeeze())[::-1]  # reverse
            top_index_global = index_global_rank[0]
            worst_index_global =index_global_rank[-1]
            worst_indices.append(top_index_global)

            fitness_goal = np.array(torch.sum(total_predicted_reward_goal, dim=1).detach())
            index_global_rank_goal = np.argsort(fitness_goal.squeeze())[::-1]  # reverse
            top_index_global_goal = index_global_rank_goal[0]
            worst_index_global_goal = index_global_rank_goal[-1]

            fitness_obstacle = np.array(torch.sum(total_predicted_reward_obstacle, dim=1).detach())
            index_global_rank_obstacle = np.argsort(fitness_obstacle.squeeze())[::-1]  # reverse
            top_index_global_obstacle = index_global_rank_obstacle[0]
            worst_index_global_obstacle = index_global_rank_obstacle[-1]

            elite_reward.append(fitness[top_index_global])
            elite_reward_goal.append(fitness_goal[top_index_global_goal])
            elite_reward_obstacle.append(fitness_obstacle[top_index_global_obstacle])

            index_local_rank = np.argsort(total_predicted_reward[top_index_global].detach().numpy().squeeze())[::-1]# tensor:1dimension step must greater than zero,so not squeeze()
            top_index_local = index_local_rank[0]

            index_local_rank_goal = np.argsort(total_predicted_reward_goal[top_index_global_goal].detach().numpy().squeeze())[::-1]
            top_index_local_goal = index_local_rank_goal[0]
            index_local_rank_obstacle = np.argsort(total_predicted_reward_obstacle[top_index_global_obstacle].detach().numpy().squeeze())[::-1]
            top_index_local_obstacle = index_local_rank_obstacle[0]

            pop_actions=torch.cat(pop_actions_log,dim=-1)# ok this way will not change requires_grad into false
            half_elite_pop_actions_goal[i]=pop_actions[top_index_global_goal] # en grad_fn,it cannot sort by local_idx [index_local_rank_obstacle]
            half_elite_pop_actions_obstacle[i] = pop_actions[top_index_global_obstacle]

            pop2 = np.stack(pop)
            pop2 = pop2.reshape(self.group, self.cfg.quads_num_agents)
            # es_params = es_params.reshape(self.group, self.cfg.quads_num_agents,-1)

            tmp_goal = pop2[top_index_global_goal][index_local_rank_goal]# esparam sort
            tmp_goal2=pop2[top_index_global_goal]# mutate by actor, donnot sort because halfaction cannot sort
            pop_actor_goal.append(tmp_goal2)#2dimension

            tmp_obstacle = pop2[top_index_global_obstacle][index_local_rank_obstacle]
            tmp_obstacle2 = pop2[top_index_global_obstacle]
            pop_actor_obstacle.append(tmp_obstacle2)  # 2dimension
            
            for j, actor in enumerate(tmp_goal):
                es_params_goal[j] = actor.get_params()
            for j, actor in enumerate(tmp_obstacle):
                es_params_obstacle[j] = actor.get_params()
            # es_params_goal=es_params[top_index_global_goal][index_local_rank_goal]
            # es_params_obstacle=es_params[top_index_global_obstacle][index_local_rank_obstacle]


            elite_params_goal.append(self.weights_local @ es_params_goal)
            elite_params_obstacle.append(self.weights_local @ es_params_obstacle)
            random.shuffle(pop)
            # indice = np.random.permutation(self.pop_size)
            # pop = pop[int(indice)]


        idx_goal = np.argsort(np.array(elite_reward_goal).squeeze())[::-1] #halfaction if a=[tensorhigh,tensorhigh] cannot sort by idx, and if cannot use torch.tensor(a) ,only np.array([numpy,numpy])
        idx_obstacle = np.argsort(np.array(elite_reward_obstacle).squeeze())[::-1]
        self.worst_index[0]=worst_indices[np.argsort(np.array(elite_reward).squeeze())[0]]
        self.elitism_param = 2*self.sigmoid(2*self.alpha_e[0]) * elite_params_goal[idx_goal[0]]+\
                             (2-2*self.sigmoid(2*self.alpha_e[0])) * elite_params_obstacle[idx_obstacle[0]]

        pop_actor_goal=np.concatenate(pop_actor_goal,axis=0)
        pop_actor_obstacle=np.concatenate(pop_actor_obstacle,axis=0)
        # print('Mutate pop1')
        if rollout_step==(self.cfg.rollout-1) and self.flags==True:
            self.proximal_mutate(half_elite_pop_actions_goal,pop_actor_goal,
                                 half_elite_pop_actions_obstacle,pop_actor_obstacle)
            self.flags[0] = False
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        
        # half_elite_pop_actions = torch.tensor(half_elite_pop_actions).reshape(self.cfg.N_iterations*self.cfg.quads_num_agents,-1,4)
        # pop_actor=np.array(pop_actor).reshape(-1)
        #
        # tot_size = len(self.elitism_param)
        # normal = Normal(torch.zeros_like(self.elitism_param),torch.ones_like(self.elitism_param)*self.cfg.mut_mag)
        #
        # jacobin = torch.zeros(self.cfg.N_iterations*self.cfg.quads_num_agents,4,tot_size).to(self.device)
        # grad_output= torch.zeros(half_elite_pop_actions[0].size()).to(self.device)
        #
        # for i in range(len(pop_params)):
        #     self.es_params[i] = copy.deepcopy(pop_params[i])
        # for i in range(len(pop_params),self.pop_size):
        #     for j in range(4):  # action_space
        #         pop_actor[i].zero_grad()
        #         grad_output.zero_()
        #         grad_output[:,j]=1
        #
        #         half_elite_pop_actions[i].backward(grad_output,retain_graph=True)
        #         jacobin[i][j]=pop_actor[i].get_grads()
        #
        #     scaling = torch.sqrt((jacobin[i]**2).sum(0))
        #     scaling[scaling==0]=1.0
        #     scaling[scaling<0.01]=0.01
        #     delta = normal.sample()
        #     delta /= scaling
        #     self.es_params[i]=self.es_params[i-64]+delta #self.es_params<->pop

        elif rollout_step == (self.cfg.rollout-1) and self.flags==False:
            # print('update pop by cem')
            self.update(np.array(elite_params_goal)[idx_goal],np.array(elite_params_obstacle)[idx_obstacle])
            self.update_pop_params()
            self.flags[0]=True
        self.es_params[-8:] = self.elitism_param  # inject elitism to evo:copy identical elitism_param into last 8 position of pop

        return state_posterior, self.elitism_param
            

    def update_rnn(self,state_post,action):
        # obs2 = obs['obs'].reshape(self.group, self.cfg.quads_num_agents, -1)
        # state_posterior = self.rssm.posterior(obs2, self.rnn_hidden)
        # rnn_hidden = self.rnn_hidden
        action = action.repeat(self.pop_size // action.shape[0], 1)
        # print('action',action.shape[0])
        _, self.rnn_hidden = self.rssm.prior(state_post.sample(),
                                             action.reshape(self.group, self.cfg.quads_num_agents, -1),
                                             self.rnn_hidden)
        return self.rnn_hidden

        
    def novelty(self,rl_actor):
        indice=[[0, 1, 2, 3, 4, 5, 6, 7],[0,0,0,0,0,0,0,0]]
        indices=tuple(np.array(indice))
        observations = self.traj_tensors['obs'].index(indices)
        obs=torch.tensor(observations['obs'],device=self.device,dtype=torch.float32)
        obs=obs.repeat(self.group+1,1) # add rl_obs, ry 8 are all the same
        obs=obs.reshape(self.group+1,self.cfg.quads_num_agents,-1)

        rnn_hidden = torch.zeros(self.group+1,self.cfg.quads_num_agents,self.rssm.rnn_hidden_dim,device=self.device)
        pop=copy.deepcopy(self.pop)

        for i in range(8):
            pop.append(rl_actor)  # add rl_actor, last 8 actor are all the same


        mu = np.zeros(self.num_params)

        Interrupt=[]
        std=np.sqrt(self.cov)
        for i in range(self.pop_size):
            I = np.random.randn(1, self.num_params)
            epsilon = std * I
            interrupt=mu + epsilon
            params=pop[i].get_params()
            interrupt=interrupt.reshape(-1)#[1,3,,,,,]
            pop[i].set_params(params+interrupt)
            Interrupt.append(I)

        Interrupt=np.array(Interrupt)
        Interrupt=Interrupt.reshape(128,-1)#128,num_params

        state_posterior = self.rssm.posterior(obs, rnn_hidden)
        pop_action,_ = self.get_action(obs.reshape(-1, obs.shape[2]), pop)
        state = state_posterior.sample()
        obs_all=[]

        for t in range(self.cfg.horizon):
            next_state_prior, rnn_hidden = self.rssm.prior(state, pop_action, rnn_hidden)
            state = next_state_prior.sample()
            obs_t = self.obs_model(state, rnn_hidden)
            pop_action,_ = self.get_action(obs_t.reshape(-1, obs.shape[2]), pop)
            obs_all.append(obs_t[:,:,:3])#self_pos, as behavior characteristic
        # bc=np.concatenate(obs_all,axis=-1)
        distance=[]
        for bc in obs_all:
            rl_mean_bc = bc[-1].mean(dim=0)
            dist=torch.sqrt(torch.sum(torch.square(bc[:-1] - rl_mean_bc),dim=-1)) #sum with respect to horizon step :2dimension
            distance.append(dist.unsqueeze(dim=-1).detach().numpy())# dist 3dimension
        distance = np.concatenate(distance,axis=-1)
        distance=np.mean(distance,axis=-1)
        distance=distance.reshape(-1, 1)  # 128,1

        novelty=distance*Interrupt*(1/(self.cfg.horizon*std))
        # novelty=torch.tensor(novelty)

        return novelty


    def update_by_novelty(self,updated_learner,grads):
        

        # rl_theta=[]
        # grads=[]
        # for k, p in updated_rl_theta.items():
        #     rl_theta.append(copy.deepcopy(p.cpu().data.numpy()))
        #     grads.append(copy.deepcopy(p.grad.cpu().data.numpy()))
        rl_theta=updated_learner.get_params()
        # rl_theta=rl_theta[: self.num_params]
        # critic_grads=np.array(grads).flatten()
        # critic_grads=critic_grads[self.num_params: -1032]
        # grads=updated_learner.get_grads()
        # rl_actor=copy.deepcopy(self.pop[0])
        # rl_actor.set_params(rl_theta)
        novelty=self.novelty(copy.deepcopy(updated_learner))
        mu=torch.tensor(rl_theta)
        cov=self.cfg.cov_init * torch.ones(self.num_params)
        distribution = Normal(mu, torch.sqrt(cov))
        for i in range(self.pop_size):
            c = distribution.sample()
            c= c.numpy()
            self.es_params[i]=rl_theta+c*grads+self.prob[0]*c*novelty[i]
        

        
        # print('n')


            



        
        
        
            


