import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.distributions import Normal
import numpy as np
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, sample_actions_log_probs2
from sample_factory.algorithms.appo.model_utils import nonlinearity
from sample_factory.algorithms.utils.action_distributions import calc_num_logits, get_action_distribution
from torch.nn.utils import clip_grad_norm_
from sample_factory.algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from sample_factory.model.common_encoder import Hypernet, Actor_QuadSelfEncoder, Actor_ObstacleEncoder, \
    Actor_QuadNeighborhoodEncoderAttention, AdvEncoderAttention, Gating_layer, PositionwiseFeedForward, \
    PositionwiseFeedForward2
from sample_factory.model.histroy_Tranformer_encoder import HistoryEncoder
from sample_factory.model.transformer_intention_layer import TransformerIntentionLayer
from sample_factory.model.transformer_latent_layer import TransformerLatentLayer


class HyperJD2TStateSpaceModel(nn.Module):
    def __init__(self, cfg, action_space, state_dim, action_dim, rnn_hidden_dim, num_agents, num_neighbor_obs,
                 num_oppo_obs,
                 hidden_dim=256, min_stddev=0.1, act=torch.tanh, device=None):
        super(RecurrentStateSpaceModel, self).__init__()
        self.cfg = cfg
        self.num_agents = num_agents  # number of ally or oppo
        self.action_space = action_space
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim  # 64
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(hidden_dim, state_dim)
        self.fc_state_action_std = nn.Linear(hidden_dim, state_dim)
        # self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim,hidden_dim)
        self.hyper_fc_delta_hidden_w = Hypernet(cfg, input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                main_input_dim=hidden_dim, main_output_dim=hidden_dim,
                                                )

        self.fc_state_mean_prior = nn.Linear(state_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(state_dim, state_dim)

        self.fc_state_mean_prior2 = nn.Linear(state_dim, state_dim)
        self.fc_state_stddev_prior2 = nn.Linear(state_dim, state_dim)

        # self.hyper_fc_rnn_hidden_embedded_obs_w = Hypernet(cfg, input_dim=54, hidden_dim=hidden_dim,
        #                                                    main_input_dim=54, main_output_dim=hidden_dim,
        #                                                    )
        # self.hyper_fc_rnn_hidden_embedded_obs_w2 = Hypernet(cfg, input_dim=54+10*cfg.num_obstacle_obs +state_dim,
        #                                                    hidden_dim=hidden_dim,
        #                                                    main_input_dim=54+10*cfg.num_obstacle_obs +state_dim,
        #                                                    main_output_dim=state_dim,
        #                                                    )
        # nn.Linear(rnn_hidden_dim + 54, hidden_dim) #if add obstacle and observe 2, change into +74

        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        # self.rnn = nn.GRUCell(hidden_dim,rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act
        # self.action_parameterization = action_parameterization

        # self.embedding_mlp = nn.Sequential(
        # nn.Linear(self.state_dim, hidden_dim),
        # nn.ReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(),
        # )

        self.hyper_embedding_mlp_w = Hypernet(cfg, input_dim=state_dim + self.action_dim, hidden_dim=hidden_dim,
                                              main_input_dim=state_dim + self.action_dim, main_output_dim=state_dim,
                                              )
        self.hyper_embedding_mlp_w2 = Hypernet(cfg, input_dim=state_dim + self.action_dim, hidden_dim=hidden_dim,
                                               main_input_dim=state_dim + self.action_dim, main_output_dim=state_dim,
                                               )

        self.neighbor_value_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.neighbor_value_mlp2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # self.hyper_neighbor_value_mlp_w = Hypernet(input_dim=hidden_dim,
        # hidden_dim=hidden_dim,
        # main_input_dim=hidden_dim,
        # main_output_dim=hidden_dim,
        # )

        # self.attention_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     # neighbor_hidden_size * 2 because we concat e_i and e_m
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, 1),
        # )
        self.self_obs_dim = 9
        self.neighbor_hidden_size = hidden_dim
        self.neighbor_obs_dim = 6
        self.adv_obs_dim = 12
        self.use_spectral_norm = cfg.use_spectral_norm
        self.obstacle_obs_dim = 12
        self.obstacle_hidden_size = hidden_dim
        self.num_use_neighbor_obs = num_neighbor_obs
        self.num_adv_obs = num_oppo_obs

        self.histroy_encoder = HistoryEncoder(cfg, hidden_dim, state_dim, self.self_obs_dim, self.neighbor_obs_dim,
                                              self.num_use_neighbor_obs, self.adv_obs_dim, self.num_agents,self.num_adv_obs)

        self.intention_encoder = nn.Sequential(
            *[TransformerIntentionLayer(cfg, hidden_dim, state_dim, cfg.num_heads) for _ in range(cfg.num_layer)])
        self.latent_encoder = nn.Sequential(
            *[TransformerLatentLayer(cfg, hidden_dim, state_dim, cfg.num_heads) for _ in range(cfg.num_layer)])

        self.self_encoder = Actor_QuadSelfEncoder(cfg, self.self_obs_dim,
                                                  self.hidden_dim, self.use_spectral_norm)

        self.obstacle_encoder = Actor_ObstacleEncoder(cfg, self.self_obs_dim,
                                                      self.obstacle_obs_dim,
                                                      self.obstacle_hidden_size,
                                                      self.use_spectral_norm)

        self.neighbor_encoder = Actor_QuadNeighborhoodEncoderAttention(cfg, self.neighbor_obs_dim,
                                                                       self.neighbor_hidden_size,
                                                                       self.use_spectral_norm,
                                                                       self.self_obs_dim,
                                                                       self.num_use_neighbor_obs)
        self.adv_encoder = AdvEncoderAttention(cfg, self.adv_obs_dim,
                                               self.neighbor_hidden_size,
                                               self.use_spectral_norm,
                                               self.self_obs_dim,
                                               self.num_adv_obs)
        self.feed_forward = nn.Sequential(
            fc_layer(4* self.neighbor_hidden_size, self.neighbor_hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(self.neighbor_hidden_size, state_dim, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )
        self.mlp = nn.Sequential(
            fc_layer(hidden_dim + 4, self.hidden_dim, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(self.hidden_dim, self.hidden_dim, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )

        self.gating = Gating_layer(self.hidden_dim)
        self.gating2 = Gating_layer(self.hidden_dim)
        self.positionforward = PositionwiseFeedForward(self.hidden_dim)
        self.positionforward2 = PositionwiseFeedForward2(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        self.mlp2 = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                  nn.Linear(self.hidden_dim, state_dim))

        self.latent_gating = Gating_layer(self.hidden_dim)
        self.latent_gating2 = Gating_layer(self.hidden_dim)
        self.latent_positionforward = PositionwiseFeedForward(self.hidden_dim)
        self.latent_positionforward2 = PositionwiseFeedForward2(self.hidden_dim)
        self.latent_layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        self.latent_mlp2 = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh(),
                                         nn.Linear(self.hidden_dim, state_dim))

        self.mlp3 = nn.Linear(self.adv_obs_dim, state_dim)
        self.mlp4 = nn.Linear(self.adv_obs_dim, state_dim)

        self.positionforward_latent = PositionwiseFeedForward(state_dim)
        self.gating_latent = Gating_layer(state_dim)
        self.mlp5_latent = nn.Linear(2 * state_dim, state_dim)
        self.mlp6_latent = nn.Linear(2 * self.hidden_dim, state_dim)

        self.positionforward_latent2 = PositionwiseFeedForward(state_dim)
        self.gating_latent2 = Gating_layer(state_dim)
        self.mlp5_latent2 = nn.Linear(2 * state_dim, state_dim)
        
        
        #self.linear = nn.Linear(2 * state_dim, self.action_dim)


    def prior(self, state, action):

        # state_embeddings = self.embedding_mlp(state)
        state_a = torch.cat([state, action], dim=-1)
        # embedding_mlp_w = self.hyper_embedding_mlp_w(state)
        # neighbor_embedding = torch.matmul(state.unsqueeze(2), embedding_mlp_w)
        # neighbor_embedding = self.act(neighbor_embedding)
        # state_embeddings = neighbor_embedding.squeeze(2)
        embedding_mlp_w = self.hyper_embedding_mlp_w(state_a)
        neighbor_embedding = torch.matmul(state_a.unsqueeze(2), embedding_mlp_w)
        neighbor_embedding = self.act(neighbor_embedding)
        state_embeddings = neighbor_embedding.squeeze(2)

        # state_embeddings_mean = torch.mean(state_embeddings, dim=1)
        # state_embeddings_mean = state_embeddings_mean.view(state.shape[0], -1, self.rnn_hidden_dim)
        # state_embeddings_mean_repeat = state_embeddings_mean.repeat(1, state.shape[1], 1)

        # state_values = self.neighbor_value_mlp(state_embeddings).reshape(-1, state_embeddings.shape[-1])
        state_values = self.neighbor_value_mlp(state_embeddings)
        # neighbor_value_mlp_w=self.hyper_neighbor_value_mlp_w(state_embeddings)
        # neighbor_value_mlp = torch.matmul(neighbor_embedding, neighbor_value_mlp_w)
        # neighbor_value_mlp = self.act(neighbor_value_mlp)
        # neighbor_values =neighbor_value_mlp.squeeze(2)

        # attention_mlp_input = torch.cat((state_embeddings, state_embeddings_mean_repeat), dim=-1)
        # attention_weights = self.attention_mlp(attention_mlp_input).view(attention_mlp_input.shape[0], -1)
        # attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        # attention_weights_softmax = attention_weights_softmax.view(-1, 1)
        attention_weights = torch.matmul(state, state_embeddings.transpose(1, 2))
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=-1)

        # final_state_embedding = attention_weights_softmax * state_values
        # final_state_embedding = final_state_embedding.view(state.shape[0], -1, state_values.shape[-1])
        # final_state_embedding = torch.sum(final_state_embedding, dim=1)
        final_state_embedding = torch.matmul(attention_weights_softmax, state_values)

        Gating_outputs = self.gating(final_state_embedding, final_state_embedding)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward2(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.positionforward(transformer_out_x)
        gate_z = self.gating(final_state_embedding, out_x)
        embedding = final_state_embedding

        all_embedding = torch.cat((gate_z, embedding), dim=-1)
        z = self.mlp2(all_embedding)
        # state_attention =state_attention.reshape(-1,self.rnn_hidden_dim)
        # states =state_attention.repeat_interleave(state.shape[1],dim=0)
        # states = states.view(state.shape[0],-1,self.rnn_hidden_dim)
        # final_state_embedding = final_state_embedding.view(state.shape[0], -1, state_values.shape[-1])
        # states = final_state_embedding.repeat(1, state.shape[1], 1)

        # z_mean = self.act(self.fc_state_action(torch.cat([states, action], dim=-1)))
        # z_std = F.softplus(self.act(self.fc_state_action_std(torch.cat([states, action], dim=-1))))+ self._min_stddev
        # z_mean = self.act(self.fc_state_action(hidden))
        # z_std = F.softplus(self.act(self.fc_state_action_std(hidden))) + self._min_stddev
        # z_distriution=Normal(z_mean, z_std)
        # z= z_distriution.rsample()
        # state_z=torch.cat([state, z],dim=-1)
        # fc_delta_w = self.hyper_fc_delta_hidden_w(state_z)
        # hidden=torch.matmul(state_z.unsqueeze(2), fc_delta_w)
        # hidden = self.act(hidden)
        # hidden = hidden.squeeze(2)
        # rnn_hidden = self.rnn(hidden.reshape(-1,hidden.shape[-1]),rnn_hidden.reshape(-1,rnn_hidden.shape[-1]))
        # rnn_hidden=rnn_hidden.reshape(-1, state.shape[1], rnn_hidden.shape[-1])
        # hidden=self.act(self.fc_rnn_hidden(rnn_hidden))
        # fc_rnn_hidden_w = self.hyper_fc_rnn_hidden_w(state)
        # fc_rnn_hidden = torch.matmul(hidden.unsqueeze(2), fc_rnn_hidden_w)
        # hidden = self.act(fc_rnn_hidden)
        # hidden = hidden.squeeze(2)

        mean = self.fc_state_mean_prior(z)
        stddev = F.softplus(self.fc_state_stddev_prior(z)) + self._min_stddev
        delta_s_distribution = Normal(mean, stddev)
        delta_s = delta_s_distribution.rsample()

        # delta_s_distribution=Normal(mean.reshape(-1,mean.shape[-1]),stddev.reshape(-1,stddev.shape[-1]))
        return delta_s, z, delta_s_distribution

    

    def posterior(self, next_obs):

        obs_self = next_obs[:, :, :self.self_obs_dim]
        batch_size = obs_self.shape[0] * obs_self.shape[1]
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        self_embed = self.self_encoder(obs_self)
        neighborhood_embedding = self.neighbor_encoder(obs_self, next_obs, all_neighbor_obs_size, batch_size)
        adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
        all_adv_obs_size = self.adv_obs_dim * self.num_adv_obs
        adv_embedding = self.adv_encoder(next_obs, adv_obs_size, all_adv_obs_size, batch_size)
        all_obs_size = all_neighbor_obs_size + all_adv_obs_size
        obstacle_mean_embed = self.obstacle_encoder(next_obs, all_obs_size, batch_size)
        embeddings = torch.cat((self_embed, neighborhood_embedding), dim=-1)
        embeddings = torch.cat((embeddings,adv_embedding, obstacle_mean_embed), dim=-1)
        hidden = self.feed_forward(embeddings)

        # hidden_w = self.hyper_fc_rnn_hidden_embedded_obs_w(next_obs)
        # hidden = torch.matmul(next_obs.unsqueeze(2), hidden_w)
        # hidden = self.act(hidden)
        # hidden= hidden.squeeze(2)
        # hidden = self.act(self.fc_rnn_hidden_embedded_obs(
        # torch.cat([rnn_hidden, next_obs], dim=-1)))

        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        state_post_distribution=Normal(mean,stddev)
        state_posterior=state_post_distribution.rsample()
        return state_posterior

    def posterior_intention_latent(self, next_obs, history):
        """
        :param next_obs: T,N,D
        :param history: N,T,D
        :return: higher level intentions and lower level latent strategy
        """
        T, N, D = next_obs.size()
        k_intention, k_latent = self.histroy_encoder(history)
        # adv_posterior, neighbors_posterior=self.posterior(next_obs)
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
        all_adv_obs_size = self.adv_obs_dim * self.num_adv_obs
        adv_obs = next_obs[:, :, adv_obs_size:adv_obs_size + all_adv_obs_size]
        adv_obs = adv_obs.reshape(next_obs.shape[0], next_obs.shape[1], -1, self.adv_obs_dim).transpose(1,
                                                                                                        2)  # # T,N,K,D-->T,K,N, D
        q_intention = torch.zeros((next_obs.shape[0], adv_obs.shape[1], adv_obs.shape[2], self.adv_obs_dim)).to(
            device=self.device)  # T,K,N, D
        adv_obs_emb = self.mlp3(adv_obs)
        q_intention = self.mlp4(q_intention)
        for intention_layer in self.intention_encoder:
            q_intention, selfcross_attention = intention_layer(adv_obs_emb, k_intention, q_intention)
        out_x = self.positionforward_latent(q_intention)
        gate_z = self.gating_latent(selfcross_attention, out_x)
        all_embedding = torch.cat((gate_z, selfcross_attention), dim=-1)
        q_latent = self.mlp5_latent(all_embedding)

        obs_self = next_obs[:, :, :self.self_obs_dim]  # T,N,D
        batch_size = obs_self.shape[0] * obs_self.shape[1]
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        self_embed = self.self_encoder(obs_self)
        neighborhood_embedding = self.neighbor_encoder(obs_self, next_obs, all_neighbor_obs_size, batch_size)
        neighbor_embedding = torch.cat((self_embed, neighborhood_embedding), dim=-1)
        neighbor_embed = self.mlp6_latent(neighbor_embedding)

        for i, latent_layer in enumerate(self.latent_encoder):
            q_latent, selfcross_attention2 = latent_layer.forward20(neighbor_embed, k_latent, q_latent, q_intention)

        out_x2 = self.positionforward_latent2(q_latent)
        gate_z2 = self.gating_latent2(selfcross_attention2, out_x2)
        all_embedding2 = torch.cat((gate_z2, selfcross_attention2), dim=-1)
        q_latent = self.mlp5_latent2(all_embedding2)

        # q_intention=q_intention.transpose(1,2).reshape(T,N,-1) # T,N,K*D
        q_intention = q_intention.transpose(1, 2).reshape(T, -1, q_intention.shape[-1])  # T,N*K,D
        q_latent = q_latent.transpose(1, 2).reshape(T, -1, q_latent.shape[-1])
        
        
        return q_intention, q_latent

    def posterior_intention_latent2(self, next_obs, history):
        """
        :param next_obs: B,N,D
        :param history: B,N，T,D
        :return: higher level intentions and lower level latent strategy
        """
        B, N, D = next_obs.size()
        k_intention, k_latent = self.histroy_encoder.forward2(history) #B, N, T, D
        # adv_posterior, neighbors_posterior=self.posterior(next_obs)
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
        all_adv_obs_size = self.adv_obs_dim * self.num_adv_obs
        adv_obs = next_obs[:, :, adv_obs_size:adv_obs_size + all_adv_obs_size]
        adv_obs = adv_obs.reshape(next_obs.shape[0], next_obs.shape[1], -1, self.adv_obs_dim).transpose(1,
                                                                                                        2)  # # B,N,K,D-->B,K,N, D
        q_intention = torch.zeros((next_obs.shape[0], adv_obs.shape[1], adv_obs.shape[2], self.adv_obs_dim)).to(
            device=self.device)  # B,K,N, D
        adv_obs_emb = self.mlp3(adv_obs)
        q_intention = self.mlp4(q_intention)
        for i, intention_layer in enumerate(self.intention_encoder):
            q_intention, selfcross_attention = intention_layer.forward2(adv_obs_emb, k_intention, q_intention)
        out_x = self.positionforward_latent(q_intention)
        gate_z = self.gating_latent(selfcross_attention, out_x)
        all_embedding = torch.cat((gate_z, selfcross_attention), dim=-1)
        q_latent = self.mlp5_latent(all_embedding) #B,K,N, D

        obs_self = next_obs[:, :, :self.self_obs_dim]  # B,N,D
        batch_size = obs_self.shape[0] * obs_self.shape[1]
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        self_embed = self.self_encoder(obs_self)
        neighborhood_embedding = self.neighbor_encoder(obs_self, next_obs, all_neighbor_obs_size, batch_size)
        neighbor_embedding = torch.cat((self_embed, neighborhood_embedding), dim=-1)
        neighbor_embed = self.mlp6_latent(neighbor_embedding)

        for i, latent_layer in enumerate(self.latent_encoder):
            q_latent, selfcross_attention2 = latent_layer.forward3(neighbor_embed, k_latent, q_latent, q_intention)

        out_x2 = self.positionforward_latent2(q_latent)
        gate_z2 = self.gating_latent2(selfcross_attention2, out_x2)
        all_embedding2 = torch.cat((gate_z2, selfcross_attention2), dim=-1)
        q_latent = self.mlp5_latent2(all_embedding2) # B,K,N,D

        # q_intention=q_intention.transpose(1,2).reshape(T,N,-1) # T,N,K*D
        q_intention = q_intention.transpose(1, 2).reshape(B, -1, q_intention.shape[-1])  # B,N*K,D
        q_latent = q_latent.transpose(1, 2).reshape(B, -1, q_latent.shape[-1])        
        
        

        return q_intention, q_latent
        
    def posterior_intention_latent3(self, next_obs, history):
        """
        :param next_obs: N,D
        :param history: T，N，D # N,T,D
        :return: higher level intentions and lower level latent strategy
        """
        N, D = next_obs.size()
        T=1
        next_obs=next_obs.unsqueeze(0) #1,N,D
        k_intention, k_latent = self.histroy_encoder.forward(history.transpose(1,0)) # N,T,D
        # adv_posterior, neighbors_posterior=self.posterior(next_obs)
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
        all_adv_obs_size = self.adv_obs_dim * self.num_adv_obs
        adv_obs = next_obs[:, :, adv_obs_size:adv_obs_size + all_adv_obs_size]
        adv_obs = adv_obs.reshape(next_obs.shape[0], next_obs.shape[1], -1, self.adv_obs_dim).transpose(1,
                                                                                                        2)  # # 1,N,K,D-->1,K,N, D
        q_intention = torch.zeros((next_obs.shape[0], adv_obs.shape[1], adv_obs.shape[2], self.adv_obs_dim)).to(
            device=self.device)  # 1,K,N, D
        adv_obs_emb = self.mlp3(adv_obs)
        q_intention = self.mlp4(q_intention)
        for intention_layer in self.intention_encoder:
            q_intention, selfcross_attention = intention_layer(adv_obs_emb, k_intention, q_intention)
        out_x = self.positionforward_latent(q_intention)
        gate_z = self.gating_latent(selfcross_attention, out_x)
        all_embedding = torch.cat((gate_z, selfcross_attention), dim=-1)
        q_latent = self.mlp5_latent(all_embedding)

        obs_self = next_obs[:, :, :self.self_obs_dim]  # 1,N,D
        batch_size = obs_self.shape[0] * obs_self.shape[1]
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        self_embed = self.self_encoder(obs_self)
        neighborhood_embedding = self.neighbor_encoder(obs_self, next_obs, all_neighbor_obs_size, batch_size)
        neighbor_embedding = torch.cat((self_embed, neighborhood_embedding), dim=-1)
        neighbor_embed = self.mlp6_latent(neighbor_embedding)

        for i, latent_layer in enumerate(self.latent_encoder):
            q_latent, selfcross_attention2 = latent_layer.forward20(neighbor_embed, k_latent, q_latent, q_intention)

        out_x2 = self.positionforward_latent2(q_latent)
        gate_z2 = self.gating_latent2(selfcross_attention2, out_x2)
        all_embedding2 = torch.cat((gate_z2, selfcross_attention2), dim=-1)
        q_latent = self.mlp5_latent2(all_embedding2) # 1,K,N,D

        # q_intention=q_intention.transpose(1,2).reshape(T,N,-1) # T,N,K*D
        q_intention = q_intention.squeeze(0).transpose(1, 0).reshape(-1,q_intention.shape[-1])  # N*K,D
        q_latent = q_latent.squeeze(0).transpose(1, 2).reshape(-1, q_latent.shape[-1])

        return q_intention, q_latent

    # def posterior_delta(self, next_obs,z):
    #     obs_z=torch.cat([next_obs, z], dim=-1)
    #     hidden_w = self.hyper_fc_rnn_hidden_embedded_obs_w2(obs_z)
    #     hidden = torch.matmul(obs_z.unsqueeze(2), hidden_w)
    #     hidden = self.act(hidden)
    #     delta_posterior = hidden.squeeze(2)
    #     return delta_posterior

    def forward(self, obs, action, next_obs):
        state = self.posterior(obs)
        delta_prior, z, _, _, _ = self.prior(state[:-1], action[:-1])
        next_state_posterior = self.posterior(next_obs)
        return delta_prior, next_state_posterior

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # def get_action(self, rl_policy, obs):
    #     action_mean = rl_policy(obs, 'pop_actor')
    #     action_distribution_params, action_distribution = self.action_parameterization(action_mean)
    #     # for non-trivial action spaces it is faster to do these together
    #     action, log_prob_action = sample_actions_log_probs(action_distribution)
    #     action = action.reshape(self.L * self.B, self.N, -1)
    #     log_prob_action = log_prob_action.reshape(self.L * self.B, self.N, -1)
    #     action_logits = action_distribution_params.reshape(self.L * self.B, self.N, -1)
    #     return action, log_prob_action, action_logits

    def calculate_discounted_sum(self, x, discount, x_last=None):
        """
        Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
        :param x: ndarray of shape [num_steps, num_envs]
        :param dones: ndarray of shape [num_steps, num_envs]
        :param discount: float in range [0,1]
        :param x_last: iterable of shape [num_envs], value at the end of trajectory. None interpreted as zero(s).
        """
        x_last = np.zeros_like(x[0]) if x_last is None else np.array(x_last, dtype=np.float32)
        cumulative = x_last

        discounted_sum = np.zeros_like(x)
        for i in reversed(range(len(x))):
            cumulative = x[i] + discount * cumulative
            discounted_sum[i] = cumulative
        return discounted_sum

    def calculate_gae(self, rewards, values, gamma, gae_lambda):
        """
        Computing discounted cumulative sum, taking episode terminations into consideration. Follows the
        Generalized Advantage Estimation algorithm.
        See unit tests for details.

        :param rewards: actual environment rewards
        :param dones: True if absorbing state is reached
        :param values: estimated values
        :param gamma: discount factor [0,1]
        :param gae_lambda: lambda-factor for GAE (discounting for longer-horizon advantage estimations), [0,1]
        :return: advantages and discounted returns
        """
        # assert len(rewards) == len(dones)
        assert len(rewards) + 1 == len(values)

        # section 3 in GAE paper: calculating advantages
        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages = self.calculate_discounted_sum(deltas, gamma * gae_lambda)

        # targets for value function - this is just a simple discounted sum of rewards
        discounted_returns = self.calculate_discounted_sum(rewards, gamma, values[-1])

        return advantages.astype(np.float32), discounted_returns.astype(np.float32)

    def _calculate_gae(self, reward_v, values_v):
        """
        Calculate advantages using Generalized Advantage Estimation.
      This is leftover the from previous version of the algorithm.
        Perhaps should be re-implemented in PyTorch tensors, similar to V-trace for uniformity.
        """

        rewards = reward_v.reshape(reward_v.shape[0], -1).transpose((1, 0))  # [E, H]
        values_arr = values_v.reshape(values_v.shape[0], -1).transpose((1, 0))  # [E, H]

        # calculating fake values for the last step in the rollout
        # this will make sure that advantage of the very last action is always zero
        # values = []
        last_value, last_reward = values_arr[:, -1], rewards[:, -1]
        next_value = (last_value - last_reward) / self.cfg.gamma
        values = np.concatenate((values_arr, next_value.reshape(-1, 1)), axis=-1)  # [H] -> [H+1]
        # for i in range(len(values_arr)):
        #     last_value, last_reward = values_arr[i][-1], rewards[i, -1]
        #     next_value = (last_value - last_reward) / self.cfg.gamma
        #     values.append(list(values_arr[i]))
        #     values[i].append(float(next_value))  # [T] -> [T+1]

        # calculating returns and GAE
        rewards = rewards.transpose((1, 0))  # [E, H] -> [H, E]

        # dones = dones.transpose((1, 0))  # [E, H] -> [H, E]
        values = np.asarray(values).transpose((1, 0))  # [E, H+1] -> [H+1, E]

        advantages, returns = self.calculate_gae(
            rewards, values, self.cfg.gamma, self.cfg.gae_lambda)

        # transpose tensors back to [E, T] before creating a single experience buffer
        # advantages = advantages.transpose((1, 0))  # [T, E] -> [E, T]
        # returns = returns.transpose((1, 0))  # [T, E] -> [E, T]
        returns = returns[:, :, np.newaxis]  # [H, E] -> [H, E, 1]
        advantages = torch.tensor(advantages).reshape(-1)
        returns = torch.tensor(returns).reshape(-1)

        return advantages, returns

    

    def rollout(self, steps, obs_model, reward_model, rl_policy, obs0, history, delta_s, state_post, lambda_t, idx):

        with torch.no_grad():
            rewards = []
            actions = []
            deltas_2 = []
            log_prob_actions = []
            log_prob_deltas = []
            action_logits = []

            obs_obstacles2 = []

            distance2 = []
            # self.action_parameterization = action_parameterization
            # action, log_prob_action, action_logit = self.get_action(rl_policy, obs)

            obs = obs0[idx]
            obs1=torch.cat(obs0,dim=1)
            #print("nnnnnn",obs1.reshape(-1, obs1.shape[-1]).shape)
            action = rl_policy.forward2(obs.reshape(-1, obs.shape[-1]), idx)[0].reshape(obs.shape[0], -1,
                                                                                        self.action_space.shape[0])
            observations = torch.zeros(steps, obs.shape[0] * obs.shape[1], obs.shape[-1])
            observations_obstacle = torch.zeros(steps, obs.shape[0] * obs.shape[1], 2 * self.cfg.num_landmarks,
                                                device=self.device)  # px,py
            Distance = torch.zeros(steps, obs.shape[0] * obs.shape[1], self.cfg.num_landmarks,
                                   device=self.device)
            for t in range(steps):
                obs_obstacles = []
                distance = []
                delta_s2, z, _ = self.prior(state_post, action)
                state_post = state_post + delta_s2
                obs2 = obs_model(state_post, z)

                intention, latent, action_oppo=self.posterior_intention_latent(obs2, history)
                action_oppo=action_oppo.reshape(obs.shape[0],obs.shape[1],-1)
                intention_latent=torch.cat((intention,latent),dim=-1).reshape(obs.shape[0],self.num_agents,-1)
                obs_state2=torch.cat((obs2,intention_latent),dim=-1)

                action0, log_prob_action, action_logit, _ = rl_policy.forward2(obs_state2.reshape(-1, obs_state2.shape[-1]), idx)
                action=torch.cat((action0.reshape(obs.shape[0],obs.shape[1],-1),action_oppo),dim=-1)

                log_prob_action = log_prob_action.reshape(-1, self.num_agents, 1)
                action_logit = action_logit.reshape(-1, self.num_agents, action_logit.shape[-1])

                reward = reward_model(state_post, z)

                
                
                delta_s_2 = delta_s2.reshape(-1, delta_s2.shape[-1])
                # log_prob_delta_s = log_prob_delta_s.reshape(-1, 1)
                obs_all2 = obs_state2.reshape(-1, obs_state2.shape[-1])
                action2 = action0.reshape(-1, action0.shape[-1])
                reward = reward.reshape(-1, reward.shape[-1])
                action_logit = action_logit.reshape(-1, action_logit.shape[-1])
                log_prob_action = log_prob_action.reshape(-1, log_prob_action.shape[-1])
                # [[T*N,dim],...]
                deltas_2.append(delta_s_2.detach().cpu().numpy())
                # log_prob_deltas.append(log_prob_delta_s.detach().cpu().numpy())
                observations[t] = obs_all2
                # obs_v.append(obs_t)
                actions.append(action2.detach().cpu().numpy())
                action_logits.append(action_logit.detach().cpu().numpy())
                rewards.append(reward.detach().cpu().numpy())
                log_prob_actions.append(log_prob_action.detach().cpu().numpy())

            # next_states=np.concatenate(next_states,axis=-1)
            
            deltas_2 = np.stack(deltas_2)
            # log_prob_deltas = np.stack(log_prob_deltas)
            # observations = np.stack(observations)  # (H,batch,dim):batch=T*N
            # observations2 = np.concatenate((obs.reshape(1,-1,obs.shape[-1]),observations),axis=0)
            actions = np.stack(actions)
            action_logits = np.stack(action_logits)
            rewards = np.stack(
                rewards) 
            # rewards_v=rewards.transpose(1,0)#(T,N)

            log_prob_actions = np.stack(log_prob_actions)

            # obs_v = observations.reshape(N, -1, obs.shape[1])  # 8as a group at every t to compute values
            # obs_v = obs_v.transpose((1, 0, 2))  # (T,N,shape[1])
            # log_prob_deltas2 = torch.tensor(log_prob_deltas.reshape(-1))
            obs_v = observations.reshape(-1, observations.shape[-1])  # [H*T*N,dim]

        return log_prob_actions, deltas_2, obs_v, actions, action_logits, rewards

    

    def rollout_policy(self, gpu_buffer, steps, obs_model, reward_model, rl_policy, obs, history,delta_s,
                       state_post, L, B, N,
                       critic_encoder, critic_linear, network_type, lambda_t, idx):
        self.L = L
        self.B = B
        self.N = N
        log_prob_actions, deltas_2, obs_v, actions, action_logits, rewards= self.rollout(steps,
                                                                                                             obs_model,
                                                                                                             reward_model,
                                                                                                             rl_policy,
                                                                                                             obs,history,
                                                                                                             delta_s,
                                                                                                             state_post,
                                                                                                             lambda_t,
                                                                                                             idx)

        # next_states=torch.tensor(next_states)#(N,T)->(N*T,) to buffer
        # observations = torch.tensor(observations.reshape(-1, obs.shape[1]))
        # observations2=torch.tensor(obs) # T,N,dim: for learner
        # deltas=torch.tensor(delta_s)
        # state_post=torch.tensor
        deltas_batch = torch.tensor(deltas_2.reshape(-1, deltas_2.shape[-1]))
        obs_v = obs_v.to(device=self.device)

        actions = torch.tensor(actions.reshape(-1, actions.shape[-1]))

        action_logits = torch.tensor(action_logits.reshape(-1, action_logits.shape[-1]))

        log_prob_actions = torch.tensor(log_prob_actions.reshape(-1))

        x, yt = critic_encoder(obs_v, network_type)
        critic_output = torch.cat((x, yt), dim=-1)
        values = critic_linear(critic_output)
        values_v = values.detach().cpu().numpy()  # [H*T*N,dim]
        values_v = values_v.reshape(steps, -1, values_v.shape[-1])  # [H,T*N,dim]
        # values_v = values_v.transpose(1, 0)  # (N,T)
        
        advantages, returns = self._calculate_gae(rewards, values_v)  # H*N
        values2 = values_v.reshape(-1)
        values2 = torch.tensor(values2)
        rewards = torch.tensor(rewards.reshape(-1))

        # gpu_buffer.log_prob_deltas = log_prob_deltas2.to(device=self.device)
        gpu_buffer.values2 = values2.to(device=self.device)
        gpu_buffer['obs']['obs2'] = obs_v.detach().to(device=self.device)
        gpu_buffer.delta_s = delta_s.detach().to(device=self.device)
        gpu_buffer.delta_s_rollout = deltas_batch.to(device=self.device)
        gpu_buffer.state_post = state_post.detach().to(device=self.device)
        gpu_buffer.rewards2 = rewards.to(device=self.device)
        gpu_buffer.rewards_cpu2 = rewards

        gpu_buffer.actions2 = actions.to(device=self.device)
        gpu_buffer.action_logits2 = action_logits.to(device=self.device)
        gpu_buffer.log_prob_actions2 = log_prob_actions.to(device=self.device)
        gpu_buffer.advantages2 = advantages.to(device=self.device)
        gpu_buffer.returns2 = returns.to(device=self.device)
        

        return gpu_buffer

    

    # def rollout_policy(self, gpu_buffer, steps, obs_self_model, obs_neighbor_model, reward_model, rl_policy, obs,
    #                    prev_state, L, B, N, action_parameterization,
    #                    critic_encoder, critic_linear):
    #     self.L = L
    #     self.B = B
    #     self.N = N
    #     next_states = []
    #     observations = []
    #     obs_v = []
    #     actions = []
    #     rewards = []
    #
    #     log_prob_actions = []
    #     action_logits = []
    #     self.action_parameterization = action_parameterization
    #     action, log_prob_action, action_logit = self.get_action(rl_policy, obs)
    #
    #     for t in range(steps):
    #         next_state_prior = self.prior(prev_state, action)
    #         prev_state = next_state_prior.sample()
    #         obs_t_self = obs_self_model(prev_state)  # T,N,DIM
    #         obs_t_neighbor = obs_neighbor_model(prev_state)
    #         obs_t = torch.cat((obs_t_self, obs_t_neighbor), dim=-1)
    #         action, log_prob_action, action_logit = self.get_action(rl_policy, obs_t.reshape(-1, obs.shape[1]))
    #         reward = reward_model(prev_state)
    #
    #         prev_state2 = prev_state.transpose(1, 0)
    #         prev_state2 = prev_state2.reshape(N, -1)
    #         obs_t2 = obs_t.transpose(1, 0)
    #         obs_t2 = obs_t2.reshape(N, -1)
    #
    #         action2 = action.transpose(1, 0)
    #         action2 = action2.reshape(N, -1)
    #         action_logit2 = action_logit.transpose(1, 0)
    #         action_logit2 = action_logit2.reshape(N, -1)
    #         log_prob_action2 = log_prob_action.transpose(1, 0)
    #         log_prob_action2 = log_prob_action2.reshape(N, -1)
    #         reward2 = reward.transpose(1, 0)
    #         reward2 = reward2.reshape(N, -1)  # (N,T)
    #
    #         # next_states.append(prev_state2)
    #         observations.append(obs_t2.detach().cpu().numpy())
    #         # obs_v.append(obs_t)
    #         actions.append(action2.detach().cpu().numpy())
    #         action_logits.append(action_logit2.detach().cpu().numpy())
    #         rewards.append(reward2.detach().cpu().numpy())
    #         log_prob_actions.append(log_prob_action2.detach().cpu().numpy())
    #
    #         # next_states=np.concatenate(next_states,axis=-1)
    #     observations = np.concatenate(observations, axis=-1)  # (N,T)
    #     actions = np.concatenate(actions, axis=-1)
    #     action_logits = np.concatenate(action_logits, axis=-1)
    #     rewards = np.concatenate(rewards, axis=-1)  # (N,T)
    #
    #     # rewards_v=rewards.transpose(1,0)#(T,N)
    #
    #     log_prob_actions = np.concatenate(log_prob_actions, axis=-1)
    #
    #     obs_v = observations.reshape(N, -1, obs.shape[1])  # 8as a group at every t to compute values
    #     obs_v = obs_v.transpose((1, 0, 2))  # (T,N,shape[1])
    #     obs_v = obs_v.reshape(-1, obs.shape[1])
    #
    #     # next_states=torch.tensor(next_states)#(N,T)->(N*T,) to buffer
    #     observations = torch.tensor(observations.reshape(-1, obs.shape[1]))
    #     obs_v = torch.tensor(obs_v, device=torch.device('cuda'))
    #     actions = torch.tensor(actions.reshape(-1, 4))
    #     action_logits = torch.tensor(action_logits.reshape(-1, 8))
    #
    #     log_prob_actions = torch.tensor(log_prob_actions.flatten())
    #
    #     x, yt = critic_encoder(obs_v, 'Critic')
    #     critic_output = torch.cat((x, yt), dim=1)
    #     values = critic_linear(critic_output)
    #     values_v = values.detach().cpu().numpy()
    #     values_v = values_v.reshape(-1, N)  # (T,N)
    #     values_v = values_v.transpose(1, 0)  # (N,T)
    #
    #     advantages, returns = self._calculate_gae(rewards, values_v)
    #     values2 = values_v.flatten()
    #     values2 = torch.tensor(values2)
    #     rewards = torch.tensor(rewards.flatten())
    #
    #     gpu_buffer.values = values2.cuda()
    #     gpu_buffer['obs']['obs'] = observations.cuda()
    #     gpu_buffer.rewards = rewards.cuda()
    #     gpu_buffer.rewards_cpu = rewards
    #
    #     gpu_buffer.actions = actions.cuda()
    #     gpu_buffer.action_logits = action_logits.cuda()
    #     gpu_buffer.log_prob_actions = log_prob_actions.cuda()
    #     gpu_buffer.advantages = advantages.cuda()
    #     gpu_buffer.returns = returns.cuda()
    #
    #     return gpu_buffer


class DenseModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    Reward model to predict reward from state and rnn hidden state
    """

    def __init__(self, cfg, state_dim, rnn_hidden_dim, hidden_dim, act=torch.tanh):
        super(DenseModel, self).__init__()
        self.use_spectral_norm = cfg.use_spectral_norm
        # self.fc = nn.Sequential(
        # nn.Linear(state_dim + rnn_hidden_dim, hidden_dim),
        # nn.ReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(),
        # )
        self.hyper_fc_w = Hypernet(cfg, input_dim=hidden_dim,
                                   hidden_dim=hidden_dim,
                                   main_input_dim=hidden_dim,
                                   main_output_dim=hidden_dim,
                                   )
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

        # self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state, z):
        # hidden = self.fc(torch.cat([state, rnn_hidden], dim=-1))

        # print('%%%%%%%%%%%%%%%%%%%%%%%%%',hidden_input)
        state_z = torch.cat([state, z], dim=-1)
        hidden_w = self.hyper_fc_w(state_z)
        hidden = torch.matmul(state_z.unsqueeze(2), hidden_w)
        hidden = self.act(hidden)
        hidden = hidden.squeeze(2)
        # reward = self.fc4(hidden)
        return hidden

    def forward2(self, state, z, num_agent):
        hidden = self.forward(state, z)
        hidden = hidden.reshape(hidden.shape[0], num_agent, -1, hidden.shape[-1])
        q = torch.mean(hidden, dim=2).unsqueeze(2)
        v = self.mlp(hidden)
        attention = (q @ hidden.transpose(-1, -2)) * (1.0 / math.sqrt(hidden.size(-1)))
        att = torch.softmax(attention, dim=-1)
        y = att @ v
        return y.squeeze(2)

    def forward3(self, state, z, num_agent):
        hidden = self.forward(state, z)
        hidden = hidden.reshape(hidden.shape[0], -1, num_agent, hidden.shape[-1])
        q = torch.mean(hidden, dim=2).unsqueeze(2)
        v = self.mlp(hidden)
        attention = (q @ hidden.transpose(-1, -2)) * (1.0 / math.sqrt(hidden.size(-1)))
        att = torch.softmax(attention, dim=-1)
        y = att @ v
        return y.squeeze(2)


class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    Reward model to predict reward from state and rnn hidden state
    """

    def __init__(self, cfg, state_dim, num_oppo, rnn_hidden_dim, hidden_dim, act=torch.tanh):
        super(RewardModel, self).__init__()
        self.num_oppo = num_oppo
        self.use_spectral_norm = cfg.use_spectral_norm
        self.dense = DenseModel(cfg, state_dim, rnn_hidden_dim, hidden_dim)
        self.fc4 = fc_layer(hidden_dim, 1, spec_norm=self.use_spectral_norm)  # reward_obstacle reward_goal
        

    def forward(self, state, z):
        hidden = self.dense(state,z)
        reward = self.fc4(hidden)

        

        return reward


class ObsModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    Obs model to predict observation from state and rnn hidden state
    """

    def __init__(self, cfg, obs_space, num_agent,num_oppo, num_neighbors, state_dim, rnn_hidden_dim, hidden_dim, act=F.relu):
        super().__init__()
        self.use_spectral_norm = cfg.use_spectral_norm
        self.obs_self_dim = 18
        self.num_agent = num_agent
        self.dense = DenseModel(cfg, state_dim, rnn_hidden_dim, hidden_dim)
        self.dense2 = DenseModel(cfg, state_dim, rnn_hidden_dim, hidden_dim)
        self.fc4 = nn.Sequential(
            fc_layer(hidden_dim, hidden_dim, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(hidden_dim, 9 + 6 * num_neighbors+2*cfg.num_oppo_obs*12, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
        )  # if obstacle: 74
        
        self._min_stddev = 0.1

    def forward(self, state, z):
        hidden = self.dense(state, z)
        obs_oppo = self.fc4(hidden)

        #hidden2 = self.dense2.forward2(latent, z2, self.num_agent)
        #obs_neighbor = self.fc5(hidden2)
        # mean = self.fc_obs_mean_posterior(hidden)
        # stddev = F.softplus(self.fc_obs_stddev_posterior(hidden)) + self._min_stddev
        # obs_distribution = Normal(mean, stddev)
        # obs = obs_distribution.rsample()

        return obs_oppo

