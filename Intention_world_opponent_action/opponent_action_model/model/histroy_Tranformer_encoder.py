import torch
from torch import nn
from torch.nn import functional as F
from sample_factory.model.common_encoder import Hypernet,Actor_QuadSelfEncoder, AdvEncoderAttention,Actor_QuadNeighborhoodEncoderAttention, Gating_layer,PositionwiseFeedForward,\
    PositionwiseFeedForward2
from sample_factory.algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from sample_factory.model.transformer_history_layer import TransformerHistoryLayer,TransformerHistoryGlobalSpaceLayer,TransformerHistoryGlobalTimeLayer
class HistoryEncoder(nn.Module):
    def __init__(self, cfg,hidden_dim,state_dim,self_obs_dim,neighbor_obs_dim,num_neigbhor_obs,oppo_obs_dim, num_agents,num_oppo_obs):
        super(HistoryEncoder, self).__init__()
        self.cfg=cfg
        self.hidden_dim=hidden_dim
        self.self_obs_dim=self_obs_dim
        self.neighbor_obs_dim=neighbor_obs_dim
        self.num_use_neighbor_obs=num_neigbhor_obs
        self.oppo_obs_dim=oppo_obs_dim
        self.num_oppo_obs=num_oppo_obs
        self.use_spectral_norm=cfg.use_spectral_norm

        self.all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        self.adv_obs_size = self.all_neighbor_obs_size + self.self_obs_dim
        self.all_adv_obs_size = self.oppo_obs_dim * self.num_oppo_obs

        self.feed_forward = nn.Sequential(
            fc_layer(2 * self.hidden_dim, self.hidden_dim, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(self.hidden_dim, state_dim, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )
        self.self_encoder = Actor_QuadSelfEncoder(cfg, self.self_obs_dim,
                                                  self.hidden_dim, self.use_spectral_norm)
        self.neighbor_encoder = Actor_QuadNeighborhoodEncoderAttention(cfg, self.neighbor_obs_dim,
                                                                       self.hidden_dim,
                                                                       self.use_spectral_norm,
                                                                       self.self_obs_dim,
                                                                       self.num_use_neighbor_obs)
        self.adv_encoder = AdvEncoderAttention(cfg, self.oppo_obs_dim,
                                               state_dim,
                                               self.use_spectral_norm,
                                               self.self_obs_dim,
                                               self.num_oppo_obs)
        if self.cfg.local_time_attention:
            self.history_intention_layers = nn.Sequential(
                *[TransformerHistoryLayer(state_dim, cfg.num_heads, 2*cfg.rollout) for _ in range(cfg.num_layer)])
            self.history_latent_layers = nn.Sequential(
                *[TransformerHistoryLayer(state_dim, cfg.num_heads, 2*cfg.rollout) for _ in range(cfg.num_layer)])
            if self.cfg.global_space_attention:
                self.history_intention_global_space = nn.Sequential(
                    *[TransformerHistoryGlobalSpaceLayer(state_dim, cfg.num_heads, 2*cfg.rollout) for _ in
                      range(cfg.num_layer)])
                self.history_latent_global_space = nn.Sequential(
                    *[TransformerHistoryGlobalSpaceLayer(state_dim, cfg.num_heads, 2*cfg.rollout) for _ in
                      range(cfg.num_layer)])
        elif self.cfg.global_time_attention:
            self.history_intention_global_time = nn.Sequential(
                *[TransformerHistoryGlobalTimeLayer(state_dim, cfg.num_heads, 2*cfg.rollout,num_agents) for _ in
                  range(cfg.num_layer)])
            self.history_latent_global_time = nn.Sequential(
                *[TransformerHistoryGlobalTimeLayer(state_dim, cfg.num_heads, 2*cfg.rollout,num_agents) for _ in
                  range(cfg.num_layer)])

        # self.gru=nn.GRUCell(state_dim,state_dim)

    def forward(self,history):
        """
        param history: N, T, D
        obs_oppo-->higher level key of intentions
        obs_self and obs_neighbors-->lower level key of latent strategy
        return: intention and latent strategy
        """
        N, T, D=history.size()
        obs_self = history[:, :, :self.self_obs_dim]
        batch_size = history.shape[0] * history.shape[1]
        # local space attention
        adv_embedding = self.adv_encoder(history, self.adv_obs_size, self.all_adv_obs_size, batch_size)
        if self.cfg.local_time_attention:
            # local time attention
            k_intention = self.history_intention_layers(adv_embedding) #N, T, D
            # global space attention
            if self.cfg.global_space_attention: # N,T*D
                k_intention = self.history_intention_global_space.forward(k_intention.reshape(k_intention.shape[0],-1))
                k_intention=k_intention.reshape(N,T,-1)
        elif self.cfg.global_time_attention:
            # global time attention
            k_intention = self.history_intention_global_time.forward(adv_embedding.transpose(0,1).reshape(T,-1))  # T, N*D
            k_intention = k_intention.reshape(T, N, -1).transpose(0,1)


        self_embed = self.self_encoder(obs_self)
        neighborhood_embedding = self.neighbor_encoder(obs_self, history, self.all_neighbor_obs_size, batch_size)
        # obstacle_mean_embed = self.obstacle_encoder(history, all_neighbor_obs_size, batch_size)
        neighbors_embedding= torch.cat((self_embed,neighborhood_embedding),dim=-1)
        neighbors_embedding=self.feed_forward(neighbors_embedding)
        # k_latent=self.history_latent_layers(neighbors_embedding)
        if self.cfg.local_time_attention:
            # local time attention
            k_latent = self.history_latent_layers(neighbors_embedding) #N, T, D
            # global space attention
            if self.cfg.global_space_attention: # N,T*D
                k_latent = self.history_latent_global_space.forward(k_latent.reshape(k_latent.shape[0],-1))
                k_latent=k_latent.reshape(N,T,-1)
        elif self.cfg.global_time_attention:
            # global time attention
            k_latent = self.history_intention_global_time.forward(neighbors_embedding.transpose(0,1).reshape(T,-1))  # T, N*D
            k_latent = k_latent.reshape(T, N, -1).transpose(0,1)

        return k_intention, k_latent

    def forward2(self,history):
        """
        param history: B, N, T, D
        obs_oppo-->higher level key of intentions
        obs_self and obs_neighbors-->lower level key of latent strategy
        return: intention and latent strategy
        """
        B, N, T, D=history.size()
        obs_self = history[:, :, :, :self.self_obs_dim]
        # history=history.reshape(-1,history.shape[-2],history.shape[-1]) # B*N,T,D
        batch_size = history.shape[0] * history.shape[1]*history.shape[2]
        # local space attention
        adv_embedding = self.adv_encoder(history.reshape(-1,history.shape[-2],history.shape[-1]), self.adv_obs_size, self.all_adv_obs_size, batch_size) # B*N,T,D
        if self.cfg.local_time_attention:
            # local time attention
            k_intention = self.history_intention_layers(adv_embedding) #B*N, T, D
            # global space attention
            if self.cfg.global_space_attention: # B, N,T*D
                for i in range(self.cfg.num_layer):
                    k_intention = self.history_intention_global_space[i].forward2(k_intention.reshape(k_intention.shape[0], N, -1))  # B, N,T*D
                k_intention=k_intention.reshape(B,N,T,-1) #B, N, T, D
        elif self.cfg.global_time_attention:
            # global time attention
            for i in range(self.cfg.num_layer):
                k_intention = self.history_intention_global_time[i].forward2(
                    adv_embedding.reshape(B, N, T, adv_embedding.shape[-1]).transpose(1, 2).reshape(B, T, -1))  # B, T, N*D
            k_intention =k_intention.reshape(B,T,N,-1).transpose(1,2) #B, N, T, D

        obs_self=obs_self.reshape(-1, obs_self.shape[-2], obs_self.shape[-1])
        self_embed = self.self_encoder(obs_self) #B*N, T, D
        neighborhood_embedding = self.neighbor_encoder(obs_self, history.reshape(-1,history.shape[-2],history.shape[-1]), self.all_neighbor_obs_size, batch_size)# B*N,T,D
        # obstacle_mean_embed = self.obstacle_encoder(history, all_neighbor_obs_size, batch_size)
        neighbors_embedding= torch.cat((self_embed,neighborhood_embedding),dim=-1)
        neighbors_embedding=self.feed_forward(neighbors_embedding) #B*N, T, D
        # k_latent=self.history_latent_layers(neighbors_embedding)
        if self.cfg.local_time_attention:
            # local time attention
            k_latent = self.history_latent_layers(neighbors_embedding) #B*N, T, D
            # global space attention
            if self.cfg.global_space_attention: #B, N,T*D
                for i in range(self.cfg.num_layer):
                    k_latent = self.history_latent_global_space[i].forward2(k_latent.reshape(k_latent.shape[0], N, -1))
                k_latent=k_latent.reshape(B,N,T,-1)
        elif self.cfg.global_time_attention:
            # global time attention
            for i in range(self.cfg.num_layer):
                k_latent = self.history_intention_global_time[i].forward2(
                    neighbors_embedding.reshape(B, N, T, neighbors_embedding.shape[-1]).transpose(1, 2).reshape(B, T, -1))  # B, T, N*D
            k_latent = k_latent.reshape(B,T,N,-1).transpose(1,2) # B,N,T,D

        return k_intention, k_latent





