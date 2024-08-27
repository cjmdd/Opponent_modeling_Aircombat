import torch
from torch import nn
from torch.nn import functional as F
from sample_factory.model.multi_head_attention import MultiHeadAttention2,MultiHeadAttention3
from sample_factory.model.common_encoder import Hypernet3,Gating_layer,PositionwiseFeedForward2
from sample_factory.model.transformer_intention_layer import TransformerIntentionLayer
from sample_factory.model.multi_head_attention import MultiHeadAttention32
class TransformerLatentLayer(TransformerIntentionLayer):
    def __init__(self,cfg,hidden_dim,state_dim,num_heads):
        super().__init__(cfg,hidden_dim, state_dim, num_heads)
        self.selfatt_k_intention = nn.Linear(state_dim, state_dim)
        self.selfatt_v_intention = nn.Linear(state_dim, state_dim)

        self.gating_intention = Gating_layer(state_dim)
        self.layer_norm_intention = nn.LayerNorm(state_dim, eps=1e-6)
        self.multi_head_attention2 = MultiHeadAttention32(state_dim, num_heads)
        self.multi_head_attention_intention=MultiHeadAttention32(state_dim, num_heads)

        self.crossatt_q_latent = Hypernet3(cfg, input_dim=state_dim, hidden_dim=hidden_dim,
                                             main_input_dim=state_dim, main_output_dim=state_dim,
                                             )  # T,K,N,D,D

        self.crossatt_k_latent = nn.Linear(state_dim, state_dim)
        self.crossatt_v_latent = nn.Linear(state_dim, state_dim)
        self.multi_head_attention_latent = MultiHeadAttention3(state_dim, num_heads)

        self.selfcrossatt_q_latent = nn.Linear(state_dim, state_dim)
        self.selfcrossatt_k_latent = nn.Linear(state_dim, state_dim)
        self.selfcrossatt_v_latent = nn.Linear(state_dim, state_dim)

    def forward20(self,neighbor_obs,k_latent,q_latent,q_intention):
        # self_attention
        # neighbor_obs=B,N,D; q_latent, q_intention=B,K,N,D; k_latent=B,N,T,D
        # self_attention
        # neighbor_obs=T,N,D; q_latent, q_intention=T,K,N,D; k_latent=N,T,D
        T, K, N, D = q_intention.size()
        q_self = self.selfatt_q_adv_obs(neighbor_obs)
        k_self = self.selfatt_k_adv_obs(neighbor_obs)
        v_self = self.selfatt_v_adv_obs(neighbor_obs)  # T,N,D

        self_attention = self.multi_head_attention2(q_self, k_self, v_self)  # T,N,D
        Gating_outputs = neighbor_obs+ self_attention
        k_intention = self.layer_norm(Gating_outputs)

        # self_intention_attention
        k_intentions = self.selfatt_k_intention(k_intention)  # T,N,D
        v_intentions = self.selfatt_v_intention(k_intentions)
        q_intention_w = self.selfatt_q_intention(q_intention)
        q_intention2 = torch.matmul(q_intention.unsqueeze(-2), q_intention_w)
        q_intention2 = F.tanh(q_intention2).squeeze(-2)  # T,K,N,D
        q_intentions = q_intention2.transpose(1, 2).reshape(T, -1, D)  # T,N*k,D

        self_intention_attention = self.multi_head_attention_intention(q_intentions, k_intentions,
                                                                       v_intentions)  # T,N*k,D
        Gating_intention = q_intention.transpose(1, 2).reshape(T, -1, D)+ self_intention_attention
        Position_input = self.layer_norm_intention(Gating_intention)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_intention = Gating_intention+ Gating_y2
        q_intention_cross = self.layer_norm2(transformer_out_intention)  # T,N*k,D

        # cross_latent_attention
        q = self.crossatt_q_intention(q_intention_cross)
        q_latent_w = self.crossatt_q_latent(q_latent)  # T,K,N,D
        q_latent = torch.matmul(q_latent.unsqueeze(-2), q_latent_w)
        q_latent = F.tanh(q_latent).squeeze(-2)  # T,K,N,D
        q_latents = q_latent.transpose(1, 2).reshape(T, -1, D)  # T,N*k,D
        q_latent_cross = q + q_latents

        k_latent = k_latent.reshape(-1, k_latent.shape[-1])  # N*T,D
        k_latent = k_latent.unsqueeze(0)  # 1,N*T,D
        k_latent_cross = self.crossatt_k_latent(k_latent)
        v_latent_cross = self.crossatt_v_latent(k_latent)
        cross_latent_attention = self.multi_head_attention_latent(q_latent_cross, k_latent_cross,
                                                                  v_latent_cross)  # T,N*K,D

        Gating_cross = q_intention_cross+ cross_latent_attention
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_cross = self.layer_norm_cross(Gating_cross)
        Gating_y2_cross = self.positionforward_cross(Position_cross)
        transformer_out_cross = Gating_cross+ Gating_y2_cross
        q_self_cross0 = self.layer_norm2_cross(transformer_out_cross).reshape(T, N, K, D).transpose(1, 2)  # T,K,N,D

        # self-cross-attention
        q_self_cross = self.selfcrossatt_q_latent(q_self_cross0)
        k_self_cross = self.selfcrossatt_k_latent(q_self_cross0)
        v_self_cross = self.selfcrossatt_v_latent(q_self_cross0)
        q_self_cross = torch.cat((q_self_cross, q_latent), dim=-1)
        k_self_cross = torch.cat((k_self_cross, q_latent), dim=-1)
        selfcross_attention = self.multi_head_attention3(q_self_cross, k_self_cross, v_self_cross)  # T,K,N,D

        Gating_self_cross = q_self_cross0+ selfcross_attention
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_self_cross = self.layer_norm_selfcross(Gating_self_cross)
        Gating_y2_self_cross = self.positionforward_selfcross(Position_self_cross)
        transformer_out_self_cross = Gating_self_cross+ Gating_y2_self_cross
        q_latent = self.layer_norm2_cross(transformer_out_self_cross)
        return q_latent, selfcross_attention

    def forward3(self,neighbor_obs,k_latent,q_latent,q_intention):
        # self_attention
        # neighbor_obs=B,N,D; q_latent, q_intention=B,K,N,D; k_latent=B,N,T,D
        T,K,N,D=q_intention.size()
        q_self = self.selfatt_q_adv_obs(neighbor_obs)
        k_self = self.selfatt_k_adv_obs(neighbor_obs)
        v_self = self.selfatt_v_adv_obs(neighbor_obs) #B,N,D

        self_attention=self.multi_head_attention2(q_self,k_self,v_self) #B,N,D
        Gating_outputs = neighbor_obs+ self_attention
        k_intention = self.layer_norm(Gating_outputs)

        # self_intention_attention
        k_intentions=self.selfatt_k_intention(k_intention) #B,N,D
        v_intentions=self.selfatt_v_intention(k_intentions)
        q_intention_w = self.selfatt_q_intention(q_intention)
        q_intention2 = torch.matmul(q_intention.unsqueeze(-2), q_intention_w)
        q_intention2 = F.tanh(q_intention2).squeeze(-2)  # B,K,N,D
        q_intentions= q_intention2.transpose(1,2).reshape(T,-1,D) # B,N*k,D

        self_intention_attention = self.multi_head_attention_intention(q_intentions, k_intentions, v_intentions) # B,N*k,D
        Gating_intention = q_intention.transpose(1,2).reshape(T,-1,D)+ self_intention_attention
        Position_input = self.layer_norm_intention(Gating_intention)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_intention = Gating_intention+ Gating_y2
        q_intention_cross = self.layer_norm2(transformer_out_intention) # B,N*k,D

        # cross_latent_attention
        q=self.crossatt_q_intention(q_intention_cross)
        q_latent_w = self.crossatt_q_latent(q_latent)  # B,K,N,D
        q_latent = torch.matmul(q_latent.unsqueeze(-2), q_latent_w)
        q_latent = F.tanh(q_latent).squeeze(-2)  # B,K,N,D
        q_latents = q_latent.transpose(1, 2).reshape(T, -1, D)  # B,N*k,D
        q_latent_cross=q+q_latents

        k_latent= k_latent.reshape(k_latent.shape[0],-1, k_latent.shape[-1])  # B,N*T,D
        # k_latent= k_latent.unsqueeze(0)  # 1,N*T,D
        k_latent_cross = self.crossatt_k_latent(k_latent)
        v_latent_cross = self.crossatt_v_latent(k_latent)
        cross_latent_attention = self.multi_head_attention_latent.forward2(q_latent_cross, k_latent_cross, v_latent_cross)  # B,N*K,D

        Gating_cross = q_intention_cross+ cross_latent_attention
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_cross = self.layer_norm_cross(Gating_cross)
        Gating_y2_cross = self.positionforward_cross(Position_cross)
        transformer_out_cross = Gating_cross+ Gating_y2_cross
        q_self_cross0 = self.layer_norm2_cross(transformer_out_cross).reshape(T, N, K, D).transpose(1, 2)  # B,K,N,D

        # self-cross-attention
        q_self_cross = self.selfcrossatt_q_latent(q_self_cross0)
        k_self_cross = self.selfcrossatt_k_latent(q_self_cross0)
        v_self_cross = self.selfcrossatt_v_latent(q_self_cross0)
        q_self_cross = torch.cat((q_self_cross, q_latent), dim=-1)
        k_self_cross = torch.cat((k_self_cross, q_latent), dim=-1)
        selfcross_attention = self.multi_head_attention3(q_self_cross, k_self_cross, v_self_cross)  # B,K,N,D

        Gating_self_cross = q_self_cross0+ selfcross_attention
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_self_cross = self.layer_norm_selfcross(Gating_self_cross)
        Gating_y2_self_cross = self.positionforward_selfcross(Position_self_cross)
        transformer_out_self_cross = Gating_self_cross+ Gating_y2_self_cross
        q_latent= self.layer_norm2_cross(transformer_out_self_cross)
        return q_latent, selfcross_attention











