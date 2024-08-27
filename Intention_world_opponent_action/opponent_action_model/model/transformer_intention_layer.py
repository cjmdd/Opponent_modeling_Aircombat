import torch
from torch import nn
from torch.nn import functional as F
from sample_factory.model.common_encoder import Gating_layer,PositionwiseFeedForward2,Hypernet3
from sample_factory.model.multi_head_attention import MultiHeadAttention2,MultiHeadAttention3

class TransformerIntentionLayer(nn.Module):
    def __init__(self,cfg,hidden_dim,state_dim,num_heads):
        super(TransformerIntentionLayer, self).__init__()
        # Self-Attention
        self.selfatt_q_adv_obs=nn.Linear(state_dim, state_dim)
        self.selfatt_k_adv_obs = nn.Linear(state_dim, state_dim)
        self.selfatt_v_adv_obs = nn.Linear(state_dim, state_dim)
        self.selfatt_q_intention=Hypernet3(cfg, input_dim=state_dim, hidden_dim=hidden_dim,
                                              main_input_dim=state_dim, main_output_dim=state_dim,
                                              )# T,K,N,D,D
        self.gating = Gating_layer(state_dim)
        self.gating2 = Gating_layer(state_dim)
        self.positionforward = PositionwiseFeedForward2(state_dim)
        self.layer_norm = nn.LayerNorm(state_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(state_dim, eps=1e-6)
        self.multi_head_attention = MultiHeadAttention2(state_dim, num_heads)
        # Cross-Attention
        self.crossatt_q_intention = nn.Linear(state_dim, state_dim)
        self.crossatt_k_intention = nn.Linear(state_dim, state_dim)
        self.crossatt_v_intention = nn.Linear(state_dim, state_dim)
        self.gating_cross = Gating_layer(state_dim)
        self.gating2_cross = Gating_layer(state_dim)
        self.positionforward_cross = PositionwiseFeedForward2(state_dim)
        self.layer_norm_cross = nn.LayerNorm(state_dim, eps=1e-6)
        self.layer_norm2_cross = nn.LayerNorm(state_dim, eps=1e-6)
        self.multi_head_attention2 = MultiHeadAttention3(state_dim, num_heads)

        # Self-Cross-Attention
        self.selfcrossatt_q_intention = nn.Linear(state_dim, state_dim)
        self.selfcrossatt_k_intention = nn.Linear(state_dim, state_dim)
        self.selfcrossatt_v_intention = nn.Linear(state_dim, state_dim)
        self.gating_selfcross = Gating_layer(state_dim)
        self.gating2_selfcross = Gating_layer(state_dim)
        self.positionforward_selfcross = PositionwiseFeedForward2(state_dim)
        self.layer_norm_selfcross = nn.LayerNorm(state_dim, eps=1e-6)
        self.layer_norm2_selfcross = nn.LayerNorm(state_dim, eps=1e-6)
        self.multi_head_attention3 = MultiHeadAttention2(state_dim, num_heads)


    def forward(self,adv_obs,k_intention,q_intention):
        # self_attention
        # adv_obs, q_intention=T,K,N,D; k_intention=N,T,D
        T,K,N,D=adv_obs.size()
        q_self=self.selfatt_q_adv_obs(adv_obs)
        k_self=self.selfatt_k_adv_obs(adv_obs)
        v_self=self.selfatt_v_adv_obs(adv_obs)

        q_intention_w=self.selfatt_q_intention(q_intention)
        q_intention=torch.matmul(q_intention.unsqueeze(-2),q_intention_w)
        q_intention=F.tanh(q_intention).squeeze(-2) # T,K,N,D

        q_self=q_self+q_intention # T,K,N,D
        k_self=k_self+q_intention
        self_attention=self.multi_head_attention(q_self,k_self,v_self)

        Gating_outputs = self.gating(adv_obs, self_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        q_cross0 = self.layer_norm2(transformer_out_x).transpose(1,2).reshape(T,-1,D) # T,N*K,D

        # cross_attention
        q=self.crossatt_q_intention(q_cross0)
        q_intentions=q_intention.transpose(1,2).reshape(T,-1,D)
        q_cross=q+q_intentions # T,N*K,D

        k_intention=k_intention.reshape(-1,k_intention.shape[-1]) # N*T,D
        k_intention=k_intention.unsqueeze(0) # 1,N*T,D
        k_cross=self.crossatt_k_intention(k_intention)
        v_cross=self.crossatt_v_intention(k_intention)
        cross_attention=self.multi_head_attention2(q_cross,k_cross,v_cross) # T,N*K,D

        Gating_cross = self.gating_cross(q_cross0, cross_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_cross = self.layer_norm_cross(Gating_cross)
        Gating_y2_cross = self.positionforward_cross(Position_cross)
        transformer_out_cross = self.gating2_cross(Gating_cross, Gating_y2_cross)
        q_self_cross0 = self.layer_norm2_cross(transformer_out_cross).reshape(T, N,K, D).transpose(1, 2)  # T,K,N,D

        # self_Cross
        q_self_cross=self.selfcrossatt_q_intention(q_self_cross0)
        k_self_cross = self.selfcrossatt_k_intention(q_self_cross0)
        v_self_cross = self.selfcrossatt_v_intention(q_self_cross0)
        q_self_cross=torch.cat((q_self_cross,q_intention),dim=-1)
        k_self_cross=torch.cat((k_self_cross,q_intention),dim=-1)
        selfcross_attention = self.multi_head_attention3(q_self_cross, k_self_cross, v_self_cross) # T,K,N,D

        Gating_self_cross = self.gating_selfcross(q_self_cross0, selfcross_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_self_cross = self.layer_norm_selfcross(Gating_self_cross)
        Gating_y2_self_cross = self.positionforward_selfcross(Position_self_cross)
        transformer_out_self_cross = self.gating2_selfcross(Gating_self_cross, Gating_y2_self_cross)
        q_intention = self.layer_norm2_cross(transformer_out_self_cross)

        return q_intention,selfcross_attention


    def forward2(self,adv_obs,k_intention,q_intention):
        # self_attention
        # adv_obs, q_intention=B,K,N,D; k_intention=B，N,T,D
        T,K,N,D=adv_obs.size()
        q_self=self.selfatt_q_adv_obs(adv_obs)
        k_self=self.selfatt_k_adv_obs(adv_obs)
        v_self=self.selfatt_v_adv_obs(adv_obs)

        q_intention_w=self.selfatt_q_intention(q_intention)
        q_intention=torch.matmul(q_intention.unsqueeze(-2),q_intention_w)
        q_intention=F.tanh(q_intention).squeeze(-2) # B,K,N,D

        q_self=q_self+q_intention # B,K,N,D
        k_self=k_self+q_intention
        self_attention=self.multi_head_attention(q_self,k_self,v_self)

        Gating_outputs = self.gating(adv_obs, self_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        q_cross0 = self.layer_norm2(transformer_out_x).transpose(1,2).reshape(T,-1,D) # B,N*K,D

        # cross_attention
        q=self.crossatt_q_intention(q_cross0)
        q_intentions=q_intention.transpose(1,2).reshape(T,-1,D)
        q_cross=q+q_intentions # B,N*K,D

        k_intention=k_intention.reshape(k_intention.shape[0],-1,k_intention.shape[-1]) # B,N*T,D
        # k_intention=k_intention.unsqueeze(0) # 1,N*T,D
        k_cross=self.crossatt_k_intention(k_intention)
        v_cross=self.crossatt_v_intention(k_intention)
        cross_attention=self.multi_head_attention2.forward2(q_cross,k_cross,v_cross) # B,N*K,D

        Gating_cross = self.gating_cross(q_cross0, cross_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_cross = self.layer_norm_cross(Gating_cross)
        Gating_y2_cross = self.positionforward_cross(Position_cross)
        transformer_out_cross = self.gating2_cross(Gating_cross, Gating_y2_cross)
        q_self_cross0 = self.layer_norm2_cross(transformer_out_cross).reshape(T, N,K, D).transpose(1, 2)  # B,K,N,D

        # self_Cross
        q_self_cross=self.selfcrossatt_q_intention(q_self_cross0)
        k_self_cross = self.selfcrossatt_k_intention(q_self_cross0)
        v_self_cross = self.selfcrossatt_v_intention(q_self_cross0)
        q_self_cross=torch.cat((q_self_cross,q_intention),dim=-1)
        k_self_cross=torch.cat((k_self_cross,q_intention),dim=-1)
        selfcross_attention = self.multi_head_attention3(q_self_cross, k_self_cross, v_self_cross) # B,K,N,D

        Gating_self_cross = self.gating_selfcross(q_self_cross0, selfcross_attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_self_cross = self.layer_norm_selfcross(Gating_self_cross)
        Gating_y2_self_cross = self.positionforward_selfcross(Position_self_cross)
        transformer_out_self_cross = self.gating2_selfcross(Gating_self_cross, Gating_y2_self_cross)
        q_intention = self.layer_norm2_cross(transformer_out_self_cross)

        return q_intention,selfcross_attention
