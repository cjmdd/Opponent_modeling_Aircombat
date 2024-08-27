import torch
from torch import nn
from torch.nn import functional as F
from sample_factory.model.common_encoder import Gating_layer,PositionwiseFeedForward2
from sample_factory.model.multi_head_attention import MultiHeadAttention,MultiHeadAttention_GlobalSpace, MultiHeadAttention_GlobalTime

class TransformerHistoryLayer(nn.Module):
    def __init__(self,hidden_dim,num_heads,n_time_sequence):
        super(TransformerHistoryLayer,self).__init__()
        self.hidden_dim=hidden_dim
        self.mlp = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gating = Gating_layer(self.hidden_dim)
        self.gating2 = Gating_layer(self.hidden_dim)
        self.positionforward = PositionwiseFeedForward2(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.multi_head_attention =MultiHeadAttention(self.hidden_dim, num_heads, n_time_sequence)

    def forward(self,adv_embedding):
        q = k = adv_embedding # N,T,D
        v = self.mlp(adv_embedding)
        attention=self.multi_head_attention(q,k,v)
        Gating_outputs = self.gating(adv_embedding, attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.layer_norm2(transformer_out_x)
        return out_x


class TransformerHistoryGlobalSpaceLayer(TransformerHistoryLayer):
    def __init__(self,hidden_dim,num_heads,n_time_sequence):
        super().__init__(hidden_dim,num_heads,n_time_sequence)
        self.mlp = nn.Linear(n_time_sequence*self.hidden_dim, n_time_sequence*self.hidden_dim)
        self.gating = Gating_layer(n_time_sequence*self.hidden_dim)
        self.gating2 = Gating_layer(n_time_sequence*self.hidden_dim)
        self.positionforward = PositionwiseFeedForward2(n_time_sequence*self.hidden_dim)
        self.layer_norm = nn.LayerNorm(n_time_sequence*self.hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(n_time_sequence*self.hidden_dim, eps=1e-6)
        self.multi_head_attention =MultiHeadAttention_GlobalSpace(self.hidden_dim, num_heads, n_time_sequence)

    def forward(self,adv_embedding):# N,T*D
        q = k = adv_embedding  
        v = self.mlp(adv_embedding)
        attention = self.multi_head_attention(q, k, v)
        Gating_outputs = self.gating(adv_embedding, attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.layer_norm2(transformer_out_x)
        return out_x


    def forward2(self,adv_embedding):# B,N,T*D
        q = k = adv_embedding
        v = self.mlp(adv_embedding)
        attention = self.multi_head_attention.forward2(q, k, v)
        Gating_outputs = self.gating(adv_embedding, attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.layer_norm2(transformer_out_x)
        return out_x


class TransformerHistoryGlobalTimeLayer(TransformerHistoryLayer):
    def __init__(self,hidden_dim,num_heads,n_time_sequence,num_agents):
        super().__init__(hidden_dim,num_heads,n_time_sequence)
        self.mlp = nn.Linear(num_agents * self.hidden_dim, num_agents * self.hidden_dim)
        self.gating = Gating_layer(num_agents  * self.hidden_dim)
        self.gating2 = Gating_layer(num_agents * self.hidden_dim)
        self.positionforward = PositionwiseFeedForward2(num_agents *self.hidden_dim)
        self.layer_norm = nn.LayerNorm(num_agents *self.hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(num_agents * self.hidden_dim, eps=1e-6)
        self.multi_head_attention =MultiHeadAttention_GlobalTime(self.hidden_dim, num_heads, n_time_sequence)

    def forward(self,adv_embedding):# T,N*D
        q = k = adv_embedding  # N,T,D
        v = self.mlp(adv_embedding)
        attention = self.multi_head_attention(q, k, v)
        Gating_outputs = self.gating(adv_embedding, attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.layer_norm2(transformer_out_x)
        return out_x

    def forward2(self,adv_embedding):# B, T,N*D
        q = k = adv_embedding
        v = self.mlp(adv_embedding)
        attention = self.multi_head_attention.forward2(q, k, v)
        Gating_outputs = self.gating(adv_embedding, attention)
        # Gating_outputs = torch.cat((self_obs_action_embed, Gating_output), dim=-1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.layer_norm2(transformer_out_x)
        return out_x