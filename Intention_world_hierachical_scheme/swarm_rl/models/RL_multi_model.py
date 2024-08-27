import torch
from torch import nn
from torch.nn.utils import spectral_norm
from sample_factory.algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

import torch.nn.functional as F


class QuadNeighborhoodEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs


class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


class Hypernet(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim, main_input_dim, main_output_dim):
        super(Hypernet, self).__init__()
        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim

        self.hyper_w = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nonlinearity(cfg),
            nn.Linear(hidden_dim, output_dim),
            nonlinearity(cfg),
        )

    def forward(self, x):
        return self.hyper_w(x).view(-1, self.main_input_dim, self.main_output_dim)


class Actor_QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        # self.embedding_mlp = nn.Sequential(
        #     fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        #     fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg)
        # )

        self.hyper_embedding_mlp_w = Hypernet(cfg, input_dim=self_obs_dim + neighbor_obs_dim,
                                              hidden_dim=neighbor_hidden_size,
                                              main_input_dim=self_obs_dim + neighbor_obs_dim,
                                              main_output_dim=neighbor_hidden_size,
                                              )

        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size, spec_norm=use_spectral_norm),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size, num_groups_neighbor):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        embedding_mlp_w = self.hyper_embedding_mlp_w(mlp_input)
        neighbor_embedding = torch.matmul(mlp_input.unsqueeze(1), embedding_mlp_w)
        neighbor_embedding = F.tanh(neighbor_embedding)
        neighbor_embeddings = neighbor_embedding.reshape(-1, self.neighbor_hidden_size)
        # neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)


        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding

class AdvEncoderAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, adv_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_adv_obs):
        super().__init__(cfg, self_obs_dim, adv_obs_dim, neighbor_hidden_size, num_adv_obs)

        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        # self.embedding_mlp = nn.Sequential(
        #     fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        #     fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg)
        # )
        self.num_adv_obs=num_adv_obs
        self.hyper_embedding_mlp_w = Hypernet(cfg, input_dim=self_obs_dim + adv_obs_dim,
                                              hidden_dim=neighbor_hidden_size,
                                              main_input_dim=self_obs_dim + adv_obs_dim,
                                              main_output_dim=neighbor_hidden_size,
                                              )

        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size, spec_norm=use_spectral_norm),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, obs, adv_obs_size, all_adv_obs_size,batch_size):
        self_obs = obs[:, :self.self_obs_dim]
        obs_neighbors = obs[:,adv_obs_size:adv_obs_size+ all_adv_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation
        self_obs_repeat = self_obs.repeat(self.num_adv_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        embedding_mlp_w = self.hyper_embedding_mlp_w(mlp_input)
        neighbor_embedding = torch.matmul(mlp_input.unsqueeze(1), embedding_mlp_w)
        neighbor_embedding = F.tanh(neighbor_embedding)
        neighbor_embeddings = neighbor_embedding.reshape(-1, self.neighbor_hidden_size)
        # neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_adv_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding

class Critic_QuadNeighborhood_MultiheadAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs, num_agents):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)
        self.num_agents = num_agents
        self.neighbor_hidden_size = neighbor_hidden_size
        self.attention_size = cfg.attention_size
        self.num_heads = cfg.num_heads

        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        self.agent_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size, num_groups):
        # agent_embedding = self.neighborhood_encoder(self_obs, obs, all_neighbor_obs_size, batch_size,num_groups)
        # agents_embedding_input = agent_embedding.reshape(num_groups, -1, self.neighbor_hidden_size)
        # agents_embedding_mean = torch.mean(agents_embedding_input, dim=1)
        # agents_embedding_mean_repeat = agents_embedding_mean.repeat(self.num_agents, 1)
        # attention_agents_input = torch.cat((agent_embedding, agents_embedding_mean_repeat), dim=1)
        # attention_agents_weights = self.attention_mlp(attention_agents_input).view(num_groups, -1)
        # attention_agents_weights_softmax = torch.nn.functional.softmax(attention_agents_weights, dim=1)
        # attention_agents_weights_softmax = attention_agents_weights_softmax.view(-1, 1)
        # final_agents_embedding = attention_agents_weights_softmax * agent_embedding
        # final_agents_embedding = final_agents_embedding.view(num_groups, -1, self.neighbor_hidden_size)
        # final_agents_embedding = torch.sum(final_agents_embedding, dim=1)
        # final_agents_embedding = final_agents_embedding.repeat(self.num_agents, 1)

        # local neighbor attention of each agent

        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)

        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)

        agent_v = neighbor_values.view(batch_size, -1, self.neighbor_hidden_size)
        agent_k = neighbor_embeddings.view(batch_size, -1, self.neighbor_hidden_size)
        agent_q = neighbor_embeddings_mean.view(batch_size, -1, self.neighbor_hidden_size)

        d = self.neighbor_hidden_size ** 0.5

        agent_scores = torch.matmul(agent_q / d, agent_k.transpose(1, 2))
        agent_attention = torch.softmax(agent_scores, dim=-1)
        agent_attention = torch.matmul(agent_attention, agent_v)
        agent_attention = agent_attention.reshape(-1, self.neighbor_hidden_size)

        # global agents multi-attention

        agent_values = self.agent_value_mlp(agent_attention)

        agent_embeddings_mean_input = agent_attention.reshape(num_groups, -1, self.neighbor_hidden_size)
        agent_embeddings_mean = torch.mean(agent_embeddings_mean_input, dim=1)

        multi_v = agent_values.view(num_groups, -1, self.num_heads, self.attention_size)
        multi_k = agent_attention.view(num_groups, -1, self.num_heads, self.attention_size)
        multi_q = agent_embeddings_mean.view(num_groups, -1, self.num_heads, self.attention_size)

        d2 = self.attention_size ** 0.5

        multi_q, multi_k, multi_v = multi_q.transpose(1, 2), multi_k.transpose(1, 2), multi_v.transpose(1, 2)

        multi_scores = torch.matmul(multi_q / d2, multi_k.transpose(2, 3))
        multi_attention = torch.softmax(multi_scores, dim=-1)
        multi_attention = torch.matmul(multi_attention, multi_v)
        multi_attention = multi_attention.transpose(1, 2)
        multi_attention = multi_attention.reshape(-1, self.num_heads * self.attention_size)

        multi_head_attention = multi_attention.repeat(self.num_agents, 1)

        return multi_head_attention, agent_attention

class Critic_QuadAdv_MultiheadAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, adv_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_adv_obs, num_agents):
        super().__init__(cfg, self_obs_dim, adv_obs_dim, neighbor_hidden_size, num_adv_obs)
        self.num_agents = num_agents
        self.num_adv_obs=num_adv_obs

        self.neighbor_hidden_size = neighbor_hidden_size
        self.attention_size = cfg.attention_size
        self.num_heads = cfg.num_heads

        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + adv_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        self.agent_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

    def forward(self, obs, adv_obs_size, all_adv_obs_size, batch_size,num_groups):
        # agent_embedding = self.neighborhood_encoder(self_obs, obs, all_neighbor_obs_size, batch_size,num_groups)
        # agents_embedding_input = agent_embedding.reshape(num_groups, -1, self.neighbor_hidden_size)
        # agents_embedding_mean = torch.mean(agents_embedding_input, dim=1)
        # agents_embedding_mean_repeat = agents_embedding_mean.repeat(self.num_agents, 1)
        # attention_agents_input = torch.cat((agent_embedding, agents_embedding_mean_repeat), dim=1)
        # attention_agents_weights = self.attention_mlp(attention_agents_input).view(num_groups, -1)
        # attention_agents_weights_softmax = torch.nn.functional.softmax(attention_agents_weights, dim=1)
        # attention_agents_weights_softmax = attention_agents_weights_softmax.view(-1, 1)
        # final_agents_embedding = attention_agents_weights_softmax * agent_embedding
        # final_agents_embedding = final_agents_embedding.view(num_groups, -1, self.neighbor_hidden_size)
        # final_agents_embedding = torch.sum(final_agents_embedding, dim=1)
        # final_agents_embedding = final_agents_embedding.repeat(self.num_agents, 1)

        # local neighbor attention of each agent
        obs_neighbors = obs[:, adv_obs_size:adv_obs_size + all_adv_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation
        self_obs=obs[:, :self.self_obs_dim]
        self_obs_repeat = self_obs.repeat(self.num_adv_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)

        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)

        agent_v = neighbor_values.view(batch_size, -1, self.neighbor_hidden_size)
        agent_k = neighbor_embeddings.view(batch_size, -1, self.neighbor_hidden_size)
        agent_q = neighbor_embeddings_mean.view(batch_size, -1, self.neighbor_hidden_size)

        d = self.neighbor_hidden_size ** 0.5

        agent_scores = torch.matmul(agent_q / d, agent_k.transpose(1, 2))
        agent_attention = torch.softmax(agent_scores, dim=-1)
        agent_attention = torch.matmul(agent_attention, agent_v)
        agent_attention = agent_attention.reshape(-1, self.neighbor_hidden_size)

        # global agents multi-attention

        agent_values = self.agent_value_mlp(agent_attention)

        agent_embeddings_mean_input = agent_attention.reshape(num_groups, -1, self.neighbor_hidden_size)
        agent_embeddings_mean = torch.mean(agent_embeddings_mean_input, dim=1)

        multi_v = agent_values.view(num_groups, -1, self.num_heads, self.attention_size)
        multi_k = agent_attention.view(num_groups, -1, self.num_heads, self.attention_size)
        multi_q = agent_embeddings_mean.view(num_groups, -1, self.num_heads, self.attention_size)

        d2 = self.attention_size ** 0.5

        multi_q, multi_k, multi_v = multi_q.transpose(1, 2), multi_k.transpose(1, 2), multi_v.transpose(1, 2)

        multi_scores = torch.matmul(multi_q / d2, multi_k.transpose(2, 3))
        multi_attention = torch.softmax(multi_scores, dim=-1)
        multi_attention = torch.matmul(multi_attention, multi_v)
        multi_attention = multi_attention.transpose(1, 2)
        multi_attention = multi_attention.reshape(-1, self.num_heads * self.attention_size)

        multi_head_attention = multi_attention.repeat(self.num_agents, 1)

        return multi_head_attention, agent_attention


class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding


class Actor_QuadSelfEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, fc_encoder_layer, use_spectral_norm):
        super().__init__()
        self.self_encoder = nn.Sequential(
            fc_layer(self_obs_dim, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

    def forward(self, obs_self, num_groups):
        self_embedding = self.self_encoder(obs_self)
        return self_embedding


class Critic_QuadSelfEncoder(Actor_QuadSelfEncoder):
    def __init__(self, cfg, self_obs_dim, fc_encoder_layer, use_spectral_norm, num_agents):
        super().__init__(cfg, self_obs_dim, fc_encoder_layer, use_spectral_norm)
        self.fc_encoder_layer = fc_encoder_layer
        self.num_agents = num_agents
        self.self_value_mlp = nn.Sequential(
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )
        self.attention_mlp = nn.Sequential(
            fc_layer(fc_encoder_layer * 2, fc_encoder_layer, spec_norm=use_spectral_norm),
            # fc_encoder_layer * 2
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, 1),
        )

    def forward(self, obs_self, num_groups):
        # self_embedding = self.self_encoder(obs_self)
        # self_embedding = self_embedding.view(num_groups, -1, self.fc_encoder_layer)
        # self_embedding = torch.sum(self_embedding, dim=1)
        # self_embedding = self_embedding.repeat(self.num_agents, 1)

        # self_agent_embedding = self.self_encoder(obs_self)
        # self_agent_embedding_input = self_agent_embedding.reshape(num_groups, -1, self.fc_encoder_layer)
        # self_agent_embedding_mean = torch.mean(self_agent_embedding_input, dim=1)
        # self_agent_embedding_mean_repeat = self_agent_embedding_mean.repeat(self.num_agents, 1)
        # attention_self_agents_input = torch.cat((self_agent_embedding, self_agent_embedding_mean_repeat), dim=1)
        # attention_self_agents_weights = self.attention_mlp(attention_self_agents_input).view(num_groups, -1)
        # attention_self_agents_weights_softmax = torch.nn.functional.softmax(attention_self_agents_weights, dim=1)
        # attention_self_agents_weights_softmax = attention_self_agents_weights_softmax.view(-1, 1)
        # final_self_agents_embedding = attention_self_agents_weights_softmax * self_agent_embedding
        # final_self_agents_embedding = final_self_agents_embedding.view(num_groups, -1, self.fc_encoder_layer)
        # final_self_agents_embedding = torch.sum(final_self_agents_embedding, dim=1)
        # final_self_agents_embedding = final_self_agents_embedding.repeat(self.num_agents, 1)

        self_agent_embedding = self.self_encoder(obs_self)
        # self_values = self.self_value_mlp(self_agent_embedding)
        #
        # self_agent_embedding_input = self_agent_embedding.reshape(num_groups, -1, self.fc_encoder_layer)
        # self_agent_embedding_mean = torch.mean(self_agent_embedding_input, dim=1)
        # self_agent_embedding_mean_repeat = self_agent_embedding_mean.repeat(self.num_agents, 1)
        #
        # attention_self_agents_input = torch.cat((self_agent_embedding, self_agent_embedding_mean_repeat), dim=1)
        # attention_self_agents_weights = self.attention_mlp(attention_self_agents_input).view(num_groups, -1)
        # attention_self_agents_weights_softmax = torch.nn.functional.softmax(attention_self_agents_weights, dim=1)
        # attention_self_agents_weights_softmax = attention_self_agents_weights_softmax.view(-1, 1)
        #
        # final_self_agents_embedding = attention_self_agents_weights_softmax * self_values
        # final_self_agents_embedding = final_self_agents_embedding.view(num_groups, -1, self.fc_encoder_layer)
        # final_self_agents_embedding = torch.sum(final_self_agents_embedding, dim=1)
        # final_self_agents_embedding = final_self_agents_embedding.repeat(self.num_agents, 1)

        return self_agent_embedding


class Actor_ObstacleEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm):
        super().__init__()
        self.cfg=cfg
        self.self_obs_dim = self_obs_dim
        self.num_obstacle_obs=cfg.num_landmarks
        self.obstacle_hidden_size = obstacle_hidden_size
        self.obstacle_obs_dim = obstacle_obs_dim
        # self.obstacle_encoder = nn.Sequential(
        #     fc_layer(self.obstacle_obs_dim+self.self_obs_dim, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        #     fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        # )
        self.hyper_embedding_mlp_w = Hypernet(cfg, input_dim=self_obs_dim + obstacle_obs_dim,
                                              hidden_dim=obstacle_hidden_size,
                                              main_input_dim=self_obs_dim + obstacle_obs_dim,
                                              main_output_dim=obstacle_hidden_size,
                                              )
        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(obstacle_hidden_size * 2, obstacle_hidden_size, spec_norm=use_spectral_norm),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, 1),
        )

    def forward(self, obs, all_neighbor_obs_size, batch_size, num_groups):
        obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:-4*self.cfg.state_dim*self.cfg.num_oppo_obs]
        obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
        self_obs_repeat = obs[:,:self.self_obs_dim].repeat(self.num_obstacle_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_obstacles), dim=1)
        #obstacle_embeds = self.obstacle_encoder(mlp_input)
        #obstacle_embeds = obstacle_embeds.reshape(batch_size, -1, self.obstacle_hidden_size)
        #obstacle_mean_embed = torch.mean(obstacle_embeds, dim=1)
        obstacle_mlp_w = self.hyper_embedding_mlp_w(mlp_input)
        obstacle_embedding = torch.matmul(mlp_input.unsqueeze(1), obstacle_mlp_w)
        obstacle_embedding = F.tanh(obstacle_embedding)
        # obstacle_embeddings = obstacle_embedding.reshape(batch_size, -1, self.obstacle_hidden_size)
        # obstacle_mean_embed = torch.mean(obstacle_embeddings, dim=1)
        neighbor_embeddings= obstacle_embedding.reshape(-1, self.obstacle_hidden_size)
        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.obstacle_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_obstacle_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.obstacle_hidden_size)
        obstacle_mean_embed = torch.sum(final_neighborhood_embedding, dim=1)

        
        return obstacle_mean_embed


class Actor_IntentionEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm):
        super().__init__()
        self.self_obs_dim = self_obs_dim
        self.num_obstacle_obs = cfg.num_oppo_obs
        self.obstacle_hidden_size = obstacle_hidden_size
        self.obstacle_obs_dim = obstacle_obs_dim
        # self.obstacle_encoder = nn.Sequential(
        #     fc_layer(self.obstacle_obs_dim+self.self_obs_dim, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        #     fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        # )
        self.hyper_embedding_mlp_w = Hypernet(cfg, input_dim=self_obs_dim + obstacle_obs_dim,
                                              hidden_dim=obstacle_hidden_size,
                                              main_input_dim=self_obs_dim + obstacle_obs_dim,
                                              main_output_dim=obstacle_hidden_size,
                                              )
        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(obstacle_hidden_size * 2, obstacle_hidden_size, spec_norm=use_spectral_norm),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(obstacle_hidden_size, 1),
        )


    def forward(self, obs, obs_intention_latent, num_adv_obs, batch_size, num_groups):

        obs_obstacles = obs_intention_latent.reshape(-1, self.obstacle_obs_dim)
        self_obs_repeat = obs[:, :self.self_obs_dim].repeat(num_adv_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_obstacles), dim=1)
        # obstacle_embeds = self.obstacle_encoder(mlp_input)
        # obstacle_embeds = obstacle_embeds.reshape(batch_size, -1, self.obstacle_hidden_size)
        # obstacle_mean_embed = torch.mean(obstacle_embeds, dim=1)
        obstacle_mlp_w = self.hyper_embedding_mlp_w(mlp_input)
        obstacle_embedding = torch.matmul(mlp_input.unsqueeze(1), obstacle_mlp_w)
        obstacle_embedding = F.tanh(obstacle_embedding)
        # obstacle_embeddings = obstacle_embedding.reshape(batch_size, -1, self.obstacle_hidden_size)
        # obstacle_mean_embed = torch.mean(obstacle_embeddings, dim=1)
        neighbor_embeddings = obstacle_embedding.reshape(-1, self.obstacle_hidden_size)
        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.obstacle_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(2*self.num_obstacle_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.obstacle_hidden_size)
        intention_embed = torch.sum(final_neighborhood_embedding, dim=1)
        return intention_embed

#
# class Critic_ObstacleEncoder(nn.Module):
#     def __init__(self, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm, num_agents):
#         super().__init__()
#         self.num_agents = num_agents
#         self.obstacle_hidden_size = obstacle_hidden_size
#
#         self.obstacle_encoders = Actor_ObstacleEncoder(self_obs_dim, obstacle_obs_dim, self.obstacle_hidden_size,
#                                                        use_spectral_norm)
#
#     def forward(self, obs, all_neighbor_obs_size, batch_size, num_groups):
#         obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:]
#         obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
#         obstacles_embeds = self.obstacle_encoders.obstacle_encoder(obs_obstacles)
#         obstacles_embeds = obstacles_embeds.view(num_groups, -1, self.obstacle_hidden_size)
#         obstacles_embeds = torch.mean(obstacles_embeds, dim=1)
#         # obstacles_embeds = obstacles_embeds.repeat(self.num_agents, 1)
#
#         return obstacles_embeds
class Critic_ObstacleEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm,num_agents):
        super().__init__()
        self.cfg=cfg
        self.num_agents = num_agents
        self.num_obstacle_obs = cfg.num_landmarks
        self.num_heads = cfg.num_heads
        self.attention_size=cfg.attention_size
        self.obstacle_hidden_size = obstacle_hidden_size
        self.self_obs_dim=self_obs_dim
        self.obstacle_obs_dim=obstacle_obs_dim

        self.obstacle_encoder = nn.Sequential(
            fc_layer(obstacle_obs_dim+self_obs_dim, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )


        #  outputs h_i from the paper
        self.obstacle_value_mlp = nn.Sequential(
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )
        self.obstacles_value_mlp = nn.Sequential(
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        self.layer_norm = nn.LayerNorm(self.obstacle_hidden_size, eps=1e-6)


    def forward(self, obs, all_neighbor_obs_size, batch_size,num_groups):
        # obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:]
        # obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
        # obstacle_embeds = self.obstacle_encoder(obs_obstacles)
        #
        # obstacle_mean_input = obstacle_embeds.reshape(batch_size, -1, self.obstacle_hidden_size)
        # obstacle_mean_embed = torch.mean(obstacle_mean_input, dim=1)

        obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:-4*self.cfg.state_dim*self.cfg.num_oppo_obs]
        obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = obs[:,:self.self_obs_dim].repeat(self.num_obstacle_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_obstacles), dim=1)

        neighbor_embeddings = self.obstacle_encoder(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.obstacle_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)

        neighbor_values = self.obstacle_value_mlp(neighbor_embeddings)

        agent_v = neighbor_values.view(batch_size, -1, self.obstacle_hidden_size)
        agent_k = neighbor_embeddings.view(batch_size, -1, self.obstacle_hidden_size)
        agent_q = neighbor_embeddings_mean.view(batch_size, -1, self.obstacle_hidden_size)

        d = self.obstacle_hidden_size ** 0.5

        agent_scores = torch.matmul(agent_q / d, agent_k.transpose(1, 2))
        agent_attention = torch.softmax(agent_scores, dim=-1)
        agent_attention = torch.matmul(agent_attention, agent_v)
        obstacle_mean_embed = agent_attention.reshape(-1, self.obstacle_hidden_size)


        # global obstacle multi-attention
        obstacle_values = self.layer_norm(self.obstacles_value_mlp(obstacle_mean_embed))

        obstacle_embeddings_mean_input = obstacle_mean_embed.reshape(num_groups, -1, self.obstacle_hidden_size)
        obstacle_embeddings_mean = torch.mean(obstacle_embeddings_mean_input, dim=1)
        d = self.attention_size ** 0.5
        obstacles_v = obstacle_values.view(num_groups, -1, self.num_heads, self.attention_size)
        obstacles_k = obstacle_mean_embed.view(num_groups, -1, self.num_heads, self.attention_size)
        obstacles_q = obstacle_embeddings_mean.view(num_groups, -1, self.num_heads, self.attention_size)

        obstacles_q, obstacles_k, obstacles_v = obstacles_q.transpose(1, 2), obstacles_k.transpose(1, 2), obstacles_v.transpose(1, 2)

        multi_scores = torch.matmul(obstacles_q / d, obstacles_k.transpose(2, 3))
        multi_attention = torch.softmax(multi_scores, dim=-1)
        multi_attention = torch.matmul(multi_attention, obstacles_v)
        multi_attention = multi_attention.transpose(1, 2)
        multi_attention = multi_attention.reshape(-1, self.num_heads * self.attention_size)

        obstacles_attention = multi_attention.repeat(self.num_agents, 1)



        return obstacles_attention, obstacle_mean_embed

class Critic_IntentionEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm,num_agents):
        super().__init__()
        self.num_adv_obs=cfg.num_oppo_obs
        self.num_agents = num_agents
        # self.num_obstacle_obs = cfg.num_landmarks
        self.num_heads = cfg.num_heads
        self.attention_size=cfg.attention_size
        self.obstacle_hidden_size = obstacle_hidden_size
        self.self_obs_dim=self_obs_dim
        self.obstacle_obs_dim=obstacle_obs_dim

        self.obstacle_encoder = nn.Sequential(
            fc_layer(obstacle_obs_dim+self_obs_dim, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )


        #  outputs h_i from the paper
        self.obstacle_value_mlp = nn.Sequential(
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )
        self.obstacles_value_mlp = nn.Sequential(
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        self.layer_norm = nn.LayerNorm(self.obstacle_hidden_size, eps=1e-6)


    def forward(self, obs, obs_intention_latent, batch_size,num_groups):
        # obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:]
        # obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
        # obstacle_embeds = self.obstacle_encoder(obs_obstacles)
        #
        # obstacle_mean_input = obstacle_embeds.reshape(batch_size, -1, self.obstacle_hidden_size)
        # obstacle_mean_embed = torch.mean(obstacle_mean_input, dim=1)

        obs_obstacles = obs_intention_latent.reshape(-1, self.obstacle_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = obs[:, :self.self_obs_dim].repeat(2*self.num_adv_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_obstacles), dim=1)

        neighbor_embeddings = self.obstacle_encoder(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.obstacle_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)

        neighbor_values = self.obstacle_value_mlp(neighbor_embeddings)

        agent_v = neighbor_values.view(batch_size, -1, self.obstacle_hidden_size)
        agent_k = neighbor_embeddings.view(batch_size, -1, self.obstacle_hidden_size)
        agent_q = neighbor_embeddings_mean.view(batch_size, -1, self.obstacle_hidden_size)

        d = self.obstacle_hidden_size ** 0.5

        agent_scores = torch.matmul(agent_q / d, agent_k.transpose(1, 2))
        agent_attention = torch.softmax(agent_scores, dim=-1)
        agent_attention = torch.matmul(agent_attention, agent_v)
        obstacle_mean_embed = agent_attention.reshape(-1, self.obstacle_hidden_size)


        # global obstacle multi-attention
        obstacle_values = self.layer_norm(self.obstacles_value_mlp(obstacle_mean_embed))

        obstacle_embeddings_mean_input = obstacle_mean_embed.reshape(num_groups, -1, self.obstacle_hidden_size)
        obstacle_embeddings_mean = torch.mean(obstacle_embeddings_mean_input, dim=1)
        d = self.attention_size ** 0.5
        obstacles_v = obstacle_values.view(num_groups, -1, self.num_heads, self.attention_size)
        obstacles_k = obstacle_mean_embed.view(num_groups, -1, self.num_heads, self.attention_size)
        obstacles_q = obstacle_embeddings_mean.view(num_groups, -1, self.num_heads, self.attention_size)

        obstacles_q, obstacles_k, obstacles_v = obstacles_q.transpose(1, 2), obstacles_k.transpose(1, 2), obstacles_v.transpose(1, 2)

        multi_scores = torch.matmul(obstacles_q / d, obstacles_k.transpose(2, 3))
        multi_attention = torch.softmax(multi_scores, dim=-1)
        multi_attention = torch.matmul(multi_attention, obstacles_v)
        multi_attention = multi_attention.transpose(1, 2)
        multi_attention = multi_attention.reshape(-1, self.num_heads * self.attention_size)

        obstacles_attention = multi_attention.repeat(self.num_agents, 1)



        return obstacles_attention, obstacle_mean_embed


class Critic_MultiHeadAttention(nn.Module):
    def __init__(self, cfg, neighbor_obs_dim, adv_obs_dim,neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs,num_adv_obs, intention_obs_dim,obstacle_obs_dim, obstacle_hidden_size, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.num_use_neighbor_obs = num_use_neighbor_obs
        self.num_adv_obs = num_adv_obs
        self.adv_obs_dim=adv_obs_dim
        self.self_obs_dim=self_obs_dim
        self.obstacle_hidden_size = obstacle_hidden_size
        self.fc_encoder_layer = cfg.hidden_size
        self.obstacle_mode = cfg.num_landmarks

        self.neighbor_encoder = Critic_QuadNeighborhood_MultiheadAttention(cfg, neighbor_obs_dim,
                                                                           neighbor_hidden_size,
                                                                           use_spectral_norm,
                                                                           self_obs_dim,
                                                                           num_use_neighbor_obs,
                                                                           num_agents)
        self.adv_encoder = Critic_QuadAdv_MultiheadAttention(cfg, adv_obs_dim,
                                               neighbor_hidden_size,
                                               use_spectral_norm,
                                               self_obs_dim,
                                               num_adv_obs,num_agents)

        self.intention_encoder=Critic_IntentionEncoder(cfg, self_obs_dim,
                               intention_obs_dim,
                               obstacle_hidden_size,
                               use_spectral_norm,num_agents)
        self.self_encoder = Critic_QuadSelfEncoder(cfg, self_obs_dim,
                                                   self.fc_encoder_layer, use_spectral_norm, num_agents)
        if self.obstacle_mode >0:
            self.obstacle_encoder = Critic_ObstacleEncoder(cfg,self_obs_dim,
                                                           obstacle_obs_dim,
                                                           obstacle_hidden_size,
                                                           use_spectral_norm,num_agents)

    def forward(self, self_obs, obs,obs_intention_latent, all_neighbor_obs_size, batch_size, num_groups):
        self_embed = self.self_encoder(self_obs, num_groups)

        if self.num_use_neighbor_obs > 0 and self.neighbor_encoder:
            neighborhood_attention, neighbor_embedding = self.neighbor_encoder(self_obs, obs,
                                                                               all_neighbor_obs_size, batch_size,
                                                                               num_groups)
            adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
            all_adv_obs_size = self.adv_obs_dim * self.num_adv_obs
            adv_attention, adv_embedding = self.adv_encoder(obs, adv_obs_size, all_adv_obs_size, batch_size,num_groups)
            intention_attention, intention_embed=self.intention_encoder(obs,obs_intention_latent, batch_size, num_groups)
            # Add_embedding= neighborhood_attention+neighbor_embedding
            # total_embeddings = torch.cat((self_embed, Add_embedding), dim=1)
            if self.obstacle_mode >0:
                all_obs_size = all_neighbor_obs_size + all_adv_obs_size
                obstacles_attention, obstacle_mean_embed = self.obstacle_encoder(obs, all_obs_size, batch_size, num_groups)

                total_embeddings = torch.cat((self_embed, obstacle_mean_embed), dim=1)
            return neighborhood_attention, neighbor_embedding, adv_attention, adv_embedding,self_embed,intention_attention, intention_embed,obstacles_attention, obstacle_mean_embed


class PositionwiseFeedForward(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.positionforward = nn.Sequential(
            nn.Linear(5 * num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),

        )

    def forward(self, x):
        return self.positionforward(x)


class PositionwiseFeedForward2(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.positionforward = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),

        )

    def forward(self, x):
        return self.positionforward(x)


class FeedForward(nn.Module):
    def __init__(self, total_encoder_out_size, num_hiddens):
        super().__init__()
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, num_hiddens),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.feed_forward(x)


class Gating_layer(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()

        self.Ur = nn.Linear(num_hiddens, num_hiddens)
        self.Wr = nn.Linear(num_hiddens, num_hiddens)
        self.Wz = nn.Linear(num_hiddens, num_hiddens)
        self.Uz = nn.Linear(num_hiddens, num_hiddens)
        self.Ug = nn.Linear(num_hiddens, num_hiddens)
        self.Wg = nn.Linear(num_hiddens, num_hiddens)
        self.bg = torch.zeros(num_hiddens).cuda()

    def forward(self, y, x):
        ## After MultiAttention: y=feed_forward(embeddings), x=neighbor_mean;
        # r = sigmoid(x @ Wr + y @ Ur)
        r = torch.sigmoid(self.Wr(x) + self.Ur(y))
        z = torch.sigmoid(self.Wz(x) + self.Uz(y) - self.bg)
        h = torch.tanh(self.Wg(x) + self.Ug((r * y)))
        g = (1 - z) * y + z * h
        return g


class Multihead_MeanEmbedding_GTrXL(nn.Module):
    def __init__(self, cfg, neighbor_obs_dim, adv_obs_dim,neighbor_hidden_size, use_spectral_norm, self_obs_dim,
                 num_use_neighbor_obs,num_adv_obs, intention_obs_dim,obstacle_obs_dim, obstacle_hidden_size, num_agents, total_encoder_out_size):
        super().__init__()

        self.multi_attention = Critic_MultiHeadAttention(cfg, neighbor_obs_dim, adv_obs_dim,neighbor_hidden_size, use_spectral_norm,
                                                         self_obs_dim,
                                                         num_use_neighbor_obs,num_adv_obs, intention_obs_dim,obstacle_obs_dim, obstacle_hidden_size,
                                                         num_agents)
        self.feedforward = FeedForward(total_encoder_out_size, neighbor_hidden_size)
        self.gating = Gating_layer(neighbor_hidden_size)
        self.gating_adv = Gating_layer(neighbor_hidden_size)
        self.gating2 = Gating_layer(5 * neighbor_hidden_size)
        self.positionforward = PositionwiseFeedForward(neighbor_hidden_size)
        self.positionforward2 = PositionwiseFeedForward2(5 * neighbor_hidden_size)
        self.layer_norm = nn.LayerNorm(5 * neighbor_hidden_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(neighbor_hidden_size, eps=1e-6)
        # self.dropout=nn.Dropout(neighbor_hidden_size)
        self.mlp = nn.Sequential(
            fc_layer(3*cfg.hidden_size, cfg.hidden_size, spec_norm=use_spectral_norm),
            nn.Tanh(), )

    def forward(self, self_obs, obs, obs_intention_latent,all_neighbor_obs_size, batch_size, num_groups):
        # neighbor_mean_embedding:Yt in Coberl or x in GTrXL
        MultiHeadAttention_embeddings, neighbor_embedding, adv_attention, adv_embedding,self_embed,intention_attention, intention_embed,obstacles_attention, obstacle_mean_embed= self.multi_attention(self_obs, obs,obs_intention_latent,
                                                                                             all_neighbor_obs_size,
                                                                                             batch_size,
                                                                                             num_groups)
        # Addlayer_attention=self.layer_norm(MultiHeadAttention_embeddings)
        #
        # Gating_y = self.feedforward(Addlayer_attention)
        Gating_output = self.gating(neighbor_embedding, MultiHeadAttention_embeddings)
        Gating_output_adv = self.gating_adv(adv_embedding, adv_attention)
        Gating_outputs = torch.cat((self_embed, Gating_output,Gating_output_adv,obstacles_attention,intention_attention), dim=1)
        Position_input = self.layer_norm(Gating_outputs)
        Gating_y2 = self.positionforward2(Position_input)
        transformer_out_x = self.gating2(Gating_outputs, Gating_y2)
        out_x = self.positionforward(transformer_out_x)
        gate_z = self.gating(neighbor_embedding, out_x)
        embedding=torch.cat((neighbor_embedding,adv_embedding,intention_embed), dim=1)
        all_embedding=self.mlp(embedding)
        return gate_z, all_embedding


class QuadMultiEncoder(EncoderBase):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, network_type, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        # internal params -- cannot change from cmd line
        # if cfg.quads_obs_repr == 'xyz_vxyz_R_omega':
        #     self.self_obs_dim = 18
        # elif cfg.quads_obs_repr == 'xyz_vxyz_R_omega_wall':
        #     self.self_obs_dim = 24
        # else:
        #     raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')
        self.self_obs_dim=9

        self.num_agents = cfg.num_good_agents+cfg.num_adversaries
        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size

        self.neighbor_obs_type = cfg.neighbor_obs_type
        self.use_spectral_norm = cfg.use_spectral_norm
        self.intention_obs_dim = 2 * cfg.state_dim
        self.obstacle_mode = cfg.num_oppo_obs
        self.obstacle_obs_dim = 12 # internal param, pos_vel_size_type, 3 * 3 + 1, note: for size, we should consider it's length in xyz direction
        self.obstacle_hidden_size = cfg.hidden_size  # internal param
        # if cfg.quads_local_obs == -1:
        #     self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        # else:
        #     self.num_use_neighbor_obs = cfg.quads_local_obs
        self.num_use_neighbor_obs = cfg.num_neighbors_obs
        self.num_adv_obs=cfg.num_oppo_obs

        self.num_adv_neighbor_obs = cfg.num_neighbors_obs
        self.num_ally_obs = cfg.num_oppo_obs
        self.neighbor_obs_dim=6
        self.adv_obs_dim = 12
        # if self.neighbor_obs_type == 'pos_vel_goals':
        #     self.neighbor_obs_dim = 9  # include goal pos info
        # elif self.neighbor_obs_type == 'pos_vel':
        #     self.neighbor_obs_dim = 6
        # elif self.neighbor_obs_type == 'pos_vel_goals_ndist_gdist':
        #     self.neighbor_obs_dim = 11
        # elif self.neighbor_obs_type == 'none':
        #     # override these params so that neighbor encoder is a no-op during inference
        #     self.neighbor_obs_dim = 0
        #     self.num_use_neighbor_obs = 0
        # else:
        #     raise NotImplementedError(f'Unknown value {cfg.neighbor_obs_type} passed to --neighbor_obs_type')

        fc_encoder_layer = cfg.hidden_size
        # encode the current drone's observations
        if network_type == 'Ally_Actor'or network_type == 'Oppo_Actor':
            self.self_encoder = Actor_QuadSelfEncoder(cfg, self.self_obs_dim,
                                                      fc_encoder_layer, self.use_spectral_norm)

        self_encoder_out_size = fc_encoder_layer

        neighbor_encoder_out_size = self.neighbor_hidden_size
        # encode the obstacle observations
        obstacle_encoder_out_size = 0
        if self.obstacle_mode >0:
            if network_type == 'Ally_Actor'or network_type == 'Oppo_Actor':
                self.obstacle_encoder = Actor_ObstacleEncoder(cfg, self.self_obs_dim,
                                                              self.obstacle_obs_dim,
                                                              self.obstacle_hidden_size,
                                                              self.use_spectral_norm)

            obstacle_encoder_out_size = cfg.hidden_size

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # encode the neighboring drone's observations
        if self.num_use_neighbor_obs > 0:
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            if neighbor_encoder_type == 'mean_embed':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(cfg, self.neighbor_obs_dim,
                                                                        self.neighbor_hidden_size,
                                                                        self.use_spectral_norm,
                                                                        self.self_obs_dim, self.num_use_neighbor_obs)

            elif network_type == 'Ally_Actor' and neighbor_encoder_type == 'attention':
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
                self.intention_encoder = Actor_IntentionEncoder(cfg, self.self_obs_dim,
                                                              self.intention_obs_dim ,
                                                              self.obstacle_hidden_size,
                                                              self.use_spectral_norm)
            elif network_type == 'Oppo_Actor' and neighbor_encoder_type == 'attention':
                self.neighbor_encoder = Actor_QuadNeighborhoodEncoderAttention(cfg, self.neighbor_obs_dim,
                                                                               self.neighbor_hidden_size,
                                                                               self.use_spectral_norm,
                                                                               self.self_obs_dim,
                                                                               self.num_adv_neighbor_obs)
                self.adv_encoder = AdvEncoderAttention(cfg, self.adv_obs_dim,
                                                                               self.neighbor_hidden_size,
                                                                               self.use_spectral_norm,
                                                                               self.self_obs_dim,
                                                                               self.num_ally_obs)
                self.intention_encoder = Actor_IntentionEncoder(cfg, self.self_obs_dim,
                                                               self.intention_obs_dim,
                                                               self.obstacle_hidden_size,
                                                               self.use_spectral_norm)
            elif neighbor_encoder_type == 'mlp':
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(cfg, self.neighbor_obs_dim,
                                                                   self.neighbor_hidden_size, self.use_spectral_norm,
                                                                   self.self_obs_dim, self.num_use_neighbor_obs)
            elif neighbor_encoder_type == 'no_encoder':
                self.neighbor_encoder = None  # blind agent
            elif network_type == 'Ally_Critic':
                self.Critic_transformer_encode = Multihead_MeanEmbedding_GTrXL(cfg, self.neighbor_obs_dim,self.adv_obs_dim,
                                                                               self.neighbor_hidden_size,
                                                                               self.use_spectral_norm,
                                                                               self.self_obs_dim,
                                                                               self.num_use_neighbor_obs,
                                                                               self.num_adv_obs,
                                                                               self.intention_obs_dim,
                                                                               self.obstacle_obs_dim,
                                                                               self.obstacle_hidden_size,
                                                                               cfg.num_adversaries,
                                                                               total_encoder_out_size)
            elif network_type == 'Oppo_Critic':
                self.Critic_transformer_encode = Multihead_MeanEmbedding_GTrXL(cfg, self.neighbor_obs_dim,self.adv_obs_dim,
                                                                               self.neighbor_hidden_size,
                                                                               self.use_spectral_norm,
                                                                               self.self_obs_dim,
                                                                               self.num_adv_neighbor_obs,
                                                                               self.num_ally_obs,
                                                                               self.intention_obs_dim,
                                                                               self.obstacle_obs_dim,
                                                                               self.obstacle_hidden_size,
                                                                               cfg.num_good_agents,
                                                                               total_encoder_out_size)

        # this is followed by another fully connected layer in the action parameterization, so we add a nonlinearity here
        self.feed_forward = nn.Sequential(
            fc_layer(5 * cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )
        self.feed_forward1 = nn.Sequential(
            fc_layer(2 * cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )
        self.feed_forward2 = nn.Sequential(
            fc_layer(2 * cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
            fc_layer(cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )
        self.mlp = nn.Sequential(
            fc_layer(2*cfg.hidden_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(), )

        self.encoder_out_size = cfg.hidden_size

    def forward(self, obs_dict, network_type):
        
        obs_self = obs_dict[:, :self.self_obs_dim]
        obs_intention_latent=obs_dict[:,-4*self.cfg.state_dim*self.cfg.num_oppo_obs:]
        # embeddings = obs_self
        batch_size = obs_self.shape[0]
        att_adv_intention =0
        # all neighbors of num_agents per group, batch_size=num_agents* neighbors per one agent * num_groups
        # num_groups = int(batch_size / self.num_agents)
        # relative xyz and vxyz for the entire minibatch (batch dimension is batch_size * num_neighbors)
        if network_type == 'Ally_Actor'or  network_type == 'Ally_Critic':
            all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
            num_adv_obs=self.num_adv_obs
            num_groups = int(batch_size / self.cfg.num_adversaries)
        else:
            all_neighbor_obs_size = self.neighbor_obs_dim * self.num_adv_neighbor_obs
            num_adv_obs = self.num_ally_obs
            num_groups = int(batch_size / self.cfg.num_good_agents)
        if network_type == 'Actor_predict' or network_type == 'Ally_Actor' or  network_type == 'Oppo_Actor':
            self_embed = self.self_encoder(obs_self, num_groups)
            embeddings = self_embed

            if self.num_use_neighbor_obs > 0 and self.neighbor_encoder:
                neighborhood_embedding = self.neighbor_encoder(obs_self, obs_dict, all_neighbor_obs_size, batch_size,
                                                               num_groups)
                adv_obs_size = all_neighbor_obs_size + self.self_obs_dim
                all_adv_obs_size = self.adv_obs_dim * num_adv_obs
                adv_embedding = self.adv_encoder(obs_dict, adv_obs_size, all_adv_obs_size,batch_size)
                embeddings = torch.cat((embeddings, neighborhood_embedding,adv_embedding), dim=1)

                all_obs_size = all_neighbor_obs_size + all_adv_obs_size
                intention_embedding=self.intention_encoder(obs_dict, obs_intention_latent,2*num_adv_obs, batch_size, num_groups)
                adv_intention = torch.cat((adv_embedding.reshape(num_groups, -1, adv_embedding.shape[-1]),
                                           intention_embedding.reshape(num_groups, -1, intention_embedding.shape[-1])),
                                          dim=-1)
                adv_intention_q = self.feed_forward1(adv_intention)
                adv_intention_k = self.feed_forward2(adv_intention)
                adv_intention_v = self.mlp(adv_intention)
                attention_weights = torch.softmax(
                    adv_intention_q @ adv_intention_k.transpose(-2, -1) / adv_intention_k.size(-1) ** 0.5,dim=-1)
                att_adv_intention = attention_weights @ adv_intention_v
                att_adv_intention=att_adv_intention.reshape(-1, att_adv_intention.shape[-1])

            if self.obstacle_mode >0: # treat missiles as obstacles
                all_obs_size = all_neighbor_obs_size + all_adv_obs_size
                obstacle_mean_embed = self.obstacle_encoder(obs_dict, all_obs_size, batch_size, num_groups)

                embeddings = torch.cat((embeddings, obstacle_mean_embed), dim=1)
                # fuse_adv_intention = adv_embedding + att_adv_intention.reshape(-1, att_adv_intention.shape[-1])
                embeddings = torch.cat((embeddings, att_adv_intention), dim=1)
            actor_out = self.feed_forward(embeddings)
            return actor_out

        else: # 'Ally/oppo: Critic'
            gate_z, mean_embed = self.Critic_transformer_encode(obs_self, obs_dict, obs_intention_latent,all_neighbor_obs_size, batch_size,
                                                                num_groups)

            return gate_z, mean_embed


def register_models():
    quad_custom_encoder_name = 'quad_multi_encoder'
    if quad_custom_encoder_name not in ENCODER_REGISTRY:
        register_custom_encoder(quad_custom_encoder_name, QuadMultiEncoder)

