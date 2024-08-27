import torch
from torch import nn
from torch.nn import functional as F
from sample_factory.algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
class Hypernet3(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim, main_input_dim, main_output_dim):
        super(Hypernet3, self).__init__()
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
        return self.hyper_w(x).view(-1, x.shape[1],x.shape[2],self.main_input_dim, self.main_output_dim)

class Hypernet2(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim, main_input_dim, main_output_dim):
        super(Hypernet2, self).__init__()
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

class QuadNeighborhoodEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs

class Actor_QuadSelfEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, fc_encoder_layer, use_spectral_norm):
        super().__init__()
        self.self_encoder = nn.Sequential(
            fc_layer(self_obs_dim, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

    def forward(self, obs_self):
        self_embedding = self.self_encoder(obs_self)
        return self_embedding

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

        self.hyper_embedding_mlp_w = Hypernet2(cfg, input_dim=self_obs_dim + neighbor_obs_dim,
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

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:,:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(1,self.num_use_neighbor_obs, 1).reshape(-1,self.self_obs_dim)
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
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1).reshape(-1,obs.shape[1],self.neighbor_hidden_size)

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

        self.hyper_embedding_mlp_w = Hypernet2(cfg, input_dim=adv_obs_dim,
                                              hidden_dim=neighbor_hidden_size,
                                              main_input_dim=adv_obs_dim,
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
        obs_adv = obs[:,:, adv_obs_size:adv_obs_size+ all_adv_obs_size]
        mlp_input  = obs_adv.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation
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
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1).reshape(-1,obs.shape[1],self.neighbor_hidden_size)

        return final_neighborhood_embedding

class Actor_ObstacleEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, obstacle_obs_dim, obstacle_hidden_size, use_spectral_norm):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.num_obstacle_obs = cfg.num_landmarks
        self.obstacle_hidden_size = obstacle_hidden_size
        self.obstacle_obs_dim = obstacle_obs_dim
        # self.obstacle_encoder = nn.Sequential(
        #     fc_layer(self.obstacle_obs_dim+self.self_obs_dim, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        #     fc_layer(obstacle_hidden_size, obstacle_hidden_size, spec_norm=use_spectral_norm),
        #     nonlinearity(cfg),
        # )
        self.hyper_embedding_mlp_w = Hypernet2(cfg, input_dim=self_obs_dim + obstacle_obs_dim,
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

    def forward(self, obs, all_neighbor_obs_size, batch_size):
        obs_obstacles = obs[:, :,self.self_obs_dim + all_neighbor_obs_size:]
        obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
        num_repeat = self.num_obstacle_obs
        self_obs_repeat = obs[:,:, :self.self_obs_dim].repeat(1,num_repeat,1).reshape(-1,self.self_obs_dim)  # +num_pred_obst
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
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(num_repeat, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.obstacle_hidden_size)
        obstacle_mean_embed = torch.sum(final_neighborhood_embedding, dim=1).reshape(-1,obs.shape[1],self.obstacle_hidden_size)

        return obstacle_mean_embed

class PositionwiseFeedForward(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.positionforward = nn.Sequential(
            nn.Linear( num_hiddens, num_hiddens),
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



class Gating_layer(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ur = nn.Linear(num_hiddens, num_hiddens)
        self.Wr = nn.Linear(num_hiddens, num_hiddens)
        self.Wz = nn.Linear(num_hiddens, num_hiddens)
        self.Uz = nn.Linear(num_hiddens, num_hiddens)
        self.Ug = nn.Linear(num_hiddens, num_hiddens)
        self.Wg = nn.Linear(num_hiddens, num_hiddens)
        # self.bg = torch.zeros(num_hiddens)

    def forward(self, y, x):
        ## After MultiAttention: y=feed_forward(embeddings), x=neighbor_mean;
        # r = sigmoid(x @ Wr + y @ Ur)
        r = torch.sigmoid(self.Wr(x) + self.Ur(y))
        z = torch.sigmoid(self.Wz(x) + self.Uz(y) )
        h = torch.tanh(self.Wg(x) + self.Ug((r * y)))
        g = (1 - z) * y + z * h
        return g

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
        return self.hyper_w(x).view(-1, x.shape[1], self.main_input_dim, self.main_output_dim)