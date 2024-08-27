# Brain-inspired Opponent modeling in Air combat

The part of codebase for training opponent model based multi-agent reinforcement learning policies for UAV swarms.




## Running experiments
### Opponent actions Scheme: Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

```
CUDA_VISIBLE_DEVICES=0 python -m swarm_rl.train --env=dogfight_multi --train_for_env_steps=900000000 --algo=APPO --use_rnn=False --rnn_type=lstm --num_heads=4 --attention_size=32 --num_layer=1 --num_workers=18 --num_envs_per_worker=2 --learning_rate=0.0003 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False 
--policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=128 --encoder_custom=quad_multi_encoder --with_pbt=False --quads_neighbor_hidden_size=128 --quads_obstacle_hidden_size=128 
--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 
--quads_neighbor_encoder_type=attention --replay_buffer_sample_prob=0.75 --anneal_collision_steps=30000000 --experiment=dogfight4v4_v10 
--num_good_agents=4 --num_adversaries=4 --num_landmarks=4 --num_neighbors_obs=3 --num_oppo_obs=4 --num_obstacle_obs=4 --use_spectral_norm=True
 --quads_num_agents=8 --seed=1 --oppo_model_ally=True --local_time_attention=False --global_time_attention=True --scenario_name=4v4/ShootMissile/Selfplay3_altitude_noheading2 -
 -intention_model=False


```


### A hierarchical world model Scheme: Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

```
CUDA_VISIBLE_DEVICES=0 python -m swarm_rl.train --env=dogfight_multi --train_for_env_steps=160000000 --algo=APPO  --use_rnn=False --rnn_type=lstm --num_heads=4 --attention_size=32 --num_layer=1 --num_workers=18 --num_envs_per_worker=2  --learning_rate=0.0003 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False 
 --policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000  --hidden_size=128 --encoder_custom=quad_multi_encoder --with_pbt=False --quads_neighbor_hidden_size=128  --quads_obstacle_hidden_size=128 --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0  --rollout=128 --batch_size=1024 --quads_episode_duration=110.0 --quads_collision_reward=5.0  --quads_neighbor_encoder_type=attention --replay_buffer_sample_prob=0.75 --anneal_collision_steps=30000000   --experiment=dogfight4v4_v10 --num_good_agents=4 --num_adversaries=4 --num_landmarks=4 --num_neighbors_obs=3 
 --num_oppo_obs=4 --num_obstacle_obs=4 --use_spectral_norm=True --quads_num_agents=8 --seed=1 --oppo_model_ally=True   --local_time_attention=False --global_time_attention=True --scenario_name=4v4/ShootMissile/Selfplay3_altitude_noheading2  --intention_model=True

```

### A Joint optimization Scheme: Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

```
CUDA_VISIBLE_DEVICES=0 python -m swarm_rl.train --env=dogfight_multi --train_for_env_steps=160000000 --algo=APPO --use_rnn=False --rnn_type=lstm --num_heads=4 --attention_size=32 --num_layer=1 --num_workers=18 --num_envs_per_worker=2 
--learning_rate=0.0003 --ppo_clip_value=5.0 --recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=128 --encoder_custom=quad_multi_encoder --with_pbt=False --quads_neighbor_hidden_size=128 -
-quads_obstacle_hidden_size=128 --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 --quads_episode_duration=110.0 --quads_neighbor_encoder_type=attention --replay_buffer_sample_prob=0.75 --anneal_collision_steps=30000000 --experiment=dogfight4v4_v10 --num_good_agents=4 
--num_adversaries=4 --num_landmarks=4 --num_neighbors_obs=3 --num_oppo_obs=4 --num_obstacle_obs=4 --use_spectral_norm=True --quads_num_agents=8 --seed=1 --oppo_model_ally=True --local_time_attention=False --global_time_attention=True --scenario_name=4v4/ShootMissile/Selfplay3_altitude_noheading2 --intention_model=False

```



We use the MSE with reparameterization trick to indirectly minimize the difference between the prior and posterior distributions and ensure the minimization of trajectory prediction errors.

Env_steps=num_workers * num_envs_per_worker *  num_agents *  max_length *  episodes=160M
Real_Train_Env_steps=160M / num_agents= 20M

If you use this repository in your work or otherwise wish to cite it, please make reference to our paper.


Github issues and pull requests are welcome. qu	
