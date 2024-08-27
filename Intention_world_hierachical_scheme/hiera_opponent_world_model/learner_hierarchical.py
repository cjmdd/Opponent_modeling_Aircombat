from typing import Tuple
import glob
import os
import shutil
import signal
import threading
import time
from collections import OrderedDict, deque
from os.path import join
from queue import Empty, Queue, Full
from threading import Thread

import numpy as np
import psutil
import torch
from torch.nn.utils.rnn import PackedSequence, invert_permutation
from torch.multiprocessing import Process, Event as MultiprocessingEvent
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue

from sample_factory.algorithms.appo.appo_utils import TaskType, list_of_dicts_to_dict_of_lists, memory_stats, \
    cuda_envvars_for_policy, \
    TensorBatcher, iter_dicts_recursively, copy_dict_structure, ObjectPool
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.aux_losses import CPCA
from sample_factory.algorithms.appo.population_based_training import PbtTask
from sample_factory.algorithms.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.algorithms.utils.algo_utils import calculate_gae, EPS
from sample_factory.algorithms.utils.pytorch_utils import to_scalar
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict, experiment_dir, ensure_dir_exists, join_or_kill, safe_get, \
    safe_put

from sample_factory.model.model import RecurrentStateSpaceModel, ObsModel, RewardModel, PredModel
from sample_factory.model.ES import CEM_Mutate_Novelty
from sample_factory.model.Pred_Conflicts import Z_pred
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import ActionParameterizationContinuousNonAdaptiveStddev


# noinspection PyPep8Naming
def _build_pack_info_from_dones(dones: torch.Tensor, T: int) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create the indexing info needed to make the PackedSequence based on the dones.

    PackedSequences are PyTorch's way of supporting a single RNN forward
    call where each input in the batch can have an arbitrary sequence length

    They work as follows: Given the sequences [c], [x, y, z], [a, b],
    we generate data [x, a, c, y, b, z] and batch_sizes [3, 2, 1].  The
    data is a flattened out version of the input sequences (the ordering in
    data is determined by sequence length).  batch_sizes tells you that
    for each index, how many sequences have a length of (index + 1) or greater.
    This method will generate the new index ordering such that you can
    construct the data for a PackedSequence from a (N*T, ...) tensor
    via x.index_select(0, select_inds)
    """

    num_samples = len(dones)

    rollout_boundaries = dones.clone().detach()
    rollout_boundaries[T - 1::T] = 1  # end of each rollout is the boundary
    rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False).squeeze(dim=1) + 1

    first_len = rollout_boundaries[0].unsqueeze(0)

    if len(rollout_boundaries) <= 1:
        log.debug(
            'Only one rollout boundary. This can happen if batch size is 1, probably not during the real training.')
        rollout_lengths = first_len
    else:
        rollout_lengths = rollout_boundaries[1:] - rollout_boundaries[:-1]
        rollout_lengths = torch.cat([first_len, rollout_lengths])

    rollout_starts_orig = rollout_boundaries - rollout_lengths

    # done=True for the last step in the episode, so done flags rolled 1 step to the right will indicate
    # first frames in the episodes
    is_new_episode = dones.clone().detach().view((-1, T))
    is_new_episode = is_new_episode.roll(1, 1)

    # roll() is cyclical, so done=True in the last position in the rollout will roll to 0th position
    # we want to avoid it here. (note to self: is there a function that does two of these things at once?)
    is_new_episode[:, 0] = 0
    is_new_episode = is_new_episode.view((-1,))

    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)
    # We will want these on the CPU for torch.unique_consecutive,
    # so move now.
    cpu_lengths = lengths.to(device='cpu', non_blocking=True)

    # We need to keep the original unpermuted rollout_starts, because the permutation is later applied
    # internally in the RNN implementation.
    # From modules/rnn.py:
    #       Each batch of the hidden state should match the input sequence that
    #       the user believes he/she is passing in.
    #       hx = self.permute_hidden(hx, sorted_indices)
    rollout_starts_sorted = rollout_starts_orig.index_select(0, sorted_indices)

    select_inds = torch.empty(num_samples, device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())
    # batch_sizes is *always* on the CPU
    batch_sizes = torch.empty((max_length,), device='cpu', dtype=torch.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)
    # Iterate over all unique lengths in reverse as they sorted
    # in decreasing order
    for i in range(len(unique_lengths) - 1, -1, -1):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum().item())

        next_len = int(unique_lengths[i])

        batch_sizes[prev_len:next_len] = num_valid_for_length

        new_inds = (
                rollout_starts_sorted[0:num_valid_for_length].view(1, num_valid_for_length)
                + torch.arange(prev_len, next_len, device=rollout_starts_sorted.device).view(next_len - prev_len, 1)
        ).view(-1)

        # for a set of sequences [1, 2, 3], [4, 5], [6, 7], [8]
        # these indices will be 1,4,6,8,2,5,7,3
        # (all first steps in all trajectories, then all second steps, etc.)
        select_inds[offset:offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    # Make sure we have an index for all elements
    assert offset == num_samples
    assert is_new_episode.shape[0] == num_samples

    return rollout_starts_orig, is_new_episode, select_inds, batch_sizes, sorted_indices


def build_rnn_inputs(x, dones_cpu, rnn_states, T: int):
    """
    Create a PackedSequence input for an RNN such that each
    set of steps that are part of the same episode are all part of
    a batch in the PackedSequence.
    Use the returned select_inds and build_core_out_from_seq to invert this.
    :param x: A (N*T, -1) tensor of the data to build the PackedSequence out of
    :param dones_cpu: A (N*T) tensor where dones[i] == 1.0 indicates an episode is done, a CPU-bound tensor
    :param rnn_states: A (N*T, -1) tensor of the rnn_hidden_states
    :param T: The length of the rollout
    :return: tuple(x_seq, rnn_states, select_inds)
        WHERE
        x_seq is the PackedSequence version of x to pass to the RNN
        rnn_states are the corresponding rnn state, zeroed on the episode boundary
        inverted_select_inds can be passed to build_core_out_from_seq so the RNN output can be retrieved
    """
    rollout_starts, is_new_episode, select_inds, batch_sizes, sorted_indices = _build_pack_info_from_dones(dones_cpu, T)
    inverted_select_inds = invert_permutation(select_inds)

    def device(t):
        return t.to(device=x.device)

    select_inds = device(select_inds)
    inverted_select_inds = device(inverted_select_inds)
    sorted_indices = device(sorted_indices)
    rollout_starts = device(rollout_starts)
    is_new_episode = device(is_new_episode)

    # x = device(x)

    x_seq = PackedSequence(x.index_select(0, select_inds), batch_sizes, sorted_indices)

    # We zero-out rnn states for timesteps at the beginning of the episode.
    # rollout_starts are indices of all starts of sequences
    # (which can be due to episode boundary or just boundary of a rollout)
    # (1 - is_new_episode.view(-1, 1)).index_select(0, rollout_starts) gives us a zero for every beginning of
    # the sequence that is actually also a start of a new episode, and by multiplying this RNN state by zero
    # we ensure no information transfer across episode boundaries.
    rnn_states = rnn_states.index_select(0, rollout_starts)
    is_same_episode = (1 - is_new_episode.view(-1, 1)).index_select(0, rollout_starts)
    rnn_states = rnn_states * is_same_episode

    return x_seq, rnn_states, inverted_select_inds


def build_core_out_from_seq(x_seq: PackedSequence, inverted_select_inds):
    return x_seq.data.index_select(0, inverted_select_inds)


class LearnerWorker:
    def __init__(
            self, worker_idx, policy_id, cfg, obs_space, action_space, report_queue, policy_worker_queues,
            shared_buffers,
            policy_lock, resume_experience_collection_cv
    ):
        log.info('Initializing the learner %d for policy %d', worker_idx, policy_id)
        self.episode_length = shared_buffers.episode_length
        self.actor_critic = None
        self.episode_length = shared_buffers.episode_length
        # self.grads=grads
        # self.episode_length[0]+=1
        self.num_rollouts = shared_buffers.num_rollouts
        self.flag = shared_buffers.flag
        # self.num_rollouts[0] += 1
        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.z_pred = None
        self.cfg = cfg
        self.L = self.cfg.chunk_length
        self.B = self.cfg.batch_size2
        self.N = self.cfg.num_good_agents + self.cfg.num_adversaries
        # self.z_pred=None
        # oppo modeling
        if self.cfg.oppo_model_ally == True:  # oppo also model their oppo(ally)
            self.n = 2
        else:
            self.n = 1
        self.agents = ['Ally', 'Oppo']
        # PBT-related stuff
        self.should_save_model = True  # set to true if we need to save the model to disk on the next training iteration
        self.load_policy_id = None  # non-None when we need to replace our parameters with another policy's parameters
        self.pbt_mutex = None  # deferred initialization
        self.new_cfg = None  # non-None when we need to update the learning hyperparameters

        self.terminate = False
        self.num_batches_processed = 0

        self.obs_space = obs_space
        self.action_space = action_space

        self.shared_buffers = shared_buffers
        self._posterior_state_tensors = shared_buffers._posterior_state_tensors
        self._rnn_tensors = shared_buffers._rnn_tensors

        # deferred initialization
        self.rollout_tensors = None
        self.policy_versions = None
        self.stop_experience_collection = None

        self.stop_experience_collection_num_msgs = self.resume_experience_collection_num_msgs = 0

        self.device = torch.device('cuda', index=0)
        self.actor_critic = None
        self.aux_loss_module = None
        self.optimizer = None

        # Es
        # self.es_params = es_agent.es_params
        # self.alpha_e =es_agent.alpha_e
        # self.prob=es_agent.prob
        # self.worst_index=es_agent.worst_index

        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.task_queue = MpQueue()
        self.report_queue = report_queue

        self.initialized_event = MultiprocessingEvent()
        self.initialized_event.clear()

        self.model_saved_event = MultiprocessingEvent()
        self.model_saved_event.clear()

        # queues corresponding to policy workers using the same policy
        # we send weight updates via these queues
        self.policy_worker_queues = policy_worker_queues

        self.experience_buffer_queue = None  # deferred initialization

        self.tensor_batch_pool = self.tensor_batcher = None

        self.with_training = True  # set to False for debugging no-training regime
        self.train_in_background = self.cfg.train_in_background_thread  # set to False for debugging

        self.training_thread = None
        self.train_thread_initialized = None

        self.is_training = False

        self.train_step = self.env_steps = 0

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 5), (100000, 120), (1000000, 240)])
        self.last_summary_time = 0

        self.last_saved_time = self.last_milestone_time = 0

        self.discarded_experience_over_time = deque([], maxlen=30)
        self.discarded_experience_timer = time.time()
        self.num_discarded_rollouts = 0

        self.process = Process(target=self._run, daemon=True)

        if is_continuous_action_space(self.action_space) and self.cfg.exploration_loss == 'symmetric_kl':
            raise NotImplementedError('KL-divergence exploration loss is not supported with '
                                      'continuous action spaces. Use entropy exploration loss')

        # deferred initialization
        self.exploration_loss_func = None
        self.kl_loss_func = None

    def start_process(self):

        self.process.start()

    def deferred_initialization(self):
        self.rollout_tensors = self.shared_buffers.tensors
        self.policy_versions = self.shared_buffers.policy_versions
        self.stop_experience_collection = self.shared_buffers.stop_experience_collection

        self.pbt_mutex = threading.Lock()
        self.experience_buffer_queue = Queue()

        self.tensor_batch_pool = ObjectPool()
        self.tensor_batcher = TensorBatcher(self.tensor_batch_pool)

        self.training_thread = Thread(target=self._train_loop) if self.train_in_background else None
        self.train_thread_initialized = threading.Event()

        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids: 0.0
        elif self.cfg.exploration_loss == 'entropy':
            self.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == 'symmetric_kl':
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f'{self.cfg.exploration_loss} not supported!')

        if self.cfg.kl_loss_coeff == 0.0:
            if is_continuous_action_space(self.action_space):
                log.warning(
                    'WARNING! It is recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks. '
                    'I.e. set --kl_loss_coeff=1.0'
                )
                time.sleep(3.0)
            self.kl_loss_func = lambda action_space, action_logits, distribution, valids: 0.0
        else:
            self.kl_loss_func = self._kl_loss

    def _init(self):
        log.info('Waiting for the learner to initialize...')
        self.train_thread_initialized.wait()
        log.info('Learner %d initialized', self.worker_idx)
        self.initialized_event.set()

    def _terminate(self):
        self.terminate = True

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _broadcast_model_weights(self):
        state_dict = self.actor_critic.state_dict()
        policy_version = self.train_step
        log.debug('Broadcast model weights for model version %d', policy_version)
        model_state = (policy_version, state_dict, self.obs_model_ally.state_dict(), self.rssm_ally.state_dict(),
                       self.reward_model_ally.state_dict(), self.predict_model_ally.state_dict(),
                       self.obs_model_oppo.state_dict(), self.rssm_oppo.state_dict(),
                       self.reward_model_oppo.state_dict(), self.predict_model_oppo.state_dict())
        # self.z_pred_p.obs_model.load_state_dict(self.obs_model.state_dict())
        # self.z_pred_p.rssm.load_state_dict(self.rssm.state_dict())
        # self.z_pred_p.reward_model.load_state_dict(self.reward_model.state_dict())
        # self.z_pred_p.pred_model.load_state_dict(self.predict_model.state_dict())
        for q in self.policy_worker_queues:
            q.put((TaskType.INIT_MODEL, model_state))

    def _calculate_gae(self, buffer):
        """
        Calculate advantages using Generalized Advantage Estimation.
        This is leftover the from previous version of the algorithm.
        Perhaps should be re-implemented in PyTorch tensors, similar to V-trace for uniformity.
        """

        rewards = np.stack(buffer.rewards).squeeze()  # [E, T]
        dones = np.stack(buffer.dones).squeeze()  # [E, T]
        values_arr = np.stack(buffer.values).squeeze()  # [E, T]

        # calculating fake values for the last step in the rollout
        # this will make sure that advantage of the very last action is always zero
        values = []
        for i in range(len(values_arr)):
            last_value, last_reward = values_arr[i][-1], rewards[i, -1]
            next_value = (last_value - last_reward) / self.cfg.gamma
            values.append(list(values_arr[i]))
            values[i].append(float(next_value))  # [T] -> [T+1]

        # calculating returns and GAE
        rewards = rewards.transpose((1, 0))  # [E, T] -> [T, E]
        dones = dones.transpose((1, 0))  # [E, T] -> [T, E]
        values = np.asarray(values).transpose((1, 0))  # [E, T+1] -> [T+1, E]

        advantages, returns = calculate_gae(rewards, dones, values, self.cfg.gamma, self.cfg.gae_lambda)

        # transpose tensors back to [E, T] before creating a single experience buffer
        buffer.advantages = advantages.transpose((1, 0))  # [T, E] -> [E, T]
        buffer.returns = returns.transpose((1, 0))  # [T, E] -> [E, T]
        buffer.returns = buffer.returns[:, :, np.newaxis]  # [E, T] -> [E, T, 1]

        buffer.advantages = [torch.tensor(buffer.advantages).reshape(-1)]
        buffer.returns = [torch.tensor(buffer.returns).reshape(-1)]

        return buffer

    def _prepare_train_buffer(self, rollouts, macro_batch_size, timing):
        trajectories = [AttrDict(r['t']) for r in rollouts]


        with timing.add_time('buffers'):
            buffer = AttrDict()

            # by the end of this loop the buffer is a dictionary containing lists of numpy arrays
            for i, t in enumerate(trajectories):
                for key, x in t.items():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(x)

            # convert lists of dict observations to a single dictionary of lists
            for key, x in buffer.items():
                if isinstance(x[0], (dict, OrderedDict)):
                    buffer[key] = list_of_dicts_to_dict_of_lists(x)

        if not self.cfg.with_vtrace:
            with timing.add_time('calc_gae'):
                buffer = self._calculate_gae(buffer)

        with timing.add_time('batching'):
            # concatenate rollouts from different workers into a single batch efficiently
            # that is, if we already have memory for the buffers allocated, we can just copy the data into
            # existing cached tensors instead of creating new ones. This is a performance optimization.
            use_pinned_memory = self.cfg.device == 'gpu'
            buffer = self.tensor_batcher.cat(buffer, macro_batch_size, use_pinned_memory, timing)

        with timing.add_time('buff_ready'):
            self.shared_buffers.free_trajectory_buffers([r.traj_buffer_idx for r in rollouts])

        with timing.add_time('tensors_gpu_float'):
            device_buffer = self._copy_train_data_to_device(buffer)

        with timing.add_time('squeeze'):
            # will squeeze actions only in simple categorical case
            tensors_to_squeeze = [
                'actions', 'log_prob_actions', 'policy_version', 'policy_id', 'values',
                'rewards', 'dones', 'rewards_cpu', 'dones_cpu',
            ]
            for tensor_name in tensors_to_squeeze:
                device_buffer[tensor_name].squeeze_()
        work=['worker_id','split_id','env_id']
        for name in work:
            device_buffer[name]=rollouts[0][name]
        # we no longer need the cached buffer, and can put it back into the pool
        self.tensor_batch_pool.put(buffer)
        return device_buffer

    def _macro_batch_size(self, batch_size):
        return self.cfg.num_batches_per_iteration * batch_size

    def _process_macro_batch(self, rollouts, batch_size, timing):
        macro_batch_size = self._macro_batch_size(batch_size)

        assert macro_batch_size % self.cfg.rollout == 0
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert macro_batch_size % self.cfg.recurrence == 0

        samples = env_steps = 0
        for rollout in rollouts:
            samples += rollout['length']
            env_steps += rollout['env_steps']

        with timing.add_time('prepare'):
            buffer = self._prepare_train_buffer(rollouts, macro_batch_size, timing)
            self.experience_buffer_queue.put((buffer, batch_size, samples, env_steps))

            if not self.cfg.benchmark and self.cfg.train_in_background_thread:
                # in PyTorch 1.4.0 there is an intense memory spike when the very first batch is being processed
                # we wait here until this is over so we can continue queueing more batches onto a GPU without having
                # a risk to run out of GPU memory
                while self.num_batches_processed < 1:
                    log.debug('Waiting for the first batch to be processed')
                    time.sleep(0.5)

    def _process_rollouts(self, rollouts, timing):
        # batch_size can potentially change through PBT, so we should keep it the same and pass it around
        # using function arguments, instead of using global self.cfg

        batch_size = self.cfg.batch_size
        rollouts_in_macro_batch = self._macro_batch_size(batch_size) // self.cfg.rollout

        if len(rollouts) < rollouts_in_macro_batch:
            return rollouts

        to_discard = 0
        to_process = []
        policy_version = self.train_step
        for r in rollouts:
            mask = r.t['policy_id'] == self.policy_id
            if np.any(mask):
                rollout_newest_version = r.t['policy_version'][mask].max().item()
            else:
                log.error(
                    'Learner %d got a rollout without any transitions produced by policy %d. This must be a bug.',
                    self.policy_id, self.policy_id,
                )
                log.error('Rollout policy ids: %r', r.t['policy_id'])
                rollout_newest_version = policy_version - self.cfg.max_policy_lag

            if policy_version - rollout_newest_version >= self.cfg.max_policy_lag:
                # the entire rollout is too old, discard it!
                to_discard += 1
                self.shared_buffers.free_trajectory_buffers([r.traj_buffer_idx])
            else:
                # There is some experience in the rollout that we can learn from.
                # Old experience (older than max policy lag), experience from other policies (in case of policy
                # change on episode boundary), and experience from inactive agents (policy id = -1) will be masked
                # out during loss calculations.
                to_process.append(r)

        if to_discard > 0:
            log.warning(
                'Discarding %d old rollouts, cut by policy lag threshold %d (learner %d)',
                to_discard, self.cfg.max_policy_lag, self.policy_id,
            )
            rollouts = to_process
            self.num_discarded_rollouts += to_discard

        if len(rollouts) >= rollouts_in_macro_batch:
            # process newest rollouts
            rollouts_to_process = rollouts[:rollouts_in_macro_batch]
            rollouts = rollouts[rollouts_in_macro_batch:]

            self._process_macro_batch(rollouts_to_process, batch_size, timing)
            # log.info('Unprocessed rollouts: %d (%d samples)', len(rollouts), len(rollouts) * self.cfg.rollout)

        return rollouts

    def _get_minibatches(self, batch_size, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % batch_size == 0, f'experience size: {experience_size}, batch size: {batch_size}'

        if self.cfg.num_batches_per_iteration == 1:
            return [None]  # single minibatch is actually the entire buffer, we don't need indices

        # indices that will start the mini-trajectories from the same episode (for bptt)
        indices = np.arange(0, experience_size, self.cfg.recurrence)
        indices = np.random.permutation(indices)

        # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
        indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
        indices = np.concatenate(indices)

        assert len(indices) == experience_size

        num_minibatches = experience_size // batch_size
        minibatches = np.split(indices, num_minibatches)
        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        mb = AttrDict()

        for item, x in buffer.items():
            if isinstance(x, (dict, OrderedDict)):
                mb[item] = AttrDict()
                for key, x_elem in x.items():
                    mb[item][key] = x_elem[indices]
            else:
                mb[item] = x[indices]

        return mb

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        self._maybe_save()
        # print('train_step',self.train_step)
        # time.sleep(1.2)

    def _maybe_save(self):
        if time.time() - self.last_saved_time >= self.cfg.save_every_sec or self.should_save_model:
            self._save()
            self.model_saved_event.set()
            self.should_save_model = False
            self.last_saved_time = time.time()

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f'checkpoint_p{policy_id}')
        checkpoint_dir2 = join(experiment_dir(cfg=cfg), f'checkpoint2_p{policy_id}')
        return ensure_dir_exists(checkpoint_dir), ensure_dir_exists(checkpoint_dir2)

    @staticmethod
    def get_checkpoints(checkpoints_dir):
        checkpoints = glob.glob(join(checkpoints_dir, 'checkpoint_*'))

        return sorted(checkpoints)

    @staticmethod
    def get_checkpoints2(checkpoints_dir2):

        checkpoints2 = glob.glob(join(checkpoints_dir2, 'allmodel_*'))
        return sorted(checkpoints2)

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        checkpoint2 = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'rssm_ally': self.rssm_ally.state_dict(),
            'obs_model_ally': self.obs_model_ally.state_dict(),
            'reward_model_ally': self.reward_model_ally.state_dict(),
            'predict_model_ally': self.predict_model_ally.state_dict(),
            # 'reward_goal_model': self.reward_goal_model.state_dict(),
            # 'reward_obstacle_model': self.reward_obstacle_model.state_dict(),
            'optimizer2_ally': self.optimizer2_ally.state_dict(),
            'rssm_oppo': self.rssm_oppo.state_dict(),
            'obs_model_oppo': self.obs_model_oppo.state_dict(),
            'reward_model_oppo': self.reward_model_oppo.state_dict(),
            'predict_model_oppo': self.predict_model_oppo.state_dict(),
            'optimizer2_oppo': self.optimizer2_oppo.state_dict(),
        }
        if self.aux_loss_module is not None:
            checkpoint['aux_loss_module'] = self.aux_loss_module.state_dict()

        return checkpoint, checkpoint2

    def _save(self):
        checkpoint, checkpoint2 = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir, checkpoint_dir2 = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, 'temp_checkpoint')
        checkpoint_name = f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth'
        filepath = join(checkpoint_dir, checkpoint_name)
        log.info('Saving %s...', tmp_filepath)
        torch.save(checkpoint, tmp_filepath)
        log.info('Renaming %s to %s', tmp_filepath, filepath)
        os.rename(tmp_filepath, filepath)

        checkpoint_name2 = f'allmodel_{self.train_step:09d}_{self.env_steps}.pth'
        filepath2 = join(checkpoint_dir2, checkpoint_name2)
        torch.save(checkpoint2, filepath2)

        while len(self.get_checkpoints(checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints(checkpoint_dir)[0]

            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

        while len(self.get_checkpoints2(checkpoint_dir2)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints2(checkpoint_dir2)[0]

            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

        if self.cfg.save_milestones_sec > 0:
            # milestones enabled
            if time.time() - self.last_milestone_time >= self.cfg.save_milestones_sec:
                milestones_dir = ensure_dir_exists(join(checkpoint_dir, 'milestones'))
                milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
                log.debug('Saving a milestone %s', milestone_path)
                shutil.copy(filepath, milestone_path)
                self.last_milestone_time = time.time()

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = torch.masked_select(loss, valids)
        loss = -loss.mean()

        return loss

    @staticmethod
    def _policy_loss2(ratio, adv, clip_ratio_low, clip_ratio_high, valids):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        clipped_ratio.retain_grad()
        loss_unclipped = ratio * adv
        loss_unclipped.retain_grad()
        loss_clipped = clipped_ratio * adv
        loss_clipped.retain_grad()
        loss = torch.min(loss_unclipped, loss_clipped)
        loss.retain_grad()
        loss = torch.masked_select(loss, valids)
        loss.retain_grad()
        loss = -loss.mean()
        loss.retain_grad()

        return loss

    def _value_loss(self, new_values, old_values, target, clip_value, valids):
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = torch.masked_select(value_loss, valids)
        value_loss = value_loss.mean()

        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def _kl_loss(self, action_space, action_logits, action_distribution, valids):
        old_action_distribution = get_action_distribution(action_space, action_logits)
        kl_loss = action_distribution.kl_divergence(old_action_distribution)
        kl_loss = torch.masked_select(kl_loss, valids)
        kl_loss = kl_loss.mean()

        kl_loss *= self.cfg.kl_loss_coeff

        return kl_loss

    def _entropy_exploration_loss(self, action_distribution, valids):
        entropy = action_distribution.entropy()
        entropy = torch.masked_select(entropy, valids)
        entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _symmetric_kl_exploration_loss(self, action_distribution, valids):
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = torch.masked_select(kl_prior, valids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _prepare_observations(self, obs_tensors, gpu_buffer_obs):
        for d, gpu_d, k, v, _ in iter_dicts_recursively(obs_tensors, gpu_buffer_obs):
            device, dtype = self.actor_critic.device_and_type_for_input_tensor(k)
            tensor = v.detach().to(device, copy=True).type(dtype)
            gpu_d[k] = tensor

    def _copy_train_data_to_device(self, buffer):
        device_buffer = copy_dict_structure(buffer)

        for key, item in buffer.items():
            if key == 'obs':
                self._prepare_observations(item, device_buffer['obs'])
            else:
                device_tensor = item.detach().to(self.device, copy=True, non_blocking=True)
                device_buffer[key] = device_tensor.float()

        device_buffer['dones_cpu'] = buffer.dones.to('cpu', copy=True, non_blocking=True).float()
        device_buffer['rewards_cpu'] = buffer.rewards.to('cpu', copy=True, non_blocking=True).float()

        return device_buffer

    def update_by_rollout(self, gpu_buffer, batch_size, experience_size, index, timing):
        clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
        # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
        clip_ratio_low = 1.0 / clip_ratio_high

        clip_value = self.cfg.ppo_clip_value
        minibatches = self._get_minibatches(batch_size, experience_size)
        for batch_num in range(len(minibatches)):
            with timing.add_time('minibatch_init'):
                indices = minibatches[batch_num]
            mb = self._get_minibatch(gpu_buffer, indices)

            # calculate policy head outside of recurrent loop
            with timing.add_time('forward_head'):
                # log_prob_actions_new, log_prob_deltas_new, obs_v = self.rssm.rollout2(self.cfg.horizon,
                #                                                                       self.obs_model,
                #                                                                       self.reward_model,
                #                                                                       self.predict_model,
                #                                                                       self.actor_critic,
                #                                                                       mb.obs['obs2'],
                #                                                                       mb.delta_s,
                #                                                                       mb.state_post,
                #                                                                       mb.actions2,
                #                                                                       mb.delta_s_rollout
                #                                                                       )
                # [H*T*N,dim]
                obs_v = mb.obs['obs2']

                actions2, _, _, action_distribution2 = self.actor_critic.forward2(obs_v, index)
                # actions2.requires_grad=True
                # actions2.retain_grad()

                log_prob_actions_new = action_distribution2.log_prob(mb.actions2)
            with timing.add_time('tail'):
                values = self.actor_critic.forward3(obs_v, index, with_action_distribution=True)
                values = values.squeeze()
                # action_distribution = result.action_distribution
                # log_prob_actions = action_distribution.log_prob(mb.actions)
                ratio_action = torch.exp(log_prob_actions_new - mb.log_prob_actions2)  # pi / pi_old
                ratio = torch.clamp(ratio_action, 0.05, 20.0)
            with torch.no_grad():
                valids = mb.policy_id2 == self.policy_id
                adv = mb.advantages2
                targets = mb.returns2

                adv_mean = adv.mean()
                adv_std = adv.std()
                adv = (adv - adv_mean) / max(1e-3, adv_std)  # normalize advantage
                adv = adv.to(self.device)
            with timing.add_time('losses'):
                policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids)
                targets = targets.to(self.device)
                old_values = mb.values2
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids)
                # self.lambda_t[index] = self.lambda_t[index] - self.cfg.learning_rate * gpu_buffer.lambda_t_gradient
                loss = policy_loss + value_loss
        return loss

    

    def update_intention_world_model(self,history, Obs,Obs_state,Action,Reward,gpu_buffer,batch_size, experience_size,timing):
        '''We use the MSE with reparameterization trick to indirectly minimize the difference between the prior and
        posterior distributions and ensure the minimization of trajectory prediction errors.'''
        next_intention_posteriors = []
        delta_intention_posteriors = []
        next_latent_posteriors = []
        delta_latent_posteriors = []
        Loss_delta_intention_prior_multi = []
        Loss_delta_latent_prior_multi = []
        Reward_loss = []
        Obs_oppo_loss = []
        Obs_neighbor_loss = []
        for i in range(self.n):  # n=1, only ally model their oppo, n=2, both of ally and oppo model their oppo
            kl_loss = 0
            intention, latent_strategy = self.rssm[i].posterior_intention_latent(Obs[i],history[i]) ## T,N*K,D
            delta_intention_posterior = intention[1:] - intention[:-1]
            delta_latent_posterior = latent_strategy[1:] - latent_strategy[:-1]
            delta_intention_prior,delta_latent_prior, z,z2, _,_ = self.rssm[i].prior_intention_latent(intention[:-1], latent_strategy[:-1],Action[i][:-1])
            # delta_s_posterior2 = self.rssm.posterior_delta(obs[1:], z)
            loss_delta_intention_prior = 0.5 * mse_loss(
                delta_intention_prior, delta_intention_posterior, reduction='none').mean([0, 1]).sum()
            loss_delta_latent_prior = 0.5 * mse_loss(
                delta_latent_prior, delta_latent_posterior, reduction='none').mean([0, 1]).sum()
            # loss_delta_s_posterior = 0.5 * mse_loss(
            #     delta_s_posterior2, delta_s_posterior, reduction='none').mean([0, 1]).sum()
            delta_intention_posteriors.append(delta_intention_posterior)
            next_intention_posteriors.append(intention)
            delta_latent_posteriors.append(delta_latent_posterior)
            next_latent_posteriors.append(latent_strategy)
            # pred_2:
            state_2 = delta_intention_prior[:-1] + intention[:-2]
            state_latent2 = delta_latent_prior[:-1] + latent_strategy[:-2]
            delta_intention_prior2, delta_latent_prior2, _, _, _, _ = self.rssm[i].prior_intention_latent(state_2, state_latent2, Action[i][1:-1])
            delta_intention_prior2_all = delta_intention_prior[:-1] + delta_intention_prior2
            delta_intention_posterior2_all = intention[2:] - intention[:-2]
            delta_latent_prior2_all = delta_latent_prior[:-1] + delta_latent_prior2
            delta_latent_posterior2_all = latent_strategy[2:] - latent_strategy[:-2]
            loss_delta_intention_prior2 = 0.5 * mse_loss(
                delta_intention_prior2_all, delta_intention_posterior2_all, reduction='none').mean([0, 1]).sum()
            loss_delta_latent_prior2 = 0.5 * mse_loss(
                delta_latent_prior2_all, delta_latent_posterior2_all, reduction='none').mean([0, 1]).sum()

            # pred_3:
            state_3 = delta_intention_prior2[:-1] + state_2[:-1]
            state_latent3 = delta_latent_prior2[:-1] + state_latent2[:-1]
            delta_intention_prior3,delta_latent_prior3, _, _, _, _ = self.rssm[i].prior_intention_latent(state_3, state_latent3, Action[i][2:-1])
            delta_intention_prior3_all = delta_intention_prior2_all[:-1] + delta_intention_prior3
            delta_intention_posterior3_all = intention[3:] - intention[:-3]
            delta_latent_prior3_all = delta_latent_prior2_all[:-1] + delta_latent_prior3
            delta_latent_posterior3_all = latent_strategy[3:] - latent_strategy[:-3]
            loss_delta_intention_prior3 = 0.5 * mse_loss(
                delta_intention_prior3_all, delta_intention_posterior3_all, reduction='none').mean([0, 1]).sum()
            loss_delta_latent_prior3 = 0.5 * mse_loss(
                delta_latent_prior3_all, delta_latent_posterior3_all, reduction='none').mean([0, 1]).sum()

            # pred_4:
            state_4 = delta_intention_prior3[:-1] + state_3[:-1]
            state_latent4 = delta_latent_prior3[:-1] + state_latent3[:-1]
            delta_intention_prior4, delta_latent_prior4,_, _,_, _ = self.rssm[i].prior_intention_latent(state_4, state_latent4, Action[i][3:-1])
            delta_intention_prior4_all = delta_intention_prior3_all[:-1] + delta_intention_prior4
            delta_intention_posterior4_all = intention[4:] - intention[:-4]
            delta_latent_prior4_all = delta_latent_prior3_all[:-1] + delta_latent_prior4
            delta_latent_posterior4_all = latent_strategy[4:] - latent_strategy[:-4]
            loss_delta_intention_prior4 = 0.5 * mse_loss(
                delta_intention_prior4_all, delta_intention_posterior4_all, reduction='none').mean([0, 1]).sum()
            loss_delta_latent_prior4 = 0.5 * mse_loss(
                delta_latent_prior4_all, delta_latent_posterior4_all, reduction='none').mean([0, 1]).sum()

            # delta_s=torch.cat([next_state_posterior[0],delta_s_posterior],dim=0)
            predict_obs_oppo, predict_obs_neighbor= self.obs_model[i](intention[1:], z,latent_strategy[1:], z2)
            predict_reward = self.reward_model[i](intention[1:], z,latent_strategy[1:], z2)
            
            obs_oppo_loss = 0.5 * mse_loss(predict_obs_oppo.reshape(Obs[i].shape[0]-1,Obs[i].shape[1],-1),Obs[i][1:,:,9+6*self.rssm[i].num_use_neighbor_obs:9+6*self.rssm[i].num_use_neighbor_obs+12*2*self.rssm[i].num_adv_obs])
            obs_neighbor_loss = 0.5 * mse_loss(predict_obs_neighbor.reshape(Obs[i].shape[0]-1,Obs[i].shape[1],-1), Obs[i][1:, :, :9+6*self.rssm[i].num_use_neighbor_obs])
            
            reward_loss = 0.5 * mse_loss(predict_reward, Reward[i][:-1], reduction='none').mean([0, 1]).sum()

            
            
            loss_delta_intention_prior_multi = loss_delta_intention_prior + loss_delta_intention_prior2+loss_delta_intention_prior3 + loss_delta_intention_prior4
            loss_delta_latent_prior_multi = loss_delta_latent_prior+loss_delta_latent_prior2+ loss_delta_latent_prior3 + loss_delta_latent_prior4
            loss_model = loss_delta_intention_prior_multi +loss_delta_latent_prior_multi+ obs_oppo_loss +obs_neighbor_loss + reward_loss
            self.optimizer2[i].zero_grad()
            Loss_delta_intention_prior_multi.append(loss_delta_intention_prior_multi)
            Loss_delta_latent_prior_multi.append(loss_delta_latent_prior_multi)
            Reward_loss.append(reward_loss)
            Obs_oppo_loss.append(obs_oppo_loss)
            Obs_neighbor_loss.append(obs_neighbor_loss)
            loss_model.backward()
            clip_grad_norm_(self.all_params[i], self.cfg.clip_grad_norm)
            self.optimizer2[i].step()

            log.debug(
                f'update_step_{self.agents[i]}: %3d loss: %.5f, loss_delta_intention_prior_multi: %.5f, loss_delta_latent_prior_multi: %.5f,obs_oppo_loss: %.5f, obs_neighbor_loss: %.5f,reward_loss: % .5f'
                % (self.num_rollouts[0],
                   loss_model.item(),
                   loss_delta_intention_prior_multi.item(),
                   loss_delta_latent_prior_multi.item(),
                   obs_oppo_loss.item(), obs_neighbor_loss.item(),reward_loss.item(),
                   ))
            fake_buffer = self.rssm[i].rollout_policy_intention(gpu_buffer, self.cfg.horizon, self.obs_model[i],
                                                                self.reward_model[i],
                                                                self.actor_critic, Obs_state[i],
                                                                delta_intention_posteriors[i],
                                                                next_intention_posteriors[i],
                                                                delta_latent_posteriors[i], next_latent_posteriors[i],
                                                                self.L, self.B, self.N,
                                                                self.actor_critic.encoders_critic[i],
                                                                self.actor_critic.critic_linears[i],
                                                                self.actor_critic.network_type_critic[i],
                                                                self.lambda_t[i], i)
            if i == 0:
                fake_buffer.policy_id2 = fake_buffer.policy_id.reshape(self.cfg.quads_num_agents, self.cfg.rollout,
                                                                       -1)[:self.cfg.num_adversaries].transpose(1,
                                                                                                                0).reshape(
                    -1).repeat(self.cfg.horizon)
                fake_buffer.policy_version2 = fake_buffer.policy_version.reshape(self.cfg.quads_num_agents,
                                                                                 self.cfg.rollout,
                                                                                 -1)[
                                              :self.cfg.num_adversaries].transpose(1, 0).reshape(
                    -1).repeat(self.cfg.horizon)
            else:
                fake_buffer.policy_id2 = fake_buffer.policy_id.reshape(self.cfg.quads_num_agents, self.cfg.rollout,
                                                                       -1)[self.cfg.num_adversaries:].transpose(1,
                                                                                                                0).reshape(
                    -1).repeat(self.cfg.horizon)
                fake_buffer.policy_version2 = fake_buffer.policy_version.reshape(self.cfg.quads_num_agents,
                                                                                 self.cfg.rollout,
                                                                                 -1)[
                                              self.cfg.num_adversaries:].transpose(1, 0).reshape(
                    -1).repeat(self.cfg.horizon)

            # statess[0] = next_states[-1].reshape(-1, self.cfg.state_dim)
            # rnns[0] = next_rnn[-1].reshape(-1, self.cfg.rnn_hidden_dim)
            loss_rollout_ac += self.update_by_rollout(fake_buffer, batch_size, experience_size, i, timing)


           
                
        
        return Obs_oppo_loss,Obs_neighbor_loss,Reward_loss,Loss_delta_intention_prior_multi,Loss_delta_latent_prior_multi

    def _train(self, gpu_buffer, batch_size, experience_size, timing):
        worker_id =gpu_buffer['worker_id']
        split_id = gpu_buffer['split_id']
        env_id=gpu_buffer['env_id']
        history_obs_ally = torch.from_numpy(self.shared_buffers.history_obs_tensor[worker_id,split_id,env_id])[:self.cfg.num_adversaries].to(self.device) # N, T, D
        history_obs_oppo = torch.from_numpy(self.shared_buffers.history_obs_tensor[worker_id, split_id, env_id])[
                           self.cfg.num_adversaries:].to(self.device)  # N, T, D
        history_obs=[history_obs_ally,history_obs_oppo]
        stats_and_summaries = None
        # print('update world model')
        obs = gpu_buffer['obs']['obs'].view(self.N, self.L * self.B, -1)
        obs_ally = obs[:self.cfg.num_adversaries]
        obs_ally = obs_ally.transpose(1, 0)
        obs_oppo = obs[self.cfg.num_adversaries:]
        obs_oppo = obs_oppo.transpose(1, 0)
        Obs = []
        Obs.append(obs_ally)
        Obs.append(obs_oppo)

        obs_state = gpu_buffer['obs_intention_latent'].view(self.N, self.L * self.B, -1)
        obs_state_ally = obs_state[:self.cfg.num_adversaries]
        obs_state_ally = obs_state_ally.transpose(1, 0)
        obs_state_oppo = obs_state[self.cfg.num_adversaries:]
        obs_state_oppo = obs_state_oppo.transpose(1, 0)

        Obs_state = []
        Obs_state.append(obs_state_ally)
        Obs_state.append(obs_state_oppo)
        # obs=obs.reshape(self.cfg.rollout,self.N,-1)
        action = gpu_buffer['actions'].view(self.N, self.L * self.B, -1)
        action_ally = action[:self.cfg.num_adversaries]
        action_ally = action_ally.transpose(1, 0)
        action_oppo = action[self.cfg.num_adversaries:]
        action_oppo = action_oppo.transpose(1, 0)
        Action = []
        Action.append(action_ally)
        Action.append(action_oppo)
        # action=action.reshape(self.cfg.rollout,self.N,-1)
        reward = gpu_buffer['rewards'].view(self.N, self.L * self.B, -1)
        reward_ally = reward[:self.cfg.num_adversaries]
        reward_ally = reward_ally.transpose(1, 0)
        reward_oppo = reward[self.cfg.num_adversaries:]
        reward_oppo = reward_oppo.transpose(1, 0)
        Reward = []
        Reward.append(reward_ally)
        Reward.append(reward_oppo)
        

        if self.cfg.intention_model: # hierachical scheme
            obs_oppo_loss, obs_neighbor_loss, reward_loss, loss_delta_intention_prior_multi, loss_delta_latent_prior_multi=self.update_intention_world_model(history_obs, Obs,Obs_state,Action,Reward,gpu_buffer,batch_size, experience_size,timing)
        


        with torch.no_grad():
            policy_version_before_train = self.train_step
            Loss_delta_intention_prior_multi_ally = loss_delta_intention_prior_multi[0] #ally
            Loss_delta_intention_prior_multi_oppo = loss_delta_intention_prior_multi[1]  # oppo
            Loss_delta_latent_prior_multi_ally = loss_delta_latent_prior_multi[0]  # ally
            Loss_delta_latent_prior_multi_oppo = loss_delta_latent_prior_multi[1]  # oppo
            Loss_obs_ally = obs_oppo_loss[0]
            Loss_obs_oppo = obs_oppo_loss[1]
            Loss_neighbor_ally = obs_neighbor_loss[0]
            Loss_neighbor_oppo = obs_neighbor_loss[1]
            Loss_reward_ally = reward_loss[0]
            Loss_reward_oppo = reward_loss[1]
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = []

            # V-trace parameters
            # noinspection PyArgumentList
            rho_hat = torch.Tensor([self.cfg.vtrace_rho])
            # noinspection PyArgumentList
            c_hat = torch.Tensor([self.cfg.vtrace_c])

            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high

            clip_value = self.cfg.ppo_clip_value
            gamma = self.cfg.gamma
            recurrence = self.cfg.recurrence

            if self.cfg.with_vtrace:
                assert recurrence == self.cfg.rollout and recurrence > 1, \
                    'V-trace requires to recurrence and rollout to be equal'

            num_sgd_steps = 0

            stats_and_summaries = None
            if not self.with_training:
                return stats_and_summaries

        for epoch in range(self.cfg.ppo_epochs):
            with timing.add_time('epoch_init'):
                if early_stop or self.terminate:
                    break

                summary_this_epoch = force_summaries = False

                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with timing.add_time('minibatch_init'):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                # calculate policy head outside of recurrent loop
                with timing.add_time('forward_head'):
                    head_outputs, mean_embed, head_outputs_oppo, mean_embed_oppo = self.actor_critic.forward_head2(
                        mb.obs_intention_latent)

                # initial rnn states
                with timing.add_time('bptt_initial'):
                    if self.cfg.use_rnn:
                        head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                            head_outputs, mb.dones_cpu, mb.rnn_states, recurrence,
                        )
                    else:
                        rnn_states = mb.rnn_states[::recurrence]

                # calculate RNN outputs for each timestep in a loop
                with timing.add_time('bptt'):
                    if self.cfg.use_rnn:
                        with timing.add_time('bptt_forward_core'):
                            core_output_seq, _ = self.actor_critic.forward_core(head_output_seq, rnn_states)
                        core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                    else:
                        core_outputs, core_outputs_oppo, _ = self.actor_critic.forward_core(head_outputs,
                                                                                            head_outputs_oppo,
                                                                                            rnn_states)

                num_trajectories = head_outputs.size(0) // recurrence

                with timing.add_time('tail'):
                    assert core_outputs.shape[0] == head_outputs.shape[0]
                    # core_outputs[1]=torch.cat((core_outputs[1],mean_embed),dim=1)

                    # calculate policy tail outside of recurrent loop
                    result, result_oppo = self.actor_critic.forward_tail2(core_outputs, mean_embed,
                                                                          core_outputs_oppo, mean_embed_oppo,
                                                                          with_action_distribution=True)

                    action_distribution = result.action_distribution
                    # a = mb.actions.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                    #     :self.cfg.num_adversaries]
                    log_prob_actions = action_distribution.log_prob(
                        mb.actions.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                        :self.cfg.num_adversaries].reshape(-1,mb.actions.shape[-1]))
                    ratio = torch.exp(
                        log_prob_actions - mb.log_prob_actions.reshape(self.cfg.quads_num_agents, self.cfg.rollout,
                                                                       -1)[:self.cfg.num_adversaries].reshape(
                            -1))  # pi / pi_old

                    # super large/small values can cause numerical problems and are probably noise anyway
                    ratio = torch.clamp(ratio, 0.05, 20.0)

                    values = result.values.squeeze()

                    action_distribution_oppo = result_oppo.action_distribution
                    log_prob_actions_oppo = action_distribution_oppo.log_prob(
                        mb.actions.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                        self.cfg.num_adversaries:].reshape(-1,mb.actions.shape[-1]))
                    ratio_oppo = torch.exp(
                        log_prob_actions_oppo - mb.log_prob_actions.reshape(self.cfg.quads_num_agents,
                                                                            self.cfg.rollout, -1)[
                                                self.cfg.num_adversaries:].reshape(-1))  # pi / pi_old

                    # super large/small values can cause numerical problems and are probably noise anyway
                    ratio_oppo = torch.clamp(ratio_oppo, 0.05, 20.0)

                    values_oppo = result_oppo.values.squeeze()

                with torch.no_grad():  # these computations are not the part of the computation graph
                    # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
                    valids = mb.policy_id == self.policy_id
                    valids_ally = valids.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                  :self.cfg.num_adversaries].reshape(-1)
                    valids_oppo = valids.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                  self.cfg.num_adversaries:].reshape(-1)
                    # ignore experience that was older than the threshold even before training started
                    valids_ally = valids_ally & (
                                policy_version_before_train - mb.policy_version.reshape(self.cfg.quads_num_agents,
                                                                                        self.cfg.rollout, -1)[
                                                              :self.cfg.num_adversaries].reshape(
                            -1) < self.cfg.max_policy_lag)
                    valids_oppo = valids_oppo & (
                            policy_version_before_train - mb.policy_version.reshape(self.cfg.quads_num_agents,
                                                                                    self.cfg.rollout, -1)[
                                                          self.cfg.num_adversaries:].reshape(
                        -1) < self.cfg.max_policy_lag)
                    if self.cfg.with_vtrace:
                        ratios_cpu = ratio.cpu()
                        values_cpu = values.cpu()
                        rewards_cpu = mb.rewards_cpu
                        dones_cpu = mb.dones_cpu

                        vtrace_rho = torch.min(rho_hat, ratios_cpu)
                        vtrace_c = torch.min(c_hat, ratios_cpu)

                        vs = torch.zeros((num_trajectories * recurrence))
                        adv = torch.zeros((num_trajectories * recurrence))

                        next_values = (values_cpu[recurrence - 1::recurrence] - rewards_cpu[
                                                                                recurrence - 1::recurrence]) / gamma
                        next_vs = next_values

                        with timing.add_time('vtrace'):
                            for i in reversed(range(self.cfg.recurrence)):
                                rewards = rewards_cpu[i::recurrence]
                                dones = dones_cpu[i::recurrence]
                                not_done = 1.0 - dones
                                not_done_times_gamma = not_done * gamma

                                curr_values = values_cpu[i::recurrence]
                                curr_vtrace_rho = vtrace_rho[i::recurrence]
                                curr_vtrace_c = vtrace_c[i::recurrence]

                                delta_s = curr_vtrace_rho * (
                                        rewards + not_done_times_gamma * next_values - curr_values)
                                adv[i::recurrence] = curr_vtrace_rho * (
                                        rewards + not_done_times_gamma * next_vs - curr_values)
                                next_vs = curr_values + delta_s + not_done_times_gamma * curr_vtrace_c * (
                                        next_vs - next_values)
                                vs[i::recurrence] = next_vs

                                next_values = curr_values

                        targets = vs
                    else:
                        # using regular GAE
                        adv = mb.advantages.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                              :self.cfg.num_adversaries].reshape(-1)
                        targets = mb.returns.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                  :self.cfg.num_adversaries].reshape(-1)
                        adv_oppo = mb.advantages.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                   self.cfg.num_adversaries:].reshape(-1)
                        targets_oppo = mb.returns.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                       self.cfg.num_adversaries:].reshape(-1)

                    adv_mean = adv.mean()
                    adv_std = adv.std()
                    adv = (adv - adv_mean) / max(1e-3, adv_std)  # normalize advantage
                    adv = adv.to(self.device)

                    adv_mean_oppo = adv_oppo.mean()
                    adv_std_oppo = adv_oppo.std()
                    adv_oppo = (adv_oppo - adv_mean_oppo) / max(1e-3, adv_std_oppo)  # normalize advantage
                    adv_oppo = adv_oppo.to(self.device)

                with timing.add_time('losses'):

                    policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids_ally)
                    exploration_loss = self.exploration_loss_func(action_distribution, valids_ally)
                    kl_loss = self.kl_loss_func(self.actor_critic.action_space,
                                                mb.action_logits[:self.cfg.num_adversaries],
                                                action_distribution, valids_ally)

                    actor_loss = policy_loss + exploration_loss + kl_loss
                    epoch_actor_losses.append(actor_loss.item())

                    targets = targets.to(self.device)
                    old_values = mb.values.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                 :self.cfg.num_adversaries].reshape(-1)
                    value_loss = self._value_loss(values, old_values, targets, clip_value, valids_ally)
                    critic_loss = value_loss

                    loss = actor_loss + critic_loss
                    # oppo
                    policy_loss_oppo = self._policy_loss(ratio_oppo, adv_oppo, clip_ratio_low, clip_ratio_high,
                                                         valids_oppo)
                    exploration_loss_oppo = self.exploration_loss_func(action_distribution_oppo, valids_oppo)
                    kl_loss_oppo = self.kl_loss_func(self.actor_critic.action_space,
                                                     mb.action_logits[self.cfg.num_adversaries:],
                                                     action_distribution_oppo, valids_oppo)

                    actor_loss_oppo = policy_loss_oppo + exploration_loss_oppo + kl_loss_oppo
                    # epoch_actor_losses.append(actor_loss_oppo.item())

                    targets_oppo = targets_oppo.to(self.device)
                    old_values_oppo = mb.values.reshape(self.cfg.quads_num_agents, self.cfg.rollout, -1)[
                                      self.cfg.num_adversaries:].reshape(-1)
                    value_loss_oppo = self._value_loss(values_oppo, old_values_oppo, targets_oppo, clip_value,
                                                       valids_oppo)
                    critic_loss_oppo = value_loss_oppo

                    loss_oppo = actor_loss_oppo + critic_loss_oppo
                    # loss = loss + loss_oppo+loss_rollout_ac
                    loss = loss + loss_oppo
                    if self.aux_loss_module is not None:
                        with timing.add_time('aux_loss'):
                            aux_loss = self.aux_loss_module(
                                mb.actions.view(num_trajectories, recurrence, -1),
                                (1.0 - mb.dones).view(num_trajectories, recurrence, 1),
                                valids.view(num_trajectories, recurrence, -1),
                                head_outputs.view(num_trajectories, recurrence, -1),
                                core_outputs.view(num_trajectories, recurrence, -1),
                            )

                            loss = loss + aux_loss

                    high_loss = 30.0
                    if abs(to_scalar(policy_loss)) > high_loss or abs(to_scalar(value_loss)) > high_loss or abs(
                            to_scalar(exploration_loss)) > high_loss or abs(to_scalar(kl_loss)) > high_loss:
                        log.warning(
                            'High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)',
                            to_scalar(loss), to_scalar(policy_loss), to_scalar(value_loss),
                            to_scalar(exploration_loss), to_scalar(kl_loss),
                        )
                        force_summaries = True

                # update the weights
                with timing.add_time('update'):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None
                    if self.aux_loss_module is not None:
                        for p in self.aux_loss_module.parameters():
                            p.grad = None

                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time('clip'):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                            if self.aux_loss_module is not None:
                                torch.nn.utils.clip_grad_norm_(self.aux_loss_module.parameters(),
                                                               self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update
                    with self.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad():
                    with timing.add_time('after_optimizer'):
                        self._after_optimizer_step()

                        # collect and report summaries
                        with_summaries = self._should_save_summaries() or force_summaries
                        if with_summaries and not summary_this_epoch:
                            stats_and_summaries = self._record_summaries(AttrDict(locals()), 'real')
                            summary_this_epoch = True
                            force_summaries = False

            # end of an epoch
            # this will force policy update on the inference worker (policy worker)
            self.policy_versions[self.policy_id] = self.train_step

            new_epoch_actor_loss = np.mean(epoch_actor_losses)
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    'Early stopping after %d epochs (%d sgd steps), loss delta %.7f',
                    epoch + 1, num_sgd_steps, loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss
            epoch_actor_losses = []

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars, type):
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()
        # stats.loss_delta_s_prior_multi = var.Loss_delta_s_prior_multi
        # stats.loss_obs = var.Loss_obs
        # stats.loss_reward = var.Loss_reward

        stats.Loss_delta_intention_prior_multi_ally=var.Loss_delta_intention_prior_multi_ally
        stats.Loss_delta_intention_prior_multi_oppo=var.Loss_delta_intention_prior_multi_oppo
        stats.Loss_delta_latent_prior_multi_ally=var.Loss_delta_latent_prior_multi_ally
        stats.Loss_delta_latent_prior_multi_oppo=var.Loss_delta_latent_prior_multi_oppo
        stats.Loss_obs_ally=var.Loss_obs_ally
        stats.Loss_obs_oppo=var.Loss_obs_oppo
        stats.Loss_neighbor_ally=var.Loss_neighbor_ally
        stats.Loss_neighbor_oppo=var.Loss_neighbor_oppo
        stats.Loss_reward_ally=var.Loss_reward_ally
        stats.Loss_reward_oppo = var.Loss_reward_oppo
        stats.valids_fraction = var.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self.policy_id).float().mean()

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.actor_critic.parameters()
            if p.grad is not None
        ) ** 0.5
        stats.grad_norm = grad_norm
        stats.loss = var.loss
        # stats.value = var.result.values.mean()
        stats.value = var.values.mean()
        # stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.policy_loss
        # stats.kl_loss = var.kl_loss
        stats.value_loss = var.value_loss
        # stats.exploration_loss = var.exploration_loss
        if self.aux_loss_module is not None:
            stats.aux_loss = var.aux_loss
        stats.adv_min = var.adv.min()
        stats.adv_max = var.adv.max()
        stats.adv_std = var.adv_std
        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        # if hasattr(var.action_distribution, 'summaries'):
        #     stats.update(var.action_distribution.summaries())

        if var.epoch == self.cfg.ppo_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
            ratio_mean = torch.abs(1.0 - var.ratio).mean().detach()
            ratio_min = var.ratio.min().detach()
            ratio_max = var.ratio.max().detach()
            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

            value_delta = torch.abs(var.values - var.old_values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            # calculate KL-divergence with the behaviour policy action distribution
            old_action_distribution = get_action_distribution(
                self.actor_critic.action_space, var.mb.action_logits,
            )
            # kl_old = var.action_distribution.kl_divergence(old_action_distribution)
            # kl_old_mean = kl_old.mean()
            # kl_old_max = kl_old.max()

            # stats.kl_divergence = kl_old_mean
            # stats.kl_divergence_max = kl_old_max
            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max
            stats.fraction_clipped = (
                        (var.ratio < var.clip_ratio_low).float() + (var.ratio > var.clip_ratio_high).float()).mean()
            stats.ratio_mean = ratio_mean
            stats.ratio_min = ratio_min
            stats.ratio_max = ratio_max
            stats.num_sgd_steps = var.num_sgd_steps

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            adam_max_second_moment = max(tensor_state['exp_avg_sq'].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        if type == 'real':
            version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self.policy_id]
            stats.version_diff_avg = version_diff.mean()
            stats.version_diff_min = version_diff.min()
            stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _update_pbt(self):
        """To be called from the training loop, same thread that updates the model!"""
        with self.pbt_mutex:
            if self.load_policy_id is not None:
                assert self.cfg.with_pbt

                log.debug('Learner %d loads policy from %d', self.policy_id, self.load_policy_id)
                self.load_from_checkpoint(self.load_policy_id)
                self.load_policy_id = None

            if self.new_cfg is not None:
                for key, value in self.new_cfg.items():
                    if self.cfg[key] != value:
                        log.debug('Learner %d replacing cfg parameter %r with new value %r', self.policy_id, key, value)
                        self.cfg[key] = value

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg.learning_rate
                    param_group['betas'] = (self.cfg.adam_beta1, self.cfg.adam_beta2)
                    log.debug('Updated optimizer lr to value %.7f, betas: %r', param_group['lr'], param_group['betas'])

                self.new_cfg = None

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                try:
                    log.warning('Loading state from checkpoint %s...', latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f'Could not load from checkpoint, attempt {attempt}')

    def _load_state(self, checkpoint_dict, checkpoint_dict2, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict['train_step']
            self.env_steps = checkpoint_dict['env_steps']
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

        self.rssm_ally.load_state_dict(checkpoint_dict2['rssm_ally'])
        self.obs_model_ally.load_state_dict(checkpoint_dict2['obs_model_ally'])
        self.reward_model_ally.load_state_dict(checkpoint_dict2['reward_model_ally'])
        self.predict_model_ally.load_state_dict(checkpoint_dict2['predict_model_ally'])
        # self.reward_goal_model.load_state_dict(checkpoint_dict2['reward_goal_model'])
        # self.reward_obstacle_model.load_state_dict(checkpoint_dict2['reward_obstacle_model'])
        self.optimizer2_ally.load_state_dict(checkpoint_dict2['optimizer2_ally'])

        self.rssm_oppo.load_state_dict(checkpoint_dict2['rssm_oppo'])
        self.obs_model_oppo.load_state_dict(checkpoint_dict2['obs_model_oppo'])
        self.reward_model_oppo.load_state_dict(checkpoint_dict2['reward_model_oppo'])
        self.predict_model_oppo.load_state_dict(checkpoint_dict2['predict_model_oppo'])
        # self.reward_goal_model.load_state_dict(checkpoint_dict2['reward_goal_model'])
        # self.reward_obstacle_model.load_state_dict(checkpoint_dict2['reward_obstacle_model'])
        self.optimizer2_oppo.load_state_dict(checkpoint_dict2['optimizer2_oppo'])

        if self.aux_loss_module is not None:
            self.aux_loss_module.load_state_dict(checkpoint_dict['aux_loss_module'])
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def init_model(self, timing):

        self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, self.device, timing)
        self.actor_critic.model_to_device(self.device)
        self.actor_critic.share_memory()
        # world model

        # world model

        self.rssm_ally = RecurrentStateSpaceModel(self.cfg, self.action_space, self.cfg.state_dim,
                                                  self.action_space.shape[0],
                                                  self.cfg.rnn_hidden_dim, self.cfg.num_adversaries,self.cfg.num_neighbors_obs,self.cfg.num_oppo_obs,
                                                  hidden_dim=self.cfg.rnn_hidden_dim,
                                                  device=self.device).to(
            self.device)
        self.rssm_ally.share_memory()
        # self.rssm_ally.gating.bg = self.rssm_ally.gating.bg.to(self.device)
        # self.rssm_ally.gating2.bg = self.rssm_ally.gating2.bg.to(self.device)

        self.obs_model_ally = ObsModel(self.cfg, self.obs_space, self.cfg.num_adversaries,self.cfg.num_good_agents,self.cfg.num_neighbors_obs,self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                       self.cfg.rnn_hidden_dim).to(
            self.device)
        self.obs_model_ally.share_memory()
        # print('learn1', id(self.obs_model))
        self.reward_model_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                             hidden_dim=self.cfg.rnn_hidden_dim).to(
            self.device)
        self.reward_model_ally.share_memory()
        self.predict_model_ally = PredModel(self.cfg, self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                            self.cfg.rnn_hidden_dim).to(self.device)
        self.predict_model_ally.share_memory()
        self.reward_goal_model_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                  hidden_dim=self.cfg.rnn_hidden_dim).to(
            self.device)
        self.reward_goal_model_ally.share_memory()
        self.reward_obstacle_model_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                      hidden_dim=self.cfg.rnn_hidden_dim).to(self.device)
        self.reward_obstacle_model_ally.share_memory()
        self.z_pred_ally = Z_pred(self.cfg, rssm=self.rssm_ally, obs_model=self.obs_model_ally,
                                  reward_model=self.reward_model_ally,
                                  pred_model=self.predict_model_ally, shared_buffers=self.shared_buffers,
                                  device=self.device)
        # oppo
        self.rssm_oppo = RecurrentStateSpaceModel(self.cfg, self.action_space, self.cfg.state_dim,
                                                  self.action_space.shape[0],
                                                  self.cfg.rnn_hidden_dim, self.cfg.num_good_agents,self.cfg.num_neighbors_obs,self.cfg.num_oppo_obs,
                                                  hidden_dim=self.cfg.rnn_hidden_dim,
                                                  device=self.device).to(
            self.device)
        self.rssm_oppo.share_memory()
        # self.rssm_oppo.gating.bg = self.rssm_oppo.gating.bg.to(self.device)
        # self.rssm_oppo.gating2.bg = self.rssm_oppo.gating2.bg.to(self.device)
        self.obs_model_oppo = ObsModel(self.cfg, self.obs_space, self.cfg.num_good_agents,self.cfg.num_adversaries,self.cfg.num_neighbors_obs, self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                       self.cfg.rnn_hidden_dim).to(
            self.device)
        self.obs_model_oppo.share_memory()
        # print('learn1', id(self.obs_model))
        self.reward_model_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                             hidden_dim=self.cfg.rnn_hidden_dim).to(
            self.device)
        self.reward_model_oppo.share_memory()
        self.predict_model_oppo = PredModel(self.cfg, self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                            self.cfg.rnn_hidden_dim).to(self.device)
        self.predict_model_oppo.share_memory()
        self.reward_goal_model_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                  hidden_dim=self.cfg.rnn_hidden_dim).to(
            self.device)
        self.reward_goal_model_oppo.share_memory()
        self.reward_obstacle_model_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                      hidden_dim=self.cfg.rnn_hidden_dim).to(self.device)
        self.reward_obstacle_model_oppo.share_memory()
        self.z_pred_oppo = Z_pred(self.cfg, rssm=self.rssm_oppo, obs_model=self.obs_model_oppo,
                                  reward_model=self.reward_model_oppo,
                                  pred_model=self.predict_model_oppo, shared_buffers=self.shared_buffers,
                                  device=self.device)
        # print('learner_z1', id(self.z_pred))
        # print('learner_p1', id(self.z_pred.obs_model))
        self.para_ally = self.rssm_ally.parameters()
        self.all_params_ally = (list(self.rssm_ally.parameters()) +
                                list(self.obs_model_ally.parameters()) +
                                list(self.reward_model_ally.parameters()) +
                                list(self.predict_model_ally.parameters())
                                # list(self.reward_goal_model.parameters())+
                                # list(self.reward_obstacle_model.parameters())
                                )
        self.optimizer2_ally = torch.optim.Adam(self.all_params_ally, lr=self.cfg.lr2, eps=self.cfg.eps2)

        self.para_oppo = self.rssm_oppo.parameters()
        self.all_params_oppo = (list(self.rssm_oppo.parameters()) +
                                list(self.obs_model_oppo.parameters()) +
                                list(self.reward_model_oppo.parameters()) +
                                list(self.predict_model_oppo.parameters())
                                # list(self.reward_goal_model.parameters())+
                                # list(self.reward_obstacle_model.parameters())
                                )
        self.optimizer2_oppo = torch.optim.Adam(self.all_params_oppo, lr=self.cfg.lr2, eps=self.cfg.eps2)

        self.rssm = []
        self.obs_model = []
        self.reward_model = []
        self.rssm.append(self.rssm_ally)
        self.rssm.append(self.rssm_oppo)
        self.obs_model.append(self.obs_model_ally)
        self.obs_model.append(self.obs_model_oppo)
        self.reward_model.append(self.reward_model_ally)
        self.reward_model.append(self.reward_model_oppo)
        self.optimizer2 = []
        self.optimizer2.append(self.optimizer2_ally)
        self.optimizer2.append(self.optimizer2_oppo)
        self.all_params = []
        self.all_params.append(self.all_params_ally)
        self.all_params.append(self.all_params_oppo)

        if self.cfg.use_cpc:
            self.aux_loss_module = CPCA(self.cfg, self.action_space)

        if self.aux_loss_module is not None:
            self.aux_loss_module.to(device=self.device)

    def load_from_checkpoint(self, policy_id):
        checkpoints_dir, checkpoints_dir2 = self.checkpoint_dir(self.cfg, policy_id)
        checkpoints = self.get_checkpoints(checkpoints_dir)
        checkpoints2 = self.get_checkpoints2(checkpoints_dir2)
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        checkpoint_dict2 = self.load_checkpoint(checkpoints2, self.device)
        # checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id))
        # checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            load_progress = policy_id == self.policy_id
            self._load_state(checkpoint_dict, checkpoint_dict2, load_progress=load_progress)
            # self._load_state(checkpoint_dict, load_progress=load_progress)

    def initialize(self, timing):
        with timing.timeit('init'):
            # initialize the Torch modules
            if self.cfg.seed is None:
                log.info('Starting seed is not provided')
            else:
                log.info('Setting fixed seed %d', self.cfg.seed)
                torch.manual_seed(self.cfg.seed)
                np.random.seed(self.cfg.seed)

            # this does not help with a single experiment
            # but seems to do better when we're running more than one experiment in parallel
            torch.set_num_threads(1)

            if self.cfg.device == 'gpu':
                torch.backends.cudnn.benchmark = True

                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            self.init_model(timing)
            params = list(self.actor_critic.parameters())

            if self.aux_loss_module is not None:
                params += list(self.aux_loss_module.parameters())

            self.optimizer = torch.optim.Adam(
                params,
                self.cfg.learning_rate,
                betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                eps=self.cfg.adam_eps,
            )
            self.lambda_t_ally = self.cfg.lambda_t
            self.lambda_t_oppo = self.cfg.lambda_t
            self.lambda_t = [self.lambda_t_ally, self.lambda_t_oppo]
            self.load_from_checkpoint(self.policy_id)

            self._broadcast_model_weights()  # sync the very first version of the weights

        self.train_thread_initialized.set()

    def _process_training_data(self, data, timing, wait_stats=None):
        self.is_training = True

        buffer, batch_size, samples, env_steps = data
        assert samples == batch_size * self.cfg.num_batches_per_iteration

        self.env_steps += env_steps
        experience_size = buffer.rewards.shape[0]
        # print('trainenv_steps:',self.env_steps)
        stats = dict(learner_env_steps=self.env_steps, policy_id=self.policy_id)

        with timing.add_time('train'):
            discarding_rate = self._discarding_rate()

            self._update_pbt()

            train_stats = self._train(buffer, batch_size, experience_size, timing)

            if train_stats is not None:
                stats['train'] = train_stats

                if wait_stats is not None:
                    wait_avg, wait_min, wait_max = wait_stats
                    stats['train']['wait_avg'] = wait_avg
                    stats['train']['wait_min'] = wait_min
                    stats['train']['wait_max'] = wait_max

                stats['train']['discarded_rollouts'] = self.num_discarded_rollouts
                stats['train']['discarding_rate'] = discarding_rate

                stats['stats'] = memory_stats('learner', self.device)

        self.is_training = False

        try:
            safe_put(self.report_queue, stats, queue_name='report')
        except Full:
            log.warning('Could not report training stats, the report queue is full!')

    def _train_loop(self):
        timing = Timing()
        self.initialize(timing)

        wait_times = deque([], maxlen=self.cfg.num_workers)
        last_cache_cleanup = time.time()
        train = 0
        while not self.terminate:
            with timing.timeit('train_wait'):
                data = safe_get(self.experience_buffer_queue)

            if self.terminate:
                break

            wait_stats = None
            wait_times.append(timing.train_wait)

            if len(wait_times) >= wait_times.maxlen:
                wait_times_arr = np.asarray(wait_times)
                wait_avg = np.mean(wait_times_arr)
                wait_min, wait_max = wait_times_arr.min(), wait_times_arr.max()
                # log.debug(
                #     'Training thread had to wait %.5f s for the new experience buffer (avg %.5f)',
                #     timing.train_wait, wait_avg,
                # )
                wait_stats = (wait_avg, wait_min, wait_max)

            self._process_training_data(data, timing, wait_stats)
            self.num_batches_processed += 1

            train += 1
            # print('train:',train)
            # time.sleep(1.2)
            if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and self.num_batches_processed < 50):
                if self.cfg.device == 'gpu':
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                last_cache_cleanup = time.time()

        time.sleep(0.3)
        log.info('Train loop timing: %s', timing)
        del self.actor_critic
        del self.device

    def _experience_collection_rate_stats(self):
        now = time.time()
        if now - self.discarded_experience_timer > 1.0:
            self.discarded_experience_timer = now
            self.discarded_experience_over_time.append((now, self.num_discarded_rollouts))

    def _discarding_rate(self):
        if len(self.discarded_experience_over_time) <= 1:
            return 0

        first, last = self.discarded_experience_over_time[0], self.discarded_experience_over_time[-1]
        delta_rollouts = last[1] - first[1]
        delta_time = last[0] - first[0]
        discarding_rate = delta_rollouts / (delta_time + EPS)
        return discarding_rate

    def _extract_rollouts(self, data):
        rollouts = []
        for rollout_data in data:
            tensors = self.rollout_tensors.index(rollout_data['traj_buffer_idx'])
            rollout_data['t'] = tensors
            rollouts.append(AttrDict(rollout_data))

        return rollouts

    def _process_pbt_task(self, pbt_task):
        task_type, data = pbt_task

        with self.pbt_mutex:
            if task_type == PbtTask.SAVE_MODEL:
                policy_id = data
                assert policy_id == self.policy_id
                self.should_save_model = True
            elif task_type == PbtTask.LOAD_MODEL:
                policy_id, new_policy_id = data
                assert policy_id == self.policy_id
                assert new_policy_id is not None
                self.load_policy_id = new_policy_id
            elif task_type == PbtTask.UPDATE_CFG:
                policy_id, new_cfg = data
                assert policy_id == self.policy_id
                self.new_cfg = new_cfg

    def _accumulated_too_much_experience(self, rollouts):
        max_minibatches_to_accumulate = self.cfg.num_minibatches_to_accumulate
        if max_minibatches_to_accumulate == -1:
            # default value
            max_minibatches_to_accumulate = 2 * self.cfg.num_batches_per_iteration

        # allow the max batches to accumulate, plus the minibatches we're currently training on
        max_minibatches_on_learner = max_minibatches_to_accumulate + self.cfg.num_batches_per_iteration

        minibatches_currently_training = int(self.is_training) * self.cfg.num_batches_per_iteration

        rollouts_per_minibatch = self.cfg.batch_size / self.cfg.rollout

        # count contribution from unprocessed rollouts
        minibatches_currently_accumulated = len(rollouts) / rollouts_per_minibatch

        # count minibatches ready for training
        minibatches_currently_accumulated += self.experience_buffer_queue.qsize() * self.cfg.num_batches_per_iteration

        total_minibatches_on_learner = minibatches_currently_training + minibatches_currently_accumulated

        return total_minibatches_on_learner >= max_minibatches_on_learner

    def _run(self):
        self.deferred_initialization()
        log.info(f'LEARNER\tpid {os.getpid()}\tparent {os.getppid()}')

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            psutil.Process().nice(self.cfg.default_niceness)
        except psutil.AccessDenied:
            log.error('Low niceness requires sudo!')

        if self.cfg.device == 'gpu':
            cuda_envvars_for_policy(self.policy_id, 'learner')

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(self.cfg.learner_main_loop_num_cores)

        timing = Timing()

        rollouts = []

        if self.train_in_background:
            self.training_thread.start()
        else:
            self.initialize(timing)
            log.error(
                'train_in_background set to False on learner %d! This is slow, use only for testing!', self.policy_id,
            )

        while not self.terminate:
            while True:
                try:
                    tasks = self.task_queue.get_many(timeout=0.005)

                    for task_type, data in tasks:
                        if task_type == TaskType.TRAIN:
                            with timing.add_time('extract'):
                                rollouts.extend(self._extract_rollouts(data))
                                # log.debug('Learner %d has %d rollouts', self.policy_id, len(rollouts))
                        elif task_type == TaskType.INIT:
                            self._init()
                        elif task_type == TaskType.TERMINATE:
                            time.sleep(0.3)
                            log.info('GPU learner timing: %s', timing)
                            self._terminate()
                            break
                        elif task_type == TaskType.PBT:
                            self._process_pbt_task(data)
                except Empty:  # first
                    break

            if self._accumulated_too_much_experience(rollouts):
                # if we accumulated too much experience, signal the policy workers to stop experience collection
                if not self.stop_experience_collection[self.policy_id]:
                    self.stop_experience_collection_num_msgs += 1
                    # TODO: add a logger function for this
                    if self.stop_experience_collection_num_msgs >= 50:
                        log.info(
                            'Learner %d accumulated too much experience, stop experience collection! '
                            'Learner is likely a bottleneck in your experiment (%d times)',
                            self.policy_id, self.stop_experience_collection_num_msgs,
                        )
                        self.stop_experience_collection_num_msgs = 0

                self.stop_experience_collection[self.policy_id] = True
            elif self.stop_experience_collection[self.policy_id]:
                # otherwise, resume the experience collection if it was stopped
                self.stop_experience_collection[self.policy_id] = False
                with self.resume_experience_collection_cv:
                    self.resume_experience_collection_num_msgs += 1
                    if self.resume_experience_collection_num_msgs >= 50:
                        log.debug('Learner %d is resuming experience collection!', self.policy_id)
                        self.resume_experience_collection_num_msgs = 0
                    self.resume_experience_collection_cv.notify_all()

            with torch.no_grad():
                rollouts = self._process_rollouts(rollouts, timing)

                # if self.num_rollouts[0] != 0 and self.flag[0] == False and self.episode_length > 0:
                #     print('novelty-search')

            if not self.train_in_background:
                while not self.experience_buffer_queue.empty():
                    training_data = self.experience_buffer_queue.get()
                    self._process_training_data(training_data, timing)

            self._experience_collection_rate_stats()

        if self.train_in_background:
            self.stop_experience_collectionf.experience_buffer_queue.put(None)
            self.training_thread.join()

    def init(self):

        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()
        self.rssm_p_ally = RecurrentStateSpaceModel(self.cfg, self.action_space, self.cfg.state_dim,
                                                    self.action_space.shape[0],
                                                    self.cfg.rnn_hidden_dim, self.cfg.num_adversaries,self.cfg.num_neighbors_obs,self.cfg.num_oppo_obs,
                                                    hidden_dim=self.cfg.rnn_hidden_dim,
                                                    device=self.device)
        self.rssm_p_ally.share_memory()
        self.obs_model_p_ally = ObsModel(self.cfg, self.obs_space, self.cfg.num_adversaries,self.cfg.num_good_agents,self.cfg.num_neighbors_obs,self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                         self.cfg.rnn_hidden_dim)
        self.obs_model_p_ally.share_memory()
        # print('learn1', id(self.obs_model_p))
        self.reward_model_p_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                               hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_model_p_ally.share_memory()
        self.predict_model_p_ally = PredModel(self.cfg, self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                              self.cfg.rnn_hidden_dim)
        self.predict_model_p_ally.share_memory()
        self.reward_goal_model_p_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                    hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_goal_model_p_ally.share_memory()
        self.reward_obstacle_model_p_ally = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                        hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_obstacle_model_p_ally.share_memory()
        self.z_pred_p_ally = Z_pred(self.cfg, rssm=self.rssm_p_ally, obs_model=self.obs_model_p_ally,
                                    reward_model=self.reward_model_p_ally,
                                    pred_model=self.predict_model_p_ally, shared_buffers=self.shared_buffers,
                                    device=self.device)
        # oppo
        self.rssm_p_oppo = RecurrentStateSpaceModel(self.cfg, self.action_space, self.cfg.state_dim,
                                                    self.action_space.shape[0],
                                                    self.cfg.rnn_hidden_dim, self.cfg.num_good_agents,self.cfg.num_neighbors_obs,self.cfg.num_oppo_obs,
                                                    hidden_dim=self.cfg.rnn_hidden_dim,
                                                    device=self.device)
        self.rssm_p_oppo.share_memory()
        self.obs_model_p_oppo = ObsModel(self.cfg, self.obs_space, self.cfg.num_good_agents,self.cfg.num_adversaries,self.cfg.num_neighbors_obs,self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                         self.cfg.rnn_hidden_dim)
        self.obs_model_p_oppo.share_memory()
        # print('learn1', id(self.obs_model_p))
        self.reward_model_p_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                               hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_model_p_oppo.share_memory()
        self.predict_model_p_oppo = PredModel(self.cfg, self.cfg.state_dim, self.cfg.rnn_hidden_dim,
                                              self.cfg.rnn_hidden_dim)
        self.predict_model_p_oppo.share_memory()
        self.reward_goal_model_p_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                    hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_goal_model_p_oppo.share_memory()
        self.reward_obstacle_model_p_oppo = RewardModel(self.cfg, self.cfg.state_dim, self.cfg.num_oppo_obs,self.cfg.rnn_hidden_dim,
                                                        hidden_dim=self.cfg.rnn_hidden_dim)
        self.reward_obstacle_model_p_oppo.share_memory()
        self.z_pred_p_oppo = Z_pred(self.cfg, rssm=self.rssm_p_oppo, obs_model=self.obs_model_p_oppo,
                                    reward_model=self.reward_model_p_oppo,
                                    pred_model=self.predict_model_p_oppo, shared_buffers=self.shared_buffers,
                                    device=self.device)
        # print('learner_z',id(self.z_pred_p))
        # print('learner_p', id(self.z_pred_p.obs_model))
        # print('learner_obs', id(self.obs_model_p))

    def save_model(self, timeout=None):
        self.model_saved_event.clear()
        save_task = (PbtTask.SAVE_MODEL, self.policy_id)
        self.task_queue.put((TaskType.PBT, save_task))
        log.debug('Wait while learner %d saves the model...', self.policy_id)
        if self.model_saved_event.wait(timeout=timeout):
            log.debug('Learner %d saved the model!', self.policy_id)
        else:
            log.warning('Model saving request timed out!')
        self.model_saved_event.clear()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))
        self.shared_buffers._stop_experience_collection[self.policy_id] = False

    def join(self):
        join_or_kill(self.process)
