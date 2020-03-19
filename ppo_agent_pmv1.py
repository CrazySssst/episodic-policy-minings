import time
from collections import deque, defaultdict
from copy import copy

import numpy as np
import psutil
import tensorflow as tf
from mpi4py import MPI
from baselines import logger
import tf_util
from recorder import Recorder
from utils import explained_variance
from console_util import fmt_row
from mpi_util import MpiAdamOptimizer, RunningMeanStd, sync_from_root

from episodic_curiosity import oracle


from replay_buffer import PrioritizedReplayBuffer

import functools
NO_STATES = ['NO_STATES']



class circular_queue_with_default_value(object):
    def __init__(self, max_size):
        self._storage = np.zeros(max_size)
        self._max_size = max_size
        self._next_idx = 0


    def __len__(self):
        return len(self._storage)

    def add(self,data):
        
        self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._max_size

    def mean(self):
        return self._storage.mean()

    def __getitem__(self, idx):
        assert 0 <= idx < self._max_size
        return self._storage[idx]        

class SemicolonList(list):
    def __str__(self):
        return '['+';'.join([str(x) for x in self])+']'

class InteractionState(object):
    """
    Parts of the PPOAgent's state that are based on interaction with a single batch of envs
    """
    def __init__(self, ob_space, ac_space, nsteps, gamma, venvs, stochpol, comm):
        self.lump_stride = venvs[0].num_envs
        self.venvs = venvs
        assert all(venv.num_envs == self.lump_stride for venv in self.venvs[1:]), 'All venvs should have the same num_envs'
        self.nlump = len(venvs)
        nenvs = self.nenvs = self.nlump * self.lump_stride
        self.reset_counter = 0
        self.env_results = [None] * self.nlump
        self.buf_vpreds_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_vpreds_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_nlps = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_ext = np.zeros((nenvs, nsteps), np.float32)

        self.buf_rews_ec = np.zeros((nenvs, nsteps), np.float32)

        self.buf_acs = np.zeros((nenvs, nsteps, *ac_space.shape), ac_space.dtype)
        self.buf_obs = { k: np.zeros(
                            [nenvs, nsteps] + stochpol.ph_ob[k].shape.as_list()[2:],
                            dtype=stochpol.ph_ob_dtypes[k])
                        for k in stochpol.ph_ob_keys }
        self.buf_ob_last = { k: self.buf_obs[k][:, 0, ...].copy() for k in stochpol.ph_ob_keys }
        self.buf_epinfos = [{} for _ in range(self.nenvs)]
        self.buf_news = np.zeros((nenvs, nsteps), np.float32)
        self.buf_ent = np.zeros((nenvs, nsteps), np.float32)
        self.mem_state = stochpol.initial_state(nenvs)
        self.seg_init_mem_state = copy(self.mem_state) # Memory state at beginning of segment of timesteps
        self.rff_int = RewardForwardFilter(gamma)
        self.rff_rms_int = RunningMeanStd(comm=comm, use_mpi=True)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_int_last = self.buf_vpreds_int[:, 0, ...].copy()
        self.buf_vpred_ext_last = self.buf_vpreds_ext[:, 0, ...].copy()
        self.step_count = 0 # counts number of timesteps that you've interacted with this set of environments
        self.t_last_update = time.time()
        self.statlists = defaultdict(lambda : deque([], maxlen=100)) # Count other stats, e.g. optimizer outputs
        self.stats = defaultdict(float) # Count episodes and timesteps
        self.stats['epcount'] = 0
        self.stats['n_updates'] = 0
        self.stats['tcount'] = 0
        self.stats['nbatch'] = 0

        self.buf_scores = np.zeros((nenvs), np.float32)
        self.buf_nsteps = np.zeros((nenvs), np.float32)
        self.buf_reset = np.zeros((nenvs), np.float32)

        self.buf_ep_raminfos = [{} for _ in range(self.nenvs)]

        self.oracle_visited_count = oracle.OracleExplorationRewardForAllEpisodes()



        self.cur_gen_idx = 0
        self.rews_found_by_ancestors={}
        self.last_gen_policy = None
        self.last_gen_rnd = None


        self.target_max_rews = 20000
        self.max_npolicies = 5
        self.max_nsteps_training_single_policy = 10 * 1e6

        self.plan_step = 3

        self.reset_for_new_generation()
        self.reset_for_new_policy()



    def reset_for_new_generation(self):

        self.cur_policy_idx = 0
        self.policy_dict = []

    def reset_for_new_policy(self):
        self.rews_found_by_contemporary = set()
        self.rews_found_by_cur_policy = {}
        self.rews_found_by_cur_policy_in_one_episode = [[] for _ in range(self.nenvs)]
        self.cur_oracle_visited_count = oracle.OracleExplorationRewardForAllEpisodes()
        self.cur_oracle_visited_count_for_next_gen = oracle.OracleExplorationRewardForAllEpisodes()
        self.cur_policy_nsteps = 0
        self.cur_found_extra_rews_count = 0
        
        self.cur_ith_rews = {}

        self.max_extra_rews_count = 10


        #over 10 episodes
        self.cur_policy_scores  = circular_queue_with_default_value(10)



    def close(self):
        for venv in self.venvs:
            venv.close()

def dict_gather(comm, d, op='mean'):
    if comm is None: return d
    alldicts = comm.allgather(d)
    size = comm.Get_size()
    k2li = defaultdict(list)
    for d in alldicts:
        for (k,v) in d.items():
            k2li[k].append(v)
    result = {}
    for (k,li) in k2li.items():
        if op=='mean':
            result[k] = np.mean(li, axis=0)
        elif op=='sum':
            result[k] = np.sum(li, axis=0)
        elif op=="max":
            result[k] = np.max(li, axis=0)
        else:
            assert 0, op
    return result



def uniqueReward(unclip_reward, pos, open_door_type):
    x, y, room_id, nkeys = pos
    room_has_swords = [6]
    room_has_torchs = [5]


    '''
    0: none reward
    1: collect key
    
    3: kill master
    4: collect gem
    5: collect sword
    6: collect torch
    7: collect mallet

    8: open left door
    9: open right door
    '''
    reward_type = 0

    if unclip_reward ==0:
        reward_type = 0
    elif unclip_reward == 100:
        # sword
        if room_id in room_has_swords:
            reward_type = 5
        else:
        # key
            reward_type = 1
    #open door
    elif unclip_reward == 300:

        if open_door_type == 1:
            reward_type = 8
        elif open_door_type == 2:
            reward_type = 9
    #gem
    elif unclip_reward ==1000:
        reward_type = 4
    # mallet
    elif unclip_reward ==200:
        reward_type = 7
    elif unclip_reward == 3000:
        #torch
        if room_id in room_has_torchs:
            reward_type = 6
        else:
        #kill frog
            reward_type = 3
    #kill master (exclude frog)
    elif unclip_reward == 2000:
        reward_type = 3

    return (reward_type, room_id, unclip_reward)


class PpoAgent(object):
    envs = None
    def __init__(self, *, scope,
                 ob_space, ac_space,
                 stochpol_fn,
                 nsteps, nepochs=4, nminibatches=1,
                 gamma=0.99,
                 gamma_ext=0.99,
                 lam=0.95,
                 ent_coef=0,
                 cliprange=0.2,
                 max_grad_norm=1.0,
                 vf_coef=1.0,
                 lr=30e-5,
                 adam_hps=None,
                 testing=False,
                 comm=None, comm_train=None, use_news=False,
                 update_ob_stats_every_step=True,
                 int_coeff=None,
                 ext_coeff=None,
                 log_interval = 1,
                 only_train_r = True,
                 rnd_type = 'rnd',
                 reset=False, reset_prob=0.2,dynamics_sample=False, save_path=''
                 ):
        self.lr = lr
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.use_news = use_news
        self.update_ob_stats_every_step = update_ob_stats_every_step
        self.abs_scope = (tf.get_variable_scope().name + '/' + scope).lstrip('/')
        
        self.rnd_type = rnd_type

        self.sess = sess = tf_util.get_session()

        self.testing = testing

        self.only_train_r = only_train_r

        self.log_interval = log_interval

        self.reset = reset
        self.reset_prob = reset_prob
        self.dynamics_sample = dynamics_sample

        self.save_path = save_path

        self.random_weight_path = '{}_{}'.format(save_path,str(1))

        self.comm_log = MPI.COMM_SELF
        if comm is not None and comm.Get_size() > 1:
            self.comm_log = comm
            assert not testing or comm.Get_rank() != 0, "Worker number zero can't be testing"
        if comm_train is not None:
            self.comm_train, self.comm_train_size = comm_train, comm_train.Get_size()
        else:
            self.comm_train, self.comm_train_size = self.comm_log, self.comm_log.Get_size()
        self.is_log_leader = self.comm_log.Get_rank()==0
        self.is_train_leader = self.comm_train.Get_rank()==0
        with tf.variable_scope(scope):
            self.best_ret = -np.inf
            self.local_best_ret = - np.inf
            self.rooms = []
            self.local_rooms = []
            self.scores = []
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.stochpol = stochpol_fn()
            self.nepochs = nepochs
            self.cliprange = cliprange
            self.nsteps = nsteps
            self.nminibatches = nminibatches
            self.gamma = gamma
            self.gamma_ext = gamma_ext
            self.lam = lam
            self.adam_hps = adam_hps or dict()
            self.ph_adv = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_int = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_ext = tf.placeholder(tf.float32, [None, None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [None, None])
            self.ph_oldvpred = tf.placeholder(tf.float32, [None, None])
            self.ph_lr = tf.placeholder(tf.float32, [])
            self.ph_lr_pred = tf.placeholder(tf.float32, [])
            self.ph_cliprange = tf.placeholder(tf.float32, [])

            #Define loss.
            neglogpac = self.stochpol.pd_opt.neglogp(self.stochpol.ph_ac)
            entropy = tf.reduce_mean(self.stochpol.pd_opt.entropy())
            vf_loss_int = (0.5 * vf_coef) * tf.reduce_mean(tf.square(self.stochpol.vpred_int_opt - self.ph_ret_int))
            vf_loss_ext = (0.5 * vf_coef) * tf.reduce_mean(tf.square(self.stochpol.vpred_ext_opt - self.ph_ret_ext))
            vf_loss = vf_loss_int + vf_loss_ext
            ratio = tf.exp(self.ph_oldnlp - neglogpac) # p_new / p_old
            negadv = - self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
            ent_loss =  (- ent_coef) * entropy
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            maxkl    = .5 * tf.reduce_max(tf.square(neglogpac - self.ph_oldnlp))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.ph_cliprange)))
            loss = pg_loss + ent_loss + vf_loss + self.stochpol.aux_loss

            #Create optimizer.
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.abs_scope)
            logger.info("PPO: using MpiAdamOptimizer connected to %i peers" % self.comm_train_size)
            trainer = MpiAdamOptimizer(self.comm_train, learning_rate=self.ph_lr, **self.adam_hps)
            grads_and_vars = trainer.compute_gradients(loss, params)
            grads, vars = zip(*grads_and_vars)
            if max_grad_norm:
                _, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            global_grad_norm = tf.global_norm(grads)
            grads_and_vars = list(zip(grads, vars))
            self._train = trainer.apply_gradients(grads_and_vars)

        #assign ph_mean and ph_var
        self.assign_op=[]
        self.assign_op.append(self.stochpol.var_ph_mean.assign(self.stochpol.ph_mean))
        self.assign_op.append(self.stochpol.var_ph_std.assign(self.stochpol.ph_std))
        self.assign_op.append(self.stochpol.var_ph_count.assign(self.stochpol.ph_count))

        #Quantities for reporting.
        self._losses = [loss, pg_loss, vf_loss, entropy, clipfrac, approxkl, maxkl, self.stochpol.aux_loss,
                        self.stochpol.feat_var, self.stochpol.max_feat, global_grad_norm]
        self.loss_names = ['tot', 'pg', 'vf', 'ent', 'clipfrac', 'approxkl', 'maxkl', "auxloss", "featvar",
                           "maxfeat", "gradnorm"]
        self.I = None
        self.disable_policy_update = None
        allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.abs_scope)
        if self.is_log_leader:
            tf_util.display_var_info(allvars)
        self.sess.run(tf.variables_initializer(allvars))
        sync_from_root(self.sess, allvars) #Syncs initialization across mpi workers.
        self.t0 = time.time()
        self.global_tcount = 0


        #save & load
        self.save = functools.partial(tf_util.save_state)
        self.load = functools.partial(tf_util.load_state)


    def start_interaction(self, venvs, disable_policy_update=False):
        self.I = InteractionState(ob_space=self.ob_space, ac_space=self.ac_space,
            nsteps=self.nsteps, gamma=self.gamma,
            venvs=venvs, stochpol=self.stochpol, comm=self.comm_train)
        self.disable_policy_update = disable_policy_update
        self.recorder = Recorder(nenvs=self.I.nenvs, score_multiple=venvs[0].score_multiple)

        if self.only_train_r:
            self.disable_policy_update = True

    def collect_statistics_from_model(self):

        mean, std, count = self.stochpol.get_ph_mean_std()

        self.stochpol.ob_rms.mean = mean
        self.stochpol.ob_rms.var = std **2
        self.stochpol.ob_rms.count = count


    def collect_random_statistics(self, num_timesteps):
        #Initializes observation normalization with data from random agent.
        all_ob = []
        for lump in range(self.I.nlump):
            all_ob.append(self.I.venvs[lump].reset())
        for step in range(num_timesteps):

            if step % 128 ==0:
                logger.info("process: {}".format(str(step/128)))

            for lump in range(self.I.nlump):
                acs = np.random.randint(low=0, high=self.ac_space.n, size=(self.I.lump_stride,))
                self.I.venvs[lump].step_async(acs)
                ob, _, _, _, _,_, _ = self.I.venvs[lump].step_wait()
                all_ob.append(ob)
                if len(all_ob) % (128 * self.I.nlump) == 0:
                    ob_ = np.asarray(all_ob).astype(np.float32).reshape((-1, *self.ob_space.shape))
                    self.stochpol.ob_rms.update(ob_[:,:,:,-1:])
                    all_ob.clear()

        feed = {self.stochpol.ph_mean: self.stochpol.ob_rms.mean, self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5 \
                    , self.stochpol.ph_count: self.stochpol.ob_rms.count}

        self.sess.run(self.assign_op, feed)


    def stop_interaction(self):
        self.I.close()
        self.I = None

    @logger.profile("update")
    def update(self):

        #Some logic gathering best ret, rooms etc using MPI.
        temp = sum(MPI.COMM_WORLD.allgather(self.local_rooms), [])
        temp = sorted(list(set(temp)))
        self.rooms = temp

        temp = sum(MPI.COMM_WORLD.allgather(self.scores), [])
        temp = sorted(list(set(temp)))
        self.scores = temp

        temp = sum(MPI.COMM_WORLD.allgather([self.local_best_ret]), [])
        self.best_ret = max(temp)

        eprews = MPI.COMM_WORLD.allgather(np.mean(list(self.I.statlists["eprew"])))
        local_best_rets = MPI.COMM_WORLD.allgather(self.local_best_ret)
        n_rooms = sum(MPI.COMM_WORLD.allgather([len(self.local_rooms)]), [])

        if MPI.COMM_WORLD.Get_rank() == 0 and self.I.stats["n_updates"] % self.log_interval ==0: 
            logger.info("Rooms visited {}".format(self.rooms))
            logger.info("Best return {}".format(self.best_ret))
            logger.info("Best local return {}".format(sorted(local_best_rets)))
            logger.info("eprews {}".format(sorted(eprews)))
            logger.info("n_rooms {}".format(sorted(n_rooms)))
            logger.info("Extrinsic coefficient {}".format(self.ext_coeff))
            logger.info("Intrinsic coefficient {}".format(self.int_coeff))
            logger.info("Gamma {}".format(self.gamma))
            logger.info("Gamma ext {}".format(self.gamma_ext))
            logger.info("All scores {}".format(sorted(self.scores)))


        '''
        to do:  
        '''
        #Normalize intrinsic rewards.
        rffs_int = np.array([self.I.rff_int.update(rew) for rew in self.I.buf_rews_int.T])
        self.I.rff_rms_int.update(rffs_int.ravel())
        rews_int = self.I.buf_rews_int / np.sqrt(self.I.rff_rms_int.var)
        self.mean_int_rew = np.mean(rews_int)
        self.max_int_rew = np.max(rews_int)

        #Don't normalize extrinsic rewards.
        rews_ext = self.I.buf_rews_ext

        rewmean, rewstd, rewmax = self.I.buf_rews_int.mean(), self.I.buf_rews_int.std(), np.max(self.I.buf_rews_int)

        #Calculate intrinsic returns and advantages.
        lastgaelam = 0
        for t in range(self.nsteps-1, -1, -1): # nsteps-2 ... 0
            if self.use_news:
                nextnew = self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last
            else:
                nextnew = 0.0 #No dones for intrinsic reward.
            nextvals = self.I.buf_vpreds_int[:, t + 1] if t + 1 < self.nsteps else self.I.buf_vpred_int_last
            nextnotnew = 1 - nextnew
            delta = rews_int[:, t] + self.gamma * nextvals * nextnotnew - self.I.buf_vpreds_int[:, t]
            self.I.buf_advs_int[:, t] = lastgaelam = delta + self.gamma * self.lam * nextnotnew * lastgaelam
        rets_int = self.I.buf_advs_int + self.I.buf_vpreds_int

        #Calculate extrinsic returns and advantages.
        lastgaelam = 0
        for t in range(self.nsteps-1, -1, -1): # nsteps-2 ... 0
            nextnew = self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last
            #Use dones for extrinsic reward.
            nextvals = self.I.buf_vpreds_ext[:, t + 1] if t + 1 < self.nsteps else self.I.buf_vpred_ext_last
            nextnotnew = 1 - nextnew
            delta = rews_ext[:, t] + self.gamma_ext * nextvals * nextnotnew - self.I.buf_vpreds_ext[:, t]
            self.I.buf_advs_ext[:, t] = lastgaelam = delta + self.gamma_ext * self.lam * nextnotnew * lastgaelam
        rets_ext = self.I.buf_advs_ext + self.I.buf_vpreds_ext

        #Combine the extrinsic and intrinsic advantages.
        self.I.buf_advs = self.int_coeff*self.I.buf_advs_int + self.ext_coeff*self.I.buf_advs_ext

        #Collects info for reporting.
        info = dict(
            advmean = self.I.buf_advs.mean(),
            advstd  = self.I.buf_advs.std(),
            retintmean = rets_int.mean(), # previously retmean
            retintstd  = rets_int.std(), # previously retstd
            retextmean = rets_ext.mean(), # previously not there
            retextstd  = rets_ext.std(), # previously not there

            rewec_mean = self.I.buf_rews_ec.mean(),
            rewec_max = np.max(self.I.buf_rews_ec),

            rewintmean_unnorm = rewmean, # previously rewmean
            rewintmax_unnorm = rewmax, # previously not there
            rewintmean_norm = self.mean_int_rew, # previously rewintmean
            rewintmax_norm = self.max_int_rew, # previously rewintmax
            rewintstd_unnorm  = rewstd, # previously rewstd
            vpredintmean = self.I.buf_vpreds_int.mean(), # previously vpredmean
            vpredintstd  = self.I.buf_vpreds_int.std(), # previously vrpedstd
            vpredextmean = self.I.buf_vpreds_ext.mean(), # previously not there
            vpredextstd  = self.I.buf_vpreds_ext.std(), # previously not there
            ev_int = np.clip(explained_variance(self.I.buf_vpreds_int.ravel(), rets_int.ravel()), -1, None),
            ev_ext = np.clip(explained_variance(self.I.buf_vpreds_ext.ravel(), rets_ext.ravel()), -1, None),
            rooms = SemicolonList(self.rooms),
            n_rooms = len(self.rooms),
            best_ret = self.best_ret,
            reset_counter = self.I.reset_counter
        )

        info['mem_available'] = psutil.virtual_memory().available

        to_record = {'acs': self.I.buf_acs,
                     'rews_int': self.I.buf_rews_int,
                     'rews_int_norm': rews_int,
                     'rews_ext': self.I.buf_rews_ext,
                     'rews_ect': self.I.buf_rews_ec,
                     'vpred_int': self.I.buf_vpreds_int,
                     'vpred_ext': self.I.buf_vpreds_ext,
                     'adv_int': self.I.buf_advs_int,
                     'adv_ext': self.I.buf_advs_ext,
                     'ent': self.I.buf_ent,
                     'ret_int': rets_int,
                     'ret_ext': rets_ext,
                     }
        if self.I.venvs[0].record_obs:
            to_record['obs'] = self.I.buf_obs[None]
        self.recorder.record(bufs=to_record,
                             infos=self.I.buf_epinfos)


        #Create feeddict for optimization.
        envsperbatch = self.I.nenvs // self.nminibatches
        ph_buf = [
            (self.stochpol.ph_ac, self.I.buf_acs),
            (self.ph_ret_int, rets_int),
            (self.ph_ret_ext, rets_ext),
            (self.ph_oldnlp, self.I.buf_nlps),
            (self.ph_adv, self.I.buf_advs),
        ]
        if self.I.mem_state is not NO_STATES:
            ph_buf.extend([
                (self.stochpol.ph_istate, self.I.seg_init_mem_state),
                (self.stochpol.ph_new, self.I.buf_news),
            ])

        verbose = False
        if verbose and self.is_log_leader:
            samples = np.prod(self.I.buf_advs.shape)
            logger.info("buffer shape %s, samples_per_mpi=%i, mini_per_mpi=%i, samples=%i, mini=%i " % (
                    str(self.I.buf_advs.shape),
                    samples, samples//self.nminibatches,
                    samples*self.comm_train_size, samples*self.comm_train_size//self.nminibatches))
            logger.info(" "*6 + fmt_row(13, self.loss_names))


        epoch = 0
        start = 0
        #Optimizes on current data for several epochs.
        while epoch < self.nepochs:
            end = start + envsperbatch
            mbenvinds = slice(start, end, None)

            fd = {ph : buf[mbenvinds] for (ph, buf) in ph_buf}
            fd.update({self.ph_lr : self.lr, self.ph_cliprange : self.cliprange})
            fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None][mbenvinds], self.I.buf_ob_last[None][mbenvinds, None]], 1)
            assert list(fd[self.stochpol.ph_ob[None]].shape) == [self.I.nenvs//self.nminibatches, self.nsteps + 1] + list(self.ob_space.shape), \
                [fd[self.stochpol.ph_ob[None]].shape, [self.I.nenvs//self.nminibatches, self.nsteps + 1] + list(self.ob_space.shape)]
            fd.update({self.stochpol.ph_mean:self.stochpol.ob_rms.mean, self.stochpol.ph_std:self.stochpol.ob_rms.var**0.5})

            ret = tf_util.get_session().run(self._losses+[self._train], feed_dict=fd)[:-1]
            if not self.testing:
                lossdict = dict(zip([n for n in self.loss_names], ret), axis=0)
            else:
                lossdict = {}
            #Synchronize the lossdict across mpi processes, otherwise weights may be rolled back on one process but not another.
            _maxkl = lossdict.pop('maxkl')
            lossdict = dict_gather(self.comm_train, lossdict, op='mean')
            maxmaxkl = dict_gather(self.comm_train, {"maxkl":_maxkl}, op='max')
            lossdict["maxkl"] = maxmaxkl["maxkl"]
            if verbose and self.is_log_leader:
                logger.info("%i:%03i %s" % (epoch, start, fmt_row(13, [lossdict[n] for n in self.loss_names])))
            start += envsperbatch
            if start == self.I.nenvs:
                epoch += 1
                start = 0

        if self.is_train_leader:
            self.I.stats["n_updates"] += 1
            info.update([('opt_'+n, lossdict[n]) for n in self.loss_names])
            tnow = time.time()
            info['tps'] = self.nsteps * self.I.nenvs / (tnow - self.I.t_last_update)
            info['time_elapsed'] = time.time() - self.t0
            self.I.t_last_update = tnow
        self.stochpol.update_normalization( # Necessary for continuous control tasks with odd obs ranges, only implemented in mlp policy,
            ob=self.I.buf_obs               # NOTE: not shared via MPI
            )
        return info

    def env_step(self, l, acs):
        self.I.venvs[l].step_async(acs)
        self.I.env_results[l] = None

    def env_get(self, l):
        """
        Get most recent (obs, rews, dones, infos) from vectorized environment
        Using step_wait if necessary
        """
        if self.I.step_count == 0: # On the zeroth step with a new venv, we need to call reset on the environment
            ob = self.I.venvs[l].reset()
            out = self.I.env_results[l] = (ob, None,None, np.ones(self.I.lump_stride, bool), {}, None,[])

            #print("aaa")
        else:
            #print("bbb")
            if self.I.env_results[l] is None:
                #print("cccc")
                out = self.I.env_results[l] = self.I.venvs[l].step_wait()
            else:
                #print("dddd")
                out = self.I.env_results[l]
        return out

    @logger.profile("step")
    def step(self):
        #Does a rollout.
        t = self.I.step_count % self.nsteps
        epinfos = []

        self.check_goto_next_policy()

        for l in range(self.I.nlump):
            obs, prevrews, ec_rews, news, infos, ram_states, monitor_rews = self.env_get(l)




            for env_pos_in_lump, info in enumerate(infos):
                if 'episode' in info:
                    #Information like rooms visited is added to info on end of episode.
                    epinfos.append(info['episode'])
                    info_with_places = info['episode']
                    try:
                        info_with_places['places'] = info['episode']['visited_rooms']
                    except:
                        import ipdb; ipdb.set_trace()
                    self.I.buf_epinfos[env_pos_in_lump+l*self.I.lump_stride][t] = info_with_places


                    self.check_episode(env_pos_in_lump+l*self.I.lump_stride)

            sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
            memsli = slice(None) if self.I.mem_state is NO_STATES else sli





            dict_obs = self.stochpol.ensure_observation_is_dict(obs)
            with logger.ProfileKV("policy_inference"):
                #Calls the policy and value function on current observation.
                acs, vpreds_int, vpreds_ext, nlps, self.I.mem_state[memsli], ent = self.stochpol.call(dict_obs, news, self.I.mem_state[memsli],
                                                                                                               update_obs_stats=self.update_ob_stats_every_step)
            self.env_step(l, acs)





            #Update buffer with transition.
            for k in self.stochpol.ph_ob_keys:
                self.I.buf_obs[k][sli, t] = dict_obs[k]
            self.I.buf_news[sli, t] = news
            self.I.buf_vpreds_int[sli, t] = vpreds_int
            self.I.buf_vpreds_ext[sli, t] = vpreds_ext
            self.I.buf_nlps[sli, t] = nlps
            self.I.buf_acs[sli, t] = acs
            self.I.buf_ent[sli, t] = ent

            if t > 0:

                prevrews = [self.filter_rew(prevrews[k], infos[k]['unclip_rew'], infos[k]['position'], infos[k]['open_door_type'],k)
                                for k in range(self.I.nenvs)]
    
                prevrews = np.asarray(prevrews)
                #print(prevrews)

                self.I.buf_rews_ext[sli, t-1] = prevrews
                self.I.buf_rews_ec[sli, t-1] = ec_rews

                if self.rnd_type=='oracle':
                    #buf_rews_int = [
                    #    self.I.oracle_visited_count.update_position(infos[k]['position'])
                    #    for k in range(self.I.nenvs)]

                    buf_rews_int = [
                        self.update_rnd(infos[k]['position'], k)
                        for k in range(self.I.nenvs)]

                    #print(buf_rews_int)

                    buf_rews_int = np.array(buf_rews_int)
                    self.I.buf_rews_int[sli, t-1] = buf_rews_int

    


        self.I.step_count += 1
        if t == self.nsteps - 1 and not self.disable_policy_update:
            #We need to take one extra step so every transition has a reward.
            for l in range(self.I.nlump):
                sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
                memsli = slice(None) if self.I.mem_state is NO_STATES else sli
                nextobs, rews, ec_rews, nextnews, infos, ram_states, monitor_rews = self.env_get(l)
                dict_nextobs = self.stochpol.ensure_observation_is_dict(nextobs)
                for k in self.stochpol.ph_ob_keys:
                    self.I.buf_ob_last[k][sli] = dict_nextobs[k]
                self.I.buf_new_last[sli] = nextnews
                with logger.ProfileKV("policy_inference"):
                    _, self.I.buf_vpred_int_last[sli], self.I.buf_vpred_ext_last[sli], _, _, _ = self.stochpol.call(dict_nextobs, nextnews, self.I.mem_state[memsli], update_obs_stats=False)
                

                rews = [self.filter_rew(rews[k], infos[k]['unclip_rew'], infos[k]['position'], infos[k]['open_door_type'],k)
                                for k in range(self.I.nenvs)]
    
                rews = np.asarray(rews)

                self.I.buf_rews_ext[sli, t] = rews
                self.I.buf_rews_ec[sli, t] = ec_rews

                if self.rnd_type=='oracle':
                    #buf_rews_int = [
                    #    self.I.oracle_visited_count.update_position(infos[k]['position'])
                    #    for k in range(self.I.nenvs)]
                    
                    buf_rews_int = [
                        self.update_rnd(infos[k]['position'], k)
                        for k in range(self.I.nenvs)]

                    buf_rews_int = np.array(buf_rews_int)
                    self.I.buf_rews_int[sli, t] = buf_rews_int

            if self.rnd_type =='rnd':
                #compute RND
                fd = {}
                fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None], self.I.buf_ob_last[None][:,None]], 1)
                fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                               self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                fd[self.stochpol.ph_ac] = self.I.buf_acs
                self.I.buf_rews_int[:] = tf_util.get_session().run(self.stochpol.int_rew, fd) * self.I.buf_rews_ec
            elif self.rnd_type =='oracle':
            #compute oracle count-based reward
                fd = {}
            else:
                raise ValueError('Unknown exploration reward: {}'.format(
                  self._exploration_reward))
            #Calcuate the intrinsic rewards for the rollout (for each step).
            '''
            envsperbatch = self.I.nenvs // self.nminibatches

            #fd = {}

            
            #[nenvs, nstep+1, h,w,stack]
            #fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None], self.I.buf_ob_last[None][:,None]], 1)
            start = 0
            while start < self.I.nenvs:
                end = start + envsperbatch
                mbenvinds = slice(start, end, None)
    
                fd = {}
        
                fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None][mbenvinds],  self.I.buf_ob_last[None][mbenvinds, None]], 1)
    
    
                fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                           self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                fd[self.stochpol.ph_ac] = self.I.buf_acs[mbenvinds]


    


                # if dead,  we set rew_int to zero
                #self.I.buf_rews_int[mbenvinds] = (1 -self.I.buf_news[mbenvinds]) * self.sess.run(self.stochpol.int_rew, fd)
                
                rews_int = tf_util.get_session().run(self.stochpol.int_rew, fd)
                self.I.buf_rews_int[mbenvinds] = rews_int * self.I.buf_rews_ec[mbenvinds]

                start +=envsperbatch

            '''
            if not self.update_ob_stats_every_step:
                #Update observation normalization parameters after the rollout is completed.
                obs_ = self.I.buf_obs[None].astype(np.float32)
                self.stochpol.ob_rms.update(obs_.reshape((-1, *obs_.shape[2:]))[:,:,:,-1:])
                feed = {self.stochpol.ph_mean: self.stochpol.ob_rms.mean, self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5\
                        , self.stochpol.ph_count: self.stochpol.ob_rms.count}
                self.sess.run(self.assign_op, feed)

            if not self.testing:


                logger.info(self.I.cur_gen_idx,self.I.rews_found_by_contemporary)
                update_info = self.update()


                self.I.oracle_visited_count.sync()

                self.I.cur_oracle_visited_count.sync()
                self.I.cur_oracle_visited_count_for_next_gen.sync()

            else:
                update_info = {}
            self.I.seg_init_mem_state = copy(self.I.mem_state)
            global_i_stats = dict_gather(self.comm_log, self.I.stats, op='sum')
            global_deque_mean = dict_gather(self.comm_log, { n : np.mean(dvs) for n,dvs in self.I.statlists.items() }, op='mean')
            update_info.update(global_i_stats)
            update_info.update(global_deque_mean)
            self.global_tcount = global_i_stats['tcount']
            for infos_ in self.I.buf_epinfos:
                infos_.clear()
        else:
            update_info = {}

        #Some reporting logic.
        for epinfo in epinfos:
            if self.testing:
                self.I.statlists['eprew_test'].append(epinfo['r'])
                self.I.statlists['eplen_test'].append(epinfo['l'])
            else:
                if "visited_rooms" in epinfo:
                    self.local_rooms += list(epinfo["visited_rooms"])
                    self.local_rooms = sorted(list(set(self.local_rooms)))
                    score_multiple = self.I.venvs[0].score_multiple
                    if score_multiple is None:
                        score_multiple = 1000
                    rounded_score = int(epinfo["r"] / score_multiple) * score_multiple
                    self.scores.append(rounded_score)
                    self.scores = sorted(list(set(self.scores)))
                    self.I.statlists['eprooms'].append(len(epinfo["visited_rooms"]))

                self.I.statlists['eprew'].append(epinfo['r'])
                if self.local_best_ret is None:
                    self.local_best_ret = epinfo["r"]
                elif epinfo["r"] > self.local_best_ret:
                    self.local_best_ret = epinfo["r"]

                self.I.statlists['eplen'].append(epinfo['l'])
                self.I.stats['epcount'] += 1
                self.I.stats['tcount'] += epinfo['l']
                self.I.stats['rewtotal'] += epinfo['r']
                # self.I.stats["best_ext_ret"] = self.best_ret


        return {'update' : update_info}



    def check_goto_next_gen(self, last_policy_find_new_rew = True):

        #
        if self.I.cur_policy_idx >= self.I.max_npolicies or last_policy_find_new_rew==False:

            self.I.cur_gen_idx = self.I.cur_gen_idx + 1

            max_performance = 0
            #select best policy for next generation
            for i in range(len(self.I.policy_dict)):
                performance, oracle_visited_count, path = self.I.policy_dict[i]
                if performance > max_performance:
                    max_performance = performance
                    self.I.last_gen_policy = path
                    self.I.last_gen_rnd = oracle_visited_count

            self.I.reset_for_new_generation()

        return
    def check_goto_next_policy(self):


        self.I.cur_policy_nsteps = self.I.cur_policy_nsteps + 1

        if self.I.cur_found_extra_rews_count >= self.I.max_extra_rews_count \
                or (self.I.cur_policy_idx > 0 and self.I.cur_policy_nsteps > self.I.max_nsteps_training_single_policy):

            self.I.cur_policy_idx = self.I.cur_policy_idx + 1

            
            cur_unique_rew  = None
            cur_unique_rew_count = 0
            for unique_rew in self.I.cur_ith_rews:
                count = self.I.cur_ith_rews[unique_rew]

                if count > cur_unique_rew_count:
                    cur_unique_rew = unique_rew
                    count = cur_unique_rew_count


            # add the rew found by the current policy to buf.
            # the rew in the buf will not be collected by other policy in the current generation
            if cur_unique_rew is not None:
                self.I.rews_found_by_contemporary.add(cur_unique_rew)

            self.record_policy_info()

            last_policy_find_new_rew = len(self.I.rews_found_by_cur_policy) >= (self.I.cur_gen_idx + self.I.plan_step)
            self.check_goto_next_gen(last_policy_find_new_rew)


            self.init_agent()
            self.init_rnd()
            self.I.reset_for_new_policy()

        return

    def filter_rew(self,rew, unclip_reward, pos, open_door_type, env_id):

        unique_rew = uniqueReward(unclip_reward, pos, open_door_type)
        if unique_rew in self.I.rews_found_by_contemporary or rew==0.:
            return 0.
        else:
            self.I.rews_found_by_cur_policy_in_one_episode[env_id].append(unique_rew)
        return rew

    def update_rnd(self, pos, env_id):

        rnd_rew = self.I.cur_oracle_visited_count.update_position(pos)

        epsiode_record = self.I.rews_found_by_cur_policy_in_one_episode[env_id]

        if len(epsiode_record) < self.I.cur_gen_idx:
            self.I.cur_oracle_visited_count_for_next_gen.update_position(pos)

        return rnd_rew


    # we check an episode when it ends
    def check_episode(self, env_id):


        epsiode_record = self.I.rews_found_by_cur_policy_in_one_episode[env_id]


        #clac total scores
        scores = 0.
        for unique_rew in epsiode_record:
            (reward_type, room_id, unclip_reward) = unique_rew
            scores = scores + unclip_reward

        self.I.cur_policy_scores.add(scores)


        #check if collect enough rews
        if len(epsiode_record) >= (self.I.cur_gen_idx + self.I.plan_step):
            self.I.cur_found_extra_rews_count = self.I.cur_found_extra_rews_count + 1

            ith_rew = epsiode_record[self.I.cur_gen_idx]

            count = self.I.cur_ith_rews.get(ith_rew, 0)
            self.I.cur_ith_rews[ith_rew] = count + 1

        self.I.rews_found_by_cur_policy_in_one_episode[env_id] = []


    def init_agent(self):

        if self.I.last_gen_policy ==None:
            if self.I.cur_gen_idx == 0:
                if self.I.cur_policy_idx > 0 :
                    #load random weight
                    self.load(self.random_weight_path)
                else:
                    pass
            else:
                raise ValueError('last_gen_policy is none in generation: {}'.format(
                  str(self.I.cur_gen_idx)))
        else:
            self.load(self.I.last_gen_policy)
        return

    def init_rnd(self):
        if self.I.last_gen_rnd ==None:
            if self.I.cur_gen_idx == 0:
                if self.I.cur_policy_idx > 0 :
                    self.I.cur_oracle_visited_count = oracle.OracleExplorationRewardForAllEpisodes()
                else:
                    pass
            else:
                raise ValueError('last_gen_policy is none in generation: {}'.format(
                  str(self.I.cur_gen_idx)))
        else:
            self.I.cur_oracle_visited_count = oracle.OracleExplorationRewardForAllEpisodes()
            self.I.cur_oracle_visited_count.copy(self.I.last_gen_rnd)
        return

    def record_policy_info(self):

        #total scores over 10 episodes
        performance = self.I.cur_policy_scores.mean()
        oracle_visited_count = self.I.cur_oracle_visited_count_for_next_gen

        path = '{}_gen{}_policy{}'.format(self.save_path,str(self.I.cur_gen_idx),str(self.I.cur_policy_idx))
        logger.log("save model:",path)
        self.save(path)

        policy_record = (performance, oracle_visited_count, path)
        self.I.policy_dict.append(policy_record)

        return

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


'''
policy mining

max total time steps T

max number of policy  NP, max time step of mine policy TP,

cur target rew i

i = 0
t = 0
cur_max_rew = 0
target_max_rew = TARGET

policy_{-1} = RandomInitialized()
rnd_{-1} = RandomInitialized()

RSet_FoundByAncestor = {}

while cur_max_rew > TARGET or t > T:
    Start policy mining for rew i:
        j = 0
        policy_i^j = policy_{i-1}
        rnd_i^j = rnd_{i-1}
        miningRewSet RSet_new = {}
        RSet_FoundByContemporary={}
        
        polocy_dict={}
        

        while j < NP:

            tp =  0
    
            while  LEN(RSet_new) > 3
    
                if tp < TP and j >0:
                    break;


                a ~ policy_i^{j}
                obs,r = env(a)
                
                tp = tp +1
                t = t + 1
                if r in RSet_FoundByContemporary
                    r = 0 
    
                if r >0 and r not in RSet_FoundByAncestor
                    add r to RSet_new
    
                update policy_i^{j} and rnd
    

                update rnd_i^j using the observation s which before get rew_i in a episode 
    
            j = j +1

            rews  <-  eval(policy_i^{j})

            if len(rews) > i:

                sum_r_j = sum(rews)

                add {sum_r_j, policy_i^{j} rnd_i^j } to polocy_dict

                add the ith rew r_{i} in rews to RSet_FoundByContemporary

            else:  
              

              print(can't find new policy)

              break


        policy_i = policy_i^{k} which has the highest rews in polocy_dict
        rnd_i = rnd_i^j
        
        cur_max_rew = sum_r_j
        
        add rew_i to RSet_FoundByAncestor


    
'''


'''policy mining version_v2

max_rew_mining_step, max_policy_mining_step

input: mining_check_point_list =[(reward_list: list[policy_model, rnd_model], ignored_reward_idx, , all_check_point_list= { (policy, rnd_model, reward_list) }

output: new_polices and its check_point

rew_step_count = 0
policy_step_count =0



ignored_reward_list =  reward_list[ignored_reward_idx:]

rnd_model = rnd_model_set[ignored_reward_idx]

pi = policy_set[ignored_reward_idx]


while rew_step_count < max_mining_step  or policy_step_count < max_policy_mining_step:
    rew_step_count++
    policy_step_count ++


    a ~ pi
    r ~ env

    r ~ rew_filter(r)


    is_save = is_check_point()


    if flag:
        save pi&rnd_model & ignored_reward_idx+1
        rew_step_count = 0

    updae pi & rnd_model


is_check_point(r):

   check the last 100 episodes rew_list

   for rew_list in each episode:
      rew_list[ignored_reward_idx] is equal
      and  rew_list[ignored_reward_idx] not in reward_list


rew_filter(r):

    if r in ignored_reward_list
        return 0