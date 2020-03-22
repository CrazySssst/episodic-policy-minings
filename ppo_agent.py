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
from utils import explained_variance, Scheduler, LinearSchedule
from console_util import fmt_row
from mpi_util import MpiAdamOptimizer, RunningMeanStd, sync_from_root
import csv
from episodic_curiosity import oracle
import matplotlib.pyplot as plt

from replay_buffer import PrioritizedReplayBuffer, CloneReplayBuffer, DDReplayBuffer, StateOnlyReplayBuffer, SAReplayBuffer, PrioritizedStateOnlyReplayBuffer, PrioritizedDDReplayBuffer,PrioritizedSAReplayBuffer


import pickle

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


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def clac_sample_prob(num_agents):

    prob_list = []
    all_prob = 0
    for i in range(num_agents):

        if i == (num_agents -1):
            p = 1 - all_prob
        else:
            p = 1. / num_agents

        all_prob = all_prob + p
        prob_list.append(p)
    logger.info("sample prob:", prob_list)
    return prob_list

def clac_base_int_weight(num_agents,min_w=0.25,max_w=1.):

    weight_list = []

    delta = (max_w - min_w) / max(num_agents - 1,1)

    w = min_w
    #w = 0.25
    weight_list.append(w) 

    for i in range(num_agents-1):

        w = w + delta
        #w = 0.25
        weight_list.append(w)
    weight_list = [1,1,1,1,1,1]
    logger.info("weight list:", weight_list)
    return weight_list

class InteractionState(object):
    """
    Parts of the PPOAgent's state that are based on interaction with a single batch of envs
    """
    def __init__(self, ob_space, ac_space, nsteps, gamma, gamma_div, venvs, stochpol, comm, rnd_type='rnd', num_agents = 1, load_ram=False, debug=False):
        self.lump_stride = venvs[0].num_envs
        self.venvs = venvs
        assert all(venv.num_envs == self.lump_stride for venv in self.venvs[1:]), 'All venvs should have the same num_envs'
        self.nlump = len(venvs)
        nenvs = self.nenvs = self.nlump * self.lump_stride
        self.reset_counter = 0
        self.env_results = [None] * self.nlump
        self.buf_vpreds_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_vpreds_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_vpreds_div = np.zeros((nenvs, nsteps), np.float32)
        self.buf_nlps = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_advs_div = np.zeros((nenvs, nsteps), np.float32)

        self.buf_rews_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_rews_div = np.zeros((nenvs, nsteps), np.float32)

        self.buf_int_weight = np.zeros((nenvs, nsteps), np.float32)
        self.buf_use_div = np.zeros((nenvs, nsteps), np.float32)

        #for sil
        self.buf_target_vpreds_ext = np.zeros((nenvs, nsteps), np.float32)
        self.buf_target_vpreds_int = np.zeros((nenvs, nsteps), np.float32)
        self.buf_target_acs = np.zeros((nenvs, nsteps, *ac_space.shape), ac_space.dtype)

        self.divexp_flag = np.zeros((nenvs),np.bool)
        self.buf_ppo_mask = np.zeros((nenvs, nsteps), np.float32)
        self.buf_sil_mask = np.zeros((nenvs, nsteps), np.float32)
        self.buf_vfdiv_mask = np.zeros((nenvs, nsteps), np.float32)
        self.buf_div_train_mask = np.zeros((nenvs, nsteps), np.float32)
        self.buf_div_rew_mask = np.zeros((nenvs, nsteps), np.int32)


        self.buf_room_infos = np.zeros((nenvs, nsteps), np.int32)
        self.buf_x_infos = np.zeros((nenvs, nsteps), np.int32)
        self.buf_y_infos = np.zeros((nenvs, nsteps), np.int32)
        self.buf_level_infos = np.zeros((nenvs, nsteps), np.int32)


        self.buf_rews_ec = np.zeros((nenvs, nsteps), np.float32)

        self.buf_acs = np.zeros((nenvs, nsteps, *ac_space.shape), ac_space.dtype)
        self.buf_step_acs = self.buf_acs.copy()

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

        '''
        self.rff_int = RewardForwardFilter(gamma)
        self.rff_rms_int = RunningMeanStd(comm=comm, use_mpi=True)
        '''
        self.rff_int_list = [RewardForwardFilter(gamma) for _ in range(num_agents)]
        self.rff_rms_int_list = [RunningMeanStd(comm=comm, use_mpi=False) for _ in range(num_agents)]

        self.rff_div_list = [RewardForwardFilter(gamma_div) for _ in range(num_agents)]
        self.rff_rms_div_list = [RunningMeanStd(comm=comm, use_mpi=False) for _ in range(num_agents)]

        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_int_last = self.buf_vpreds_int[:, 0, ...].copy()
        self.buf_vpred_ext_last = self.buf_vpreds_ext[:, 0, ...].copy()
        self.buf_vpred_div_last = self.buf_vpreds_div[:, 0, ...].copy()


        self.step_count = 0 # counts number of timesteps that you've interacted with this set of environments
        self.t_last_update = time.time()

        assert num_agents > 0

        self.statlists = defaultdict(lambda : [deque([], maxlen=100) for _ in range(num_agents)]) # Count other stats, e.g. optimizer outputs
        self.stats = defaultdict(float) # Count episodes and timesteps
        self.stats['epcount'] = 0
        self.stats['n_updates'] = 0
        self.stats['tcount'] = 0
        self.stats['nbatch'] = 0

        self.buf_scores = np.zeros((nenvs), np.float32)
        self.buf_nsteps = np.zeros((nenvs), np.float32)
        self.buf_reset = np.zeros((nenvs), np.float32)

        self.buf_ep_raminfos = [{} for _ in range(self.nenvs)]


        self.buf_rnd_weights = np.zeros((nenvs, nsteps), np.float32)
        self.buf_value_weights = np.zeros((nenvs, nsteps), np.float32)

        self.buf_rnd_mask = np.zeros((nenvs, nsteps, num_agents), np.float32)

        

        self.ep_scores = np.zeros((nenvs), np.float32)
        self.buf_ep_scores = np.zeros((nenvs, nsteps), np.float32)
        self.buf_last_rew_ob = self.buf_obs[None].copy()
        self.ep_last_rew_ob = np.full_like(self.buf_ob_last[None], 128)
        self.init_last_rew_ob = np.full_like(self.ep_last_rew_ob[0], 128)

        self.novel_value = np.zeros((num_agents))


        self.num_agents = num_agents
        self.div_scores = [deque([], maxlen=10) for _ in range(num_agents)]
        self.agent_ep_div_score = np.zeros(nenvs,np.float32)


        self.buf_unique_rews = [ [None]*nsteps for _ in range (nenvs)  ]
        self.ignored_reward_set_list = [{} for _ in range(num_agents)]
        self.small_reward_set_list = [{} for _ in range(num_agents)]
        self.ignored_reward_set_divscore_list = [{} for _ in range(num_agents)]
        self.reward_set_divscore_list = [{} for _ in range(num_agents)]
        self.reward_set_base_divscore_list = [{} for _ in range(num_agents)]

        self.p = clac_sample_prob(num_agents)

        self.cur_agent_idx = 0
        self.cur_agent_idx_count = 0

        if debug:
            self.step_thresold = 0
        else:
            self.step_thresold = 0

        self.agent_idx_count={}

        self.oracle_visited_count_list = [oracle.OracleExplorationRewardForAllEpisodes() for _ in range(num_agents)] 
        self.oracle_reward_count_list = [oracle.OracleRewardForAllEpisodes() for _ in range(num_agents)]

        self.buf_agent_idx = np.zeros((nenvs, nsteps), np.int32)
        self.agent_idx = np.zeros((nenvs), np.int32)
        self.buf_last_agent_idx = np.zeros((nenvs), np.int32)
        self.buf_first_agent_idx = np.zeros((nenvs), np.int32)


        self.buf_agent_change = np.zeros((nenvs, nsteps), np.int32)
        self.buf_last_agent_change = np.zeros((nenvs), np.int32)

        self.buf_agent_ob = [[] for _ in range(num_agents)]
        self.buf_ph_mean = np.zeros(([nenvs, nsteps] + list(stochpol.ob_space.shape[:2])+[1]), np.float32)
        self.buf_ph_std =  np.zeros(([nenvs, nsteps] + list(stochpol.ob_space.shape[:2])+[1]), np.float32)

        self.sample_agent_prob = np.zeros((nenvs), np.float32)
        self.buf_sample_agent_prob = np.zeros((nenvs,nsteps), np.float32)
        #for policy mining
        self.gen_idx = 0
        self.policy_idx = 0
        self.reward_set_list = [{} for _ in range(num_agents)]
        self.ingore_reward_set_list = [{} for _ in range(num_agents)]
        self.rews_found_by_cur_policy_in_one_episode = [[] for _ in range(self.nenvs)]

        self.init_ram_state = None
        self.init_monitor_rews = []

        self.int_exclude_rooms = [] #[(5,0)]
        self.div_exclude_rooms = []

        self.exclude_rooms = []
        self._exclude_rews  = [] #[(3,2,2000.0,0,0),(3,2,2000.0,1,0),(0,5,300.0,2,0),(9,5,300.0,2,0),(9,5,300.0,0,0),(0,5,300.0,0,0),(6,5,3000.0,0,0),(6,5,3000.0,2,0)]

        self.agents_local_rooms = [ [] for _ in range(num_agents) ]


        self.min_vpred_ext_thresold = 0.1

        self.group_coeff = [[1.,2.,0.],[1.,2.,0.],[1.,2.,1.],[1.,2.,1.]]

        self.int_coef = np.zeros((nenvs, nsteps), np.float32)
        self.ext_coef = np.zeros((nenvs, nsteps), np.float32)


        self.buf_base_vpred_ext = np.zeros((nenvs, nsteps), np.float32)
        
        self.nupdates_divdisc = 200 

        self.nupdates_divend = 2000

        self.divrew_weight = LinearSchedule(schedule_timesteps=int(self.nupdates_divend),
                             initial_p=2.0,
                             final_p=0.)

        self.sil_loss_weight = LinearSchedule(schedule_timesteps=int(200), initial_p=2.0, final_p=0.)

        self.pos_buffer = CloneReplayBuffer( 4500 * 20)
        self.neg_buffer = CloneReplayBuffer( 4500 * 20)

        self.div_discr_background_buffer_list = [ DDReplayBuffer(1) for _ in range(num_agents) ]
        self.div_discr_neg_buffer_list = [ PrioritizedDDReplayBuffer(1, 0.7) for _ in range(num_agents) ]
        self.div_discr_pos_buffer_list = [ PrioritizedDDReplayBuffer(1, 0.7) for _ in range(num_agents) ]

        self.idle_agent_discr_neg_buffer_list = [ StateOnlyReplayBuffer(1) for _ in range(num_agents) ]
        self.idle_agent_discr_pos_buffer_list = [ PrioritizedStateOnlyReplayBuffer(1, 0.7) for _ in range(num_agents) ]

        self.idle_agent_flag = np.zeros((nenvs), np.int32)

        self.novel_rew_pos_buffer = PrioritizedStateOnlyReplayBuffer(1, 0.7)

        self.oralce_rew_div_rewset = [ {} for _ in range(num_agents) ]
        self.oracle_rew_div_idleset = [ {} for _ in range(num_agents) ]

        self.unlock_agent_index = -1
        self.replay_buffer_count_list = [ {} for _ in range(num_agents) ]

        self.div_rooms = [0,3,4,5,8,9,10,11,15,16,17,18,19,20]


        self.minimum_div_score = 0.01
        self.maximum_div_score = 0.8


        self.img_count = 0



        self.trajector_list = [ [] for _ in range(nenvs) ]
        self.sil_buffer_list = [ PrioritizedSAReplayBuffer(50000,0.6) for _ in range(num_agents) ]
        self.num_trajectors= np.zeros((num_agents), np.int32)

        self.buffer_rand_em = np.zeros((nenvs, nsteps, 64), np.float32)
        self.buffer_ent_coef = np.zeros((nenvs, nsteps), np.float32)


        self.cluster_list = [ Clusters(1) for _ in range(num_agents) ]

        self.neg_rew_flag = np.zeros((nenvs),np.bool)

        self.ep_step_count = np.zeros((nenvs), np.int32) 

        self.ep_explore_step = np.zeros((nenvs), np.int32)
        self.ep_explore_step[:] = 4500

        self.int_exclude_rooms = []
        self.div_exclude_rooms =  [(7,0),(12,0),(13,0),(14,0),(21,0),(22,0),(23,0)]
        self.divrew_exclude_rooms = [1,6,7,13,14,23]
        #self.div_exclude_rooms = [(1,0),(2,0),(5,0),(6,0)]

        self.buf_div_weights = np.zeros((nenvs, nsteps), np.float32)

        self.buf_stage_label = np.zeros((nenvs, nsteps), np.float32)

        self.agent_detial_room_info = [ {} for _ in range(num_agents) ]

        self.div_room_set = [set() for _ in range(num_agents)]

        self.socre_baseline =  8.5 #7.5 #0 #7.5i

        self.base_int_weight = clac_base_int_weight(num_agents)
        self.ext_weight = [2,2,2,2,2,2]

        self.sd_rms = RunningMeanStd(shape=list(stochpol.ob_space.shape[:2])+[1], use_mpi= False)

        if load_ram:


            #ram_path='./ram_state_400_6room'
            #path='./ram_state_400_monitor_rews_6room'

            #ram_path='./ram_state_6700_13room'
            #path='./ram_state_6700_monitor_rews_13room'

            #ram_path='./ram_state_6700_21room'
            #path='./ram_state_6700_monitor_rews_21room'

            #ram_path = './ram_state_500_7room'
            #path = './ram_state_500_monitor_rews_7room'

            #ram_path='./ram_state_6700'
            #path='./ram_state_6700_monitor_rews'

            #ram_path='./ram_state_6600_13room'
            #path='./ram_state_6600_monitor_rews_13room'

            #ram_path='./ram_state_6600_21room'
            #path='./ram_state_6600_monitor_rews_21room'

            #ram_path='./ram_state_6600_23room'
            #path='./ram_state_6600_monitor_rews_23room'

            #ram_path='./ram_state_6700_7room'
            #path='./ram_state_6700_monitor_rews_7room'


            ram_path='./ram_state_7700_10room'
            path='./ram_state_7700_monitor_rews_10room'

            f = open(ram_path,'rb')
            self.init_ram_state = pickle.load(f)
            f.close()

            if rnd_type=='oracle':
                assert ram_path == './ram_state_6700'


            
            f = open(path,'rb')
            self.init_monitor_rews = pickle.load(f)
            f.close()


            self.int_exclude_rooms = [] #[(1,0),(2,0),(6,0),(14,0),(21,0),(22,0),(23,0)]
            self.div_exclude_rooms = [(10,0)]  #[(10,0),(11,0),(12,0),(13,0)] #[(1,0),(2,0),(6,0),(7,0),(12,0),(13,0),(14,0),(21,0),(22,0),(23,0)]

            '''
            self.int_exclude_rooms = [(1,0),(2,0)]

            
            if '13room' in ram_path:
                self.int_exclude_rooms = [(1,0),(2,0),(6,0),(14,0),(21,0),(22,0),(23,0)]

            if '21room' in ram_path:
                self.int_exclude_rooms = [(1,0),(2,0),(6,0),(14,0),(22,0),(23,0)]
            '''
            #self.div_exclude_rooms = [1,2,6,7,12,13,14,21,22,23]
            #if '6700' in path:
            #    self.int_exclude_rooms = [(1,0),(2,0),(6,0),(7,0),(12,0),(13,0),(14,0),(21,0),(22,0),(23,0)]
            #elif '7room' in path:
            #    self.int_exclude_rooms = [(1,0),(2,0),(5,0),(6,0)]
            #else:
            #    self.exclude_rooms = [1,2,6,7,12,13,14,21,22,23]
            

            self.exclude_rooms = [12]

            for i in range(num_agents):
                self.oracle_visited_count_list[i]._exclude_room = self.exclude_rooms

            #self._exclude_rews=[(3,2,2000.0,0,0),(3,2,2000.0,1,0)]


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
            res = np.mean(li, axis=0)
            if res.shape ==():
                result[k] = res
            else:
                result[k] = SemicolonList(res)
        elif op=='sum':
            result[k] = np.sum(li, axis=0)
        elif op=="max":
            result[k] = np.max(li, axis=0)
        else:
            assert 0, op
    return result


def preprocess_statlists(data):
    mean = np.mean(data)

    if np.isnan(mean):
        mean = 0.
    else:
        mean = np.round(mean,1)
    return mean


def uniqueReward(unclip_reward, pos, open_door_type):
    x, y, room_id, nkeys, level = pos
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

    return (reward_type, room_id, unclip_reward, nkeys, level)


def save_rews_list(rews_list, path):
    f = open(path,'wb')
    pickle.dump(rews_list,f)
    f.close()
def load_rews_list(path):
    f = open(path,'rb')
    rews_list = pickle.load(f)
    f.close()
    return rews_list

class PpoAgent(object):
    envs = None
    def __init__(self, *, scope,
                 ob_space, ac_space,
                 stochpol_fn,
                 nsteps, nepochs=4, nminibatches=1,
                 gamma=0.99,
                 gamma_ext=0.99,
                 gamma_div=0.99,
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
                 reset=False, 
                 reset_prob=0.2,
                 dynamics_sample=False, 
                 save_path='', 
                 num_agents = 1,
                 div_type = 'oracle',
                 load_ram=False,
                 debug = False,
                 rnd_mask_prob = 1.,
                 rnd_mask_type = 'prog',
                 sd_type = 'oracle',
                 from_scratch = False,
                 use_kl = False,
                 indep_rnd= False
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

        self.num_agents = num_agents
        self.div_type = div_type
        self.load_ram = load_ram
        self.debug = debug

        self.rnd_mask_prob = rnd_mask_prob
        self.rnd_mask_type = rnd_mask_type

        self.indep_rnd = indep_rnd

        self.sd_type = sd_type

        self.from_scratch = from_scratch

        self.sd_interval =  64


        self.eprew_f = [ open('{}_{}_agent{}'.format(save_path,str('eprew_f'), str(i)), "wt") for i in range(num_agents) ] 

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
            self.best_ret = [ -np.inf for _ in range(num_agents) ] 
            self.local_best_ret = [ -np.inf for _ in range(num_agents) ] 
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
            self.gamma_div = gamma_div
            self.lam = lam
            self.adam_hps = adam_hps or dict()
            self.ph_adv = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_int = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_ext = tf.placeholder(tf.float32, [None, None])
            self.ph_ret_div = tf.placeholder(tf.float32, [None, None])

            self.ph_oldnlp = tf.placeholder(tf.float32, [None, None])
            self.ph_oldvpred = tf.placeholder(tf.float32, [None, None])
            self.ph_lr = tf.placeholder(tf.float32, [])
            self.ph_lr_pred = tf.placeholder(tf.float32, [])
            self.ph_cliprange = tf.placeholder(tf.float32, [])

            '''
            self.target_vpred_int = tf.placeholder(tf.float32, [None, None], name='target_vpred_int')
            self.target_vpred_ext = tf.placeholder(tf.float32, [None, None], name='target_vpred_ext')
            self.target_ac = self.stochpol.pdtype.sample_placeholder([None, None], name='target_ac')
            self.sil_mask = tf.placeholder(tf.float32, [None, None], name='sil_mask')
            '''

            self.target_ac = self.stochpol.pdtype.sample_placeholder([None, None], name='target_ac')
            self.traget_vpred_ext = tf.placeholder(tf.float32, [None, None], name='traget_vpred_ext')

            self.sep_ent_coef = tf.placeholder(tf.float32, [None, None], name='ent_coef')

            self.div_train_mask = tf.placeholder(tf.float32, [None, None], name='div_train_mask')
            #Define ppo loss.
            

            neglogpac = self.stochpol.pd_opt.neglogp(self.stochpol.ph_ac)


            entropy = tf.reduce_mean(self.stochpol.pd_opt.entropy())
            
            vf_loss_int = (0.5 * vf_coef) * tf.reduce_mean(tf.square(self.stochpol.vpred_int_opt - self.ph_ret_int)) 
            vf_loss_ext = (0.5 * vf_coef) * tf.reduce_mean( tf.square(self.stochpol.vpred_ext_opt - self.ph_ret_ext))

            
            #agent 0 should ignore div brach
            #vf_div_mask = tf.cast(self.stochpol.ph_agent_idx > 0, tf.float32) #* self.vfdiv_mask
            #vf_loss_div = (0.5 * vf_coef) * tf.reduce_sum(vf_div_mask * tf.square(self.stochpol.vpred_div_opt - self.ph_ret_div)) / tf.maximum(tf.reduce_sum(vf_div_mask), 1.)

            vf_loss = vf_loss_int + vf_loss_ext #+ vf_loss_div
            ratio = tf.exp(self.ph_oldnlp - neglogpac) # p_new / p_old
            negadv = - self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss = tf.reduce_mean( tf.maximum(pg_losses1, pg_losses2))
            ent_loss =  (- ent_coef) * entropy
            #ent_loss = -1 *  tf.reduce_mean(self.sep_ent_coef * self.stochpol.pd_opt.entropy())
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            maxkl    = .5 * tf.reduce_max(tf.square(neglogpac - self.ph_oldnlp))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.ph_cliprange)))


            kl_loss = kl =tf.constant(0.)
            if use_kl:
                kl = tf.reduce_sum(self.stochpol.kl_mask * self.stochpol.pd_opt.kl_v2(self.stochpol.other_pdparam)) / tf.maximum(tf.reduce_sum(self.stochpol.kl_mask), 1.)
                kl_loss = -0.01 * kl

            loss = pg_loss + ent_loss + vf_loss + kl_loss

            aux_loss = tf.constant(0.)
            feat_var = tf.constant(0.)
            max_feat = tf.constant(0.)


            self.disc_loss = tf.constant(0.)
            self.idle_loss = tf.constant(0.)
            self.div_loss = tf.constant(0.)
            self.ep_loss = tf.constant(0.)
            self.stage_loss = tf.constant(0.)
            self.disc_ent = tf.constant(0.)
            self.rew_disc_loss = tf.constant(0.)
            #define RND loss
            if self.rnd_type =='rnd':

                aux_loss = self.stochpol.aux_loss #+ self.stochpol.new_aux_loss

                feat_var = self.stochpol.feat_var

                max_feat = self.stochpol.max_feat




            if div_type =='cls':
                div_train_mask = tf.reshape(self.div_train_mask,[-1])
                self.disc_loss = tf.reduce_sum(div_train_mask * self.stochpol.disc_nlp) / tf.maximum(tf.reduce_sum(self.div_train_mask), 1.)  #tf.reduce_mean(self.stochpol.disc_loss)
                #self.disc_loss = tf.reduce_mean( self.stochpol.disc_nlp)
                #self.rew_disc_loss = tf.reduce_sum(div_train_mask * self.stochpol.rew_disc_nlp) / tf.maximum(tf.reduce_sum(self.div_train_mask), 1.) 
                #self.rew_disc_loss = tf.reduce_mean(self.stochpol.rew_disc_nlp)

                self.disc_loss = self.disc_loss + self.rew_disc_loss
                
                #self.disc_ent = tf.reduce_sum(div_train_mask * self.stochpol.disc_pd.entropy()) / tf.maximum(tf.reduce_sum(self.div_train_mask), 1.)
                #self.idle_loss = tf.reduce_mean(self.stochpol.idle_agent_disc_loss)
                #self.novel_loss = tf.reduce_mean(self.stochpol.novel_loss)
            elif div_type =='rnd':
                self.div_loss = self.stochpol.div_loss



            loss = loss + aux_loss + self.div_loss  + self.disc_loss # + 0.01 * self.disc_ent


            #Create optimizer for loss.
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


            #Create optimizer for div discr loss
            if div_type =='cls':
                div_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div')
                grads_and_vars = trainer.compute_gradients(self.disc_loss, div_params)
                grads, vars = zip(*grads_and_vars)
                if max_grad_norm:
                    _, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                div_global_grad_norm = tf.global_norm(grads)
                grads_and_vars = list(zip(grads, vars))
                self._div_train = trainer.apply_gradients(grads_and_vars)

                '''
                idle_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/idle')
                grads_and_vars = trainer.compute_gradients(self.idle_loss, idle_params)
                grads, vars = zip(*grads_and_vars)
                if max_grad_norm:
                    _, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                idle_global_grad_norm = tf.global_norm(grads)
                grads_and_vars = list(zip(grads, vars))
                self._idle_train = trainer.apply_gradients(grads_and_vars)
                
                novel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/novel')
                grads_and_vars = trainer.compute_gradients(self.novel_loss, novel_params)
                grads, vars = zip(*grads_and_vars)
                if max_grad_norm:
                    _, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                novel_global_grad_norm = tf.global_norm(grads)
                grads_and_vars = list(zip(grads, vars))
                self._novel_train = trainer.apply_gradients(grads_and_vars)
                '''

            #define self-imitation learning loss


            #sil_adv =   tf.maximum(self.traget_vpred_ext - self.stochpol.vpred_ext_opt,0)
            #sil_pg_weight = tf.stop_gradient(sil_adv)


            sil_pg_weight = tf.stop_gradient(
                          tf.clip_by_value(self.traget_vpred_ext,0.0, 1.0)
                                )
            
            self.sil_pg_weight = sil_pg_weight
            sil_batch_pg_loss = self.stochpol.pd_opt.neglogp(self.target_ac)
            sil_batch_pg_loss = tf.stop_gradient(tf.minimum(sil_batch_pg_loss, 1) - sil_batch_pg_loss)  + sil_batch_pg_loss

            self.sil_batch_pg_loss = sil_pg_weight * sil_batch_pg_loss

            self.sil_pg_loss = tf.reduce_mean( self.sil_batch_pg_loss)


            delta = tf.clip_by_value(self.stochpol.vpred_ext_opt - self.traget_vpred_ext, -1, 0)
            self.sil_vf_loss_ext = tf.constant(0.) #0.5 * tf.reduce_mean( self.stochpol.vpred_ext_opt * tf.stop_gradient(delta) )

            self.sil_entropy =  tf.reduce_mean(self.stochpol.pd_opt.entropy())


            self.sil_loss = self.sil_pg_loss

            sil_grads_and_vars = trainer.compute_gradients(self.sil_loss, params)
            grads, vars = zip(*sil_grads_and_vars)
            grads, _grad_norm = tf.clip_by_global_norm(grads, 1.0)
            sil_global_grad_norm = tf.global_norm(grads)
            grads_and_vars = list(zip(grads, vars))
            self._sil_train = trainer.apply_gradients(grads_and_vars)

            #State Discriminator loss

            if self.sd_type=='sd':
                sd_loss = tf.reduce_mean(self.stochpol.stage_loss)
                extra_loss = tf.constant(0.)
    
                sd_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.abs_scope+'/sd')
                sd_trainer = MpiAdamOptimizer(self.comm_train, learning_rate=self.ph_lr, **self.adam_hps)
                sd_grads_and_vars = sd_trainer.compute_gradients(sd_loss, sd_params)
                sd_grads, sd_vars = zip(*sd_grads_and_vars)
                if max_grad_norm:
                    _, sd__grad_norm = tf.clip_by_global_norm(sd_grads, max_grad_norm)
                sd_global_grad_norm = tf.global_norm(sd_grads)
                sd_grads_and_vars = list(zip(sd_grads, sd_vars))
                self._sd_train = trainer.apply_gradients(sd_grads_and_vars)

                self._sd_losses = [sd_loss]
                self.sd_loss_names = ['sd_loss']

        #assign ph_mean and ph_var
        
        #self.assign_op=[]
        #self.assign_op.append(self.stochpol.var_ph_mean.assign(self.stochpol.ph_mean))
        #self.assign_op.append(self.stochpol.var_ph_std.assign(self.stochpol.ph_std))
        #self.assign_op.append(self.stochpol.var_ph_count.assign(self.stochpol.ph_count))


        '''
        scam_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/CAM/scam')
        fixed_scam_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/fixed_CAM/scam')
        update_fixed_scam_op=[]

        for var, fixed_var in zip(sorted(scam_vars, key=lambda v: v.name),
                                 sorted(fixed_scam_vars, key=lambda v: v.name)):
        
                update_fixed_scam_op.append(fixed_var.assign(var))
        '''
        '''
        #clone div network
        self.clone_divnet_op = []
        fixed_div_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/')
        train_div_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/train_div/')

        for var, fixed_var in zip(sorted(train_div_vars, key=lambda v: v.name),
                                 sorted(fixed_div_vars, key=lambda v: v.name)):
        
                self.clone_divnet_op.append(fixed_var.assign(var))

        self.clone_divnet_op = tf.group(*self.clone_divnet_op)
        '''


        self.clone_base_op = [ [] for _ in range(self.num_agents) ]
        for i in range(self.num_agents):
            src_scope = 'ppo/pol/agent_{}'.format(i)
            src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/base/')


            tmp_op=[]
            for src_var, target_var in zip(sorted(src_vars, key=lambda v: v.name),
                                 sorted(target_vars, key=lambda v: v.name)):
        
                    tmp_op.append(target_var.assign(src_var))

            self.clone_base_op[i] = tf.group(*tmp_op)


        #clone policy network
        self.clone_policy_op=[ [ [] for _ in range(self.num_agents) ] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            src_scope = 'ppo/pol/agent_{}'.format(i)
            for j in range(self.num_agents):
                target_scope = 'ppo/pol/agent_{}'.format(j)
                src = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope)
                target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)

                tmp_op = []
                for src_var, target_var in zip(sorted(src, key=lambda v: v.name),
                                         sorted(target, key=lambda v: v.name)):
                
                        tmp_op.append(target_var.assign(src_var))

                self.clone_policy_op[i][j] = tf.group(*tmp_op)



        #clone baseline policy
        src = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/c1') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/c2') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/c3') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/fc1') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/fc_additional') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/fc2val') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/fc2act') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/pd') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/vf_int') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_0/vf_ext')

        target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/c1') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/c2') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/c3') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/fc1') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/fc_additional') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/fc2val') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/fc2act') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/pd') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/vf_int') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/baseline_agent/vf_ext')

        tmp_op = []
        for src_var, target_var in zip(sorted(src, key=lambda v: v.name),
                                 sorted(target, key=lambda v: v.name)):
        
                tmp_op.append(target_var.assign(src_var))
        self.clone_baseline_agent_op = tf.group(*tmp_op)


        #clone rnd net
        self.clone_rnd_op=[ [ [] for _ in range(self.num_agents) ] for _ in range(self.num_agents)]
        for i in range(self.num_agents):

            src_target_scope = 'ppo/target_net_{}'.format(i)
            src_pred_scope = 'ppo/pred_net_{}'.format(i)

            for j in range(self.num_agents):

                target_target_scope = 'ppo/target_net_{}'.format(j)
                target_pred_scope = 'ppo/pred_net_{}'.format(j)

                src_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_target_scope)
                src_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_pred_scope)

                target_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_target_scope)
                target_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_pred_scope)

                tmp_op = []
                for src_var, target_var in zip(sorted(src_target, key=lambda v: v.name),
                                         sorted(target_target, key=lambda v: v.name)):
                
                        tmp_op.append(target_var.assign(src_var))

                for src_var, target_var in zip(sorted(src_pred, key=lambda v: v.name),
                                         sorted(target_pred, key=lambda v: v.name)):
                
                        tmp_op.append(target_var.assign(src_var))

                self.clone_rnd_op[i][j] = tf.group(*tmp_op)


        # insert noise
        '''
        self.insert_noise_op = [ [] for _ in range(self.num_agents) ]
        for i in range(self.num_agents):

            tmp = []

            vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/pd'.format(i)) + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/vf_int'.format(i)) + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/vf_ext'.format(i))

            for var, in zip(sorted(vars_list, key=lambda v: v.name)):
                logger.info('  {} + noise'.format(var.name))
                tmp.append(tf.assign(var, var + tf.random_normal(tf.shape(var), mean=0., stddev=0.2)))

            self.insert_noise_op[i] = tf.group(*tmp)
        '''

        #Quantities for reporting.
        self._losses = [loss, pg_loss, vf_loss, entropy, clipfrac, approxkl, maxkl, aux_loss,kl,self.stage_loss,
                        feat_var, max_feat, global_grad_norm, self.div_loss, self.disc_loss, self.disc_ent]
        self.loss_names = ['tot', 'pg', 'vf', 'ent', 'clipfrac', 'approxkl', 'maxkl', "auxloss", "kl","stage_loss",
                            "featvar","maxfeat", "gradnorm", "div_rnd_loss","div_cls_loss","disc_ent","div_cls_loss_offline", "div_idle_loss",'novel_loss',
                            'sil_pgloss','sil_vf_loss_ext']


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
        self.load_variables = functools.partial(tf_util.load_variables) 


    def start_interaction(self, venvs, disable_policy_update=False):
        self.I = InteractionState(ob_space=self.ob_space, ac_space=self.ac_space,
            nsteps=self.nsteps, gamma=self.gamma, gamma_div = self.gamma_div,
            venvs=venvs, stochpol=self.stochpol, comm=self.comm_train, rnd_type=self.rnd_type , num_agents = self.num_agents, load_ram= self.load_ram, debug=self.debug)
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
                ob, _, _, news, _,_, _ = self.I.venvs[lump].step_wait()

                if self.I.init_ram_state is not None:

                    for k in range(self.I.lump_stride):
                        if news[k]:
                            self.I.venvs[lump].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
                            #restore ram
                            ob[k] = self.I.venvs[lump].restore_full_state_by_idx(self.I.init_ram_state,k)

                all_ob.append(ob)
                if len(all_ob) % (128 * self.I.nlump) == 0:
                    ob_ = np.asarray(all_ob).astype(np.float32).reshape((-1, *self.ob_space.shape))
                    #self.stochpol.ob_rms.update(ob_[:,:,:,-1:])

                    for i in range(self.I.num_agents):
                        self.stochpol.ob_rms_list[i].update(ob_[:,:,:,-1:])
                    all_ob.clear()



        '''
        feed = {self.stochpol.ph_mean: self.stochpol.ob_rms.mean, self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5 \
                    , self.stochpol.ph_count: self.stochpol.ob_rms.count}

        self.sess.run(self.assign_op, feed)
        '''

    def stop_interaction(self):
        self.I.close()
        self.I = None
        for i in range(self.num_agents):
            self.eprew_f[i].close()

    @logger.profile("update")
    def update(self):

        #Some logic gathering best ret, rooms etc using MPI.
        temp = sum(MPI.COMM_WORLD.allgather(self.local_rooms), [])
        temp = sorted(list(set(temp)))
        self.rooms = temp

        temp = sum(MPI.COMM_WORLD.allgather(self.scores), [])
        temp = sorted(list(set(temp)))
        self.scores = temp

        temp = [ sum(MPI.COMM_WORLD.allgather([self.local_best_ret[k]]), []) for k in range(self.I.num_agents) ]
        self.best_ret = [ max(temp[k]) for k in range(self.I.num_agents) ]

        eprews = [ MPI.COMM_WORLD.allgather(np.mean(self.I.statlists["eprew"][k]))[0] for k in range(self.I.num_agents)]
        local_best_rets = MPI.COMM_WORLD.allgather(self.local_best_ret)
        n_rooms = sum(MPI.COMM_WORLD.allgather([len(self.local_rooms)]), [])


        for i in range(self.num_agents):
            self.eprew_f[i].write(str(eprews[i])+'\n')
            self.eprew_f[i].flush()

        #divrew_weight = 2 #self.I.divrew_weight.value_n()

        if self.I.stats["n_updates"] < self.I.nupdates_divend:
            sil_loss_weight = 0.01
        else:
            sil_loss_weight = self.I.sil_loss_weight.value_n()

        if MPI.COMM_WORLD.Get_rank() == 0 and self.I.stats["n_updates"] % self.log_interval ==0: 
            logger.info("Rooms visited {}".format(self.rooms))
            logger.info("Best return {}".format(self.best_ret))
            logger.info("Best local return {}".format(sorted(local_best_rets)))
            logger.info("eprews {}".format(eprews))
            logger.info("n_rooms {}".format(sorted(n_rooms)))
            logger.info("Extrinsic coefficient {}".format(self.ext_coeff))
            logger.info("Intrinsic coefficient {}".format(self.int_coeff))
            logger.info("Gamma {}".format(self.gamma))
            logger.info("Gamma ext {}".format(self.gamma_ext))
            logger.info("All scores {}".format(sorted(self.scores)))
            logger.info("group_coeff {}".format(self.I.group_coeff))
            logger.info("sil_loss_weight {}".format(sil_loss_weight))
            #logger.info("divrew_weight {}".format(divrew_weight))


        '''
        to do:  
        '''
        #Normalize intrinsic rewards.

        '''
        rffs_int = np.array([self.I.rff_int.update(rew) for rew in self.I.buf_rews_int.T])
        self.I.rff_rms_int.update(rffs_int.ravel())
        rews_int = self.I.buf_rews_int / np.sqrt(self.I.rff_rms_int.var)
        self.mean_int_rew = np.mean(rews_int)
        self.max_int_rew = np.max(rews_int)
        '''

        self.mean_int_rew = []
        self.max_int_rew = []


        rewmean = []
        rewstd = []
        rewmax = []


        #assert self.I.nenvs % self.I.num_agents == 0
        envs_per_agent = self.I.nenvs // self.num_agents
        start = 0

        rews_int = np.zeros_like(self.I.buf_rews_int)

        for i in range(self.num_agents):
            end = start + envs_per_agent


            mbenvinds = slice(start, end, None)



            agent_idx = i
            #[env,nstep] -> [nstep, env]

            #int
            rffs_int = np.array([self.I.rff_int_list[agent_idx].update(rew) for rew in self.I.buf_rews_int[mbenvinds].T])
            self.I.rff_rms_int_list[agent_idx].update(rffs_int.ravel())

            if self.rnd_type =='oracle':
                rews_int[mbenvinds] = self.I.buf_rews_int[mbenvinds]
            else:
                rews_int[mbenvinds] = self.I.buf_rews_int[mbenvinds]  / np.sqrt(self.I.rff_rms_int_list[agent_idx].var)

            self.mean_int_rew.append(np.mean(rews_int[mbenvinds]))
            self.max_int_rew.append(np.max(rews_int[mbenvinds]))


            rewmean.append(self.I.buf_rews_int[mbenvinds].mean())
            rewstd.append(self.I.buf_rews_int[mbenvinds].std())
            rewmax.append(np.max(self.I.buf_rews_int[mbenvinds]))


            start += envs_per_agent


        rews_int =self.I.buf_int_weight *rews_int

        #rews_div = rews_div * self.I.buf_vfdiv_mask


        #Don't normalize extrinsic rewards.
        rews_ext = self.I.buf_rews_ext



        #Calculate intrinsic returns and advantages.
        lastgaelam = 0
        for t in range(self.nsteps-1, -1, -1): # nsteps-2 ... 0
            if self.use_news:
                nextnew = self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last
            else:
                #No dones for intrinsic reward. if change agent, done for intrinsic reward.
                nextnew = 0. #self.I.buf_news[:, t + 1] if t + 1 < self.nsteps else self.I.buf_new_last #self.I.buf_agent_change[:,t+1] if t + 1 < self.nsteps else self.I.buf_last_agent_change

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
        self.I.buf_advs = self.I.int_coef*self.I.buf_advs_int + self.I.ext_coef*self.I.buf_advs_ext
        #Collects info for reporting.
        info = dict(
            advmean = self.I.buf_advs.mean(),
            advstd  = self.I.buf_advs.std(),
            retintmean = rets_int.mean(), # previously retmean
            retintstd  = rets_int.std(), # previously retstd
            retextmean = rets_ext.mean(), # previously not there
            retextstd  = rets_ext.std(), # previously not there


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
                     'step_acs':self.I.buf_step_acs,
                     'rews_int': self.I.buf_rews_int,
                     'rews_int_norm': rews_int,
                     'agent_idx':self.I.buf_agent_idx,
                     'vpred_int': self.I.buf_vpreds_int,
                     'vpred_ext': self.I.buf_vpreds_ext,
                     'adv_int': self.I.buf_advs_int,
                     'adv_ext': self.I.buf_advs_ext,
                     'ent': self.I.buf_ent,
                     'ret_int': rets_int,
                     'ret_ext': rets_ext
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


        #train diversity discriminator
        disc_loss = [0]
        idle_loss = [0]
        novel_loss = [0]

        '''
        if self.div_type=='cls':
            for i in range(4):

                buffer_obs, buffer_last_rew_obs, buffer_game_score, buffer_labels, buffer_agent_idx, batch_idxes_list = self.sample_dd_training_data(12)
                

                fd={
                    self.stochpol.ph_ob[None]: buffer_obs,
                    self.stochpol.last_rew_ob: buffer_last_rew_obs,
                    self.stochpol.game_score: buffer_game_score,
                    self.stochpol.rew_agent_label: buffer_labels,
                    self.stochpol.ph_agent_idx:buffer_agent_idx,
                    self.ph_lr : self.lr
                }



                disc_loss, batch_loss = tf_util.get_session().run([self.disc_loss, self.stochpol.disc_loss, self._div_train],feed_dict=fd)[:-1]
                disc_loss = [disc_loss]

            #for i in range(1):

            #    buffer_last_rew_ob, buffer_labels,  buffer_agent_idx, batch_idxes_list = self.sample_idle_agent_training_data(16,16)
            #    fd={
            #        self.stochpol.last_rew_ob: buffer_last_rew_ob,
            #        self.stochpol.idle_agent_label: buffer_labels,
            #        self.stochpol.ph_agent_idx:buffer_agent_idx,
            #        self.ph_lr : 1e-4
            #    }

            #    idle_loss, batch_loss = tf_util.get_session().run([self.idle_loss, self.stochpol.idle_agent_disc_loss, self._idle_train],feed_dict=fd)[:-1]
            #    idle_loss = [idle_loss]
            
            #    new_priorities = np.abs(batch_loss) + 1e-6

            #    start = 0
            #    for k in range(self.I.num_agents):
            #        end = 16*self.nsteps + start
                    
            #        if len(batch_idxes_list[k]) > 0:
            #            left = end
            #            right = left + 16*self.nsteps
            #            self.I.idle_agent_discr_pos_buffer_list[k].update_priorities(batch_idxes_list[k], new_priorities[left:right])

            #        start = end
        '''
        sil_pgloss = [0]
        sil_vfloss = [0]

        num_samples = 0

        for k in range(self.num_agents):
            num_samples += len(self.I.sil_buffer_list[k])

        epoch = 0
        start = 0
        with logger.ProfileKV("policy_update"):

            #if self.I.stats["n_updates"] % 1 == 0:
            #    tf_util.get_session().run(self.clone_divnet_op) 
            #    logger.info("clone div net")

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
    
                fd.update({self.stochpol.sep_ph_mean: self.I.buf_ph_mean[mbenvinds], self.stochpol.sep_ph_std: self.I.buf_ph_std[mbenvinds]})
                fd.update({self.stochpol.ph_agent_idx: self.I.buf_agent_idx[mbenvinds]})
                fd.update({self.stochpol.rnd_mask: self.I.buf_rnd_mask[mbenvinds]})
    
                #fd.update({self.stochpol.div_train_mask: self.I.buf_div_train_mask[mbenvinds]})
                #fd.update({self.vfdiv_mask: self.I.buf_vfdiv_mask[mbenvinds]})
    
                fd[self.stochpol.game_score] = self.I.buf_ep_scores[mbenvinds]
                fd[self.stochpol.last_rew_ob] = self.I.buf_last_rew_ob[mbenvinds]

                
                fd[self.stochpol.rew_agent_label] = self.I.buf_agent_idx[mbenvinds]

                fd[self.sep_ent_coef] = self.I.buffer_ent_coef[mbenvinds]

                fd[self.div_train_mask] = self.I.buf_div_train_mask[mbenvinds]
                

                fd[self.stochpol.stage_label] = self.I.buf_stage_label[mbenvinds]
                fd[self.stochpol.new_rnd_mask] = self.I.buf_use_div[mbenvinds]
                '''
                fd.update({self.target_vpred_int: self.I.buf_target_vpreds_int[mbenvinds],
                           self.target_vpred_ext: self.I.buf_target_vpreds_ext[mbenvinds],
                           self.target_ac: self.I.buf_target_acs[mbenvinds],
                           self.sil_mask: self.I.buf_sil_mask[mbenvinds],
                           self.ppo_mask: self.I.buf_ppo_mask[mbenvinds],
                           self.sil_loss_weight: sil_loss_weight
                        })
                '''
    
                ret = tf_util.get_session().run(self._losses+[self._train], feed_dict=fd)[:-1]


                if num_samples > 10:
                    
                    buf_obs, buf_acs, buf_vf_exts, buf_agent_idx, batch_idxes_list = self.sample_sil_training_data(8)
                    fd={
                        self.stochpol.ph_ob[None]: buf_obs,
                        self.target_ac: buf_acs,
                        self.traget_vpred_ext: buf_vf_exts,
                        self.stochpol.ph_agent_idx:buf_agent_idx,
                        self.ph_lr : self.lr
                    }
                    _, sil_pgloss, sil_vfloss, prior  = tf_util.get_session().run([self.sil_loss, self.sil_pg_loss, self.sil_vf_loss_ext, self.sil_batch_pg_loss ,self._sil_train],feed_dict=fd)[:-1]
                    sil_pgloss = [sil_pgloss]
                    sil_vfloss = [sil_vfloss]
    
                    prior = prior.flatten()
                    new_priorities = np.abs(prior) + 1e-6
    
                    sil_start = 0
                    sil_end = 0
                    for k in range(self.I.num_agents):
                        
                        if len(batch_idxes_list[k]) > 0:
                            sil_end = sil_start + 8*self.nsteps
                            self.I.sil_buffer_list[k].update_priorities(batch_idxes_list[k], new_priorities[sil_start:sil_end])
    
                        sil_start = sil_end


                ret = ret + disc_loss + idle_loss + novel_loss + sil_pgloss + sil_vfloss
 

                #print(len(ret),self.loss_names)
    
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


    def train_sd(self, max_nepoch, max_neps):


        buf_vpreds_int = [[] for _ in range(self.I.nenvs)]
        buf_vpreds_ext = [[] for _ in range(self.I.nenvs)]
        buf_vpreds_div = [[] for _ in range(self.I.nenvs)]
        buf_acs = [[] for _ in range(self.I.nenvs)]


        buf_labels = [[] for _ in range(self.I.nenvs)]
        buf_obs = [[] for _ in range(self.I.nenvs)]


        agent_idx = np.zeros((self.I.nenvs), np.int32)
        agent_idx[:] = 0

        buf_room_infos = [[] for _ in range(self.I.nenvs)]
        buf_x_infos = [[] for _ in range(self.I.nenvs)]
        buf_y_infos = [[] for _ in range(self.I.nenvs)]
        buf_nkeys_infos = [[] for _ in range(self.I.nenvs)]




        

        rews_statistics = []


        visited_room = set()

        ep_count = 0

        step = 0

        img_count = 0
        while True:

            t = step % self.nsteps
            for l in range(self.I.nlump):

                if step == 0:
                    ob = self.I.venvs[l].reset()

                    if self.I.init_ram_state is not None:
                        for k in range(self.I.lump_stride):
    
                            self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
                                #restore ram
                            ob[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)
                    news = np.ones(self.I.lump_stride, bool)
                    mem_state = self.stochpol.initial_state(self.I.nenvs)

                sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)

                dict_obs = self.stochpol.ensure_observation_is_dict(ob)

                memsli = slice(None) if mem_state is NO_STATES else sli



                acs, vpreds_int, vpreds_ext, nlps, _, ent = self.stochpol.call(dict_obs, news, mem_state,agent_idx[:,None],update_obs_stats=False)

                self.I.venvs[l].step_async(acs)


                for j in range(self.I.lump_stride):
                    env_idx = l * self.I.lump_stride + j

                    buf_obs[env_idx].append(dict_obs[None][j])
                    buf_vpreds_ext[env_idx].append(vpreds_ext[j])
                    buf_vpreds_int[env_idx].append(vpreds_int[j])
                    buf_acs[env_idx].append(acs[j])


                    x = infos[j]['position'][0] if step > 0 else -1
                    y = infos[j]['position'][1] if step > 0 else -1
                    room = infos[j]['position'][2] if step >0 else -1
                    nkeys =  infos[j]['position'][3] if step >0 else -1
                    room_level = infos[j]['position'][4] if step >0 else -1
                    
                    buf_x_infos[env_idx].append(x)
                    buf_y_infos[env_idx].append(y)
                    buf_room_infos[env_idx].append(room)
                    buf_nkeys_infos[env_idx].append(nkeys)
                    visited_room.add((room,nkeys))



                ob, _, _, news, infos,_, _ = self.I.venvs[l].step_wait()

                for env_pos_in_lump, info in enumerate(infos):
                    if 'episode' in info:
                        rews_statistics.append(info['episode']['r'])

                for j in range(self.I.lump_stride):


                    if news[j] and step > 0:

                        if np.min(buf_vpreds_ext[j]) >= self.I.min_vpred_ext_thresold:
                            break;
                        ep_count = ep_count + 1
                        #clac entropy weights and pgloss weights
                        ep_len = len(buf_obs[j])
                        logger.info("ep_len:", ep_len)

                        sd_interval = 64
                        for step_idx in range(ep_len):
                            left = max(step_idx - sd_interval,0)
                            right = min(step_idx+ sd_interval, ep_len)
                            #logger.info("left: {} right: {}".format(str(left),str(right) ))
                            min_vpred_ext = np.max(buf_vpreds_ext[j][left:right])

                            room = buf_room_infos[j][step_idx]
                            #increase entropy
                            #if min_vpred_ext < 0.9:

                            if min_vpred_ext > 0.2 :


                                if buf_vpreds_ext[j][step_idx] < 0.2:


                                    vis_obs = buf_obs[j][step_idx][:,:,-1]
                                    plt.imshow(vis_obs, cmap=plt.cm.gray)
                                    plt.savefig('/home/xupei/RL/rnd_model/POS/vis_obs_{}.png'.format(str(img_count)))
                                    plt.close()
                                    img_count = img_count + 1

                                    self.I.pos_buffer.add(buf_obs[j][step_idx], buf_acs[j][step_idx], buf_vpreds_ext[j][step_idx], buf_vpreds_int[j][step_idx])
                            else:

                                vis_obs = buf_obs[j][step_idx][:,:,-1]
                                plt.imshow(vis_obs, cmap=plt.cm.gray)
                                plt.savefig('/home/xupei/RL/rnd_model/NEG/vis_obs_{}.png'.format(str(img_count)))
                                plt.close()
                                img_count = img_count + 1

                                if room not in [11]:
                                    self.I.neg_buffer.add(buf_obs[j][step_idx], buf_acs[j][step_idx], buf_vpreds_ext[j][step_idx], buf_vpreds_int[j][step_idx])
                                    
                                    obs_ = buf_obs[j][step_idx]
                                    self.I.sd_rms.update(obs_[:,:,-1:])



                        buf_obs[j] = []
                        buf_vpreds_ext[j] = []
                        buf_vpreds_int[j] = []
                        buf_acs[j] = []


                if self.I.init_ram_state is not None:

                    for k in range(self.I.lump_stride):
                        if news[k]:
                            self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
                            #restore ram
                            ob[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)
            


            if t == 0:
                logger.info("ep_count: {}".format(str(ep_count)))
                logger.info("rews_statistics: {}".format(rews_statistics))

            if ep_count >= max_neps and len(self.I.neg_buffer) > 0:
                logger.info("collect data done")
                logger.info("ep_count: {}".format(str(ep_count)))
                logger.info("rews_statistics: {}".format(rews_statistics))
                logger.info("visited_room: {}".format(visited_room))
                
                break

            step = step + 1 

        envsperbatch = self.I.nenvs // self.nminibatches
        envsperbatch = 32

        assert envsperbatch%2==0

        for epoch in range(max_nepoch):

            pos_obs, _, _, _ = self.I.pos_buffer.sample(envsperbatch * self.nsteps)
            pos_obs = np.reshape(pos_obs,(envsperbatch,self.nsteps) + self.I.buf_obs[None].shape[2:])
            pos_labels = np.ones((envsperbatch, self.nsteps), np.int32)


            #pos_obs, _, _, _ = self.I.neg_buffer.sample(envsperbatch * self.nsteps)
            #pos_obs = np.reshape(pos_obs,(envsperbatch,self.nsteps) + self.I.buf_obs[None].shape[2:])
            #pos_labels = np.ones((envsperbatch, self.nsteps), np.int32)

            neg_obs, _, _, _ = self.I.neg_buffer.sample(envsperbatch * self.nsteps)
            neg_obs = np.reshape(neg_obs,(envsperbatch,self.nsteps) + self.I.buf_obs[None].shape[2:])
            neg_labels = np.zeros((envsperbatch, self.nsteps), np.int32)



            obses_t = np.concatenate([pos_obs,neg_obs],0)

            obses_t = np.concatenate([obses_t, obses_t[:,0:1]], 1)

            labels = np.concatenate([pos_labels,neg_labels],0)              
            fd = {}
            fd.update({self.ph_lr : self.lr})
            fd[self.stochpol.ph_ob[None]] = obses_t
            fd[self.stochpol.stage_label] = labels
            fd[self.stochpol.sd_ph_mean] = self.I.sd_rms.mean
            fd[self.stochpol.sd_ph_std] = self.I.sd_rms.var** 0.5
            
            ret = tf_util.get_session().run(self._sd_losses+[self._sd_train], feed_dict=fd)[:-1]

            lossdict = dict(zip([n for n in self.sd_loss_names], ret), axis=0)
            #print(lossdict)
            logger.info("nepoch: {}".format(str(epoch)))
            logger.info(" "*6 + fmt_row(13, self.sd_loss_names))
            logger.info(" %s" % (fmt_row(13, [lossdict[n] for n in self.sd_loss_names])))


        path = '{}_sd_trained_sd_rms'.format(self.save_path)
        self.I.sd_rms.save(path)


    def collect_rnd_info(self, max_nstep):
        #Does a rollout.
        
        buf_room_infos = np.zeros((self.I.nenvs, self.nsteps), np.int32)
        buf_x_infos = np.zeros((self.I.nenvs, self.nsteps), np.int32)
        buf_y_infos = np.zeros((self.I.nenvs, self.nsteps), np.int32)

        room_set =[[1,2,6,7,12,13,14,21,22,23],[11]]

        max_int = np.zeros((2),np.float32)
        min_int = np.zeros((2),np.float32)
        min_int[:] = 1e7

        visited_rooms = set()

        count = 0

        for step in range(max_nstep):

            t = self.I.step_count % self.nsteps

            if self.I.step_count == 0:
                agent_idx = [ 0 for k in range(self.I.nenvs)]
                agent_idx = np.asarray(agent_idx) 
                self.I.agent_idx = agent_idx
    
            for l in range(self.I.nlump):

                obs, prevrews, ec_rews, news, infos, ram_states, monitor_rews = self.env_get(l)
    
                sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
                memsli = slice(None) if self.I.mem_state is NO_STATES else sli
    
                for k in range(self.I.lump_stride):
                    if news[k]:
                        if self.I.init_ram_state is not None:
                            self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
                            obs[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)
    
    
                dict_obs = self.stochpol.ensure_observation_is_dict(obs)
                with logger.ProfileKV("policy_inference"):
                    #Calls the policy and value function on current observation.
                    acs, vpreds_int, vpreds_ext, vpreds_div, nlps, self.I.mem_state[memsli], ent = self.stochpol.call(dict_obs, news, self.I.mem_state[memsli],self.I.agent_idx[:,None],
                                                                                                                   update_obs_stats=self.update_ob_stats_every_step)
                self.env_step(l, acs)
    

                #Update buffer with transition.
                for k in self.stochpol.ph_ob_keys:
                    self.I.buf_obs[k][sli, t] = dict_obs[k]
                
                self.I.buf_agent_idx[sli,t] = self.I.agent_idx


                buf_x_infos[sli,t] =  np.asarray([ -1 if infos=={} else infos[k]['position'][0] for k in range(self.I.nenvs)])
                buf_y_infos[sli,t] =  np.asarray([ -1 if infos=={} else infos[k]['position'][1] for k in range(self.I.nenvs)])
                buf_room_infos[sli,t] = np.asarray([ -1 if infos=={} else infos[k]['position'][2] for k in range(self.I.nenvs)])
    
    
                self.I.buf_ph_mean[sli,t] = np.asarray([ self.stochpol.ob_rms_list[self.I.agent_idx[k]].mean for k in range(self.I.nenvs)])
                self.I.buf_ph_std[sli,t] =  np.asarray([ self.stochpol.ob_rms_list[self.I.agent_idx[k]].var ** 0.5 for k in range(self.I.nenvs)])
    
    
            
            if t == self.nsteps - 1:
                #We need to take one extra step so every transition has a reward.
                for l in range(self.I.nlump):
                    sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
                    memsli = slice(None) if self.I.mem_state is NO_STATES else sli
                    nextobs, rews, ec_rews, nextnews, infos, ram_states, monitor_rews = self.env_get(l)
    
    
                    for k in range(self.I.lump_stride):
                        if nextnews[k]:
                            if self.I.init_ram_state is not None:
                                self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
            
                                #restore ram
                                nextobs[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)
        
                            self.I.agent_idx[k]= 0
    
    
                    dict_nextobs = self.stochpol.ensure_observation_is_dict(nextobs)
                    for k in self.stochpol.ph_ob_keys:
                        self.I.buf_ob_last[k][sli] = dict_nextobs[k]
    
    
                    with logger.ProfileKV("policy_inference"):
                        _, self.I.buf_vpred_int_last[sli], self.I.buf_vpred_ext_last[sli], self.I.buf_vpred_div_last[sli], _, _, _, _ \
                                     = self.stochpol.call(dict_nextobs, nextnews, self.I.mem_state[memsli], 
                                            self.I.agent_idx[:,None],update_obs_stats=False)
    
    
                if self.rnd_type =='rnd':
                    '''
                    #compute RND
                    fd = {}
                    fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None], self.I.buf_ob_last[None][:,None]], 1)
                    fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                                   self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                    fd[self.stochpol.ph_ac] = self.I.buf_acs
                    self.I.buf_rews_int[:] = tf_util.get_session().run(self.stochpol.int_rew, fd) * self.I.buf_rnd_weights
                    '''
                    count = count + 1
                    start = 0
                    envsperbatch = self.I.nenvs // self.nminibatches
                    while start < self.I.nenvs:
                        end = start + envsperbatch
                        mbenvinds = slice(start, end, None)
            
                        fd = {}
                
                        fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None][mbenvinds],  self.I.buf_ob_last[None][mbenvinds, None]], 1)
            
            
                        #fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                        #           self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                        fd.update({self.stochpol.sep_ph_mean: self.I.buf_ph_mean[mbenvinds],
                                    self.stochpol.sep_ph_std: self.I.buf_ph_std[mbenvinds]})
    
    
                        fd[self.stochpol.ph_agent_idx] = self.I.buf_agent_idx[mbenvinds]
    
    
    
                        #print(self.I.buf_sample_agent_prob[mbenvinds])
                        
                        rews_int = tf_util.get_session().run(self.stochpol.int_rew, fd)
                        
                        self.I.buf_rews_int[mbenvinds] = rews_int
        
                        start +=envsperbatch

                    for env_idx in range(self.I.nenvs):
                        for step_idx in range(self.nsteps):
                            room = buf_room_infos[env_idx, step_idx]
                            rnd_rew_int = self.I.buf_rews_int[env_idx, step_idx]

                            visited_rooms.add(room)
                            for cluser_idx in range(len(room_set)):
                                if room in room_set[cluser_idx]:
                                    if rnd_rew_int > max_int[cluser_idx]:
                                        max_int[cluser_idx] = rnd_rew_int
                                    if rnd_rew_int < min_int[cluser_idx]:
                                        min_int[cluser_idx] = rnd_rew_int

                    logger.info("process: {}".format(count))
                    logger.info("visted_rooms: {}".format(sorted(list(visited_rooms))))
                    for cluser_idx in range(len(room_set)):
                        logger.info("cluser {}: maxint {}, minint {}".format(cluser_idx, max_int[cluser_idx], min_int[cluser_idx]))
    
                elif self.rnd_type =='oracle':
                #compute oracle count-based reward
                    fd = {}
                else:
                    raise ValueError('Unknown exploration reward: {}'.format(
                      self._exploration_reward))
    
            self.I.step_count += 1


    def sample_dd_training_data(self, envs_per_agent):
        


        background_count = 0
        neg_count = 0
        pos_count = 0

        batch_idxes_list = []

        buffer_obs = None
        buffer_last_rew_obs = None
        buffer_game_score = None
        buffer_labels = None
        buffer_agent_idx = None

        for k in range(self.num_agents):


            if len(self.I.div_discr_pos_buffer_list[k]) > 0:
                obs, last_rew_obs, game_score = self.I.div_discr_pos_buffer_list[k].sample_uniform(envs_per_agent * self.nsteps)
                obs = np.reshape(obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                last_rew_obs =  np.reshape(last_rew_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                game_score = np.reshape(game_score,(envs_per_agent,self.nsteps))
                labels = np.full_like(game_score, k)

                pos_count += 1

                batch_idxes_list.append([])
                 
            else:

                obs, last_rew_obs, game_score = self.I.div_discr_background_buffer_list[k].sample(envs_per_agent  * self.nsteps)
                obs = np.reshape(obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                last_rew_obs =  np.reshape(last_rew_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                game_score = np.reshape(game_score,(envs_per_agent,self.nsteps))
                labels = np.full_like(game_score, k)
    
                background_count = background_count +1
                batch_idxes_list.append([])           


            

            agent_idx = np.full_like(labels, k)

            if k ==0:
                buffer_obs = obs
                buffer_last_rew_obs = last_rew_obs
                buffer_labels = labels
                buffer_game_score = game_score
                buffer_agent_idx = agent_idx
            else:
                buffer_obs = np.concatenate([buffer_obs, obs], 0)
                buffer_last_rew_obs = np.concatenate([buffer_last_rew_obs, last_rew_obs], 0)
                buffer_labels = np.concatenate([buffer_labels, labels], 0)
                buffer_game_score = np.concatenate([buffer_game_score, game_score], 0)
                buffer_agent_idx = np.concatenate([buffer_agent_idx, agent_idx], 0)   

        buffer_obs = np.concatenate([buffer_obs, buffer_obs[:,0:1]], 1)
        logger.info("dd_training neg_cout: {}  pos_count: {} background_count:{}".format(str(neg_count),str(pos_count), str(background_count)))
        return buffer_obs, buffer_last_rew_obs, buffer_game_score, buffer_labels, buffer_agent_idx, batch_idxes_list

    def sample_dd_training_data_v2(self, envs_per_agent):
        


        background_count = 0
        neg_count = 0
        pos_count = 0

        batch_idxes_list = []

        buffer_obs = None
        buffer_last_rew_obs = None
        buffer_game_score = None
        buffer_labels = None
        buffer_agent_idx = None

        bg_envs = envs_per_agent // 3

        for k in range(self.num_agents):

            obs, last_rew_obs, game_score = self.I.div_discr_background_buffer_list[k].sample(bg_envs  * self.nsteps)
            obs = np.reshape(obs,(bg_envs,self.nsteps) + self.I.buf_obs[None].shape[2:])
            last_rew_obs =  np.reshape(last_rew_obs,(bg_envs,self.nsteps) + self.I.buf_obs[None].shape[2:])
            game_score = np.reshape(game_score,(bg_envs,self.nsteps))
            labels = np.full_like(game_score, k)

            background_count = background_count +1

            if len(self.I.div_discr_pos_buffer_list[k]) > 0:
                pos_obs, pos_last_rew_obs, pos_game_score = self.I.div_discr_pos_buffer_list[k].sample_uniform(envs_per_agent * self.nsteps)
                pos_obs = np.reshape(pos_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                pos_last_rew_obs =  np.reshape(pos_last_rew_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                pos_game_score = np.reshape(pos_game_score,(envs_per_agent,self.nsteps))
                pos_labels = np.full_like(pos_game_score, k)

                pos_count += 1

                batch_idxes =[]
                batch_idxes_list.append(batch_idxes)
                 
                obs = np.concatenate([pos_obs, obs], 0)
                last_rew_obs = np.concatenate([pos_last_rew_obs, last_rew_obs], 0)
                game_score = np.concatenate([pos_game_score, game_score], 0)
                labels = np.concatenate([pos_labels, labels], 0)
            else:
                pos_obs, pos_last_rew_obs, pos_game_score = self.I.div_discr_background_buffer_list[k].sample(envs_per_agent * self.nsteps)
                pos_obs = np.reshape(pos_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                pos_last_rew_obs =  np.reshape(pos_last_rew_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                pos_game_score = np.reshape(pos_game_score,(envs_per_agent,self.nsteps))
                pos_labels = np.full_like(pos_game_score, k)

                background_count += 1

                batch_idxes =[]
                batch_idxes_list.append(batch_idxes)
                 
                obs = np.concatenate([pos_obs, obs], 0)
                last_rew_obs = np.concatenate([pos_last_rew_obs, last_rew_obs], 0)
                game_score = np.concatenate([pos_game_score, game_score], 0)
                labels = np.concatenate([pos_labels, labels], 0)
                batch_idxes_list.append([])           


            

            agent_idx = np.full_like(labels, k)

            if k ==0:
                buffer_obs = obs
                buffer_last_rew_obs = last_rew_obs
                buffer_labels = labels
                buffer_game_score = game_score
                buffer_agent_idx = agent_idx
            else:
                buffer_obs = np.concatenate([buffer_obs, obs], 0)
                buffer_last_rew_obs = np.concatenate([buffer_last_rew_obs, last_rew_obs], 0)
                buffer_labels = np.concatenate([buffer_labels, labels], 0)
                buffer_game_score = np.concatenate([buffer_game_score, game_score], 0)
                buffer_agent_idx = np.concatenate([buffer_agent_idx, agent_idx], 0)   

        buffer_obs = np.concatenate([buffer_obs, buffer_obs[:,0:1]], 1)
        logger.info("dd_training neg_cout: {}  pos_count: {} background_count:{}".format(str(neg_count),str(pos_count), str(background_count)))
        return buffer_obs, buffer_last_rew_obs, buffer_game_score, buffer_labels, buffer_agent_idx, batch_idxes_list


    def sample_novel_training_data(self, envs_per_agent):
        
        background_count = 0
        neg_count = 0
        pos_count = 0

        batch_idxes_list = []

        agent_idx = np.random.choice(self.I.num_agents,size=1, replace=False,p=self.I.p)[0]

        obs, last_rew_obs, game_score = self.I.div_discr_background_buffer_list[agent_idx].sample(envs_per_agent  * self.nsteps)
        background_count = background_count +1


        obs = np.reshape(obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
        labels = np.zeros((envs_per_agent,self.nsteps))


        if len(self.I.novel_rew_pos_buffer) > 0:
            pos_obs, weights, batch_idxes = self.I.novel_rew_pos_buffer.sample(envs_per_agent * self.nsteps, beta=0.1)
            pos_obs = np.reshape(pos_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
            pos_labels = np.ones((envs_per_agent,self.nsteps))
            pos_count += 1

            obs = np.concatenate([pos_obs, obs], 0)
            labels = np.concatenate([pos_labels, labels], 0)
            batch_idxes_list.append(batch_idxes)
        else:
            batch_idxes_list.append([])


        buffer_obs = obs
        buffer_labels = labels
 


        buffer_obs = np.concatenate([buffer_obs, buffer_obs[:,0:1]], 1)

        logger.info("novel_training neg_cout: {}  pos_count: {} background_count:{}".format(str(neg_count),str(pos_count), str(background_count)))
        return buffer_obs, buffer_labels, batch_idxes_list

    def sample_idle_agent_training_data(self, envs_per_agent_neg, envs_per_agent_pos):


        neg_count = 0
        pos_count = 0


        batch_idxes_list = []

        for k in range(self.num_agents):

            obs = self.I.idle_agent_discr_neg_buffer_list[k].sample(envs_per_agent_neg * self.nsteps)
            obs = np.reshape(obs,(envs_per_agent_neg,self.nsteps) + self.I.buf_obs[None].shape[2:])
            labels = np.zeros((envs_per_agent_neg,self.nsteps))

            neg_count += 1


            if len(self.I.idle_agent_discr_pos_buffer_list[k]) > 0:
                (pos_obs, weights, batch_idxes) = self.I.idle_agent_discr_pos_buffer_list[k].sample(envs_per_agent_pos * self.nsteps, beta=0.1)
                pos_obs = np.reshape(pos_obs,(envs_per_agent_pos,self.nsteps) + self.I.buf_obs[None].shape[2:])
                pos_labels = np.ones((envs_per_agent_pos,self.nsteps))

                obs = np.concatenate([pos_obs, obs], 0)
                labels = np.concatenate([pos_labels, labels], 0)
                

                pos_count +=1


                batch_idxes_list.append(batch_idxes)
            else:
                batch_idxes_list.append([])


            agent_idx = np.full_like(labels, k)
            '''

            else:
                neg_obs = self.I.idle_agent_discr_neg_buffer_list[k].sample(envs_per_agent * self.nsteps)
                neg_obs = np.reshape(neg_obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                #neg label
                neg_labels = np.zeros((envs_per_agent,self.nsteps))

                obs = np.concatenate([neg_obs, obs], 0)
                labels = np.concatenate([neg_labels, labels], 0)
                agent_idx = np.full_like(labels, k)

                neg_count += 1
            '''

            if k ==0:
                buffer_last_rew_obs = obs
                buffer_labels = labels
                buffer_agent_idx = agent_idx
            else:
                buffer_last_rew_obs = np.concatenate([buffer_last_rew_obs, obs], 0)
                buffer_labels = np.concatenate([buffer_labels, labels], 0)
                buffer_agent_idx = np.concatenate([buffer_agent_idx, agent_idx], 0)


        logger.info("idle_agent_training neg_cout: {}  pos_count: {}".format(str(neg_count),str(pos_count)))

        return buffer_last_rew_obs, buffer_labels, buffer_agent_idx, batch_idxes_list

    def sample_sil_training_data(self, envs_per_agent):

        buf_obs = None
        batch_idxes_list = []
        for k in range(self.num_agents):
            if len(self.I.sil_buffer_list[k]) > 0:
                obs, acs, vf_exts, rooms_id, _, batch_idxes = self.I.sil_buffer_list[k].sample(batch_size=envs_per_agent*self.nsteps, beta=0.01)
                
                obs = np.reshape(obs,(envs_per_agent,self.nsteps) + self.I.buf_obs[None].shape[2:])
                acs = np.reshape(acs,(envs_per_agent,self.nsteps))
                vf_exts = np.reshape(vf_exts,(envs_per_agent,self.nsteps))

                agent_idx = np.full_like(acs,k)

                batch_idxes_list.append(batch_idxes)

                if buf_obs is None:
                    buf_obs = obs
                    buf_acs = acs
                    buf_vf_exts = vf_exts
                    buf_agent_idx = agent_idx
                else:
                    buf_obs = np.concatenate([buf_obs, obs], axis=0)
                    buf_acs = np.concatenate([buf_acs, acs], axis=0)
                    buf_vf_exts = np.concatenate([buf_vf_exts, vf_exts], axis=0)
                    buf_agent_idx = np.concatenate([buf_agent_idx, agent_idx], axis=0)
            else:
                batch_idxes_list.append([])


        buf_obs = np.concatenate([buf_obs, buf_obs[:,0:1]], 1)

        return buf_obs, buf_acs, buf_vf_exts, buf_agent_idx, batch_idxes_list


    @logger.profile("step")
    def step(self):
        #Does a rollout.
        t = self.I.step_count % self.nsteps
        epinfos = []
        epinfos_agent_idx = []

        if self.I.step_count == 0:
            agent_idx = [ self.sample_agent_idx(k) for k in range(self.I.nenvs)]
            agent_idx = np.asarray(agent_idx) 
            self.I.agent_idx = agent_idx


        for l in range(self.I.nlump):
            with logger.ProfileKV("env_get"):
                obs, prevrews, ec_rews, news, infos, ram_states, monitor_rews = self.env_get(l)



            for env_pos_in_lump, info in enumerate(infos):
                if 'episode' in info:
                    #Information like rooms visited is added to info on end of episode.
                    epinfos.append(info['episode'])
                    epinfos_agent_idx.append(self.I.agent_idx[env_pos_in_lump+l*self.I.lump_stride])
                    #print(epinfos_agent_idx, env_pos_in_lump+l*self.I.lump_stride)
                    info_with_places = info['episode']
                    try:
                        info_with_places['places'] = info['episode']['visited_rooms']
                    except:
                        import ipdb; ipdb.set_trace()
                    self.I.buf_epinfos[env_pos_in_lump+l*self.I.lump_stride][t] = info_with_places


            sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
            memsli = slice(None) if self.I.mem_state is NO_STATES else sli



            room_info = np.asarray([ -1 if infos=={} else infos[k]['position'][2] for k in range(self.I.nenvs)])

            
            for k in range(self.I.lump_stride):
                if news[k]:
                    self.I.ep_scores[k] = 0
                    self.I.ep_last_rew_ob[k] = np.full_like(self.I.ep_last_rew_ob[k],128)
                    self.I.divexp_flag[k] = False

                    self.I.ep_step_count[k] = 0

                    self.I.ep_explore_step[k] = 4500

                    

                    if self.I.init_ram_state is not None:
                        self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
    
                        #restore ram
                        obs[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)



                    self.I.sample_agent_prob[k] = self.I.p[self.I.agent_idx[k]]

                if self.I.ep_scores[k] > self.I.socre_baseline or self.load_ram:
                    key = (room_info[k],self.I.ep_scores[k])
                    key_count = self.I.agent_detial_room_info[self.I.agent_idx[k]].get( key, 0)
                    self.I.agent_detial_room_info[self.I.agent_idx[k]][key] = key_count + 1

                self.I.agent_idx[k]= self.sample_agent_idx(k +l*self.I.lump_stride, room_info[k], self.I.ep_scores[k])


                self.I.buf_agent_ob[self.I.agent_idx[k]].append(obs[k])


                


            dict_obs = self.stochpol.ensure_observation_is_dict(obs)
            with logger.ProfileKV("policy_inference"):
                #Calls the policy and value function on current observation.
                acs, vpreds_int, vpreds_ext, nlps, self.I.mem_state[memsli], ent, base_vpreds_ext = self.stochpol.call(dict_obs, news, self.I.mem_state[memsli],self.I.agent_idx[:,None],
                                                                                                               update_obs_stats=self.update_ob_stats_every_step)



                '''
                if self.sd_type == 'sd':
                    state_stages_prob = self.stochpol.state_stage(dict_obs)
                    state_stages = state_stages_prob[:,1]

                    divexp_stages = state_stages > 0.5
                '''
                #determine stage by room info
                if self.I.step_count == 0:
                    divexp_stages = np.asarray( [True  for k in range(self.I.nenvs) ])
                else:
                    divexp_stages = np.asarray( [True if infos[k]['position'][2] in self.I.div_rooms else False  for k in range(self.I.nenvs) ])




            for k in range(self.I.lump_stride):
                if vpreds_ext[k] < 0.1:

                    if self.I.ep_explore_step[k] == 4500:
                        self.I.ep_explore_step[k] = self.I.ep_step_count[k]
                    else:
                        detla_step_count = self.I.ep_step_count[k] - self.I.ep_explore_step[k]
                        if detla_step_count < 300:
                            self.I.buf_stage_label[k,t] = 1
                        else:
                            self.I.buf_stage_label[k,t] = 0

            

 
            step_acs = acs
            self.env_step(l, step_acs)


            #Update buffer with transition.
            for k in self.stochpol.ph_ob_keys:
                self.I.buf_obs[k][sli, t] = dict_obs[k]
            self.I.buf_news[sli, t] = news

            self.I.buf_vpreds_int[sli, t] = vpreds_int
            self.I.buf_vpreds_ext[sli, t] = vpreds_ext
            #self.I.buf_target_vpreds_ext[sli,t] = target_vpred_ext

            self.I.buf_nlps[sli, t] = nlps
            self.I.buf_acs[sli, t] = acs
            #self.I.buf_target_acs[sli,t] = target_acs
            self.I.buf_step_acs[sli,t] = step_acs
            self.I.buf_ent[sli, t] = ent
            
            self.I.buf_agent_idx[sli,t] = self.I.agent_idx
            

            self.I.buf_sample_agent_prob[sli,t] = self.I.sample_agent_prob



            self.I.buf_rnd_mask[sli,t] = np.asarray([ self.get_rnd_mask(self.I.agent_idx[k], room_info[k]) for k in range(self.I.nenvs)])
            self.I.buf_ph_mean[sli,t] = np.asarray([ self.stochpol.ob_rms_list[self.I.agent_idx[k]].mean for k in range(self.I.nenvs)])
            self.I.buf_ph_std[sli,t] =  np.asarray([ self.stochpol.ob_rms_list[self.I.agent_idx[k]].var ** 0.5 for k in range(self.I.nenvs)])
            self.I.buf_x_infos[sli,t] =  np.asarray([ -1 if infos=={} else infos[k]['position'][0] for k in range(self.I.nenvs)])
            self.I.buf_y_infos[sli,t] =  np.asarray([ -1 if infos=={} else infos[k]['position'][1] for k in range(self.I.nenvs)])
            self.I.buf_room_infos[sli,t] = room_info
            self.I.buf_level_infos[sli,t] = np.asarray([ -1 if infos=={} else infos[k]['position'][3] for k in range(self.I.nenvs)])


            self.I.buffer_ent_coef[sli,t] = np.full_like(acs,0.001)

            self.I.buf_last_rew_ob[sli,t] = self.I.ep_last_rew_ob
            self.I.buf_ep_scores[sli, t] = self.I.ep_scores


            self.I.int_coef[sli,t] = 1
            self.I.ext_coef[sli,t] = 2

            self.I.buf_base_vpred_ext[sli,t] =  base_vpreds_ext

            if self.I.step_count > 0:
                self.I.ep_scores = self.I.ep_scores + prevrews

                self.I.ep_last_rew_ob[prevrews > 0] =  dict_obs[None][prevrews > 0]



            if t == 0:
                self.I.buf_first_agent_idx[sli] = self.I.buf_agent_idx[sli,self.nsteps - 1]

                self.I.buf_agent_change[sli,t] = self.I.buf_last_agent_change[sli] 

            if t > 0:



                fixed_prevrews = np.zeros_like(prevrews)

                for k in range(self.I.lump_stride):
                    dd_obs = self.I.buf_obs[None][k, t-1]
                    dd_agent_idx = self.I.buf_agent_idx[k,t-1]
                    dd_last_rew_ob = self.I.buf_last_rew_ob[k,t-1]
                    dd_game_score = self.I.buf_ep_scores[k,t-1]

                    fixed_prevrews[k], self.I.buf_unique_rews[k][t-1] =self.filter_rew(prevrews[k], infos[k]['unclip_rew'], infos[k]['position'], infos[k]['open_door_type'],k, dd_agent_idx )

                    x, y, room_id, nkeys, level = infos[k]['position']
                    
                    #if  room_id ==5 and fixed_prevrews[k]>0 and dd_agent_idx==0:
                    #   logger.info("error!")
                    
                    '''
                    if self.div_type=='cls':

                        if (room_id, level) in self.I.div_exclude_rooms:
                            continue
                        if fixed_prevrews[k]==0:
                            self.I.div_discr_neg_buffer.add(dd_obs, dd_last_rew_ob, dd_game_score)
                            self.I.idle_agent_discr_neg_buffer_list[dd_agent_idx].add(dd_obs)
                    '''

                prevrews = fixed_prevrews
                #print(prevrews)

                self.I.buf_rews_ext[sli, t-1] = prevrews
                self.I.buf_rews_ec[sli, t-1] = ec_rews


                if self.rnd_type=='oracle':
                    buf_rews_int = [
                        self.update_oracle_count(self.I.buf_agent_idx[k,t-1],infos[k]['position'])
                        for k in range(self.I.nenvs)]
                    #print(buf_rews_int)

                    buf_rews_int = np.array(buf_rews_int)
                    self.I.buf_rews_int[sli, t-1] = buf_rews_int

                #clac rnd weights:
                buf_rnd_weights = [ self.clac_rnd_weight(infos[k]['position'], self.I.buf_agent_idx[k,t-1], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                self.I.buf_rnd_weights[sli, t-1] = np.asarray(buf_rnd_weights)



                buf_div_mask = [ self.clac_div_mask(infos[k]['position'] ,self.I.agent_idx[k], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                
                self.I.buf_use_div[sli,t-1] = buf_div_mask
                self.I.buf_div_train_mask[sli,t-1] = [ self.clac_div_mask(infos[k]['position'] ,self.I.agent_idx[k], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                self.I.buf_div_rew_mask[sli, t-1] = prevrews * buf_div_mask


                


                self.I.buf_agent_change[sli,t] = self.I.buf_agent_idx[sli,t] != self.I.buf_agent_idx[sli,t-1]

        
        if t == self.nsteps - 1 and not self.disable_policy_update:
            #We need to take one extra step so every transition has a reward.
            for l in range(self.I.nlump):
                sli = slice(l * self.I.lump_stride, (l + 1) * self.I.lump_stride)
                memsli = slice(None) if self.I.mem_state is NO_STATES else sli
                with logger.ProfileKV("env_get"):
                    nextobs, rews, ec_rews, nextnews, infos, ram_states, monitor_rews = self.env_get(l)


                for k in range(self.I.lump_stride):
                    if nextnews[k]:
                        if self.I.init_ram_state is not None:
                            self.I.venvs[l].set_cur_monitor_rewards_by_idx(self.I.init_monitor_rews,k)
        
                            #restore ram
                            nextobs[k] = self.I.venvs[l].restore_full_state_by_idx(self.I.init_ram_state,k)


                self.I.buf_last_agent_idx[sli] = self.I.agent_idx

                self.I.buf_last_agent_change[sli] = self.I.buf_last_agent_idx[sli] != self.I.buf_agent_idx[sli,t]


                dict_nextobs = self.stochpol.ensure_observation_is_dict(nextobs)
                for k in self.stochpol.ph_ob_keys:
                    self.I.buf_ob_last[k][sli] = dict_nextobs[k]
                self.I.buf_new_last[sli] = nextnews


                buf_rnd_weights = [ self.clac_rnd_weight(infos[k]['position'], self.I.buf_agent_idx[k,t], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                self.I.buf_rnd_weights[sli, t] = np.asarray(buf_rnd_weights)



                buf_div_mask = [ self.clac_div_mask(infos[k]['position'], self.I.agent_idx[k], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                
                self.I.buf_use_div[sli,t] = buf_div_mask
                #self.I.buf_div_train_mask[sli,t] = np.asarray(buf_div_mask)

                self.I.buf_div_train_mask[sli,t] = [ self.clac_div_mask(infos[k]['position'] ,self.I.agent_idx[k], self.I.ep_scores[k]) for k in range(self.I.lump_stride)]
                self.I.buf_div_rew_mask[sli,t] = rews * buf_div_mask

                with logger.ProfileKV("policy_inference"):
                    _, self.I.buf_vpred_int_last[sli], self.I.buf_vpred_ext_last[sli], _, _, _,_ \
                                 = self.stochpol.call(dict_nextobs, nextnews, self.I.mem_state[memsli], 
                                        self.I.agent_idx[:,None],update_obs_stats=False)
                





                fixed_rews = np.zeros_like(rews)

                for k in range(self.I.lump_stride):
                    dd_obs = self.I.buf_obs[None][k, t]
                    dd_agent_idx = self.I.buf_agent_idx[k,t]
                    dd_last_rew_ob = self.I.buf_last_rew_ob[k,t]
                    dd_game_score = self.I.buf_ep_scores[k,t]

                    fixed_rews[k], self.I.buf_unique_rews[k][t] =self.filter_rew(rews[k], infos[k]['unclip_rew'], infos[k]['position'], infos[k]['open_door_type'],k, dd_agent_idx)
                   
                    x, y, room_id, nkeys, level = infos[k]['position']
                    #if room_id ==5 and fixed_rews[k]>0 and dd_agent_idx==0:
                    #   logger.info("error")

                    '''
                    if self.div_type=='cls':
                        x, y, room_id, nkeys, level = infos[k]['position']
                        if (room_id, level) in self.I.div_exclude_rooms:
                            continue
                        if fixed_rews[k]==0:
                            self.I.div_discr_neg_buffer.add(dd_obs, dd_last_rew_ob, dd_game_score)
                            self.I.idle_agent_discr_neg_buffer_list[dd_agent_idx].add(dd_obs)
                    '''
    
                rews =  fixed_rews


                self.I.buf_rews_ext[sli, t] = rews
                self.I.buf_rews_ec[sli, t] = ec_rews


                if self.rnd_type=='oracle':
                    buf_rews_int = [
                        self.update_oracle_count(self.I.buf_agent_idx[k,t],infos[k]['position'])
                        for k in range(self.I.nenvs)]
                    

                    buf_rews_int = np.array(buf_rews_int)
                    self.I.buf_rews_int[sli, t] = buf_rews_int * self.I.buf_rnd_weights[sli, t]

            if self.rnd_type =='rnd':
                '''
                #compute RND
                fd = {}
                fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None], self.I.buf_ob_last[None][:,None]], 1)
                fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                               self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                fd[self.stochpol.ph_ac] = self.I.buf_acs
                self.I.buf_rews_int[:] = tf_util.get_session().run(self.stochpol.int_rew, fd) * self.I.buf_rnd_weights
                '''

                with logger.ProfileKV("inference_ir_div"):
                    start = 0
                    envsperbatch = self.I.nenvs // self.nminibatches
                    while start < self.I.nenvs:
                        end = start + envsperbatch
                        mbenvinds = slice(start, end, None)
            
                        fd = {}
                
                        fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None][mbenvinds],  self.I.buf_ob_last[None][mbenvinds, None]], 1)
            
            
                        fd.update({self.stochpol.ph_mean: self.stochpol.ob_rms.mean,
                                   self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5})
                        fd.update({self.stochpol.sep_ph_mean: self.I.buf_ph_mean[mbenvinds],
                                    self.stochpol.sep_ph_std: self.I.buf_ph_std[mbenvinds]})
    
                        fd[self.stochpol.ph_ac] = self.I.buf_acs[mbenvinds]
                        fd[self.stochpol.ph_agent_idx] = self.I.buf_agent_idx[mbenvinds]
    
                        fd[self.stochpol.sample_agent_prob] = self.I.buf_sample_agent_prob[mbenvinds]
    
    
                        fd[self.stochpol.game_score] = self.I.buf_ep_scores[mbenvinds]
                        fd[self.stochpol.last_rew_ob] = self.I.buf_last_rew_ob[mbenvinds]
    
    
                        if self.div_type =='cls':

                            rews_int, stage_rnd, rews_div, div_prob, sp_prob  = tf_util.get_session().run([self.stochpol.int_rew,
                                                                            self.stochpol.stage_rnd,
                                                                            self.stochpol.div_rew,
                                                                            self.stochpol.all_div_prob,
                                                                            self.stochpol.sp_prob], fd)

                            '''
                            max_div = np.zeros_like(rews_div)

                            div_interval = 128
                            for c in range(envsperbatch):
                                for n in range(self.nsteps):
                                    left = max(n - div_interval, 0)
                                    right = min(n + div_interval, self.nsteps)
                                    max_div[c,n] = np.max(rews_div[c,left:right])
                            '''

                            rews_int = np.clip(rews_int,0,1)

                            stage_explore = stage_rnd > 0.1

                            stage_explore =  self.I.buf_use_div[mbenvinds] #stage_explore * self.I.buf_use_div[mbenvinds]


                            base_int_weight = np.ones_like(stage_explore)
                            for c in range(envsperbatch):
                                for n in range(self.nsteps):
                                    room_id = self.I.buf_room_infos[c+start][n]
                                    agent_idx = self.I.buf_agent_idx[c+start][n]
                                    agent_score =  self.I.buf_ep_scores[c+start][n]
                                    base_vpred_ext = self.I.buf_base_vpred_ext[c+start][n] 

                                    #if room_id ==5 and self.I.buf_rews_ext[c+start][n] > 0:
                                    #    logger.info( base_vpred_ext)
                                    
                                    if  room_id not in  self.I.divrew_exclude_rooms:
                                        if self.I.buf_rews_ext[c+start][n] > 0 :
                                             #logger.info(rews_div[c,n], agent_idx, room_id)
                                             div_rew = (np.log(sp_prob[c,n]) - np.log(0.5))  * 2
                                             #if room_id ==5  or room_id ==10:
                                             #    div_rew = 0
                                             logger.info(base_vpred_ext,rews_div[c,n], div_rew ,agent_idx, room_id)
                                             self.I.buf_rews_ext[c+start][n] = self.I.buf_rews_ext[c+start][n]*np.clip(div_rew,-0.1,1)

                                    if stage_explore[c,n]:
                                        self.I.div_room_set[agent_idx].add(room_id)
                                        #if self.I.buf_rews_ext[c+start][n] > 0 :
                                        #    logger.info(rews_div[c,n], agent_idx, room_id)
                                         

                                        #div_rew = np.log(sp_prob[c,n]) - np.log(0.5)
                                        #div_rew = rews_div[c,n] * 2
                                        #self.I.buf_rews_ext[c+start][n] = self.I.buf_rews_ext[c+start][n]*np.clip(div_rew,-1,1)

                                        base_int_weight[c,n] = self.I.base_int_weight[agent_idx] #0
                                        #rews_int[c,n] = max(0.01,rews_int[c,n])
                                        self.I.int_coef[c+start][n] = self.int_coeff
                                        self.I.ext_coef[c+start][n] = self.I.ext_weight[agent_idx]
                                        
                                        #if room_id == 5 or room_id == 10:
                                        #    rews_div[c,n] = -1
                                        #rews_int[c,n] *= 5
                                    
                                    else:
                                        base_int_weight[c,n] =  1#self.I.base_int_weight[agent_idx]
                                        self.I.int_coef[c+start][n] = 1
                                        self.I.ext_coef[c+start][n] = self.I.ext_weight[agent_idx]
                                        #rews_int[c,n] = 0

                            #rews_div[rews_div<0] = -1
                            #rews_div[rews_div>0] = rews_div[rews_div>0] * 4
                            int_weight = np.clip(rews_div * 4 * stage_explore,0, 4) + base_int_weight #* (1-stage_explore)
                          
                            #rews_int = rews_int * int_weight
                            #int_weight[:] = 1  
                            #int_weight = np.clip(int_weight,0,0.75)


                            #int_weight[self.I.buf_agent_idx[mbenvinds]==0] = 1

                            #int_weight[self.I.buf_rnd_weights[mbenvinds]==0] = 1
                           
                            self.I.buf_int_weight[mbenvinds] = int_weight



                        elif self.div_type=='oracle':
                            rews_int = tf_util.get_session().run(self.stochpol.int_rew, fd)



                            for c in range(envsperbatch):
                                for n in range(self.nsteps):
                                    s = self.I.buf_obs[None][c+start][n]
                                    a = self.I.buf_acs[c+start][n]
                                    r = self.I.buf_rews_ext[c+start][n]
                                    agent_idx = self.I.buf_agent_idx[c+start][n]
                                    room_id = self.I.buf_room_infos[c+start][n]
                                    new = self.I.buf_news[c+start][n]



                                    
                                    self.I.buf_int_weight[c+start][n] = 1



                        else:
                            rews_int = tf_util.get_session().run(self.stochpol.int_rew, fd)
                        
        
    
                        self.I.buf_target_vpreds_int[mbenvinds] = 0 #vpreds_int_baseline
                        self.I.buf_rews_int[mbenvinds] = rews_int * self.I.buf_rnd_weights[mbenvinds] #* self.I.buf_int_weight[mbenvinds]


                        '''
                        for c in range(envsperbatch):
                            for n in range(self.nsteps):
                                s = self.I.buf_obs[None][c+start][n]
                                a = self.I.buf_acs[c+start][n]
                                r = self.I.buf_rews_ext[c+start][n]
                                agent_idx = self.I.buf_agent_idx[c+start][n]
                                room_id = self.I.buf_room_infos[c+start][n]
                                new = self.I.buf_news[c+start][n]

                                if new:
                                    self.clac_return(self.I.trajector_list[c+start], self.I.sil_buffer_list[agent_idx], agent_idx)
                                    self.I.trajector_list[c+start] = []

                                self.I.trajector_list[c+start].append([s,a,r, room_id])
                        '''
                        start +=envsperbatch




                self.I.unlock_agent_index = -1
            elif self.rnd_type =='oracle':
            #compute oracle count-based reward

                start = 0
                envsperbatch = self.I.nenvs // self.nminibatches
                while start < self.I.nenvs:
                    end = start + envsperbatch
                    mbenvinds = slice(start, end, None)
        
                    fd = {}
            
                    fd[self.stochpol.ph_ac] = self.I.buf_acs[mbenvinds]
                    fd[self.stochpol.ph_agent_idx] = self.I.buf_agent_idx[mbenvinds]

                    fd[self.stochpol.sample_agent_prob] = self.I.buf_sample_agent_prob[mbenvinds]


                    fd[self.stochpol.game_score] = self.I.buf_ep_scores[mbenvinds]
                    fd[self.stochpol.last_rew_ob] = self.I.buf_last_rew_ob[mbenvinds]

                    fd[self.stochpol.ph_ob[None]] = np.concatenate([self.I.buf_obs[None][mbenvinds],  self.I.buf_ob_last[None][mbenvinds, None]], 1)



                    for c in range(envsperbatch):
                        for n in range(self.nsteps):
                            s = self.I.buf_obs[None][c+start][n]
                            a = self.I.buf_acs[c+start][n]
                            r = self.I.buf_rews_ext[c+start][n]
                            agent_idx = self.I.buf_agent_idx[c+start][n]
                            room_id = self.I.buf_room_infos[c+start][n]
                            new = self.I.buf_news[c+start][n]
                            
                            self.I.buf_int_weight[c+start][n] = 1


                    if self.div_type =='cls':
                        stage_rnd, rews_div, div_prob  = tf_util.get_session().run([self.stochpol.stage_rnd,
                                                                        self.stochpol.div_rew,
                                                                        self.stochpol.all_div_prob], fd)
                        stage_explore = stage_rnd > 0.1
                        stage_explore =  self.I.buf_use_div[mbenvinds] #stage_explore * self.I.buf_use_div[mbenvinds]
                        base_int_weight = np.ones_like(stage_explore)
                        for c in range(envsperbatch):
                            for n in range(self.nsteps):
                                room_id = self.I.buf_room_infos[c+start][n]
                                agent_idx = self.I.buf_agent_idx[c+start][n]
                                if stage_explore[c,n]:
                                    self.I.div_room_set[agent_idx].add(room_id)
                                    #self.I.buf_rews_ext[c+start][n] = self.I.buf_rews_ext[c+start][n]*(div_prob[c,n,agent_idx]>0.5)  # (rews_div[c,n] > 0)
                                    base_int_weight[c,n] = self.I.base_int_weight[agent_idx] #0
                                else:
                                    base_int_weight[c,n] = self.I.base_int_weight[agent_idx]
                        int_weight = np.clip(rews_div * 2  * stage_explore,-0.5,0.5) + base_int_weight# * (1-stage_explore)
                        self.I.buf_int_weight[mbenvinds] = int_weight
    



                    start +=envsperbatch

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

                for i in range(self.I.num_agents):

                    if len(self.I.buf_agent_ob[i]) > 0:
                        obs_ = np.array(self.I.buf_agent_ob[i], copy=False)
                        #print(obs_.shape)
                        self.update_obs_rms(obs_[:,:,:,-1:],i)
    
                        self.I.buf_agent_ob[i] = []



                obs_ = self.I.buf_obs[None].astype(np.float32)
                self.stochpol.ob_rms.update(obs_.reshape((-1, *obs_.shape[2:]))[:,:,:,-1:])
                feed = {self.stochpol.ph_mean: self.stochpol.ob_rms.mean, self.stochpol.ph_std: self.stochpol.ob_rms.var ** 0.5\
                        , self.stochpol.ph_count: self.stochpol.ob_rms.count}
                #self.sess.run(self.assign_op, feed)


            if not self.testing:

                logger.info("sample prob: ",self.I.p)
                for i in range(self.I.num_agents):
                    logger.info("agent{} : {}".format(i, self.I.reward_set_list[i]))
              

                
                for i in range(self.I.num_agents):
                    logger.info("agent{}: {}".format(i, self.I.agents_local_rooms[i]))
                    self.I.agents_local_rooms[i] = []

                logger.info("div room_set:")
                for i in range(self.I.num_agents):
                    logger.info("agent{}: {}".format(i, self.I.div_room_set[i]))
                    self.I.div_room_set[i].clear()


                for i in range(self.I.num_agents):
                    detial_room_info = self.I.agent_detial_room_info[i]#sorted(self.I.agent_detial_room_info[i])
                    logger.info("agent{}: {}".format(i, detial_room_info))
                    self.I.agent_detial_room_info[i] = {}



                logger.info(self.I.agent_idx_count)
                update_info = self.update()

                for i in range(self.I.num_agents):
                    self.I.oracle_visited_count_list[i].sync()
                    self.I.oracle_reward_count_list[i].sync()

            else:
                update_info = {}
            self.I.seg_init_mem_state = copy(self.I.mem_state)
            global_i_stats = dict_gather(self.comm_log, self.I.stats, op='sum')
            global_deque_mean = dict_gather(self.comm_log, { n : [preprocess_statlists(dvs[m]) for m in range(self.I.num_agents)] for n,dvs in self.I.statlists.items() }, op='mean')
            update_info.update(global_i_stats)
            update_info.update(global_deque_mean)
            self.global_tcount = global_i_stats['tcount']
            for infos_ in self.I.buf_epinfos:
                infos_.clear()
        else:
            update_info = {}

        self.I.step_count += 1


        self.I.ep_step_count = self.I.ep_step_count + 1

        #Some reporting logic.
        for i, epinfo in enumerate(epinfos):

            agent_idx = epinfos_agent_idx[i]
            if self.testing:
                self.I.statlists['eprew_test'][agent_idx].append(epinfo['r'])
                self.I.statlists['eplen_test'][agent_idx].append(epinfo['l'])
            else:
                if "visited_rooms" in epinfo:

                    self.I.agents_local_rooms[agent_idx] += list(epinfo["visited_rooms"])
                    self.I.agents_local_rooms[agent_idx] = sorted(list(set(self.I.agents_local_rooms[agent_idx])))

                    self.local_rooms += list(epinfo["visited_rooms"])
                    self.local_rooms = sorted(list(set(self.local_rooms)))
                    score_multiple = self.I.venvs[0].score_multiple
                    if score_multiple is None:
                        score_multiple = 1000
                    rounded_score = int(epinfo["r"] / score_multiple) * score_multiple
                    self.scores.append(rounded_score)
                    self.scores = sorted(list(set(self.scores)))
                    self.I.statlists['eprooms'][agent_idx].append(len(epinfo["visited_rooms"]))

                self.I.statlists['eprew'][agent_idx].append(epinfo['r'])
                if self.local_best_ret[agent_idx] is None:
                    self.local_best_ret[agent_idx] = epinfo["r"]
                elif epinfo["r"] > self.local_best_ret[agent_idx]:
                    self.local_best_ret[agent_idx] = epinfo["r"]

                self.I.statlists['eplen'][agent_idx].append(epinfo['l'])
                self.I.stats['epcount'] += 1
                self.I.stats['tcount'] += epinfo['l']
                self.I.stats['rewtotal'] += epinfo['r']
                # self.I.stats["best_ext_ret"] = self.best_ret


        return {'update' : update_info}



    # True: if the state is in divese exploration stage
    # False: if the state is in imitation stage
    def is_divexp_stage(self, rough_divexp_stage, base_vfext, agent_idx, divexp_flag, cur_rews):

        if agent_idx == 0:
            return True

        if divexp_flag:
            return True

        if (rough_divexp_stage == True and base_vfext < 3 ):
            return True
        else:
            return False


    def adjug_sample_prob_by_diversity_score(self):

        mean_div_score_list = []
        for i in range(self.I.num_agents):
            mean_div_score_list.append(np.mean(self.I.div_scores[i]))

        prob = softmax(mean_div_score_list)



        logger.info("sample prob:", prob)

        self.p = pro
      

    def clac_rnd_weight(self, pos, agent_idx=-1, score=0):

        x, y, room_id, nkeys, level = pos

        if (room_id, level) in self.I.int_exclude_rooms:
            return 0
        else: 
            return 1.

    def clac_div_mask(self, pos, agent_idx, score):
        x, y, room_id, nkeys, level = pos

        #if agent_idx == 0 :
        #    return 0
        if (room_id, level) not in self.I.div_exclude_rooms and (score >  self.I.socre_baseline or self.load_ram):
            return 1
        else:
            return 0

        #if (room_id, level) in self.I.div_exclude_rooms or score < 7:
        #    return 0.
        #else:
        #    return 1


        #if self.I.stats["n_updates"] < 5 and  agent_idx !=0:
        #    return 0.
        #else:
        #    return 1
    def clac_train_div_mask(self, pos, agent_idx, score):
        x, y, room_id, nkeys, level = pos

        #return 1

        if (room_id, level) not in self.I.div_exclude_rooms and  (score > self.I.socre_baseline or self.load_ram):
            return 1
        else:
            return 0

    def clac_oracle_div(self, pos):
        x, y, room_id, nkeys, level = pos

        if room_id == 5:
            return -1
        else:
            return 0

    def get_rnd_mask(self, agent_idx, room_id):

        mask = np.zeros(self.I.num_agents)

        rnd_mask_prob = self.rnd_mask_prob

        #main_agent_count = self.I.agent_idx_count.get(0, 0)
        #if main_agent_count < self.I.epsiode_thresold:
        if self.I.step_count < self.I.step_thresold:
            rnd_mask_prob = 1.

        if self.rnd_mask_type=='prog':
            for i in range(agent_idx, self.I.num_agents):
                mask[i] = np.random.uniform() < rnd_mask_prob
        elif self.rnd_mask_type =='indep':
            mask[agent_idx] = np.random.uniform() < rnd_mask_prob
        elif self.rnd_mask_type == 'shared':
            for i in range(self.I.num_agents):
                mask[i] = np.random.uniform() < rnd_mask_prob


        #if room_id in [1,2,6,7,12,13,14,21,22,23]:
        #    mask[:] = 0

        return mask

    def update_obs_rms(self, x, agent_idx):
        
        if self.rnd_mask_type=='prog':
            for i in range(agent_idx, self.I.num_agents):
                self.stochpol.ob_rms_list[i].update(x)
        elif self.rnd_mask_type =='indep':
            self.stochpol.ob_rms_list[agent_idx].update(x)
        elif self.rnd_mask_type == 'shared':
            for i in range(self.I.num_agents):
                self.stochpol.ob_rms_list[i].update(x)



    def sample_agent_idx(self, env_idx, room_id=-1, score=-1):


        #self.I.ep_step_count[k] % 1000

        if score==3 or self.I.divexp_flag[env_idx]:
        #if room_id ==6 or self.I.divexp_flag[env_idx]:

            envs_per_agent = self.I.nenvs // self.I.num_agents

            agent_idx = env_idx // envs_per_agent

            self.I.divexp_flag[env_idx] = True
        else:
            agent_idx = 0



        envs_per_agent = self.I.nenvs // self.I.num_agents
        agent_idx = env_idx // envs_per_agent

        #agent_idx = (self.I.ep_step_count[env_idx] // self.nsteps) % self.I.num_agents

        count = self.I.agent_idx_count.get(agent_idx, 0)
        self.I.agent_idx_count[agent_idx] = count + 1


        return agent_idx

    def update_oracle_count(self,agent_idx, pos):


        if self.indep_rnd==False:
            agent_idx = 0

        rew = self.I.oracle_visited_count_list[agent_idx].update_position(pos)
        return rew

    def update_oracle_rew_count(self, agent_idx, rew_key):

        '''
        if self.div_type =='rew':
            rew = self.I.oracle_reward_count_list[agent_idx].get_reward(rew_key)
    
            for i in range(agent_idx + 1, self.I.num_agents):
                self.I.oracle_reward_count_list[i].update_key(rew_key)
        else:
        '''
        rew = 1.0

        return rew

    @logger.profile("filter_rew")
    def filter_rew(self,rew, unclip_reward, pos, open_door_type, env_id, agent_idx):


        if rew > 0:
            unique_rew = uniqueReward(unclip_reward, pos, open_door_type)

            rew_weight = self.update_oracle_rew_count(agent_idx, unique_rew)

            rew = rew * rew_weight

            x, y, room_id, nkeys, level = pos


            ''' 
            if unique_rew in self.I._exclude_rews or room_id ==5:
                rew = 0
                logger.info("ignored_rew: {} {}".format(unique_rew,str(agent_idx)))
            '''

    


            rew_key = unique_rew #+ (agent_idx,)

            rew_count = self.I.reward_set_list[agent_idx].get(rew_key,0) + 1
            self.I.reward_set_list[agent_idx][rew_key] = rew_count
    
            self.I.rews_found_by_cur_policy_in_one_episode[env_id].append(unique_rew)
    

            #if rew_count == 100:
            #    self.save_check_point(self.I.rews_found_by_cur_policy_in_one_episode[env_id])
        else:
            unique_rew = {0,0,0,0}

        return rew, unique_rew

    def save_help_info(self, save_path, n_updates):

        for i in range(self.I.num_agents):
            path = '{}_{}_agent_{}_rff_int'.format(save_path,str(n_updates),str(i))
            self.I.rff_int_list[i].save(path)
            path = '{}_{}_agent_{}_rff_rms'.format(save_path,str(n_updates),str(i))
            self.I.rff_rms_int_list[i].save(path)
            path = '{}_{}_agent_{}_ob_rms'.format(save_path,str(n_updates),str(i))
            self.stochpol.ob_rms_list[i].save(path)

            path = '{}_{}_agent_{}_rff_div'.format(save_path,str(n_updates),str(i))
            self.I.rff_div_list[i].save(path)
            path = '{}_{}_agent_{}_rff_rms_div'.format(save_path,str(n_updates),str(i))
            self.I.rff_rms_div_list[i].save(path)

    def load_help_info(self, agent_idx, load_path):

        envs_per_agent = self.I.nenvs // self.I.num_agents
        #for i in range(self.I.num_agents):
        path = '{}_agent_{}_rff_int'.format(load_path,str(agent_idx))
        self.I.rff_int_list[agent_idx].load(path, envs_per_agent)
        #print(self.I.rff_int_list[i].rewems)
        path = '{}_agent_{}_rff_rms'.format(load_path,str(agent_idx))
        self.I.rff_rms_int_list[agent_idx].load(path)
        #print(self.I.rff_rms_int_list[i].count)
        path = '{}_agent_{}_ob_rms'.format(load_path,str(agent_idx))
        self.stochpol.ob_rms_list[agent_idx].load(path)

        self.stochpol.ob_rms.load(path)
        #print(self.stochpol.ob_rms_list[i].count)

    def load_agent(self, agent_idx, load_path):



        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/c1'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/c2'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/c3'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/fc1'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/fc_additional'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/fc2val'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/fc2act'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/pd'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/vf_int'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pol/agent_{}/vf_ext'.format(agent_idx))
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/c1r') + \
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/c2r') + \
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/c3r') + \
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/fc1r') + \
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div/fc2r')
                   #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div')
 
        self.load_variables(load_path, var_list)

    def load_rnd(self, agent_idx, load_path):

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/target_net_{}'.format(agent_idx)) + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/pred_net_{}'.format(agent_idx)) 
        
        self.load_variables(load_path, var_list)

    def load_sd(self,load_path):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/sd')
        logger.info("load sd weights from {}".format(load_path))
        self.load_variables(load_path, var_list)

    def clone_baseline_agent(self,src):
        #load baseline policy from agent 0
        self.sess.run(self.clone_base_op[src])

    def clone_agent(self, agent_idx, rnd=True, policy=True, help_info=True):


        #clone policy&&rnd
        for i in range(self.I.num_agents):
            if i != agent_idx:
                if rnd:
                    self.sess.run([self.clone_rnd_op[agent_idx][i]])
                if policy:
                    self.sess.run([self.clone_policy_op[agent_idx][i]])

                #insert noise
                #self.sess.run([self.insert_noise_op[i]])

        #clone help info

        if help_info==False:
            return

        for i in range(self.I.num_agents):
            if i != agent_idx:

                self.I.rff_int_list[i].copy(self.I.rff_int_list[agent_idx])
                self.I.rff_rms_int_list[i].copy(self.I.rff_rms_int_list[agent_idx])
                self.stochpol.ob_rms_list[i].copy(self.stochpol.ob_rms_list[agent_idx])

        
        
        
        
    def initialize_discriminator(self):

        old_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div')

        '''
        for old in sorted(old_vars, key=lambda v: v.name):
            print(old.name, old.eval())
            if 'w' in old.name:
                break
        '''
        self.sess.run(tf.variables_initializer(old_vars))

        new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/div')


        '''
        for new in sorted(new_vars, key=lambda v: v.name):
            print(new.name, new.eval())
            if 'w' in new.name:
                break
        '''

    def save_check_point(self,rews_list):
        #save policy
        path = '{}_gen{}_policy{}'.format(self.save_path,str(self.I.gen_idx),str(self.I.policy_idx))
        logger.log("save model:",path)
        self.save(path)
        #save rnd
        path='{}_gen{}_policy{}_rnd'.format(self.save_path,str(self.I.gen_idx),str(self.I.policy_idx))
        self.I.oracle_visited_count.save(path)
        #save rews_list

        logger.info(rews_list)
        path='{}_gen{}_policy{}_rewslist'.format(self.save_path,str(self.I.gen_idx),str(self.I.policy_idx))
        save_rews_list(rews_list, path)

        self.I.policy_idx = self.I.policy_idx + 1

    # we check an episode when it ends
    def check_episode(self, env_id):


        epsiode_record = self.I.rews_found_by_cur_policy_in_one_episode[env_id]


        self.I.rews_found_by_cur_policy_in_one_episode[env_id] = []

    def clac_return(self, trajectory, traj_buffer, agent_idx, gamma=0.99):
        traj_len = len(trajectory)
        ret = 0
        flag = False

        has_add = False

        ret_list = np.zeros(traj_len)

        for i in range(traj_len-1, -1, -1):
            s,a,r, room_id = trajectory[i]

            if r > 0.01:
                r = 0.5

            ret = r + ret * gamma
         
            if room_id in [-1,0,1,2,6]:
                w_ret = 0.01
            else:
                w_ret = ret



            if self.div_type=='oracle':
                if r > 0.01 and room_id in [7]:
                    logger.info(room_id,r, w_ret, agent_idx)
                    flag = True
            else:                   
                if r > 0.01 and room_id not in [-1,0,1,2,6]:
                    logger.info(room_id,r, w_ret, agent_idx)
                    flag = True

            if flag and ret > 0:
                has_add = True
                #logger.info(room_id,r, ret)
                traj_buffer.add(s,a, w_ret, room_id)


        self.I.num_trajectors[agent_idx] += has_add

    def clac_l2(self, obs_em, agent_idx):
        obs_buffer = self.I.div_discr_pos_buffer_list[agent_idx]
        data_store = obs_buffer._storage

        len_data = len(data_store)

        min_dis = 1000
        for i in range(len_data):
            _, target_em ,_ ,_ = data_store[i]
            dis = np.linalg.norm(obs_em - target_em) / 64.

            if dis < min_dis:
                min_dis = dis
        return min_dis
class Clusters(object):
    def __init__(self, t):
        self._cluster_list = []
        self._room_set = []
        self._cluster_mean = []
        self._t =t
    def has(self, obs_em, debug=False):

        for i in range(len(self._cluster_list)):
            dis = np.linalg.norm(obs_em - self._cluster_mean[i]) / 32.
            if dis < self._t:
                if debug:
                    logger.info(dis)
                return True
        return False

    def update(self, obs_em, room_id):

        is_new = True

        min_dis = 100
        min_k = -1
        for i in range(len(self._cluster_list)):
            tmp = np.asarray(self._cluster_list[i])
            cluster_mean = self._cluster_mean[i]  
            dis = np.linalg.norm(obs_em - cluster_mean) / 32.

            if dis < min_dis:
                min_dis = dis
                min_k = i

            if dis < self._t:

                is_new = False
        if is_new:
            #print("new cluster")
            self._cluster_list.append([obs_em])
            self._cluster_mean.append(obs_em)
            self._room_set.append(set())
            self._room_set[-1].add(room_id)
        else:
            self._cluster_list[min_k].append(obs_em)
            self._cluster_mean[min_k] = np.mean(np.asarray(self._cluster_list[min_k]),axis=0)
            self._room_set[min_k].add(room_id)
            #print("belong to ",min_k)            

        #print("min_dis:", min_dis, " min_k: ", min_k)

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

    def save(self,path):
        f = open(path,'wb')
        pickle.dump(self,f)
        f.close()
    
    def load(self,path, nenvsperagents):
        f = open(path,'rb')
        t = pickle.load(f)

        src = t.rewems.copy()
        src_nenvsperagents = src.shape[0]
        if src_nenvsperagents < nenvsperagents:
            tmp = np.zeros((nenvsperagents,), np.float32)
            tmp[:src_nenvsperagents] = src
            tmp[src_nenvsperagents:] = src[0]
            src = tmp

        self.rewems = src[:nenvsperagents]
        f.close()

    def copy(self,others):
        self.rewems = others.rewems.copy()

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
