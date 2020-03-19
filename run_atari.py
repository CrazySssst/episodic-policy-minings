#!/usr/bin/env python3
import functools
import os

from baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent, load_rews_list, Clusters
from utils import set_global_seeds
from vec_env import VecFrameStack, CuriosityEnvWrapperFrameStack
from episodic_curiosity import episodic_memory
from episodic_curiosity import r_network
from episodic_curiosity import r_network_training

from episodic_curiosity import oracle

from third_party.keras_resnet import models

import numpy as np
import time

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def train(*, env_id, num_env, hps, num_timesteps, seed):

    venv = VecFrameStack(
        make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(),
                       start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                       max_episode_steps=hps.pop('max_episode_steps')),
        hps.pop('frame_stack'))

    # Size of states when stored in the memory.
    only_train_r =  hps.pop('only_train_r')

    online_r_training = hps.pop('online_train_r') or only_train_r

    r_network_trainer = None
    save_path = hps.pop('save_path')
    r_network_weights_path = hps.pop('r_path')

    '''
    ec_type = 'none' # hps.pop('ec_type')

    venv = CuriosityEnvWrapperFrameStack(
        make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(),
                       start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                       max_episode_steps=hps.pop('max_episode_steps')),
        vec_episodic_memory = None,
        observation_embedding_fn = None,
        exploration_reward = ec_type,
        exploration_reward_min_step = 0,
        nstack = hps.pop('frame_stack'),
        only_train_r = only_train_r
        )
    '''
    
    # venv.score_multiple = {'Mario': 500,
    #                        'MontezumaRevengeNoFrameskip-v4': 100,
    #                        'GravitarNoFrameskip-v4': 250,
    #                        'PrivateEyeNoFrameskip-v4': 500,
    #                        'SolarisNoFrameskip-v4': None,
    #                        'VentureNoFrameskip-v4': 200,
    #                        'PitfallNoFrameskip-v4': 100,
    #                        }[env_id]
    venv.score_multiple = 1
    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')

    log_interval = hps.pop('log_interval')


    nminibatches=hps.pop('nminibatches')

    play = hps.pop('play')

    if play:
        nsteps=1


    rnd_type = hps.pop('rnd_type')
    div_type = hps.pop('div_type')

    num_agents = hps.pop('num_agents')

    load_ram = hps.pop('load_ram')

    debug = hps.pop('debug')

    rnd_mask_prob = hps.pop('rnd_mask_prob')

    rnd_mask_type = hps.pop('rnd_mask_type')


    indep_rnd = hps.pop('indep_rnd')
    logger.info("indep_rnd:",indep_rnd)
    indep_policy = hps.pop('indep_policy')

    sd_type = hps.pop('sd_type')

    from_scratch = hps.pop('from_scratch')

    use_kl = hps.pop('use_kl')

    save_interval = 100

    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]
    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
                scope='pol',
                ob_space=ob_space,
                ac_space=ac_space,
                update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
                proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
                dynamics_bonus = hps.pop("dynamics_bonus"),
                num_agents = num_agents,
                rnd_type = rnd_type,
                div_type= div_type,
                indep_rnd = indep_rnd,
                indep_policy = indep_policy,
                sd_type = sd_type,
                rnd_mask_prob = rnd_mask_prob
            ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        gamma_div=hps.pop('gamma_div'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=nminibatches,
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=5 if debug else 128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
        log_interval = log_interval,
        only_train_r = only_train_r,
        rnd_type= rnd_type,
        reset = hps.pop('reset'),
        dynamics_sample = hps.pop('dynamics_sample'),
        save_path = save_path,
        num_agents = num_agents,
        div_type = div_type,
        load_ram = load_ram,
        debug = debug,
        rnd_mask_prob = rnd_mask_prob,
        rnd_mask_type = rnd_mask_type,
        sd_type = sd_type,
        from_scratch = from_scratch,
        use_kl = use_kl,
        indep_rnd = indep_rnd
    )

    load_path = hps.pop('load_path')
    base_load_path = hps.pop('base_load_path')


    agent.start_interaction([venv])
    if load_path is not None:


        if play:
            agent.load(load_path)
        else:
            #agent.load(load_path)
            #agent.load_help_info(0, load_path)
            #agent.load_help_info(1, load_path)

            #load diversity agent
            #base_agent_idx = 1
            #logger.info("load base  agents weights from {}  agent {}".format(base_load_path, str(base_agent_idx)))
            #agent.load_agent(base_agent_idx, base_load_path)
            #agent.clone_baseline_agent(base_agent_idx)
                #agent.load_help_info(0, dagent_load_path)
                #agent.clone_agent(0)

            #load main agen1
            src_agent_idx = 1

            logger.info("load main agent weights from {} agent {}".format(load_path, str(src_agent_idx)))
            agent.load_agent(src_agent_idx, load_path)

           
            if indep_rnd==False:
                rnd_agent_idx = 1
            else:
                rnd_agent_idx = src_agent_idx
            #rnd_agent_idx = 0
            logger.info("load rnd weights from {} agent {}".format(load_path, str(rnd_agent_idx)))
            agent.load_rnd(rnd_agent_idx,load_path)
            agent.clone_agent(rnd_agent_idx, rnd=True, policy=False, help_info = False)
            
            logger.info("load help info from {} agent {}".format(load_path, str(src_agent_idx)))
            agent.load_help_info(src_agent_idx, load_path)
        
            agent.clone_agent(src_agent_idx,rnd=False, policy=True, help_info = True) 

            #logger.info("load main agent weights from {} agent {}".format(load_path, str(2)))
            
            #load_path = '/data/xupeifrom7700_1000/seed1_log0.5_clip-0.5~0.5_3agent_hasint4_2divrew_-1~1/models'
            #agent.load_agent(1, load_path)            
            
           
            #agent.clone_baseline_agent()
            #if sd_type =='sd':
            #    agent.load_sd("save_dir/models_sd_trained")

        #agent.initialize_discriminator()

    
    update_ob_stats_from_random_agent = hps.pop('update_ob_stats_from_random_agent')
    if play==False:

        if load_path is not None:
            pass#agent.collect_statistics_from_model()
        else:
            if update_ob_stats_from_random_agent and rnd_type=='rnd':
                agent.collect_random_statistics(num_timesteps=128*5 if debug else 128*50)
        assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())
    

        #agent.collect_rnd_info(128*50)
        '''
        if sd_type=='sd':
            agent.train_sd(max_nepoch=300, max_neps=5)
            path = '{}_sd_trained'.format(save_path)
            logger.log("save model:",path)
            agent.save(path)

            return
            #agent.update_diverse_agent(max_nepoch=1000)
            #path = '{}_divupdated'.format(save_path)
            #logger.log("save model:",path)
            #agent.save(path)
        '''
        counter = 0
        while True:
            info = agent.step()
    
            n_updates = agent.I.stats["n_updates"]
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
                counter += 1
    
    
            if info['update'] and save_path is not None and (n_updates%save_interval==0 or n_updates ==1):
                path = '{}_{}'.format(save_path,str(n_updates))
                logger.log("save model:",path)
                agent.save(path)
                agent.save_help_info(save_path,n_updates)
    
            if agent.I.stats['tcount'] > num_timesteps:
                path = '{}_{}'.format(save_path,str(n_updates))
                logger.log("save model:",path)
                agent.save(path)
                agent.save_help_info(save_path,n_updates)
                break
        agent.stop_interaction()
    else:

        '''
        check_point_rews_list_path ='{}_rewslist'.format(load_path)
        check_point_rnd_path ='{}_rnd'.format(load_path)
        oracle_rnd = oracle.OracleExplorationRewardForAllEpisodes()
        oracle_rnd.load(check_point_rnd_path)
        #print(oracle_rnd._collected_positions_writer)
        #print(oracle_rnd._collected_positions_reader)

        rews_list = load_rews_list(check_point_rews_list_path)
        print(rews_list)
        '''

        istate = agent.stochpol.initial_state(1)
        #ph_mean, ph_std = agent.stochpol.get_ph_mean_std()

        last_obs, prevrews, ec_rews, news, infos, ram_states, _ = agent.env_get(0)
        agent.I.step_count += 1

        flag = False
        show_cam = True

        last_xr = 0

        restore = None

        '''
        #path = 'ram_state_500_7room'
        #path='ram_state_400_6room'
        #path='ram_state_6700' 
        path='ram_state_7700_10room'
        f = open(path,'rb')
        restore = pickle.load(f)
        f.close()
        last_obs[0] = agent.I.venvs[0].restore_full_state_by_idx(restore,0)
        print(last_obs.shape)

        #path = 'ram_state_400_monitor_rews_6room'
        #path = 'ram_state_500_monitor_rews_7room'
        #path='ram_state_6700_monitor_rews'
        path='ram_state_7700_monitor_rews_10room'
        f = open(path,'rb')
        monitor_rews = pickle.load(f)
        f.close()
        
        agent.I.venvs[0].set_cur_monitor_rewards_by_idx(monitor_rews,0)
        '''

        agent_idx = np.asarray([0])
        sample_agent_prob = np.asarray([0.5])

        ph_mean = agent.stochpol.ob_rms_list[0].mean
        ph_std = agent.stochpol.ob_rms_list[0].var ** 0.5

        buf_ph_mean = np.zeros(([1, 1] + list(agent.stochpol.ob_space.shape[:2])+[1]), np.float32)
        buf_ph_std =  np.zeros(([1, 1] + list(agent.stochpol.ob_space.shape[:2])+[1]), np.float32)

        buf_ph_mean[0,0] = ph_mean
        buf_ph_std[0,0] = ph_std

        vpreds_ext_list = []

        ep_rews = np.zeros((1))
        divexp_flag = False
        step_count = 0
        stage_prob = True

        last_rew_ob = np.full_like(last_obs,128)

        clusters = Clusters(1.0)



        #path = '{}_sd_rms'.format(load_path)
        #agent.I.sd_rms.load(path)

        while True:
            
            dict_obs = agent.stochpol.ensure_observation_is_dict(last_obs)

            #acs= np.random.randint(low=0, high=15, size=(1))
            acs, vpreds_int, vpreds_ext, nlps, istate, ent = agent.stochpol.call(dict_obs, news, istate, agent_idx[:,None])






            step_acs = acs
            t=''
            #if show_cam==True:
            t = input("input:")
            if t!='':
                t=int(t)
                if t<=17:
                    step_acs=[t]


            agent.env_step(0, step_acs)
           
            obs, prevrews, ec_rews, news, infos, ram_states, monitor_rews = agent.env_get(0)

            if news[0] and restore is not None:
                obs[0] = agent.I.venvs[0].restore_full_state_by_idx(restore,0)
                agent.I.venvs[0].set_cur_monitor_rewards_by_idx(monitor_rews,0)


            ep_rews = ep_rews + prevrews

            print(ep_rews)

            last_rew_ob[prevrews>0] = obs[prevrews>0]

            room = infos[0]['position'][2]
            vpreds_ext_list.append([vpreds_ext,room])
            #print(monitor_rews[0])
            #print(len(monitor_rews[0]))
            #print(infos[0]['open_door_type'])


            stack_obs = np.concatenate([last_obs[:,None],obs[:,None]],1)

            fd = {}
    
            fd[agent.stochpol.ph_ob[None]] = stack_obs

            fd.update({agent.stochpol.sep_ph_mean: buf_ph_mean,
                        agent.stochpol.sep_ph_std: buf_ph_std})
            fd[agent.stochpol.ph_agent_idx] = agent_idx[:,None]
            fd[agent.stochpol.sample_agent_prob] = sample_agent_prob[:,None]

            fd[agent.stochpol.last_rew_ob] = last_rew_ob[:,None]
            fd[agent.stochpol.game_score] = ep_rews[:,None]

            fd[agent.stochpol.sd_ph_mean] = agent.I.sd_rms.mean
            fd[agent.stochpol.sd_ph_std] = agent.I.sd_rms.var ** 0.5
            


            div_prob = 0

            all_div_prob = tf_util.get_session().run([agent.stochpol.all_div_prob ], fd)
            
            '''
            if prevrews[0] > 0:
                clusters.update(rnd_em,room)
    
                num_clusters = len(clusters._cluster_list)
                for i in range(num_clusters):
                    print("{} {}".format(str(i),list(clusters._room_set[i])))
            '''
            print( "vpreds_int: ", vpreds_int, "vpreds_ext:", vpreds_ext, "ent:", ent, "all_div_prob:", all_div_prob, "room:", room,"step_count:", step_count)
            
            #aaaa = np.asarray(vpreds_ext_list)
            #print(aaaa[-100:])

            '''
            if step_acs[0]==0:
                ram_state = ram_states[0]
                path='ram_state_7700_10room'
                f = open(path,'wb')
                pickle.dump(ram_state,f)
                f.close()

                path='ram_state_7700_monitor_rews_10room'
                f = open(path,'wb')
                pickle.dump(monitor_rews[0],f)
                f.close()
            '''
            '''
            if  restore is None:
                restore = ram_states[0]

            
            if np.random.rand() < 0.1:
                print("restore")
                obs = agent.I.venvs[0].restore_full_state_by_idx(restore,0)
                prevrews = None
                ec_rews = None
                news= True
                infos = {}
                ram_states = ram_states[0]

                #restore = ram_states[0]
            '''



            img = agent.I.venvs[0].render()

            last_obs = obs

            step_count =step_count +1






            time.sleep(0.04)

def add_env_params(parser):
    #VentureNoFrameskip-v4 MontezumaRevengeNoFrameskip-v4 GravitarNoFrameskip-v4
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    parser = arg_parser()
    add_env_params(parser)
    parser.add_argument('--num_timesteps', type=float, default=100e6)
    parser.add_argument('--num_env', type=int, default=128)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_ext', type=float, default=0.99)
    parser.add_argument('--gamma_div', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=1)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_updated', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='cnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=2.)
    parser.add_argument('--dynamics_bonus', type=int, default=0)
    parser.add_argument('--save_dir',help="dir to save and log",type=str, default="save_dir")
    parser.add_argument('--load_path',help="dir to load model",type=str, default=None)
    parser.add_argument('--base_load_path',help="dir to load model",type=str, default=None)
    parser.add_argument('--r_path',help="dir to load r network",type=str, default=None)

    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--only_train_r', default=False, action='store_true')
    parser.add_argument('--online_train_r', default=False, action='store_true')
    #parser.add_argument('--ec_type', type=str, default='episodic_curiosity', choices=['episodic_curiosity', 'none','oracle'])
    parser.add_argument('--rnd_type', type=str, default='rnd', choices=['rnd','oracle'])
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--dynamics_sample', default=False, action='store_true')
    
    parser.add_argument('--num_agents', type=int, default=1)

    parser.add_argument('--div_type', type=str, default='oracle', choices=['oracle','cls','rnd'])
    parser.add_argument('--load_ram', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--rnd_mask_prob', type=float, default=1.)
    parser.add_argument('--rnd_mask_type', type=str, default='indep', choices=['prog','indep','shared'])
    parser.add_argument('--indep_rnd', default=False ,action='store_true')
    parser.add_argument('--indep_policy', default=True, action='store_true')
    parser.add_argument('--sd_type', type=str, default='oracle', choices=['oracle','sd'])
    parser.add_argument('--from_scratch', default=False, action='store_true')

    parser.add_argument('--kl', default=False, action='store_true')
    

    
    
    

    args = parser.parse_args()

    log_path = os.path.join(args.save_dir,'logs')
    save_path = os.path.join(args.save_dir,'models')

    logger.configure(dir=log_path, format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
            f.write(args.tag)
        # shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(logger.get_dir(), 'code'))

    mpi_util.setup_mpi_gpus()

    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        gamma_div=args.gamma_div,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_updated,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        dynamics_bonus = args.dynamics_bonus,
        log_interval = 10,
        save_path = save_path,
        load_path = args.load_path,
        r_path = args.r_path,
        play = args.play,
        only_train_r = args.only_train_r,
        online_train_r = args.online_train_r,
        #ec_type = args.ec_type,
        rnd_type = args.rnd_type,
        reset = args.reset,
        dynamics_sample = args.dynamics_sample,
        num_agents = args.num_agents,
        div_type= args.div_type,
        load_ram= args.load_ram,
        debug = args.debug,
        rnd_mask_prob = args.rnd_mask_prob,
        rnd_mask_type = args.rnd_mask_type,
        indep_rnd = args.indep_rnd,
        indep_policy = args.indep_policy,
        sd_type = args.sd_type,
        from_scratch = args.from_scratch,
        base_load_path = args.base_load_path,
        use_kl = args.kl
    )

    if args.play:
        args.num_env = 1

    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
        num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
    main()
