import numpy as np
import tensorflow as tf
from baselines import logger
from utils import fc, conv, ortho_init
from stochastic_policy import StochasticPolicy
from tf_util import get_available_gpus
from mpi_util import RunningMeanStd

from baselines.common.distributions import CategoricalPdType

import tf_util

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))

def _fcnobias(x, scope, nh, *, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        return tf.matmul(x, w)
def _normalize(x):
    eps = 1e-5
    mean, var = tf.nn.moments(x, axes=(-1,), keepdims=True)
    return (x - mean) / tf.sqrt(var + eps)


class CnnPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 dynamics_bonus = False, num_agents = 1, rnd_type='rnd', div_type='oracle',
                 indep_rnd = False, indep_policy = False, sd_type='oracle',rnd_mask_prob=1.
                 ):
        StochasticPolicy.__init__(self, scope, ob_space, ac_space)
        self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update
        enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[policy_size]
        rep_size = 512

        self.rnd_mask = tf.placeholder(dtype=tf.float32, shape=(None,None,num_agents), name="rnd_mask")
        self.new_rnd_mask = tf.placeholder(dtype=tf.float32, shape=(None,None), name="new_rnd_mask")
        self.div_train_mask = tf.placeholder(dtype=tf.float32, shape=(None,None), name="div_train_mask")
        self.sample_agent_prob = tf.placeholder(dtype=tf.float32, shape=(None,None,), name="sample_agent_prob")
        self.stage_label = tf.placeholder(dtype=tf.int32, shape=(None,None), name="stage_label")


        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obstd")
        self.ph_count = tf.placeholder(dtype=tf.float32, shape=(), name="obcount")


        self.sep_ph_mean = tf.placeholder(dtype=tf.float32, shape= (None, None,) + ob_space.shape[:2] +(1,) , name="sep_obmean")
        self.sep_ph_std = tf.placeholder(dtype=tf.float32, shape= (None, None,) + ob_space.shape[:2] + (1,), name="sep_obstd")
        self.sep_ph_count = tf.placeholder(dtype=tf.float32, shape=(), name="sep_obcount")

        self.game_score = tf.placeholder(dtype=tf.float32, shape= (None, None) , name="game_score")
        self.last_rew_ob = tf.placeholder(dtype=ob_space.dtype, shape= (None, None) + tuple(ob_space.shape) , name="last_rew_ob")
        
        self.div_ph_mean = tf.placeholder(dtype=tf.float32, shape= list(ob_space.shape[:2])+[1] , name="div_obmean")
        self.div_ph_std = tf.placeholder(dtype=tf.float32, shape= list(ob_space.shape[:2])+[1], name="div_obstd")

        self.idle_agent_label = tf.placeholder(dtype=tf.int32, shape= (None, None,) , name="idle_agent_label")
        self.rew_agent_label = tf.placeholder(dtype=tf.int32, shape= (None, None,) , name="rew_agent_label")
       

        #self.var_ph_mean = tf.get_variable("var_ph_mean", list(ob_space.shape[:2])+[1], initializer=tf.constant_initializer(0.0))
        #self.var_ph_std = tf.get_variable("var_ph_std", list(ob_space.shape[:2])+[1], initializer=tf.constant_initializer(0.0))
        #self.var_ph_count = tf.get_variable("var_ph_count", (), initializer=tf.constant_initializer(0.0))



        self.sd_ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="sd_obmean")
        self.sd_ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="sd_obstd")

        memsize *= enlargement
        hidsize *= enlargement
        convfeat = 16*enlargement

        self.ob_rms_list = [RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi= not update_ob_stats_independently_per_gpu) \
                                for _ in range(num_agents)]
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi= not update_ob_stats_independently_per_gpu)

        self.diversity_ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi= not update_ob_stats_independently_per_gpu)

        ph_istate = tf.placeholder(dtype=tf.float32,shape=(None,memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]

        self.memsize = memsize
        self.num_agents = num_agents
        self.indep_rnd = indep_rnd
        self.indep_policy = indep_policy

        #Inputs to policy and value function will have different shapes depending on whether it is rollout
        #or optimization time, so we treat separately.

        if  num_agents <= 0:

            self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
                self.apply_policy(self.ph_ob[None][:,:-1],
                                  reuse=False,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps - 1,
                                  pdparamsize=pdparamsize
                                  )
            self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
                self.apply_policy(self.ph_ob[None],
                                  reuse=True,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps,
                                  pdparamsize=pdparamsize
                                  )

        else:

            self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt, self.all_pdparam  = \
                self.apply_multi_head_policy(self.ph_ob[None][:,:-1],
                                  reuse=False,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps - 1,
                                  pdparamsize=pdparamsize
                                  )
            self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout, _ = \
                self.apply_multi_head_policy(self.ph_ob[None],
                                  reuse=True,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps,
                                  pdparamsize=pdparamsize
                                  )



        '''
        self.base_pdparam,self.base_vpred_int, self.base_vpred_ext =  self._build_baseline_policy(self.ph_ob[None][:,:],
                                    reuse=False,
                                  scope='baseline_agent',
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps,
                                  pdparamsize=pdparamsize
                                  )
        '''
        if dynamics_bonus:
            self.define_dynamics_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
        else:
            #self.define_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
            logger.info("self.indep_rnd:",self.indep_rnd)
            if self.indep_rnd:
                aux_loss, self.int_rew, self.feat_var, self.max_feat = self.define_multi_head_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
    
                #[env*step, num_agents ,1] 
                rnd_mask =  tf.reshape(self.rnd_mask,(-1, self.num_agents,1))
                rnd_mask = tf.cast(rnd_mask, tf.float32)
                #[env*step, num_agents , 1] -> [env*step]
                self.aux_loss = tf.reduce_sum(rnd_mask * aux_loss) / tf.maximum(tf.reduce_sum(rnd_mask), 1.)
            else:
                aux_loss, self.int_rew, self.feat_var, self.max_feat = self.single_head_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
 

                mask = tf.random_uniform(shape=tf.shape(aux_loss), minval=0., maxval=1., dtype=tf.float32)
                mask = tf.cast(mask < rnd_mask_prob, tf.float32)
                self.aux_loss = tf.reduce_sum(mask * aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)


        self.div_rew = tf.constant(0.)

        if div_type =='cls':
            with tf.variable_scope("div", reuse=False):
                #self.define_rew_discriminator(convfeat=convfeat, rep_size=256)
                with tf.variable_scope("int", reuse=False):
                   self.disc_logits, self.all_div_prob, self.sp_prob, self.div_rew,  self.disc_pd , self.disc_nlp  =  self.define_rew_discriminator_v2(convfeat=convfeat, rep_size=512, use_rew = True)
                #with tf.variable_scope("rew", reuse=False):
                #self.define_rew_discriminator(convfeat=convfeat, rep_size=256)
                #   self.rew_disc_logits, self.rew_all_div_prob, self.rew_sp_prob, self.rew_div_rew,  self.rew_disc_pd , self.rew_disc_nlp =  self.define_rew_discriminator_v2(convfeat=convfeat, rep_size=512, use_rew = True)


        self.stage_rnd = tf.constant(1.)
        self.stage_prob = tf.constant(1.)

        if sd_type=='sd':
          with tf.variable_scope("sd", reuse=False):
            #self.define_stage_rnd(convfeat=convfeat, rep_size=512, enlargement=enlargement)
            self.define_stage_discriminator(convfeat=convfeat, rep_size=512)


        #with tf.variable_scope("rand_em"):
        #  self.rnd_em = self.define_state_embedding(self.ph_ob[None][:,:-1])

        pd = self.pdtype.pdfromflat(self.pdparam_rollout)
        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)

        self.ph_istate = ph_istate




        one_hot_gidx = tf.one_hot(1-self.ph_agent_idx, self.num_agents, axis=-1)
        #[batch,nstep, ngroups] -> [batch * nstep, ngroups,1]
        one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_agents,1))
        self.other_pdparam = tf.reduce_sum(one_hot_gidx * self.all_pdparam,axis=1)
        self.other_pdparam = tf.reshape(self.other_pdparam, (self.sy_nenvs, self.sy_nsteps - 1, pdparamsize))

        '''
        self.agent0_pdparam = self.all_pdparam[:,0] #tf.reduce_sum(defulat_one_hot * self.all_pdparam,axis=1)
        self.agent0_pdparam = tf.reshape(self.agent0_pdparam, (self.sy_nenvs, self.sy_nsteps -1, pdparamsize))

        self.agent1_pdparam = self.all_pdparam[:,1]  #tf.reduce_sum(defulat_one_hot * self.all_pdparam,axis=1)
        self.agent1_pdparam = tf.reshape(self.agent1_pdparam, (self.sy_nenvs, self.sy_nsteps - 1, pdparamsize))
        '''
  
    @staticmethod
    def apply_policy(ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())
        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            mix_other_observations = [X]
            X = tf.concat(mix_other_observations, axis=1)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            additional_size = 448
            X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
            snext = tf.zeros((sy_nenvs, memsize))
            mix_timeout = [X]

            Xtout = tf.concat(mix_timeout, axis=1)
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                X     = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int   = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext   = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext


    #@staticmethod
    def apply_multi_head_policy(self, ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C

        #
        goal_ph = ph #tf.concat([ph,self.goal_ob[:,:,:,:,-1:]], axis=-1)

        logger.info("CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(goal_ph.shape)))
        X = tf.cast(goal_ph, tf.float32) / 255.
        input_X = X = tf.reshape(X, (-1, *goal_ph.shape.as_list()[-3:]))

        #X = tf.cast(ph, tf.float32)
        #X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))#[:, :, :, -1:]
        #X = tf.clip_by_value((X - self.ph_mean) / self.ph_std, -5.0, 5.0)

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())

        #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        #    mse = self.RND_backup(ph_ob, convfeat=self.convfeat, rep_size=self.rep_size, enlargement=self.enlargement)


        #one_hot_gidx = tf.one_hot(self.ph_groups_idx, self.num_groups, axis=-1)
        #one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_groups))

        additional_size = 448
        snext = tf.zeros((sy_nenvs, memsize))
        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):


            if self.indep_policy==False:

              X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
              X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
              X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
  
              X = to2d(X)
  
              X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
  
              #X = tf.concat([X, one_hot_gidx], axis=1)
              
              X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
              
              #mix_timeout = [X]
  
              if extrahid:
                  Xtout = X
                  Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                  X     = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))

            all_pdparam=[]
            all_vint=[]
            all_vext=[]

            #vpred_ext   = self._build_val(Xtout, scope="Vext", reuse=False, nh=additional_size)
            for i in range(self.num_agents):

                if self.indep_policy: 
                    scope = 'agent_{}'.format(str(i))
                    pdparam, vpred_int, vpred_ext = self._build_policy_net(input_X, scope=scope, reuse=False, 
                                                                                    hidsize=hidsize,
                                                                                    memsize=memsize, 
                                                                                    extrahid=extrahid, 
                                                                                    sy_nenvs=sy_nenvs,
                                                                                    sy_nsteps=sy_nsteps,
                                                                                    pdparamsize=pdparamsize)

                else:
                    scope = 'agent_{}'.format(str(i))
    
                    pdparam, vpred_int, vpred_ext = self._build_policy_head(X, Xtout,scope=scope, reuse=False,
                                                                                 additional_size=additional_size, 
                                                                                 pdparamsize=pdparamsize)

                if i==0:
                    #[batch,naction] - > [batch, 1, naction]
                    all_pdparam = tf.expand_dims(pdparam, axis=1)
                    #[batch,1] -> [batch,1,1]
                    all_vint = tf.expand_dims(vpred_int, axis=1)
                    all_vext = tf.expand_dims(vpred_ext, axis=1)
                else:
                    all_pdparam = tf.concat([all_pdparam, tf.expand_dims(pdparam, axis=1)], axis=1)
                    all_vint = tf.concat([all_vint,  tf.expand_dims(vpred_int, axis=1)], axis=1)
                    all_vext = tf.concat([all_vext, tf.expand_dims(vpred_ext, axis=1)], axis=1)

            #[batch, nstep] -> [batch,nstep, ngroups]
            one_hot_gidx = tf.one_hot(self.ph_agent_idx, self.num_agents, axis=-1)
            #[batch,nstep, ngroups] -> [batch * nstep, ngroups,1]
            one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_agents,1))



            pdparam = tf.reduce_sum(one_hot_gidx * all_pdparam,axis=1)
            vpred_int = tf.reduce_sum(one_hot_gidx * all_vint,axis=1)
            vpred_ext = tf.reduce_sum(one_hot_gidx * all_vext,axis=1)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))


        return pdparam, vpred_int, vpred_ext, snext, all_pdparam

    def _build_baseline_policy(self, ph_ob, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):


        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C



        logger.info("baseline-CnnPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        input_X = X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        scope = 'base'
        pdparam, vpred_int, vpred_ext = self._build_policy_net(input_X, scope=scope, reuse=reuse, 
                                                                        hidsize=hidsize,
                                                                        memsize=memsize, 
                                                                        extrahid=extrahid, 
                                                                        sy_nenvs=sy_nenvs,
                                                                        sy_nsteps=sy_nsteps,
                                                                        pdparamsize=pdparamsize)


        pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
        vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
        vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))

        return pdparam, vpred_int, vpred_ext

    def _build_policy_net(self, X, scope ,reuse, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize):

        activ = tf.nn.relu

        data_format = 'NHWC'

        with tf.variable_scope(scope, reuse=reuse):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            mix_other_observations = [X]
            X = tf.concat(mix_other_observations, axis=1)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            additional_size = 448
            X = activ(fc(X, 'fc_additional', nh=additional_size, init_scale=np.sqrt(2)))
            snext = tf.zeros((sy_nenvs, memsize))
            mix_timeout = [X]

            Xtout = tf.concat(mix_timeout, axis=1)
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=additional_size, init_scale=0.1))
                X     = X + activ(fc(X, 'fc2act', nh=additional_size, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int   = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext   = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

        return pdparam, vpred_int, vpred_ext

    def _build_policy_head(self, X, Xtout, scope, reuse, additional_size, pdparamsize):

        with tf.variable_scope(scope, reuse=reuse):

            policy_scope = '{}_policy'.format(str(scope))
            pdparam = self._build_policy(X, scope=policy_scope, reuse=reuse, nh=additional_size, pdparamsize=pdparamsize)

            Vint_scope = '{}_Vint'.format(str(scope))
            vpred_int   = self._build_val(Xtout, scope=Vint_scope, reuse=reuse, nh=additional_size)

            Vext_scope = '{}_Vext'.format(str(scope))
            vpred_ext   = self._build_val(Xtout, scope=Vext_scope, reuse=reuse, nh=additional_size)

        return pdparam, vpred_int, vpred_ext

    def _build_val(self,Xval, scope, reuse, nh):

        activ = tf.nn.relu
        with tf.variable_scope(scope, reuse=reuse):
            Xval = Xval + activ(fc(Xval, 'hval', nh=nh, init_scale=0.1))
    
            vpred   = fc(Xval, 'vf', nh=1, init_scale=0.01)

        return vpred


    def _build_policy(self,Xact, scope, reuse, nh, pdparamsize):

        activ = tf.nn.relu
        with tf.variable_scope(scope, reuse=reuse):
            Xact  = activ(fc(Xact, 'hact', nh=nh, init_scale=0.1))
            pdparam = fc(Xact, 'pd', nh=pdparamsize, init_scale=0.01)

        return pdparam




   
    def define_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using RND BONUS ****************************************************")

        #RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]

                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)


                with tf.variable_scope("target_net_0"):
                  xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                  xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                  xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                  rgbr = [to2d(xr)]
                  X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]


                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                with tf.variable_scope('pred_net_0'):
                  xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                  xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                  xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                  rgbrp = to2d(xrp)
                  # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                  X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                  X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                  X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def define_dynamics_prediction_rew(self, convfeat, rep_size, enlargement):
        #Dynamics loss with random features.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        ac_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        assert ac_one_hot.get_shape().ndims == 3
        assert ac_one_hot.get_shape().as_list() == [None, None, self.ac_space.n], ac_one_hot.get_shape().as_list()
        ac_one_hot = tf.reshape(ac_one_hot, (-1, self.ac_space.n))
        def cond(x):
            return tf.concat([x, ac_one_hot], 1)

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,:-1]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
                # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)

                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(rgbrp), 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(X_r_hat), 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(cond(X_r_hat), 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)


    def _build_target_net(self, target_x, scope, reuse, convfeat, rep_size, enlargement):

        with tf.variable_scope(scope, reuse=reuse):
            xr = tf.nn.leaky_relu(conv(target_x, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
            xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbr = [to2d(xr)]
            X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        return X_r

    def _build_pred_net(self, pred_x, scope, reuse, convfeat, rep_size, enlargement):

        with tf.variable_scope(scope, reuse=reuse):
            xrp = tf.nn.leaky_relu(conv(pred_x, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
            xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
            rgbrp = to2d(xrp)
            # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
            X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
            X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
            X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        return X_r_hat


    def single_head_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using single-head RND BONUS ****************************************************")

        #assert self.indep_rnd==False
        #RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]

                #last_rew_ob = self.last_rew_ob
                #last_rew_ob = tf.cast(last_rew_ob, tf.float32)
                #last_rew_ob = tf.reshape(last_rew_ob, (-1, *last_rew_ob.shape.as_list()[-3:]))[:, :, :, -1:]

                #xr = tf.concat([xr,last_rew_ob], axis=-1)


                ph_mean = tf.reshape(self.sep_ph_mean,(-1, *self.sep_ph_mean.shape.as_list()[-3:]))
                ph_std = tf.reshape(self.sep_ph_std,(-1, *self.sep_ph_std.shape.as_list()[-3:]))

                target_x = xr = tf.clip_by_value((xr - ph_mean) / ph_std, -5.0, 5.0)

                scope = 'target_net_{}'.format(str(1))
                target_out = self._build_target_net(target_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))


                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]


                #last_rew_ob = self.last_rew_ob
                #last_rew_ob = tf.cast(last_rew_ob, tf.float32)
                #last_rew_ob = tf.reshape(last_rew_ob, (-1, *last_rew_ob.shape.as_list()[-3:]))[:, :, :, -1:]

                #xrp = tf.concat([xrp,last_rew_ob], axis=-1)

                ph_mean = tf.reshape(self.sep_ph_mean,(-1, *self.sep_ph_mean.shape.as_list()[-3:]))
                ph_std = tf.reshape(self.sep_ph_std,(-1, *self.sep_ph_std.shape.as_list()[-3:]))

                pred_x = xrp = tf.clip_by_value((xrp - ph_mean) / ph_std, -5.0, 5.0)

                scope = 'pred_net_{}'.format(str(1))
                pred_out = self._build_pred_net(pred_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)



        int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(target_out) - pred_out), axis=-1, keep_dims=True)
        int_rew = tf.reshape(int_rew, (self.sy_nenvs, self.sy_nsteps - 1))



        aux_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target_out) - pred_out), -1)


        feat_var = tf.reduce_mean(tf.nn.moments(target_out, axes=[0])[1])
        max_feat = tf.reduce_max(tf.abs(target_out))


        return aux_loss, int_rew, feat_var, max_feat

    def define_multi_head_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using multi-head RND BONUS ****************************************************")

        assert self.indep_rnd#==True
        #RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]

                ph_mean = tf.reshape(self.sep_ph_mean,(-1, *self.sep_ph_mean.shape.as_list()[-3:]))
                ph_std = tf.reshape(self.sep_ph_std,(-1, *self.sep_ph_std.shape.as_list()[-3:]))

                target_x = xr = tf.clip_by_value((xr - ph_mean) / ph_std, -5.0, 5.0)

                all_target_out = []

                #target_out = self._build_target_net(target_x, 'target_net', False, convfeat, rep_size, enlargement)
                for i in range(self.num_agents):

                    scope = 'target_net_{}'.format(str(i))
                    target_out = self._build_target_net(target_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)#self._build_target_head(xr, scope, tf.AUTO_REUSE, rep_size)
                    if i==0:
                        #[env*step, rep_size] -> [env*step, 1, rep_size]
                        all_target_out = tf.expand_dims(target_out, axis=1)
                    else:
                        #[env*step, 1, rep_size] -> [env*step, num_agents , rep_size]
                        all_target_out = tf.concat([all_target_out, tf.expand_dims(target_out, axis=1)], axis=1)

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))


                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]

                ph_mean = tf.reshape(self.sep_ph_mean,(-1, *self.sep_ph_mean.shape.as_list()[-3:]))
                ph_std = tf.reshape(self.sep_ph_std,(-1, *self.sep_ph_std.shape.as_list()[-3:]))

                pred_x = xrp = tf.clip_by_value((xrp - ph_mean) / ph_std, -5.0, 5.0)

                all_pred_out = []
                for i in range(self.num_agents):

                    scope = 'pred_net_{}'.format(str(i))
                    pred_out = self._build_pred_net(pred_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)#self._build_pred_head(xrp, scope, tf.AUTO_REUSE, enlargement, rep_size)

                    if i==0:
                        #[env*step, rep_size] -> [env*step, 1, rep_size]
                        all_pred_out = tf.expand_dims(pred_out, axis=1)
                    else:
                        #[env*step, 1, rep_size] -> [env*step, num_agents , rep_size]
                        all_pred_out = tf.concat([all_pred_out, tf.expand_dims(pred_out, axis=1)], axis=1)

        #[env*step, num_agents , rep_size] -> [env*step, num_agents , 1]
        all_loss = tf.reduce_mean(tf.square(tf.stop_gradient(all_target_out) - all_pred_out), axis=-1, keep_dims=True)


        #[batch, nstep] -> [batch,nstep, ngroups]
        one_hot_gidx = tf.one_hot(self.ph_agent_idx, self.num_agents, axis=-1)
        #[batch,nstep, ngroups] -> [batch * nstep, ngroups,1]
        one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_agents,1))

        X_r = tf.reduce_sum(one_hot_gidx * all_target_out, axis=1)


        feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        max_feat = tf.reduce_max(tf.abs(X_r))
        #[env*step, num_agents , 1] -> [env*step, 1]
        int_rew = tf.reduce_sum(one_hot_gidx * all_loss, axis=1)
        int_rew = tf.reshape(int_rew, (self.sy_nenvs, self.sy_nsteps - 1))



        return all_loss, int_rew, feat_var, max_feat

    def _build_target_head(self, target_x, scope, reuse, rep_size):

        with tf.variable_scope(scope, reuse=reuse):
            x  = fc(target_x, 'fc2r', nh=rep_size, init_scale=np.sqrt(2))
        return x

    def _build_pred_head(self, pred_x, scope, reuse, enlargement, rep_size):

        with tf.variable_scope(scope, reuse=reuse):
            
            x = tf.nn.relu(fc(pred_x, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
            x = fc(x, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))
        return x        



    def define_rew_discriminator_v2(self, convfeat, rep_size,use_rew = False):

        output_shape = [self.sy_nenvs * (self.sy_nsteps - 1)]
        
        sample_prob = tf.reshape(self.sample_agent_prob,tf.stack(output_shape))
        game_score =  tf.reshape(self.game_score, tf.stack([self.sy_nenvs * (self.sy_nsteps - 1), 1]))


        rew_agent_label = tf.reshape(self.rew_agent_label, tf.stack([self.sy_nenvs * (self.sy_nsteps - 1), 1]))

        #rew_agent_label = tf.one_hot(self.rew_agent_label, self.num_agents, axis=-1)
        #rew_agent_label = tf.reshape(rew_agent_label,(-1,self.num_agents ))

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C

                phi = ph[:,1:]
                phi = tf.cast(phi, tf.float32)
                phi = tf.reshape(phi, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                phi = phi /255.
                

                last_rew_ob = self.last_rew_ob
                last_rew_ob = tf.cast(last_rew_ob, tf.float32)
                last_rew_ob = tf.reshape(last_rew_ob, (-1, *last_rew_ob.shape.as_list()[-3:]))[:, :, :, -1:]
                last_rew_ob = last_rew_ob /255.

                if use_rew:
                    phi = tf.concat([phi,last_rew_ob], axis=-1)

                phi = tf.nn.leaky_relu(conv(phi, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                #[20,20] [8,8]
                phi = tf.nn.leaky_relu(conv(phi, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                #[9,9] [7,7]
                phi = tf.nn.leaky_relu(conv(phi, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                phi = to2d(phi)
        
                phi = tf.nn.relu(fc(phi, 'fc1r', nh=rep_size, init_scale=np.sqrt(2)))
                phi = tf.nn.relu(fc(phi, 'fc2r', nh=rep_size, init_scale=np.sqrt(2)))
                disc_logits = fc(phi, 'fc3r', nh= self.num_agents, init_scale=np.sqrt(2))

        one_hot_gidx = tf.one_hot(self.ph_agent_idx, self.num_agents , axis=-1)
        one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_agents))



        flatten_all_div_prob = tf.nn.softmax(disc_logits, axis=-1)
        all_div_prob = tf.reshape(flatten_all_div_prob, (self.sy_nenvs, self.sy_nsteps - 1, self.num_agents))

        sp_prob = tf.reduce_sum(one_hot_gidx * flatten_all_div_prob, axis=1)
        sp_prob = tf.reshape(sp_prob, (self.sy_nenvs, self.sy_nsteps - 1))


        div_rew = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_logits, labels=one_hot_gidx)
        base_rew = tf.log(0.01)
        div_rew = div_rew - tf.log(sample_prob)

        div_rew = tf.reshape(div_rew, (self.sy_nenvs, self.sy_nsteps - 1))
        

        disc_pdtype = CategoricalPdType(self.num_agents)
        disc_pd = disc_pdtype.pdfromflat(disc_logits)

        disc_nlp = disc_pd.neglogp(rew_agent_label)

        return disc_logits, all_div_prob, sp_prob, div_rew, disc_pd , disc_nlp

        #self.disc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.disc_logits, labels=rew_agent_label)


    def define_stage_discriminator(self, convfeat, rep_size):

        output_shape = [self.sy_nenvs * (self.sy_nsteps - 1)]
        

        stage_label = tf.one_hot(self.stage_label, 2, axis=-1)
        stage_label = tf.reshape(stage_label,(-1,2 ))

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C

                phi = ph[:,1:]
                phi = tf.cast(phi, tf.float32)
                phi = tf.reshape(phi, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                phi = phi /255.
                

                phi = tf.nn.leaky_relu(conv(phi, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                #[20,20] [8,8]
                phi = tf.nn.leaky_relu(conv(phi, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                #[9,9] [7,7]
                phi = tf.nn.leaky_relu(conv(phi, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                phi = to2d(phi)
        
                phi = tf.nn.relu(fc(phi, 'fc1r', nh=rep_size, init_scale=np.sqrt(2)))
                self.stage_logits = fc(phi, 'fc2r', nh= 2, init_scale=np.sqrt(2))




        stage_prob = tf.nn.softmax(self.stage_logits, axis=-1)
        self.stage_prob = tf.reshape(stage_prob, (self.sy_nenvs, self.sy_nsteps - 1, 2))

        
        self.stage_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.stage_logits, labels=stage_label)

    def define_stage_rnd(self, convfeat, rep_size, enlargement):
        logger.info("Using RND BONUS ****************************************************")

        #RND bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]

                xr = tf.clip_by_value((xr - self.sd_ph_mean) / self.sd_ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]


                xrp = tf.clip_by_value((xrp - self.sd_ph_mean) / self.sd_ph_std, -5.0, 5.0)


                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)
                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))



        self.stage_rnd = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.stage_rnd = tf.reshape(self.stage_rnd, (self.sy_nenvs, self.sy_nsteps - 1))

        targets = tf.stop_gradient(X_r)
        # self.stage_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
        self.stage_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)

    def initial_state(self, n):
        return np.zeros((n, self.memsize), np.float32)

    def call(self, dict_obs, new, istate, agent_idx, update_obs_stats=False):
        for ob in dict_obs.values():
            if ob is not None:
                if update_obs_stats:
                    raise NotImplementedError
                    ob = ob.astype(np.float32)
                    ob = ob.reshape(-1, *self.ob_space.shape)
                    self.ob_rms.update(ob)
        # Note: if it fails here with ph vs observations inconsistency, check if you're loading agent from disk.
        # It will use whatever observation spaces saved to disk along with other ctor params.
        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        feed2 = { self.ph_istate: istate, self.ph_new: new[:,None].astype(np.float32) }
        #feed1.update({self.ph_mean: self.ob_rms.mean, self.ph_std: self.ob_rms.var ** 0.5})

        feed1.update({self.ph_agent_idx: agent_idx})
        # for f in feed1:
        #     print(f)
        a, vpred_int,vpred_ext, nlp, newstate, ent = tf_util.get_session().run(
            [self.a_samp, self.vpred_int_rollout,self.vpred_ext_rollout, \
              self.nlp_samp, self.snext_rollout, self.entropy_rollout], 
            feed_dict={**feed1, **feed2})
        base_vpred_ext =  np.ones_like(vpred_ext)
        return a[:,0], vpred_int[:,0],vpred_ext[:,0], nlp[:,0], newstate, ent[:,0], base_vpred_ext[:,0]
        
    def taget_step(self, dict_obs):

        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        vpred_ext, vpred_int , target_a = tf_util.get_session().run(
            [self.vpred_ext_rollout, self.vpred_int_rollout, self.a_samp], 
            feed_dict={**feed1})

        return target_a[:,0], vpred_ext[:,0], vpred_int[:,0]

    def state_stage(self, dict_obs):

        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        stage_prob = tf_util.get_session().run(
            self.stage_prob, 
            feed_dict={**feed1})

        return stage_prob[:,0]

    '''   
    def get_ph_mean_std(self):
        mean, std, count = tf.get_default_session().run([self.var_ph_mean,self.var_ph_std, self.var_ph_count])

        return mean, std, count
    '''
