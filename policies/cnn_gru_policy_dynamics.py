import numpy as np
import tensorflow as tf
from baselines import logger
from utils import fc, conv
from stochastic_policy import StochasticPolicy
from tf_util import get_available_gpus
from mpi_util import RunningMeanStd

from baselines.common.distributions import CategoricalPdType

import tf_util

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))



class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
    def __init__(self, num_units, rec_gate_init=-1.0):
        tf.nn.rnn_cell.RNNCell.__init__(self)
        self._num_units = num_units
        self.rec_gate_init = rec_gate_init
    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        x, new = inputs
        h = state
        h *= (1.0 - new)
        hx = tf.concat([h, x], axis=1)
        mr = tf.sigmoid(fc(hx, nh=self._num_units * 2, scope='mr', init_bias=self.rec_gate_init))
        # r: read strength. m: 'member strength
        m, r = tf.split(mr, 2, axis=1)
        rh_x = tf.concat([r * h, x], axis=1)
        htil = tf.tanh(fc(rh_x, nh=self._num_units, scope='htil'))
        h = m * h + (1.0 - m) * htil
        return h, h

class CnnGruPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 dynamics_bonus = False, num_agents = 1, rnd_type='rnd', div_type='oracle',
                 indep_rnd = False, indep_policy = False, sd_type='oracle', rnd_mask_prob =1.
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
        self.sample_agent_prob = tf.placeholder(dtype=tf.float32, shape=(None,None,), name="sample_agent_prob")

        self.game_score = tf.placeholder(dtype=tf.float32, shape= (None, None) , name="game_score")
        self.last_rew_ob = tf.placeholder(dtype=ob_space.dtype, shape= (None, None) + tuple(ob_space.shape) , name="last_rew_ob")
        self.rew_agent_label = tf.placeholder(dtype=tf.int32, shape= (None, None,) , name="rew_agent_label")

        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obstd")
        self.ph_count = tf.placeholder(dtype=tf.float32, shape=(), name="obcount")

        self.sep_ph_mean = tf.placeholder(dtype=tf.float32, shape= (None, None,) + ob_space.shape[:2] +(1,) , name="sep_obmean")
        self.sep_ph_std = tf.placeholder(dtype=tf.float32, shape= (None, None,) + ob_space.shape[:2] + (1,), name="sep_obstd")
        self.sep_ph_count = tf.placeholder(dtype=tf.float32, shape=(), name="sep_obcount")


        self.var_ph_mean = tf.get_variable("var_ph_mean", list(ob_space.shape[:2])+[1], initializer=tf.constant_initializer(0.0))
        self.var_ph_std = tf.get_variable("var_ph_std", list(ob_space.shape[:2])+[1], initializer=tf.constant_initializer(0.0))
        self.var_ph_count = tf.get_variable("var_ph_count", (), initializer=tf.constant_initializer(0.0))

        memsize *= enlargement
        hidsize *= enlargement
        convfeat = 16*enlargement


        self.ob_rms_list = [RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi= not update_ob_stats_independently_per_gpu) \
                                for _ in range(num_agents)]        
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi=not update_ob_stats_independently_per_gpu)
        ph_istate = tf.placeholder(dtype=tf.float32,shape=(None,memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]
        self.memsize = memsize

        self.num_agents = num_agents

        if  num_agents <= 0:

            self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
                self.apply_policy(self.ph_ob[None][:,:-1],
                                  ph_new=self.ph_new,
                                  ph_istate=ph_istate,
                                  reuse=False,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps - 1,
                                  pdparamsize=pdparamsize,
                                  rec_gate_init=rec_gate_init
                                  )
            self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
                self.apply_policy(self.ph_ob[None],
                                  ph_new=self.ph_new,
                                  ph_istate=ph_istate,
                                  reuse=True,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps,
                                  pdparamsize=pdparamsize,
                                  rec_gate_init=rec_gate_init
                                  )
        else:

            self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
                self.apply_multi_head_policy(self.ph_ob[None][:,:-1],
                                  ph_new=self.ph_new,
                                  ph_istate=ph_istate,
                                  reuse=False,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps - 1,
                                  pdparamsize=pdparamsize,
                                  rec_gate_init=rec_gate_init
                                  )
            self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
                self.apply_multi_head_policy(self.ph_ob[None],
                                  ph_new=self.ph_new,
                                  ph_istate=ph_istate,
                                  reuse=True,
                                  scope=scope,
                                  hidsize=hidsize,
                                  memsize=memsize,
                                  extrahid=extrahid,
                                  sy_nenvs=self.sy_nenvs,
                                  sy_nsteps=self.sy_nsteps,
                                  pdparamsize=pdparamsize,
                                  rec_gate_init=rec_gate_init
                                  )

        if dynamics_bonus:
            self.define_dynamics_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
        else:
            #self.define_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
            self.aux_loss, self.int_rew, self.feat_var, self.max_feat = self.define_multi_head_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)



        self.stage_rnd = tf.constant(1.)
        self.stage_prob = tf.constant(1.)


        if div_type =='cls':
            with tf.variable_scope("div", reuse=False):
                #self.define_rew_discriminator(convfeat=convfeat, rep_size=256)
                with tf.variable_scope("int", reuse=False):
                   self.disc_logits, self.all_div_prob, self.sp_prob, self.div_rew,  self.disc_pd , self.disc_nlp  =  self.define_rew_discriminator_v2(convfeat=convfeat, rep_size=512, use_rew = True)
        else:
            self.div_rew = tf.constant(0.)

        pd = self.pdtype.pdfromflat(self.pdparam_rollout)
        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)

        self.ph_istate = ph_istate

    @staticmethod
    def apply_policy(ph_ob, ph_new, ph_istate, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize, rec_gate_init):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnGruPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())

        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            X = tf.reshape(X, [sy_nenvs, sy_nsteps, hidsize])
            X, snext = tf.nn.dynamic_rnn(
                GRUCell(memsize, rec_gate_init=rec_gate_init), (X, ph_new[:,:,None]),
                dtype=tf.float32, time_major=False, initial_state=ph_istate)
            X = tf.reshape(X, (-1, memsize))
            Xtout = X
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=memsize, init_scale=0.1))
                X = X + activ(fc(X, 'fc2act', nh=memsize, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext

    def _build_policy_net(self, X, ph_new, ph_istate, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize, rec_gate_init):
        activ = tf.nn.relu
        data_format = 'NHWC'

        with tf.variable_scope(scope, reuse=reuse):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            X = tf.reshape(X, [sy_nenvs, sy_nsteps, hidsize])
            X, snext = tf.nn.dynamic_rnn(
                GRUCell(memsize, rec_gate_init=rec_gate_init), (X, ph_new[:,:,None]),
                dtype=tf.float32, time_major=False, initial_state=ph_istate)
            X = tf.reshape(X, (-1, memsize))
            Xtout = X
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=memsize, init_scale=0.1))
                X = X + activ(fc(X, 'fc2act', nh=memsize, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

        return pdparam, vpred_int, vpred_ext, snext

    def apply_multi_head_policy(self, ph_ob, ph_new, ph_istate, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize, rec_gate_init):


        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        logger.info("CnnGruPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        yes_gpu = any(get_available_gpus())

        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):


            all_pdparam = []
            all_vint = []
            all_vext = []
            all_snext = []

            for i in range(self.num_agents):

                scope = 'agent_{}'.format(str(i))
                pdparam, vpred_int, vpred_ext, snext = self._build_policy_net(X=X, 
                                                                  ph_new=ph_new,
                                                                  ph_istate = ph_istate,
                                                                  scope=scope, 
                                                                  reuse=False, 
                                                                  hidsize=hidsize,
                                                                  memsize=memsize, 
                                                                  extrahid=extrahid, 
                                                                  sy_nenvs=sy_nenvs,
                                                                  sy_nsteps=sy_nsteps,
                                                                  pdparamsize=pdparamsize,
                                                                  rec_gate_init=rec_gate_init)

                if i==0:
                    #[batch,naction] - > [batch, 1, naction]
                    all_pdparam = tf.expand_dims(pdparam, axis=1)
                    #[batch,1] -> [batch,1,1]
                    all_vint = tf.expand_dims(vpred_int, axis=1)
                    all_vext = tf.expand_dims(vpred_ext, axis=1)
                    all_snext = tf.expand_dims(snext, axis=1)
                else:
                    all_pdparam = tf.concat([all_pdparam, tf.expand_dims(pdparam, axis=1)], axis=1)
                    all_vint = tf.concat([all_vint, tf.expand_dims(vpred_int, axis=1)], axis=1)
                    all_vext = tf.concat([all_vext, tf.expand_dims(vpred_ext, axis=1)], axis=1)
                    all_snext = tf.concat([all_snext, tf.expand_dims(snext, axis=1)], axis=1)

            #[batch, nstep] -> [batch,nstep, ngroups]
            one_hot_gidx = tf.one_hot(self.ph_agent_idx, self.num_agents, axis=-1)
            #[batch,nstep, ngroups] -> [batch * nstep, ngroups,1]
            one_hot_gidx = tf.reshape(one_hot_gidx,(-1,self.num_agents,1))



            pdparam = tf.reduce_sum(one_hot_gidx * all_pdparam,axis=1)
            vpred_int = tf.reduce_sum(one_hot_gidx * all_vint,axis=1)
            vpred_ext = tf.reduce_sum(one_hot_gidx * all_vext,axis=1)
            snext = tf.reduce_sum(one_hot_gidx * all_snext,axis=1)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
            snext = tf.reshape(snext, (sy_nenvs, memsize))


        return pdparam, vpred_int, vpred_ext, snext

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

    def define_multi_head_self_prediction_rew(self, convfeat, rep_size, enlargement):
        logger.info("Using multi-head RND BONUS ****************************************************")

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
                    target_out = self._build_target_net(target_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)

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
                    pred_out = self._build_pred_net(pred_x, scope, tf.AUTO_REUSE, convfeat, rep_size, enlargement)
                    
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

        #[env*step, num_agents ,1] 
        rnd_mask =  tf.reshape(self.rnd_mask,(-1, self.num_agents,1))
        rnd_mask = tf.cast(rnd_mask, tf.float32)

        #[env*step, num_agents , 1] -> [env*step]
        mask_loss = tf.reduce_sum(rnd_mask * all_loss, axis=[1,2]) / tf.maximum(tf.reduce_sum(rnd_mask, axis=[1,2]), 1.)
        aux_loss = mask_loss
        mask = tf.random_uniform(shape=tf.shape(aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        aux_loss = tf.reduce_sum(mask * aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

        return aux_loss, int_rew, feat_var, max_feat

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

    def define_self_prediction_rew(self, convfeat, rep_size, enlargement):
        #RND.
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
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)
                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def define_dynamics_prediction_rew(self, convfeat, rep_size, enlargement):
        #Dynamics based bonus.

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
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

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
            [self.a_samp, self.vpred_int_rollout,self.vpred_ext_rollout, self.nlp_samp, self.snext_rollout, self.entropy_rollout],
            feed_dict={**feed1, **feed2})

        base_vpred_ext =  np.ones_like(vpred_ext)

        return a[:,0], vpred_int[:,0],vpred_ext[:,0], nlp[:,0], newstate, ent[:,0], base_vpred_ext[:,0]
        
    def get_ph_mean_std(self):
        mean, std = tf.get_default_session().run([self.var_ph_mean,self.var_ph_std])

        return mean, std
