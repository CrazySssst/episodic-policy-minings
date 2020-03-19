from episodic_curiosity import oracle

from ppo_agent import save_rews_list

import pickle

import numpy as np 
from utils import explained_variance, Scheduler

sil_loss_weight = Scheduler(v=1., nvalues=200, schedule='linear')

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

a=[[ 2.0141983, 12]
 ,[ 2.0334325, 12]
 ,[ 2.0381384, 12]
 ,[ 1.9757961, 12]
 ,[ 1.9958011, 12]
 ,[ 2.0272028, 12]
 ,[ 2.073726, 12]
 ,[ 2.0448227, 11]
 ,[ 1.9994664, 11]
 ,[ 2.0019026, 11]
 ,[ 1.9926736, 11]
 ,[ 1.9864389, 11]
 ,[ 1.9718688, 11]
 ,[ 2.0127394, 11]
 ,[ 1.9853333, 11]
 ,[ 2.0017598, 11]
 ,[ 1.9949136, 11]
 ,[ 2.0054865, 11]
 ,[ 2.011146, 11]
 ,[ 2.0157328, 11]
 ,[ 2.0109742, 11]
 ,[ 2.008641, 11]
 ,[ 2.015577, 11]
 ,[ 2.0362294, 11]
 ,[ 1.9807872, 11]
 ,[ 2.0120518, 11]
 ,[ 2.0055106, 11]
 ,[ 2.0182347, 11]
 ,[ 1.9996722, 11]
 ,[ 1.9860988, 11]
 ,[ 1.9973494, 5]
 ,[ 2.0244813, 5]
 ,[ 2.0276628, 5]
 ,[ 2.0495253, 5]
 ,[ 2.0535953, 5]
 ,[ 2.0413113, 5]
 ,[ 2.0329642, 5]
 ,[ 1.9419564, 5]
 ,[ 2.0182881, 5]
 ,[ 2.059166, 5]
 ,[ 2.032584, 5]
 ,[ 2.0150206, 5]
 ,[ 2.0751114, 5]
 ,[ 2.07085, 5]
 ,[ 1.1862005, 5]
 ,[ 0.9858524, 5]
 ,[ 1.0547862, 5]
 ,[ 1.0487491, 5]
 ,[ 1.0313374, 5]
 ,[ 1.0725129, 5]
 ,[ 1.0928918, 5]
 ,[ 1.1120985, 5]
 ,[ 1.0593233, 5]
 ,[ 1.0870851, 5]
 ,[ 1.0676547, 5]
 ,[ 1.0513232, 5]
 ,[ 0.9622736, 5]
 ,[ 1.0657601, 5]
 ,[ 1.0958743, 5]
 ,[ 1.0575311, 5]
 ,[ 1.0628878, 5]
 ,[ 1.0270221, 5]
 ,[ 0.9719502, 5]
 ,[ 1.0362406, 5]
 ,[ 1.0391208, 5]
 ,[ 1.0963031, 5]
 ,[ 1.0698969, 5]
 ,[ 1.0903025, 5]
 ,[ 1.0805593, 5]
 ,[ 1.1108366, 5]
 ,[ 1.055312, 5]
 ,[ 1.0873837, 5]
 ,[ 1.0830232, 5]
 ,[ 1.0523834, 5]
 ,[ 1.015652, 5]
 ,[ 1.0688956, 5]
 ,[ 1.0131334, 5]
 ,[ 1.0132617, 5]
 ,[ 1.004796, 5]
 ,[ 1.0495577, 5]
 ,[ 1.0588346, 5]
 ,[ 1.0779662, 5]
 ,[ 1.0168692, 5]
 ,[ 1.0865914, 5]
 ,[ 1.0747453, 5]
 ,[ 1.1289225, 5]
 ,[ 1.0175143, 5]
 ,[ 1.0118997, 5]
 ,[ 1.1129273, 5]
 ,[ 1.0916731, 5]
 ,[ 1.0784872, 5]
 ,[ 1.0814176, 5]
 ,[ 1.1147596, 5]
 ,[ 1.045303, 5]
 ,[ 1.0664393, 5]
 ,[ 0.9433172, 5]
 ,[ 0.43083483, 5]
 ,[ 0.07331575, 5]
 ,[ 0.07617247, 5]
 ,[ 0.0741353, 5]]

rff = RewardForwardFilter(0.99)


a = np.asarray(a)

lens = a.shape[0]

for t in range(lens-1, -1, -1):
	print(rff.update(a[t,0])/ 100., a[t,1])