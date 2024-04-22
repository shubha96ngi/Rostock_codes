
# it is still not giving results for GABAa from projection class 
# it is still not working for pre and post .. it only works for same neurons ( either pre pre or post post not for pre post or post pre) 
# more is synaptic conductance sooner it achieves the synchronisation pattern #( increase from 0.1 to 0.8) 
# current and reversal potential does not matter much ( it also synchronises for E=0 and E = -75 inhi and exci both) 
# later result visible hone k baad we can tune time step
# 200 or more par synchronisation easily dikh raha hai but 100 pe nahi 

import brainpy as bp
import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
bp.math.set_dt(0.04)

class A(bp.dyn.WangBuzsakiHH):
    def __init__(self, size):
        super(A, self).__init__(size=size)

        self.V = bm.Variable(bm.ones(size) * -65.)
        self.h = bm.Variable(bm.ones(size) * 0.6)
        self.n = bm.Variable(bm.ones(size) * 0.32)
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    def m_inf(self,V):
        alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
        print('m_inf*****')
        beta = 4 * bm.exp(-(V + 60) / 18)
        return alpha / (alpha + beta)
    def dn(self, n, t, V):
        alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
        print('dn_alpha*****')
        beta = 0.125 * bm.exp(-(V + 44) / 80)
        dndt = alpha * (1 - n) - beta * n
        return self.phi * dndt

    def update(self, tdi):


        V, h, n = self.integral(self.V, self.h, self.n, tdi.t, self.input, tdi.dt)
        # V += self.sum_delta_inputs()
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V.value = V
        self.h.value = h
        self.n.value = n
        self.input[:] = 6


class GABAa(bp.Projection):
    def __init__(self, pre, post, delay, prob, g_max, E=-75.):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPreMg2(
            pre=pre,
            delay=delay,
            syn=bp.dyn.GABAa.desc(pre.num, alpha=0.98, beta=0.18, T=1.0, T_dur=1.0),
            comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
            out=bp.dyn.COBA(E=E),
            post=post,
        )

import copy
from copy import deepcopy
bm.random.seed(123)
a =  bm.random.normal(size=100)
class SimpleNet(bp.DynSysGroup):
    def __init__(self, E=-75.):
        super().__init__()

        self.pre = A(400) #bp.neurons.WangBuzsakiModel(100) #bp.dyn.HH(100) #bp.dyn.WangBuzsakiHH(100)  # Changed to HH
        self.pre.V = -70. +  bm.random.normal(size=400) * 20 #bm.ones(10)*-70

        self.post = A(100) #bp.neurons.WangBuzsakiModel(100) #bp.dyn.HH(100) #WangBuzsakiHH(100)  # Changed to HH
        self.post.V =  -70. + bm.random.normal(size=100) * 20  #self.pre.V#-70. + bm.random.normal(size=100) * 20 #bm.ones(10)*-70
        # self.syn = GABAa(self.pre,self.post, delay=0, prob=1., g_max=0.1/100, E=-75.)
        # self.syn = GABAa(self.post, self.post, delay=0, prob=1., g_max=0.1/100, E=-80)

        # self.syn = bp.synapses.GABAa(pre=self.pre, post=self.post, conn=bp.connect.All2All(include_self=False),stop_spike_gradient=False)
        self.syn = bp.synapses.GABAa(pre=self.post, post=self.post, conn=bp.connect.All2All(include_self=False),stop_spike_gradient=False)
      # it is still not working for pre and post .. it only works for same neurons 

        # print('g_max =, 0.04=', self.syn.g_max)
        self.syn.g_max = 0.1/100  
      # more is synaptic conductance sooner it achieves the synchronisation pattern

    def update(self, I_pre):
        self.pre.update()
        self.syn()
        self.post.update()

        # monitor the following variables
        conductance = self.syn.g #proj.refs['syn'].g
        current = self.post.sum_current_inputs(self.post.V)
        return conductance, current, self.post.V,self.post.spike.value

duration = 40000
indices = np.arange(int(duration/bm.get_dt())).reshape(-1,100)
net = SimpleNet(E=-75)
runner = bp.DSRunner(net)
conductances, currents, potentials,spks = runner.run(duration=1000)
print('spikes.shape=', spks.shape)
ts = indices * bm.get_dt()
fig, gs = bp.visualize.get_figure(1, 2, 3.5, 8)
fig.add_subplot(gs[0, 0])
# plt.title('Syn conductance')
# plt.plot(ts, conductances[:,0])
# plt.plot(ts, conductances[:,5])
# fig.add_subplot(gs[0, 1])
# # plt.plot(ts, currents[:,0])
# plt.title('Syn current')
# fig.add_subplot(gs[0, 2])
plt.plot(runner.mon.ts, potentials[:,0])
plt.plot(runner.mon.ts, potentials[:,10])
fig.add_subplot(gs[0, 1])
# bp.visualize.raster_plot(ts, spks, show=True)
bp.visualize.raster_plot(runner.mon.ts, spks, show=True)
plt.show()

