# removing update function from batista HH .. then it works 

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from brainpy._src.integrators import JointEq
from brainpy._src.context import share
import numpy as np 

bp.math.set_dt(0.05)


class A2(bp.neurons.HH):
    def __init__(self, size, ENa=120., EK=-12., EL=10.6, C=1.0, gNa=120.,
               gK=36, gL=0.3, V_th=0., method='exp_auto'):
        super().__init__(size=size, method=method)
        self.size = size

        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.C = C
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.V_th = V_th

        self.input = bm.Variable(bm.ones(size) * 9)
        self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    def dm(self, m, t, V):
        alpha =  0.1*(25-V) / (bm.exp(2.5 - 0.1*V)-1) #0.1 * (25-V) / (bm.exp(-0.1 * (V - 25)) - 1)
        beta = 4.0*bm.exp(-V/18.0) # 4 * bm.exp(-V / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dn(self, n, t, V):
        alpha = 0.01*(10.0-V)/ (bm.exp(1.0-0.1*V )-1) #0.01 * (10-V) / (bm.exp(0.1 * (10-V)) - 1)
        beta = 0.125*bm.exp(-V/80.0) #0.125 * bm.exp(-V / 80)
        dndt = alpha * (1 - n) - beta * n
        return  dndt

    def dh(self, h, t, V):
        alpha = 0.07*bm.exp(-V/20.0) #0.07 * bm.exp(-V / 20)
        beta = 1/(1+bm.exp(3.0-0.1*V)) #1 / (bm.exp(0.1 * (-V + 30)) + 1)
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dV(self, V, t, m, h, n, I):
        INa = self.gNa * m ** 3 * h * (V - self.ENa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + I) / self.C
        return dVdt

    @property
    def derivative(self):
        return JointEq(self.dV, self.dm, self.dh, self.dn)  # , self.dh, self.dn, self.ds, self.dc, self.dq)

#     def update(self, tdi):
#         # t = share.load('t')
#         # dt = share.load('dt')
#         # x = 1.2 if x is None else x
#         # print('x=', x )

#         V, m, h, n = self.integral(self.V.value, self.m.value, self.h.value, self.n.value, tdi.t, self.input, tdi.dt)
#         # V += self.sum_delta_inputs()
#         self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
#         # self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)
#         self.V.value = V
#         self.h.value = h
#         self.n.value = n
#         # self.input[:] = x
#         return self.spike.value


num = 100
# dyn neurons
neu =  A2(num) #A1(num) #bp.neurons.HH(num)#bp.dyn.HH(10) #HH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num) #bp.dyn.WangBuzsakiHH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num)
# neu = bp.neurons.HH(100)
neu.V[:] = 0 #-70. + bm.random.normal(size=num)*20


class SimpleNet(bp.DynSysGroup):
    def __init__(self, E=-70.):
        super().__init__()
        self.neu =  neu #bp.neurons.HH(200) #A(100)
        # self.neu.V[:] = -70. + bm.random.normal(size=100) * 20
        self.syn = bp.synapses.GABAa(pre=self.neu, post=self.neu, conn=bp.connect.All2All(include_self=False),stop_spike_gradient=False)#GABAa(self.pre,self.post, delay=0, prob=1., g_max=0.1/100, E=-75.)
        # self.syn.g_max= 0.1/100

    def update(self):

        self.neu.update(bm.random.uniform(9,10,100))
        self.syn()

        #self.post.update()

        conductance =  self.syn.g#self.syn.proj.refs['syn'].g
        current = self.neu.sum_current_inputs(self.neu.V)
        return conductance, current, self.neu.V,self.neu.spike.value

net = SimpleNet(E=20.)
runner = bp.DSRunner(net)
conductances, currents, potentials, spikes = runner.run(duration=500)


fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, potentials[:, 0], label='Neuron 0')
bp.visualize.line_plot(runner.mon.ts, potentials[:, 1],label='Neuron 0')

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(runner.mon.ts, spikes, show=True)
plt.show()
