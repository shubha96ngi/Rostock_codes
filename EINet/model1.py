# similar code 
#https://github.com/brainpy/examples/blob/main/ei_nets/Brette_2007_COBAHH.ipynb

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from brainpy._src.integrators import JointEq
from brainpy._src.context import share
bp.math.set_dt(0.05)

class HH2(bp.dyn.CondNeuGroupLTC):
  def __init__(self, size):
    super(HH2, self).__init__(size)
    self.INa = bp.channels.INa_TM1991(size, g_max=100., V_sh=-63.)
    self.IK = bp.channels.IK_TM1991(size, g_max=30., V_sh=-63.)
    self.IL = bp.channels.IL(size, E=-60., g_max=0.05)

num_exc = 3200
num_inh = 800
Cm = 200  # Membrane Capacitance [pF]

gl = 0.03  # Leak Conductance   [nS]
g_Na = 120. #* 1000
g_Kd = 36. #* 1000  # K Conductance      [nS]
El = -54.4  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -77.  # reversal potential (Potassium) [mV]
# VT = -63.
V_th = -20.

# Time constants
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]

# Reversal potentials
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]

# excitatory synaptic weight
we = 6.  # excitatory synaptic conductance [nS]

# inhibitory synaptic weight
wi = 67.  # inhibitory synaptic conductance [nS]


class HH(bp.dyn.NeuDyn):
  def __init__(self, size, method='exp_auto'):
    super(HH, self).__init__(size)

    # variables
    self.V = bm.Variable(El + (bm.random.randn(self.num) * 5 - 5))
    self.m = bm.Variable(bm.zeros(self.num))
    self.n = bm.Variable(bm.zeros(self.num))
    self.h = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))

    def dV(V, t, m, h, n, Isyn):
      gna = g_Na * (m * m * m) * h
      gkd = g_Kd * (n * n * n * n)
      dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
      return dVdt

    def dm(m, t, V, ):
      m_alpha = (0.1*V+4) / (-bm.exp(-4 - 0.1*V)+1) # 0.32 * (13 - V + VT) / (bm.exp((13 - V + VT) / 4) - 1.)
      m_beta = 4.0*bm.exp(-V-65/18.0) #0.28 * (V - VT - 40) / (bm.exp((V - VT - 40) / 5) - 1)
      dmdt = (m_alpha * (1 - m) - m_beta * m)
      return dmdt

    def dh(h, t, V):
      h_alpha =  0.07*bm.exp(-V-65/20.0) #0.128 * bm.exp((17 - V + VT) / 18)
      h_beta =  1/(1+bm.exp(-3.5-0.1*V))  #4. / (1 + bm.exp(-(V - VT - 40) / 5))
      dhdt = (h_alpha * (1 - h) - h_beta * h)
      return dhdt

    def dn(n, t, V):
      
      n_alpha = 0.01*(55.0+V)/ (-bm.exp(-5.5-0.1*V )+1) #0.032 * c / (bm.exp(c / 5) - 1.)
      n_beta =  0.125*bm.exp(-V-65/80.0)  #.5 * bm.exp((10 - V + VT) / 40)
      dndt = (n_alpha * (1 - n) - n_beta * n)
      return dndt

 

    
    # functions
    self.integral = bp.odeint(bp.JointEq([dV, dm, dh, dn]), method=method)

  def update(self):
    tdi = bp.share.get_shargs()
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, tdi.t, Isyn=self.input, dt=tdi.dt)
    self.spike.value = bm.logical_and(self.V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    self.input[:] = 12.


class ExpCOBA(bp.synapses.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))

    # function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

  def update(self):
    self.g.value = self.integral(self.g, bp.share['t'], dt=bp.share['dt'])
    post_sps = bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.g.value += post_sps
    self.post.input += self.g * (self.E - self.post.V)

class COBAHH(bp.Network):
  def __init__(self, scale=1., method='exp_auto'):
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    E = HH(num_exc)
    I = HH(num_inh)
    E2E = ExpCOBA(pre=E, post=E, conn=bp.conn.FixedProb(prob=0.02),
                  E=Ee, g_max=we / scale, tau=taue, method=method)
    E2I = ExpCOBA(pre=E, post=I, conn=bp.conn.FixedProb(prob=0.02),
                  E=Ee, g_max=we / scale, tau=taue, method=method)
    I2E = ExpCOBA(pre=I, post=E, conn=bp.conn.FixedProb(prob=0.02),
                  E=Ei, g_max=wi / scale, tau=taui, method=method)
    I2I = ExpCOBA(pre=I, post=I, conn=bp.conn.FixedProb(prob=0.02),
                  E=Ei, g_max=wi / scale, tau=taui, method=method)

    super(COBAHH, self).__init__(E2E, E2I, I2I, I2E, E=E, I=I)

net = COBAHH()
runner = bp.DSRunner(net, monitors=['E.spike'])
t = runner.run(4000.)
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)










