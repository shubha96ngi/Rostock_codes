#  you can not name target neu.input according to your will it is presefined in model , spike, V , input these terms are used so use that name 
# dont use x in place of input 
# although there is one example of how to use x 

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from brainpy._src.integrators import JointEq
from brainpy._src.context import share

bp.math.set_dt(0.05)
class HH(bp.NeuGroup):
  def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=20., phi=5, method='exp_auto'):
    super(HH, self).__init__(size=size)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th
    self.phi = phi

    # variables
    self.V = bm.Variable(bm.ones(size) * -65.)
    self.h = bm.Variable(bm.ones(size) *0.6)
    self.n = bm.Variable(bm.ones(size) * 0.32)
    self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))
    self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

    # integral
    self.integral = bp.odeint(bp.JointEq([self.dV, self.dh, self.dn]), method=method)

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 58) / 20)
    beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * bm.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt

  # def m(self, V):
  #   alpha =  -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
  #   beta = 4 * bm.exp(-(V + 60) / 18)
  #   mt = alpha / (alpha + beta) #alpha * (1 - m) - beta * m
  #   return  mt

  def dV(self, V, t, h, n, Iext):
    m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * bm.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = self.gNa * m ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + Iext) / self.C

    return dVdt

  def update(self, tdi):
    V, h, n = self.integral(self.V, self.h, self.n, tdi.t,self.input, tdi.dt)
    # m =  -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1) / (-0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1) +4 * bm.exp(-(V + 60) / 18) )
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)
    self.V.value = V
    self.h.value = h
    self.n.value = n
    # self.m.value = mclass A2(bp.neurons.HH):
  def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35.,
               gK=9., gL=0.1, V_th=20., method='exp_auto'):
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

    self.input = bm.Variable(bm.ones(size) * 1.2)
    # self.t_last_spike = bm.Variable(bm.ones(size) * -1e7)

  def m_inf(self, V):
    alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    # print('m_inf*****')
    beta = 4 * bm.exp(-(V + 60) / 18)
    return alpha / (alpha + beta)

  def dn(self, n, t, V):
    alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    # print('dn_alpha*****')
    beta = 0.125 * bm.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return 5 * dndt

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 58) / 20)
    beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return 5 * dhdt

  def dV(self, V, t, h, n, I):
    # print('n,m_inf,h=', self.m_inf(V),n,h)
    INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
    # print('I ki value =', INa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + I) / self.C
    return dVdt

  @property
  def derivative(self):
    return JointEq(self.dV, self.dh, self.dn)  # , self.dh, self.dn, self.ds, self.dc, self.dq)

  def update(self, tdi):
    # t = share.load('t')
    # dt = share.load('dt')
    # x = 1.2 if x is None else x
    # print('x=', x )

    V, h, n = self.integral(self.V.value, self.h.value, self.n.value, tdi.t, self.input, tdi.dt)
    # V += self.sum_delta_inputs()
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    # self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)
    self.V.value = V
    self.h.value = h
    self.n.value = n
    self.input[:] = 0 #1.2
    return self.spike.value


num = 100
# dyn neurons
neu =  A2(num) #A1(num) #bp.neurons.HH(num)#bp.dyn.HH(10) #HH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num) #bp.dyn.WangBuzsakiHH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num)
neu.V[:] = -70. + bm.random.normal(size=num) * 20


   

class BaseAMPASyn(bp.SynConn):
  def __init__(self, pre, post, conn, delay=0, g_max=0.1/100, E=-80, alpha=0.53,
               beta=0.18, T=1, T_duration=1, method='exp_auto'):
    super(BaseAMPASyn, self).__init__(pre=pre, post=post, conn=conn)

    # check whether the pre group has the needed attribute: "spike"
    self.check_pre_attrs('spike')

    # check whether the post group has the needed attribute: "input" and "V"
    self.check_post_attrs('input', 'V')

    # parameters
    self.delay = delay
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

    # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
    self.delay_step = int(delay/bm.get_dt())
    self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step)

    # store the arrival time of the pre-synaptic spikes
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)

    # integral function
    self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

class AMPAAll2All(BaseAMPASyn):
  def __init__(self, *args, **kwargs):
    super(AMPAAll2All, self).__init__(*args, **kwargs)

    # synapse gating variable
    # -------
    # The synapse variable has the shape of the post-synaptic group
    self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))

  def update(self, tdi, x=None):
    _t, _dt = tdi.t, tdi.dt
    delayed_spike = self.pre_spike(self.delay_step)
    self.pre_spike.update(self.pre.spike)
    self.spike_arrival_time.value = bm.where(delayed_spike, _t, self.spike_arrival_time)
    TT = ((_t - self.spike_arrival_time) < self.T_duration) * self.T
          #*1/(1+bm.exp(-(self.pre.V-20)/2)))
    # TT = 1/(1+bm.exp(-(self.pre.V-20)/2))
    TT = TT.reshape((-1, 1))  # NOTE: here is the difference
    self.g.value = self.integral(self.g, _t, TT, dt=_dt)
    g_post = self.g.sum(axis=0) # NOTE: here is also different

    self.post.input += self.g_max * g_post * (self.E - self.post.V)
    print('output =', self.post.input)
syn = AMPAAll2All(pre=neu, post=neu, conn=bp.connect.All2All(include_self=False)) #include_self=False)) #,stop_spike_gradient=False)
# syn = bp.dyn.GABAa.desc(pre=neu, post=neu, prob=1) #conn=bp.connect.All2All(include_self=False)) #,stop_spike_gradient=False)
syn.g_max = 0.1/100
print('='*100)
# neu = HH(10)
# V, spike

# net = bp.Network(pre=neu, post=neu ,syn=syn)
net = bp.Network(neu=neu, syn=syn)
# runner = bp.DSRunner(net, monitors=['pre.spike', 'pre.V','post.spike', 'post.V'], inputs=[('pre.input',1.2,'fix', '+'), ('post.input',120)]) #,'fix', '=']) # bm.random.normal(1,0.02,num)])
# runner = bp.DSRunner(neu, monitors=['V','spike'], inputs=['input',1.2]) #,'fix','+']) # bm.random.normal(1,0.02,num)])
runner = bp.DSRunner(net, monitors=['neu.spike', 'neu.V'], inputs=['neu.input',1.2]) #
runner.run(duration=500.) #,inputs=inputs )


# a = bp.DynamicalSystem() #bp.Dynamic
# #a.input
fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon.ts, runner.mon['neu.V'], ylabel='Membrane potential (N0)')
bp.visualize.line_plot(runner.mon.ts, runner.mon['neu.V'][:,10])

# bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,0], ylabel='Membrane potential (N0)')
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,1])

fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['neu.spike'], show=True)
plt.show()

# fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
#
# fig.add_subplot(gs[0, 0])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['V'][:,0], ylabel='Membrane potential (N0)')
# bp.visualize.line_plot(runner.mon.ts, runner.mon['V'][:,1])
# bp.visualize.line_plot(runner.mon.ts, runner.mon['V'][:,10])
#
# # bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,0], ylabel='Membrane potential (N0)')
# # bp.visualize.line_plot(runner.mon.ts, runner.mon.V[:,1])
#
# fig.add_subplot(gs[1, 0])
# # bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.spike'], show=True)
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['spike'], show=True)
# plt.show()


