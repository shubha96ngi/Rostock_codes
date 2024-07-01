import math 
import numpy as np
def alpha_n(V):
    return 0.01*(10.0-V)/ (np.exp(1.0-0.1*V )-1)
def beta_n(V):
    return 0.125*np.exp(-V/80.0)
def alpha_m(V):
    return  0.1*(25-V) / (np.exp(2.5 - 0.1*V)-1)
def beta_m(V): 
    return 4.0*np.exp(-V/18.0)
def alpha_h(V): 
    return 0.07*np.exp(-V/20.0)
def beta_h(V): 
    return 1/(1+np.exp(3.0-0.1*V))

Cm = 1; # uF/cm^2
E_Na = 120  # mV
E_K = -12  #mV
E_Leak = 10.6; # mV
g_Na = 120; # mS/cm^2
g_K = 36; # mS/cm^2
g_Leak = 0.3; # mS/cm^2

# V = 0;s = 0 
m=0;n=0;h=0
V = np.zeros(100) #np.ones(100)*-65
s = np.zeros(100)
dt = 0.05

# connections with prob 0.1 
norm_dist = np.random.normal(0.1,0.02,100) 
# I_DBS = np.random.normal(9,10,100)  # distribution of IDBS
I_DBS = 10
#ew = np.zeros((100,100))


# selecting 10 random index for 0.1 probability 
import random
def random_without_repeat(nums, target_size):
    if target_size > len(nums):
        raise ValueError("Target size is larger than the original list")

    chosen_nums = []
    for _ in range(target_size):
        while True:
            random_index = random.randint(0, len(nums) - 1)
            if random_index not in chosen_nums:
                chosen_nums.append(random_index)
                break

    return [nums[i] for i in chosen_nums]
def remaining_elements_from_list(list1, sublist):
    remaining_elements = []
    for element in list1:
        if element not in sublist:
            remaining_elements.append(element)
    return remaining_elements
'''
# fill the index 
nums = list(range(100))
for i in range(100):
    chosen_nums = random_without_repeat(nums, 10)
    remaining_elements = remaining_elements_from_list(nums, chosen_nums)
    c = 0 
    for idx in chosen_nums:
        ew[i,idx] = norm_dist[c]
        c+= 1
        
print('ew= ', np.unique(ew))
'''
# ew = np.ones((100,100))*0.01  # when matrix 
ew = 0.01 # scalar value 

def HH(V,m,n,h,s,I_DBS):
    I_Na = g_Na*m**3*h*(V-E_Na);
    I_K = g_K*n**4*(V-E_K);
    I_Leak = g_Leak*(V-E_Leak);
    Isyn = ((20-V)*np.squeeze((np.dot(ew, np.array([s]).T))))/10
    Input  = (I_DBS-(I_Na+I_K+I_Leak)+Isyn)# _exci+Isyn_inhi)
    V  = V + dt *Input* (1/Cm) # -spk*(v+60)
    m = m + dt*((alpha_m(V)*(1-m))-beta_m(V)*m)
    n = n + dt*((alpha_n(V)*(1-n))-beta_n(V)*n)
    h = h + dt*((alpha_h(V)*(1-h))-beta_h(V)*h)
    #s =  s + dt* (-s + ((1-s)*5)/(1+np.exp(-(V+3)/8)))
    alpha = 0.98 ; beta = 0.18 
    s =  s + dt* 0.98 *(1 - s) - 0.18 * s
    return V,m,n,h,s,Isyn 

s_pre = np.zeros((5000,100))
t_pre = np.zeros((5000,100))
V1 = np.zeros((5000,100))
s1 = np.zeros((5000,100))
Isyn_all = np.zeros((5000,100))
for t in range(5000):
    V,m,n,h,s,Isyn=  HH(V,m,n,h,s,I_DBS)
    V1[t,:] = V
    Isyn_all[t] = Isyn
    s1[t] = s
    if t>2:
        for i in range(100):
            if V1[t-2,i]<V1[t-1,i] and V1[t-1,i]> V1[t,i]:
                t_pre[t,i] = t 
                s_pre[t,i] =1 
                
plt.plot(V1[:,0])
plt.show()



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

        self.input = bm.Variable(bm.ones(size) * 1)
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

    def update(self, tdi):
        # t = share.load('t')
        # dt = share.load('dt')
        # x = 1.2 if x is None else x
        # print('x=', x )

        V, m, h, n = self.integral(self.V.value, self.m.value, self.h.value, self.n.value, tdi.t, self.input, tdi.dt)
        # V += self.sum_delta_inputs()
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        # self.t_last_spike.value = bm.where(self.spike, tdi.t, self.t_last_spike)
        self.V.value = V
        self.h.value = h
        self.n.value = n
        self.input[:] = 0
        return self.spike.value


num = 1
# dyn neurons
neu =  A2(num) #A1(num) #bp.neurons.HH(num)#bp.dyn.HH(10) #HH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num) #bp.dyn.WangBuzsakiHH(num) #bp.neurons.WangBuzsakiModel(num) #HH(num)
# neu = bp.neurons.HH(100)
neu.V[:] = 0 #-70. + bm.random.normal(size=num)*20


class BaseAMPASyn(bp.SynConn):
    def __init__(self, pre, post, conn, delay=0, g_max=0.001, E=20, alpha=0.98,
               beta=0.18, 
               T=1, T_duration=1, method='exp_auto'):
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
       
        # dg = -g + ((1-g)*5)/(1+bm.exp(-(self.pre.V+3)/8))  # according to paper 
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
        TT = TT.reshape((-1, 1))  # NOTE: here is the difference
        self.g.value = self.integral(self.g, _t, TT, dt=_dt)
        g_post = self.g.sum(axis=0) # NOTE: here is also different
        #self.post.input += bm.dot(self.g_max,g_post) * (self.E - self.post.V)
        self.post.input += self.g_max * g_post * (self.E - self.post.V) 
        #bm.dot(self.g_max,g_post) * (self.E - self.post.V)/99

def show_syn_model(model):
    pre =  neu #bp.neurons.HH(100) #LIF(100, V_rest=-60., V_reset=-60., V_th=-40.)
    #post = neu #bp.neurons.HH(100) #LIF(100, V_rest=-60., V_reset=-60., V_th=-40.)
    syn = model(pre, pre, conn=bp.conn.All2All(include_self=True))
    # shape = (1,1) #(100,100)
    # syn.g_max = bp.init.Normal(mean=0.1, scale=0.02)(shape) 
    net = bp.Network(pre=pre, post=pre, syn=syn)

    runner = bp.DSRunner(net,
                       monitors=['pre.V', 'post.V', 'syn.g','pre.spike'],
                       inputs=['pre.input',10]) # bm.random.uniform(9,10,100)+10])
    runner.run(300.)

    fig, gs = bp.visualize.get_figure(1, 3, 3, 4)
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(runner.mon.ts, runner.mon['syn.g'], legend='syn.g')
    fig.add_subplot(gs[0, 1])
    bp.visualize.line_plot(runner.mon.ts, runner.mon['pre.V'], legend='pre.V')
    # bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], legend='post.V', show=True)
    fig.add_subplot(gs[0, 2])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.spike'], show=True)


show_syn_model(AMPAAll2All)
