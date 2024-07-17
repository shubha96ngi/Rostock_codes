# testing 3 variations of Dualexponential model 
# all are giving same result 
# Note: If want to avold the factor of A ( that is some expression of tau_decay and rise just use default mode , which is None
# can put some value for A

#Note: 
#while checking A ka value display nahi ho rahi thi so we can define denominator by using divide by g_max such as 
#gmax/(tau_decay-tau_rise)
#################################################################
# Method 1 
import brainpy as bp
import brainpy.math as bm
import brainpy.math as bm
import matplotlib.pyplot as plt

def run_syn(syn_model, title, run_duration=200., sp_times=(10, 20, 30), **kwargs):
  # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
  neu1 = bp.neurons.LIF(1) #bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
  neu2 = bp.neurons.LIF(1) #bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
  syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
  net = bp.Network(pre=neu1, syn=syn1, post=neu2)

  # 运行模拟
  # runner = bp.DSRunner(net, monitors=['pre.spike', 'post.V', 'syn.g', 'post.input'])
  runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.spike', 'post.V', 'syn.g', 'post.input'])
  runner.run(run_duration)

  # 可视化
  fig, gs = bp.visualize.get_figure(7, 1, 0.8, 6.)

  ax = fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
  plt.legend(loc='upper right')
  plt.title(title)
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[1:3, 0])
  plt.plot(runner.mon.ts, runner.mon['syn.g'], label=r'$g$', color=u'#d62728')
  plt.legend(loc='upper right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[3:5, 0])
  plt.plot(runner.mon.ts, runner.mon['post.input'], label='PSC', color=u'#d62728')
  plt.legend(loc='upper right')
  plt.xticks([])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  ax = fig.add_subplot(gs[5:7, 0])
  plt.plot(runner.mon.ts, runner.mon['post.V'], label='post.V')
  plt.legend(loc='upper right')
  plt.xlabel(r'$t$ (ms)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # plt.savefig('../img/DeltaSynapse.pdf', transparent=True, dpi=500)
  plt.show()


class DualExponential(bp.synapses.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., tau_decay=10., tau_rise=1., delay_step=1,
               E=0., syn_type='CUBA', method='exp_auto', **kwargs):
    super(DualExponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # 初始化参数
    self.tau_decay = tau_decay
    self.tau_rise = tau_rise
    self.g_max = g_max
    self.delay_step = delay_step
    self.E = E

    assert syn_type == 'CUBA' or syn_type == 'COBA'  # current-based 或 conductance-based
    self.type = syn_type

    # 获取关于连接的信息
    self.pre2post = self.conn.require('pre2post')  # 获取从pre到post的连接信息

    # 初始化变量
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.h = bm.Variable(bm.zeros(self.post.num))
    self.delay = bm.LengthDelay(self.pre.spike, delay_step)  # 定义一个延迟处理器

    # 定义微分方程及其对应的积分函数
    self.int_h = bp.odeint(method=method, f=lambda h, t: -h / self.tau_rise)
    self.int_g = bp.odeint(method=method, f=lambda g, t, h: -g / self.tau_decay + h)

  def update(self):
    tdi = bp.share.get_shargs()

    # 将突触前神经元传来的信号延迟delay_step的时间步长
    delayed_pre_spike = self.delay(self.delay_step)
    self.delay.update(self.pre.spike)

    # 根据连接模式计算各个突触后神经元收到的信号强度
    # at that time point it will sum up all spike from pre neuorns ( this will be collective spikes)
    # is using 7 pre and 5 post and all 7 are firing at same time 
    # it will give 5 columns of 7 
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
    # g和h的更新包括常规积分和突触前脉冲带来的跃变
    self.h.value = self.int_h(self.h, tdi.t, tdi.dt) + post_sp
    self.g.value = self.int_g(self.g, tdi.t, self.h, tdi.dt)

    # 根据不同模式计算突触后电流
    if self.type == 'CUBA':
      self.post.input += self.g  #* (self.E - (-65.))  # E - V_rest
    else:
      self.post.input += self.g * (self.E - self.post.V)


if __name__ == '__main__':
  run_syn(DualExponential,
          syn_type='CUBA',
          title='Dual Exponential Synapse Model (Current-Based)',
          sp_times=[25, 50, 75, 100, 150], g_max=1.)
  # run_syn(DualExponential,
  #         syn_type='COBA',
  #         title='Dual Exponential Synapse Model (Conductance-Based)',
  #         sp_times=[25, 50, 75, 100, 150], g_max=5.)


############################################################

# Method 2 
import brainpy as bp
from brainpy import neurons, synapses, synouts
import matplotlib.pyplot as plt

neu1 = neurons.LIF(1)
neu2 = neurons.LIF(1)
syn1 = synapses.DualExponential(neu1, neu2, bp.connect.All2All(), output=synouts.CUBA())
print(syn1.g_max,syn1.tau_decay, syn1.tau_rise)
net = bp.Network(pre=neu1, syn=syn1, post=neu2)

runner = bp.DSRunner(net, inputs=[('pre.input', 0.)], monitors=['pre.V','pre.spike', 'post.V', 'post.input' ,'syn.g', 'syn.h'])
runner.run(200.)

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre-V')
plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
# plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
plt.plot(runner.mon.ts, runner.mon['post.input'], label='h')
plt.legend()
plt.show()


########################################################################
# Method3 
import numpy as np
import brainpy as bp
import brainpy.math as bm

import matplotlib.pyplot as plt

class DualExpSparseCOBA(bp.Projection):
    def __init__(self, pre, post, delay, prob, g_max, tau_decay, tau_rise, E):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPreMg2(
            pre=pre,
            delay=delay,
            syn=bp.dyn.DualExpon.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise),
            comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
            out=bp.dyn.CUBA(), #COBA(E=E),
            post=post,
        )

class SimpleNet(bp.DynSysGroup):
    def __init__(self, syn_cls, E=0.):
        super().__init__()
        self.pre = bp.neurons.LIF(1) #bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
        self.post = bp.neurons.LIF(1) #bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                #  V_initializer=bp.init.Constant(-60.))
        self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1.,
                           tau_decay=10., tau_rise=1., E=E)

    def update(self,I_pre):
        self.pre(I_pre)
        self.syn()
        self.post()
        # monitor the following variables
        conductance = self.syn.proj.refs['syn'].g
        current = self.post.sum_inputs(self.post.V)
        return conductance, current, self.post.V

# I_pre = bp.inputs.constant_input(25.)
I_pre, duration = bp.inputs.section_input(values=[25.],
                                             durations=[200],
                                             return_length=True,
                                             dt=0.1)

indices = np.arange(200)  # 100 ms, dt= 0.1 ms
net = SimpleNet(DualExpSparseCOBA, E=0.)
# conductances, currents, potentials = bm.for_loop(net.step_run, indices, progress_bar=True)

def run(i, I_pre):
  conductances, currents, potentials = net.step_run(i, I_pre)
  return conductances, currents, potentials

duration=200.
indices = bm.arange(0, duration, bm.dt)
conductances, currents, potentials = bm.for_loop(run, [indices, I_pre])



ts = indices * bm.get_dt()
fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
fig.add_subplot(gs[0, 0])
plt.plot(ts, conductances)
plt.title('Syn conductance')
fig.add_subplot(gs[0, 1])
plt.plot(ts, currents)
plt.title('Syn current')
fig.add_subplot(gs[0, 2])
plt.plot(ts, potentials)
plt.title('Post V')
plt.show()

