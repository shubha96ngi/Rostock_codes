# trying to matxh STDP code with neuromatch academy example 
# generated spike train same as what they are generating poisson_generator for presynaptic neuron

import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np


pre1 = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0), times=(183, 290, 349, 539, 919, 943, 951, 954, 1104, 1247, 1257,
                                                               1331, 1555, 2004, 2228, 2250, 2251, 2506, 2531, 2538,
                                                               2614, 2619,
                                                               2632, 2672, 2678, 2787, 2792, 3031, 3059, 3273, 3440,
                                                               3562, 3655,
                                                               3667, 3847, 3894, 3906, 4015, 4022, 4023, 4035, 4084,
                                                               4198, 4293,
                                                               4383, 4868, 4873, 4888, 4959))  # , 30., 50., 70.))

post1 = bp.neurons.LIF(1, V_rest=-75., V_reset=-75., V_th=-55., V_initializer=bp.init.Constant(-65.), tau=10.,
                      tau_ref=2.)

class STDPNet(bp.DynamicalSystem):
   def __init__(self, num_pre, num_post):
     super().__init__()
     self.pre = pre1#bp.dyn.LifRef(num_pre)
     self.post = post1 #bp.dyn.LifRef(num_post)
     self.syn = bp.dyn.STDP_Song2000(
       pre=self.pre,
       delay=1.,
       comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                                  weight=bp.init.Uniform(max_val=0.024)),
       syn=bp.dyn.Expon.desc(self.post.varshape, tau=5.),
       out=bp.dyn.COBA.desc(E=0.),
       post=self.post,
       tau_s=20.,
       tau_t=20.,
       A1=0.008,
       A2=0.008*1.10,
     )

   def update(self):
     self.syn()
     self.pre()
     self.post()
     conductance = self.syn.refs['syn'].g
     Apre = self.syn.refs['pre_trace'].g
     Apost = self.syn.refs['post_trace'].g
     current = self.post.sum_inputs(self.post.V)
     return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, self.syn.comm.weight,self.post.V

I_pre = 0
I_post = 0
net = STDPNet(1, 1)

runner = bp.DSRunner(target=net,
                     monitors=['pre.spike','post.V'],
                     jit=True)
# run the simulation
runner.run(duration=1000.)
fig, gs = bp.visualize.get_figure(1, 1, 1, 10)
bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], ax=fig.add_subplot(gs[0, 0]),color='orange')
bp.visualize.raster_plot(runner.mon.ts, runner.mon['pre.spike'],show=True)
print(runner.mon.ts.shape)
