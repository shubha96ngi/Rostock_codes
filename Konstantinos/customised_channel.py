# first example 
https://brainpy.readthedocs.io/en/latest/tutorial_building/build_conductance_neurons_v2.html
'''
class IK(bp.dyn.IonChannel):
  master_type = bp.dyn.HHTypedNeuron

  def __init__(self, size, E=-77., g_max=36., phi=1., method='exp_auto'):
    super().__init__(size)
    self.g_max = g_max
    self.E = E
    self.phi = phi

    self.integral = bp.odeint(self.dn, method=method)

  def dn(self, n, t, V):
    alpha_n = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta_n = 0.125 * bm.exp(-(V + 65) / 80)
    return self.phi * (alpha_n * (1. - n) - beta_n * n)

  def reset_state(self, V, batch_or_mode=None, **kwargs):
    self.n = bp.init.variable_(bm.zeros, self.num, batch_or_mode)

  def update(self, V):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    self.n.value = self.integral(self.n, t, V, dt=dt)

  def current(self, V):
    return self.g_max * self.n ** 4 * (self.E - V)

'''

#from example 
https://github.com/brainpy/BrainPy/blob/5e75f78ad27a8c761029779f3f76d262bf54ab0f/docs/tutorial_building/build_conductance_neurons_v2.ipynb#L364 
'''
class HTC(bp.dyn.CondNeuGroupLTC):
  def __init__(self, size, gKL=0.01, V_initializer=bp.init.OneInit(-65.)):
    super().__init__(size, A=2.9e-4, V_initializer=V_initializer, V_th=20.)
    self.IL = bp.dyn.IL(size, g_max=0.01, E=-70.)
    self.INa = bp.dyn.INa_Ba2002(size, V_sh=-45)
    self.Ih = bp.dyn.Ih_HM1992(size, g_max=0.01, E=-43)

    self.Ca = bp.dyn.CalciumDetailed(size, C_rest=5e-5, tau=10., d=0.5)
    self.Ca.add_elem(bp.dyn.ICaL_IS2008(size, g_max=0.5))
    self.Ca.add_elem(bp.dyn.ICaN_IS2008(size, g_max=0.5))
    self.Ca.add_elem(bp.dyn.ICaT_HM1992(size, g_max=2.1))
    self.Ca.add_elem(bp.dyn.ICaHT_HM1992(size, g_max=3.0))

    self.K = bp.dyn.PotassiumFixed(size, E=-90.)
    self.K.add_elem(bp.dyn.IKDR_Ba2002v2(size, V_sh=-30., phi=0.25))
    self.K.add_elem(bp.dyn.IK_Leak(size, g_max=gKL))

    self.KCa = bp.dyn.MixIons(self.K, self.Ca)
    self.KCa.add_elem(bp.dyn.IAHP_De1994v2(size))
'''

class IK1(bp.dyn.IKNI_Ya1989v2):
or do I need to write like this following first example 
#class IK(bp.dyn.IonChannel):
  master_type = bp.dyn.IKNI_Ya1989v2 #bp.dyn.HHTypedNeuron
 # master_type = bp.dyn.HHTypedNeuron

    def __init__(self, size) #, gKL=0.01, V_initializer=bp.init.OneInit(-65.)):
    
    def f_p_inf(self, V):
        return 1. / (1. + bm.exp(-(V +12.4) / 6.8))

    def f_p_tau(self, V):
        return (0.087 + 11.4 / (1. + bm.exp((V +14.6) / 8.6)))*(0.087 + 11.4 / (1. + bm.exp(-(V -1.3) / 18.7)))
    
    # but do not want to return current 
    def current(self, V, C, E):
       
    should I put pass here or 
    return 1. 
    #return self.g_max * self.p * (E - V)

class HH_FSI(bp.dyn.CondNeuGroupLTC):
    def __init__(self, size, gKL=0.01, V_initializer=bp.init.OneInit(-65.)):
        super().__init__(size, A=2.9e-4, V_initializer=V_initializer, V_th=20.)
        self.IK = bp.dyn.PotassiumFixed(size, E=-90.)
        # how to use add_elem  for IK1 ?
        # self.K.add_elem(bp.dyn.IKDR_Ba2002v2(size, V_sh=-30., phi=0.25))
        self.K.add_elem(IK1)  ??
        
        
        
