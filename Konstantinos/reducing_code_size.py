import brainpy as bp 
#bp.dyn.IKNI_Ya1989 from 
#https://github.com/brainpy/BrainPy/blob/5e75f78ad27a8c761029779f3f76d262bf54ab0f/brainpy/_src/dyn/channels/potassium_compatible.py
class IKdr(bp.dyn.IKNI_Ya1989):
    def __init__(self, size):
        super().__init__(size,g_max=225, E=-90.) 

    def f_p_inf(self, V):
        return 1. / (1. + bm.exp(-(V +12.4) / 6.8))

    def f_p_tau(self, V):
        return (0.087 + 11.4 / (1. + bm.exp((V +14.6) / 8.6)))*(0.087 + 11.4 / (1. + bm.exp(-(V -1.3) / 18.7)))

    def current(self, V):
        return self.g_max * self.p **2 * (self.E - V)

# Is it equivalent to this 

class HH_FSI(bp.dyn.CondNeuGroupLTC):
    def __init__(self, size):
        super().__init__(size,g_max,E) 
      
        self.IKdr = bp.channels.IKNI_Ya1989(size,g_max=112.5, E=50.)#IKdr(size)  # done I guess # M current channel 
        self.IKdr.f_p_inf = 1. / (1. + bm.exp(-(V +12.4) / 6.8))
        self.IKdr.f_p_tau =  (0.087 + 11.4 / (1. + bm.exp((V +14.6) / 8.6)))*(0.087 + 11.4 / (1. + bm.exp(-(V -1.3) / 18.7)))
        #but then how will it access self.p or will it ?
        self.IKdr.current = self.g_max * self.p **2 * (self.E - V)
        
        
