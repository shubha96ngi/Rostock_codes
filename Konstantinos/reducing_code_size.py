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
        self.IKdr.f_p_inf = 1. / (1. + bm.exp(-(V +58.3) / 6.7))
        self.IKdr.f_p_tau = 0.5 + 14. / (1. + bm.exp((V + 60) / 12))
        # but then how it will access self.p ?
        self.IKdr.current = self.g_max * self.p **3 *  1. / (1. + bm.exp(-(V +24) / 11.5))* (self.E - V)
        
