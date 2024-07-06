import numpy as np
import brainpy as bp
import brainpy.math as bm

# for MSN neurons 
class IM(bp.dyn.IonChannel):
    master_type = bp.dyn.HHTypedNeuron

    def __init__(self, size, E=-100., g_max=1.3, phi=1., method='exp_auto'):
        super(IM, self).__init__(size)
        self.g_max = g_max
        self.E = E
        self.phi = phi
        self.integral = bp.odeint(self.dm, method=method)

    def dm(self, m, t, V):
        Q10 = 2.3
        Qs = bm.power(Q10, 0.1*(37-23))
        
        alpha_m = Qs *0.0001* (V + 30) / (1 - bm.exp(-(V + 30) / 9))
        beta_m =  -Qs *0.0001* (V + 30) / (1 - bm.exp(-(V + 30) / 9))
        return self.phi * (alpha_m * (1. - m) - beta_m * m)

   
    def reset_state(self, V, batch_or_mode=None, **kwargs):
        self.m = bp.init.variable_(bm.zeros, self.num, batch_or_mode)
        

    def update(self, V):
        t = bp.share.load('t')
        dt = bp.share.load('dt')
        self.m.value= self.integral(self.m,  t, V, dt=dt)

    def current(self, V):
        return self.g_max * self.m * (self.E - V)


class HH_MSN(bp.dyn.CondNeuGroupLTC):
    def __init__(self, size):
        super().__init__(size) 
    #     self.K = bp.dyn.SodiumFixed(size, E = 50)
    #     self.INa = bp.dyn.INa_Ba2002(size, g_max = 100., V_sh=-67.,phi=1.) # done 
    #     self.K = bp.dyn.PotassiumFixed(size, E= -100.)
    #     self.IK = bp.dyn.IKDR_Ba2002v2(size, g_max = 80., V_sh=-67., phi=1.) # done
        self.IL = bp.channels.IL(size, g_max=0.1, E=-67.) # done 
        self.INa = bp.channels.INa_Ba2002(size, g_max = 100.,E=50., V_sh=-67.,phi=1.) # done 
        self.IK = bp.channels.IKDR_Ba2002v2(size, g_max = 80.,E=-100., V_sh=-67., phi=1.) # done
        self.IM_type = IM(size)  # done I guess # M current channel 

#==============================================================================
# for FSI neurons 
class HH_FSI(bp.dyn.CondNeuGroupLTC):
    def __init__(self, size):
        super().__init__(size) 

        self.ILF = bp.channels.IL(size, g_max=0.25, E=-70.) # done 
      
        self.INaF = bp.channels.IKNI_Ya1989(size,g_max=112.5, E=50.) # done
        self.INaF.f_p_inf =  1. / (1. + bm.exp(-(self.V +58.3) / 6.7))
        self.INaF.f_p_tau =  0.5 + 14. / (1. + bm.exp((self.V + 60) / 12))
        self.INaF.m_inf = 1. / (1. + bm.exp(-(self.V +24) / 11.5))
        # always access inner variables like this .
        self.INaF.current = self.INaF.g_max * self.INaF.p **3 *self.INaF.m_inf* (self.INaF.E - self.V)
      
        self.IKF = bp.channels.IKK2A_HM1992(size,g_max=6, E=-90.) # done
        self.IKF.f_p_inf = 1. / (1. + bm.exp(-(self.V +50) / 20))
        self.IKF.f_p_tau =  -50
        self.IKF.f_q_inf = 1. / (1. + bm.exp(-(self.V +70) / 6))
        self.IKF.f_q_tau =  150
        self.IKF.current = self.IKF.g_max * self.IKF.p **3 *self.IKF.q* (self.IKF.E - self.V)
      
        self.IKdr = bp.channels.IKNI_Ya1989(size,g_max=225, E=-90.) #IKdr(size)  # done I guess # M current channel 
        self.IKdr.f_p_inf = 1. / (1. + bm.exp(-(self.V +12.4) / 6.8))
        self.IKdr.f_p_tau =  (0.087 + 11.4 / (1. + bm.exp((self.V +14.6) / 8.6)))*(0.087 + 11.4 / (1. + bm.exp(-(self.V -1.3) / 18.7)))
        self.IKdr.current = self.IKdr.g_max * self.IKdr.p **2 * (self.IKdr.E - self.V)

# now lets define connections for FSI, MSN 
        
