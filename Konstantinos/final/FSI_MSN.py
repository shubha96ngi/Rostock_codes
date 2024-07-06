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
        self.IL = bp.channels.IL(size, g_max=0.1, E=-67.) # done 
        # ERROR in passing values for phi and E
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



MSN = HH_MSN(1856)
FSI = HH_FSI(99)

# define connectivity matrix # import the matrix data 
a = sio.loadmat('striat2netround.mat') #
# adjacency matrix which describes connection between FSI-FSI, FSI-MSN and MSN-MSN neurons 
Adj2, iMSN, FS = a['Adj2'], a['iMSN'], a['FS']  
# Network topology # calc connectivity matrix # FSI and MSN neuron network connectivity structure  
# fully or arbitrarily connected network  #with weak coupling epsilon = k/n
def connectSTR2(matrix, iMSN, FS):
    N1 = len(iMSN[0])
    N2 = len(FS[0])

    AM2M = np.zeros((N1, N1))
    AF2M = np.zeros((N2, N1))
    AM2F = np.zeros((N1, N2))
    AF2F = np.zeros((N2, N2))

    for i in range(N1):
        x = iMSN[0][i]
        ind1 = np.where(matrix[x-1, :] > 0)[0]
        for kk in range(len(ind1)-2):
            yy = np.where(iMSN == ind1[kk])[1]
            if np.size(yy)>0:
                AM2M[i, yy[0]+1] = 1

    for i in range(N1):
        x = iMSN[0][i]
        ind1 = np.where(matrix[x-1, :] > 0)[0]

        for kk in range(len(ind1)):
            yy = np.where(FS == ind1[kk])[1]
            if np.size(yy)>0:

                AM2F[i, yy[0]] = 1 

    for i in range(N2):
        x = FS[0][i]
        ind1 = np.where(matrix[x-1, :] > 0)[0]
        for kk in range(len(ind1)-1):   
            yy = (np.where(iMSN == ind1[kk])[1])
            if np.size(yy)>0:
                AF2M[i,yy[0]+1] = 1

    for i in range(N2):
        x = FS[0][i]
        ind1 = np.where(matrix[x-1, :] > 0)[0]
        for kk in range(len(ind1)):
            yy = np.where(FS == ind1[kk])[1]

            if np.size(yy)>0:
                AF2F[i, yy[0]] = 1

    return AM2M, AF2M, AM2F, AF2F
AM2M, AF2M, AM2F, AF2F = connectSTR2(Adj2, iMSN, FS)    #striatal_connectiviyt matrix
'''
from scipy.sparse import csr_matrix
# connectivity matrix from MSN to MSN 
conn_mat_M2M = AM2M[:10,:10].astype(bool) #np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
sparse_mat_M2M = csr_matrix(conn_mat_M2M)
conn_M2M = bp.conn.SparseMatConn(sparse_mat_M2M)
conn_M2M = conn_M2M(pre_size=sparse_mat_M2M.shape[0], post_size=sparse_mat_M2M.shape[1])
#res = conn_M2M.requires('pre_ids', 'post_ids','pre2post', 'pre2syn', 'conn_mat_M2M')
#resM2M = conn_M2M.requires('conn_mat')

# connectivity matrix from MSN to FSI 
conn_mat_M2F = AM2F[:10,:10].astype(bool) #np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
sparse_mat_M2F = csr_matrix(conn_mat_M2F)
conn_M2F = bp.conn.SparseMatConn(sparse_mat_M2F)
conn_M2F = conn_M2F(pre_size=sparse_mat_M2F.shape[0], post_size=sparse_mat_M2F.shape[1])
# res = conn_M2F.requires('pre_ids', 'post_ids','pre2post', 'pre2syn', 'conn_mat_M2F')
# res = conn_M2F.requires('conn_mat')

# connectivity matrix from FSI to FSI
conn_mat_F2F = AF2F[:10,:10].astype(bool) #np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
sparse_mat_F2F = csr_matrix(conn_mat_F2F)
conn_F2F = bp.conn.SparseMatConn(sparse_mat_F2F)
conn_F2F = conn_F2F(pre_size=sparse_mat_F2F.shape[0], post_size=sparse_mat_F2F.shape[1])
# res = conn_F2F.requires('pre_ids', 'post_ids','pre2post', 'pre2syn', 'conn_mat_F2F')
# res = conn_F2F.requires('conn_mat')

# connectivity matrix from FSI to MSN 
conn_mat = AF2M[:10,:10].astype(bool) #np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
sparse_mat_F2M = csr_matrix(conn_mat)
conn_F2M = bp.conn.SparseMatConn(sparse_mat_F2M)
conn_F2M = conn_F2M(pre_size=sparse_mat_F2M.shape[0], post_size=sparse_mat_F2M.shape[1])
# reF = conn_F2M.requires('pre_ids', 'post_ids','pre2post', 'pre2syn', 'conn_mat_M2F')
'''
# from dedicated operators 
'''
conn_F2M.requires('pre_ids')
AM2M  =bm.array(AM2M[:10,:10])
data = AM2M[bm.nonzero(AM2M)]
connection = bp.conn.MatConn(AM2M.value)
indices, indptr = connection(AM2M.shape[0],AM2M.shape[1]).require('pre2post')
#bm.sparse.csrmv(data, indices=indices, indptr=indptr, vector=pre_activity, shape= AM2M.shape, transpose=True)
'''

# now lets define connections for FSI, MSN sparse network ( pre2syn and post2syn_event_sum) to make if fast 

class BaseAMPASyn(bp.SynConn):
  def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0., alpha=0.98,
               beta=0.18, T=0.5, T_duration=0.5, method='exp_auto'):
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

class AMPASparse(BaseAMPASyn):
  def __init__(self, *args, **kwargs):
    super(AMPASparse, self).__init__(*args, **kwargs)

    # connection matrix
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # synapse gating variable
    # -------
    # NOTE: Here the synapse shape is (num_syn,)
    self.g = bm.Variable(bm.zeros(len(self.pre_ids)))

  def update(self, tdi, x=None):
    _t, _dt = tdi.t, tdi.dt
    delayed_spike = self.pre_spike(self.delay_step)
    self.pre_spike.update(self.pre.spike)
    # get the time of pre spikes arrive at the post synapse
    self.spike_arrival_time.value = bm.where(delayed_spike, _t, self.spike_arrival_time)
    # get the arrival time with the synapse dimension
    arrival_times = bm.pre2syn(self.spike_arrival_time, self.pre_ids)
    # get the neurotransmitter concentration at the current time
    TT = ((_t - arrival_times) < self.T_duration) * self.T
    # integrate the synapse state
    self.g.value = self.integral(self.g, _t, TT, dt=_dt)
    # get the post-synaptic current
    # why not pre2post eventsum? is it only for expsparse ?
    g_post = bm.syn2post(self.g, self.post_ids, self.post.num)
    self.post.input += self.g_max * g_post * (self.E - self.post.V)

# for sparse connectivity 
class AMPA(bp.Projection):
    def __init__(self, pre, post, delay, prob, weight, E=0.):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPreMg2(
          pre=pre,
          delay=delay,
          syn=bp.dyn.AMPA.desc(pre.num, alpha=0.98, beta=0.18, T=0.5, T_dur=0.5),
          comm=bp.dnn.CSRLinear(weight,conn=conn)
          #comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), weight),
          out=bp.dyn.COBA(E=E),
          post=post,
        )
Esyn_inhi = -80
IsynF2M = AMPA(FSI, MSN, delay=0, conn=conn_F2M, weight=0.005, E=E_syn_inhi)
IsynF2F = AMPA(FSI, FSI, delay=0, conn=conn_F2F, weight=0.005, E=E_syn_inhi)
IsynM2M = AMPA(MSN, MSN, delay=0, conn=conn_M2M, weight=0.02, E=E_syn_inhi)
IsynM2F = AMPA(MSN, FSI, delay=0, conn=conn_M2F, weight=0.02, E=E_syn_inhi)

'''
IsynF2M = AMPAsparse(pre=FSI, post=MSN, conn=AF2M) #bp.connect.All2All
IsynF2F = AMPAsparse(pre=FSI, post=FSI, conn=AF2F)
IsynM2M = AMPAsparse(pre=MSN, post=MSN, conn=AM2M)
IsynM2F = AMPAsparse(pre=MSN, post=FSI, conn=AM2F)
'''

# now define potential .. but use EI balanced network .. EInet 
I1 = 5 ; I2= 7
pre_input = IDBS -IsynM2M - IsynF2M -I1 
post_input = I2 -IsynF2F - IsynM2F
runner = bp.DSRunner(net,
                       monitors=['pre.V', 'post.V', 'syn.g','pre.spike'],
                       inputs=['pre.input',pre_input, 'post_input', post_input])
# and what else 
# use STP equation 
# synapse mn alpha beta wali equation replace hoke it will become like STP 
# it means it is exponential synapse 
# lets check STP code once 







        
