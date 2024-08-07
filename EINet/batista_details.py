#from the paper https://www.researchgate.net/publication/340162380_Effect_of_interpopulation_spike-timing-dependent_plasticity_on_synchronized_rhythms_in_neuronal_networks_with_inhibitory_and_excitatory_populations
#it is clear that we will use  inhibitory synapse (GABA, Vsyn = -80) and excitatory synapse (AMPA, Vsyn= 0)
# some parameters are also provided for this please do check 

# parameters are from paper https://www.researchgate.net/publication/354251409_The_Hodgkin-Huxley_neuron_model_for_motion_detection_in_image_sequences


#https://colab.ws/articles/10.1016%2Fj.conb.2023.102707 
# a good website for reading important paper related to HH model 

# some useful links
#https://brainpy.readthedocs.io/en/latest/apis/auto/brainpy-changelog.html#id36
'''
class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size)

    self.INa = channels.INa_HH1952(size, )
    self.IK = channels.IK_HH1952(size, )
    self.IL = channels.IL(size, E=-54.387, g_max=0.03)
#https://github.com/brainpy/BrainPy/blob/5e75f78ad27a8c761029779f3f76d262bf54ab0f/tests/simulation/test_neu_HH.py#L11

'''
# same as batista model but gl =0.03 which is correct one  instead of 0.3 
# one paper 
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2638500/

# it has gaba synapse and ampa synapse with E= 20 and E= -75
#https://github.com/brainpy/BrainPy/blob/5e75f78ad27a8c761029779f3f76d262bf54ab0f/brainpy/_src/tests/test_access_methods.py#L9

# similar model paper 
# http://staff.ustc.edu.cn/~hzhlj/paper/44.pdf

# following citation of paper (Exci-Inhi Network model) 
#https://www.researchgate.net/publication/362417564_Synchronization_and_oscillation_behaviors_of_excitatory_and_inhibitory_populations_with_spike-timing-dependent_plasticity
#, I found the link
#https://github.com/JonathanAMichaels/hebbRNN



