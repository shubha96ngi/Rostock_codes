# This file contains all the parameters.
# Additional comments are provided in the header of main.py

print '********'
print '* Init *'
print '********'

import os
isunix = lambda: os.name == 'posix'
import matplotlib   
if isunix():
    matplotlib.use('Agg') # no graphic link with Unix cluster for me

import brian_no_units
from brian import * 
from scipy.io import *
from scipy import weave
from time import time
from time import localtime
from customrefractoriness import *
import pickle
import glob


set_global_preferences(useweave=True) # to compile C code

globalStartTime=time()*second


#*************************
# COMPUTATION PARAMETERS *
#*************************
neuronTimeCompression = 2.0**0 # to compress all the neuronal time constants
pbTimeCompression = 2.0**0 # to compress the problem and oscillation time constants

get_default_clock().set_dt(.1*ms/neuronTimeCompression) # set time step

# random state
randState = 28 # use this to specify it
seed(randState)

imposedEnd = 3*second/pbTimeCompression # imposed end time
N = 2000 # number of presynaptic neurons
nG = 1 # number of gmax values
nR = 1 # number of ratio LTD/LTP
M = nG*nR # number of postsynaptic neurons, numbered like that [ (r_0,g_0)...(r_0,g_nG),(r_1,g_0)...(r_1,g_nG),...,(r_nR,g_0)...(r_nR,g_nG)]

recomputeSpikeList = True # recompute input spike trains as opposed to load appropriate mat files
dumpSpikeList = False # save input spike list in mat files
computeOutput = True # compute output (STDP) layer

useSavedWeight = False # load previously dumped weights
timeOffset = 0*second # simulation starts at t=timeOffset

useReset = False # impose resets on input layer at dates specified in reset.###.mat file

graph = True # graph output
monitorInput = True # monitor input spikes
monitorOutput = True # monitor output spikes
monitorPot = True # monitor potential in output layer
monitorCurrent = False # monitor potential in output layer
monitorInputPot = False # monitor potential in input layer
monitorRate = False # monitor rates in output layer
isMonitoring = False # flag saying if currently monitoring
monitorTime = (imposedEnd-6/pbTimeCompression)*second # start monitoring only at that time (to save memory)
analyzePeriod = Inf # periodically launches analyze.py

if not recomputeSpikeList and dumpSpikeList:
    print 'Warning: dumping a spike list which is not re-computed makes no sense. Setting dumpSpikeList to False'
    dumpSpikeList = False

if not computeOutput and ( monitorOutput or monitorPot or monitorCurrent or monitorRate):
    print 'Warning: can not monitor output, which is not computed. Setting monitorOutput, monitorPot and monitorCurrent to False'
    monitorOutput = False
    monitorPot = False
    monitorCurrent = False
    monitorRate = False

# load pattern values (the values are only useful for plotting), but the length of the vector is used to scale gmax
if os.path.exists(os.path.join('..','data','realValuedPattern.'+'%03d' % (randState)+'.mat')):
    realValuedPattern=loadmat(os.path.join('..','data','realValuedPattern.'+'%03d' % (randState)+'.mat'))
    realValuedPattern=realValuedPattern['realValuedPattern']
else:
    realValuedPattern=zeros(round(.5*N))
    
#**************************
# NEURON MODEL PARAMETERS *
#**************************

# Types of input and output neurons
poissonInput = False # poisson input neurons (as opposed to LIF)
conductanceOutput = False # conductance-base output neurons (as opposed to LIF). Tim: 9/2008: has never been critical so far
poissonOutput = False # stochastic (Poisson) output neurons (as opposed to deterministic). Note that their differential equations are the same as the LIF, but firing is stochastic.

#neurons (Dayan&Abbott 2001)
refractoryPeriod = 1*ms/neuronTimeCompression
R=10*Mohm
if poissonOutput:
    vt=400 # not used (just for graph scaling)
    vr=0 # not used
    El=-450 # resting
#    El=-280 # resting (use that value to have more false alarms)
else:
    vt=-54*mV # threshold
    vr=-60*mV # reset
    El=-70*mV # resting
    Ee=0*mV # for excitatory conductance
taum = 20*ms/neuronTimeCompression # membrane time constant
taue=taum/4 # synapse time constant

sigma = 0.015*(vt-vr) # white Gaussian noise. Applies to input and output neurons

# array of max conductance values
unitaryEffect = taue/(taum-taue)*((taum/taue)**(-taue/(taum-taue))-(taum/taue)**(-taum/(taum-taue))) # this corresponds to the maximum of the kernel (exp(-t/taum) - exp(-t/taue)) taue / (taum-taue)
if poissonOutput:  
    gmax=1.05**-10*2.5*1500/(1.0*size(realValuedPattern))*1.0/unitaryEffect*exp(2*log(1.05)*(array(nR*range(nG))-nG/2)) # appropriate for (1000/2000 afferents, PLoS patten)
else:
    gmax = 1.05**-2/(size(realValuedPattern))*(vt-El)/unitaryEffect*exp(2*log(1.05)*(array(nR*range(nG))-nG/2)) # x=10%

if conductanceOutput:
    gmax /= (Ee-vt)

print 'gmax=' + str(gmax)

#********************
# WEIGHT PARAMETERS *
#********************
# initial synaptic weight are randomly picked (uniformly) between those two bounds
if poissonOutput:
    initialWeight_min = 0
    initialWeight_max = 10    
else:
    initialWeight_min = 0*volt
    initialWeight_max = 2*8.6e-5*volt
burstingCriterion = .5 # unplug neurons whose mean normalized synaptic weight is above this value. This allows to save memory by not computing neurons whose normalized synaptic weights all go to 1.

#***************************
# INPUT CURRENT PARAMETERS *
#***************************    
# used only if recomputeSpikeList = True

# input current (normalization with respect to raw values in input file)
if poissonInput: # in this case I values are in fact firing rates
    Imin = 0
    Imax = 100
else:
    Imax = (1.07)*(vt-El)/R*ones(N) # Tim 12/08: appropriate for (8Hz sin oscillations). 1-3 spikes/cycle
    Imin = (0.95)*(vt-El)/R*ones(N) # Tim 12/08: appropriate for (8Hz sin oscillations). 1-3 spikes/cycle

a = .075*(vt-El)/R*ones(N)
# oscillation fequence
oscilFreq = 8*Hz*pbTimeCompression

#******************
# STDP PARAMETERS *
#******************    
nearestSpike = False # Tim 08/2008: not critical with low input spike rates (10Hz)
tau_post=33.7*ms #(source: Bi & Poo 2001)
tau_pre=16.8*ms #(source: Bi & Poo 2001)
a_pre= .005 # Tim 12/08: no more that .005
w_out = - 0*.005/3 # see Matt Gilson
w_in=zeros(M) # array of w_in
for i in range(nR): # here nR corresponds to number of w_in/w_out ratios (and not number of LTD/LTP ratios)
    # see Matt Gilson
    w_in[i*nG:(i+1)*nG] = -w_out*1.05**0*exp((i-nR/2)*.5*log(2))
a_post=zeros(M) # array of LTD/LTP ratios
for i in range(nR):
   a_post[i*nG:(i+1)*nG] = -a_pre*1.05**8*exp((i-nR/2)*1*log(1.05)) # x=10%

mu = 0.0 # see Gutig et al. 2003.
print 'normalized a_post/a_pre ratios' + str(a_post/a_pre)

#***********
# FUNCIONS *
#***********

def printtime(mess):
    t = localtime()
    print  '%02d' % t[3] + ':' + '%02d' % t[4] + ' ' + mess

def listMatFile(directory,randState):                                        
    "get list of spike lists" 
    fileList = os.listdir(directory)
    fileList.sort()
    fileList = [f 
               for f in fileList
                # e.g. spikeList.200.010.mat
                #      01234567890123456789
                if f[0:9] in ['spikeList'] and f[10:13] in ['%03d' % (randState)] and f[-4:] in ['.mat'] ]
    return fileList


# Called whenever output neurons fire. Resets the potential and trigger STDP updates.
# Includes C code, will be compiled the first time it is called
# Param:
# P: NeuronGroup (output neurons)
# spikes: list of the indexes of the neuron that fire.
def neurons_reset(P,spikes):
    if size(spikes):
        if not poissonOutput:
            P.v_[spikes]=vr # reset pot
        if nearestSpike:
            nspikes = size(spikes)
            A_pre = mirror.A_pre_
            code = '''
            for(int si=0;si<nspikes;si++)
            {
                int i = spikes(si);
                for(int j=0;j<N;j++)
                {
                    if(!_alreadyPotentiated(j,i))
                    {
                        double wnew;
                        if(mu==0) { /* additive. requires hard bound */
                            wnew = _synW(j,i)+_gmax(i)*(w_out+A_pre(j));
                            if(wnew>_gmax(i)) wnew = _gmax(i);
                            if(wnew<0) wnew = 0.0;
                        }
                        else { /* soft bound */
                            wnew = _synW(j,i)+_gmax(i)*(w_out+A_pre(j)*exp(mu*log(1-_synW(j,i)/_gmax(i))));
                            if(wnew>_gmax(i)) wnew = _gmax(i);
                            if(wnew<0) wnew = 0.0;
						}
                        _synW(j,i) = wnew;
                        _alreadyPotentiated(j,i) = true;
                    }
                }
            }
            '''
            weave.inline(code,
                        ['spikes', 'nspikes', 'N', '_alreadyPotentiated', '_synW', '_gmax', 'A_pre', 'mu','w_out'],
                        compiler='gcc',
                        type_converters=weave.converters.blitz,
                        extra_compile_args=['-O3'])
            _alreadyDepressed[:,spikes] = False
            P.A_post_[spikes]=a_post[spikes] # reset A_post (~start a timer for future LTD)

        else: # all spikes
            nspikes = size(spikes)
            A_pre = mirror.A_pre_
            code = '''
            for(int si=0;si<nspikes;si++)
            {
                int i = spikes(si);
                for(int j=0;j<N;j++)
                {
                        double wnew;
                        if(mu==0){ /* additive. requires hard bound */
                            wnew = _synW(j,i)+_gmax(i)*(w_out+A_pre(j));
                            if(wnew>_gmax(i)) wnew = _gmax(i);
                            if(wnew<0) wnew = 0.0;
                        }
                        else { /* soft bound */
                            wnew = _synW(j,i)+_gmax(i)*(w_out+A_pre(j)*exp(mu*log(1-_synW(j,i)/_gmax(i))));
                            if(wnew>_gmax(i)) wnew = _gmax(i);
                            if(wnew<0) wnew = 0.0;
						}
                        _synW(j,i) = wnew;
                }
            }
            '''
            weave.inline(code,
                        ['spikes', 'nspikes', 'N', '_synW', '_gmax', 'A_pre', 'mu','w_out'],
                        compiler='gcc',
                        type_converters=weave.converters.blitz,
                        extra_compile_args=['-O3'])
            P.A_post_[spikes]+=a_post[spikes] # reset A_post (~start a timer for future LTD)

# Called whenever input neurons fire. Resets the potentials and trigger STDP updates.
# Note that mirror is a fake group that mirrors input neuron, only used for implementation issues.
# Includes C code, will be compiled the first time it is called
# Param:
# P: NeuronGroup (input neurons)
# spikes: list of the indexes of the neuron that fire.
def mirror_reset(P,spikes):
    if size(spikes):
        P.v_[spikes] = 0
        if nearestSpike:
            nspikes = size(spikes)
            A_post = neurons.A_post_
            code = '''
            for(int si=0;si<nspikes;si++)
            {
                int i = spikes(si);
                for(int j=0;j<M;j++)
                {
                    if(!_alreadyDepressed(i,j))
                    {
                        double wnew;
                        if(mu==0) { /* additive. requires hard bound */
                            wnew = _synW(i,j)+_gmax(j)*(w_in(j)+A_post(j)); 
                            if(wnew>_gmax(j)) wnew = _gmax(j);
                            if(wnew<0.0) wnew = 0.0;
                        }
                        else { /* soft bound */
                            wnew = _synW(i,j)+_gmax(j)*(w_in(j)+A_post(j)*exp(mu*log(_synW(i,j)/_gmax(j))));
                            if(wnew>_gmax(j)) wnew = _gmax(j);
                            if(wnew<0.0) wnew = 0.0;
						}
                        _synW(i,j) = wnew;
                        _alreadyDepressed(i,j) = true;
                    }
                }
            }
            '''
            weave.inline(code,
                        ['spikes', 'nspikes', 'M', '_alreadyDepressed', '_synW', '_gmax', 'A_post', 'mu','w_in'],
                        compiler='gcc',
                        type_converters=weave.converters.blitz,
                        extra_compile_args=['-O3'])
            _alreadyPotentiated[spikes,:]=False
            P.A_pre_[spikes]=a_pre  # reset A_pre (~start a timer for future LTP)
            
        else: # all spikes
            nspikes = size(spikes)
            A_post = neurons.A_post_
            code = '''
            for(int si=0;si<nspikes;si++)
            {
                int i = spikes(si);
                for(int j=0;j<M;j++)
                {
                        double wnew;
                        if(mu==0) { /* additive. requires hard bound*/
                            wnew = _synW(i,j)+_gmax(j)*(w_in(j)+A_post(j));
                            if(wnew>_gmax(j)) wnew = _gmax(j);
                            if(wnew<0.0) wnew = 0.0;
                        }
                        else { /* soft bound */
                            wnew = _synW(i,j)+_gmax(j)*(w_in(j)+A_post(j)*exp(mu*log(_synW(i,j)/_gmax(j))));
                            if(wnew>_gmax(j)) wnew = _gmax(j);
                            if(wnew<0.0) wnew = 0.0;
						}
                        _synW(i,j) = wnew;
                }
            }
            '''
            weave.inline(code,
                        ['spikes', 'nspikes', 'M', '_synW', '_gmax', 'A_post', 'mu','w_in'],
                        compiler='gcc',
                        type_converters=weave.converters.blitz,
                        extra_compile_args=['-O3'])
            P.A_pre_[spikes]+=a_pre  # reset A_pre (~start a timer for future LTP)
