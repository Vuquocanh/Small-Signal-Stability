# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import control.statesp as stsp
import matplotlib.pyplot as plt
import stsp_functions as f_stsp


""" Parameter definitions """
# Parameters for frequency sweep
fmin = 0 ; fmax = 2
f = np.arange(fmin,fmax,.001)

# PSS design parameters
gen = 4 #Specify Generator where the PSS should be tuned
Ks=20;Tw=10;Tn1=.05;Td1=.02;Tn2=3;Td2=5.4; #initial PSS parameters provided in the assignment


""" Load initial System Data """
sys_data = sio.loadmat('system_q2.mat',squeeze_me=True) #squeeze_me=True gets rid of unnecessary nesting
A = sys_data['A']
B = sys_data['B']
C = sys_data['C']
D = sys_data['D']
strsps = sys_data['strsps']
statename = sys_data['StateName']

""" Form State space object of the original system """
sys = stsp.StateSpace(A,B,C,D)


"""###################### Your code starts here ############################"""


""" Create PSS state space object """
"""############################################################################

the function pss_stsp in stsp_functions.py can be used to construct a 
PSS state space object based on the provided settings

############################################################################"""



""" Find the location of relevant indices in the original system """
"""############################################################################

statename contains the location of the states:
statename == 'psif'
    returns a vector of size (no. of states) which is "True" for all indices related to the field flux linkage states
    
np.logical_or() can be used to return True for multiple states
np.logical_not() would return all states but the indicated ones (convenient for excluding states)

strsps contains the indices related to the in- and outputs of the system:
    strsps['pe'].item() : returns the indices associated with the electric power output
All indices occur in generator order, i.e the first index is related to machine 1, etc.

############################################################################"""



""" Manipulate the system by removing the dynamic states """
"""############################################################################

Extract submatrices from the original systems A,B,C,D matrices and form a new (reduced) state space object sys_red

############################################################################"""



""" Determine the frequency response of the system and the PSS"""
"""############################################################################

mag, sys_phase, omega = sys_red.frequency_response(wsys, squeeze=None)
returns magnitude and phase of the state space system (single input single output) for the frequencies specified in wsys

############################################################################"""



"""Plot the frequency response curves"""
"""############################################################################

It might be a good idea to invert the system phase.
The ideal pss phase should fully compensate the phase of the system (pss_phase-sys_phase=0). 
Additionally you could plot the phase mismatch between the ideal compensation and the compensation using your parameters.

############################################################################"""




