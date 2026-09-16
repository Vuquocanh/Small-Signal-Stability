# -*- coding: utf-8 -*-
import scipy.io as sio
import numpy as np
import control.statesp as stsp
import stsp_functions as f_stsp

""" Load System Data """
sys_data = sio.loadmat('system_q2.mat',squeeze_me=True) # event. use squeeze_me=True to get rid of unnecessary nesting
A = sys_data['A']
B = sys_data['B']
C = sys_data['C']
D = sys_data['D']
strsps = sys_data['strsps']
statename = sys_data['StateName']

""" Form State space object of the original system """
sys = stsp.StateSpace(A,B,C,D)

""" Add a PSS to the system """
"""############################################################################

Form a PSS state space object "pss" based on the desired settings
Add the pss object to an existing state space system "sys"
The function "pss_sys=addPSS(sys,pss,strsps,gen)" in stsp_functions.py returns a new state space system "pss_sys" 
where the PSS is added to machine "gen".

If another PSS should be added the function can be called again using the system that already contains 
a pss as input. For example: "pss2_sys=addPSS(pss_sys,pss2,strsps,gen2)"

############################################################################"""



""" Compare the systems """
"""############################################################################

Determine the damping and frequency of the different systems

For a start that could be:
    Original System
    System with PSS settings provided
    Improved PSS tuning

############################################################################"""





