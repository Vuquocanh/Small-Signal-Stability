# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np


""" Load System Data """
sys_data = sio.loadmat('system_q2.mat',squeeze_me=True) #squeeze_me=True gets rid of unnecessary nesting
A = sys_data['A']
B = sys_data['B']
C = sys_data['C']
D = sys_data['D']
strsps = sys_data['strsps']
statename = sys_data['StateName']

""" Determine Eigenvalues and calculate Eigenvectors and participation matrix """

""" Find the location of relevant indices in the original system """

""" reduce analysis to electromechanical modes """
"""############################################################################

To ease the analysis the participation matrix can be reduced to reflect only the states of interest in the modes of interest.
To provide a ranging the resulting participation matrix can further be normalized such that the
magitude of the generator with the highest participation is "1".   

############################################################################"""


""" Calculate transfer function residues  """
"""############################################################################

calculate the observability and controllability
reduce the matrices to observability & controllability of the relevant modes and the appropriate in- and outputs of the system 
 
res=em_w_obs*em_Vr_contr.T 

#should then provide a suitable format where the residues for all indicated modes appear column-wise 
and the rows reflect the different locations 

Normalization can be applied similar to the participation matrix.
############################################################################"""


