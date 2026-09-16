# -*- coding: utf-8 -*-
import numpy as np
import control.statesp as stsp

# Forms a washout filter state space object
def wo_stsp (Tw):
    A = -1/Tw; B = 1/Tw
    C = -1; D = 1
    sys_Tw = stsp.StateSpace(A,B,C,D)
    return sys_Tw
   
# Forms a lead lag state space object 
def ldlg_stsp(Tn,Td):
    A =-1/Td; B = (1-Tn/Td)/Td
    C =1; D = Tn/Td
    sys_ldlg = stsp.StateSpace(A,B,C,D)
    return sys_ldlg

# forms a gain state space object
def gain_stsp(K):
    A =[]; B = []; C = []; D = K
    sys_gain = stsp.StateSpace(A,B,C,D)
    return sys_gain

# Connect two state space objects in series: 
# The input of the resulting system is the input of the first system
# The output of the resulting system is the output of the second system
# The input of the second system is the output of the first
def series_connect_siso_stsp (sys1,sys2):
    A1 = sys1.A; B1 = sys1.B; C1 = sys1.C; D1 = sys1.D
    A2 = sys2.A; B2 = sys2.B; C2 = sys2.C; D2 = sys2.D
    n1 = len(A1);n2 = len(A2)   
    A = np.zeros((n1+n2,n1+n2))
    B = np.zeros((n1+n2,1));
    C = np.zeros((1,n1+n2));
    D = 0;
    A1_idx = np.arange(0,n1,1); A2_idx = np.arange(n1,n1+n2,1);
    A[np.ix_(A1_idx,A1_idx)] = A1; A[np.ix_(A2_idx,A2_idx)] = A2; #diagonal elements
    A[np.ix_(A2_idx,A1_idx)] = B2*C1; #linking element
    B[np.ix_(A1_idx)] = B1; B[np.ix_(A2_idx)] = B2*D1;
    C[0,A1_idx] = D2*C1; C[0,A2_idx] = C2;
    D = D1*D2;
    con_sys = stsp.StateSpace(A,B,C,D)
    return con_sys

#Creates a PSS state space object based on the settings provided
# The structure here is: Gain - Washout - Lead-Lag 1 - Lead-Lag 2
# gain can either be applied to the in- or the output of the system
def pss_stsp(K,Tw,Tn1,Td1,Tn2,Td2):
    wo = wo_stsp(Tw); 
    ldlg1 = ldlg_stsp(Tn1,Td1); 
    ldlg2 = ldlg_stsp(Tn2,Td2); 
    gain = gain_stsp(K)
    pss = series_connect_siso_stsp(gain,wo)
    pss = series_connect_siso_stsp (pss,ldlg1) 
    pss = series_connect_siso_stsp(pss,ldlg2)   
    return pss

# Adds a single power system stabilizer state space object (pss) to an existing system state space object (sys)
# Stabilizer states are appended to the original states
# Inputs and outputs of the original system are retained
# gen is the generator number where the pss is installed , 
#strsps contains data about the locations of in- and output variables in the original system
def addPSS(sys,pss,strsps,gen):
    Asys = sys.A; Bsys = sys.B; Csys = sys.C; Dsys = sys.D
    Apss = pss.A; Bpss = pss.B; Cpss = pss.C; Dpss = pss.D
    n1 = len(Asys);n2 = len(Apss)  
    ni = len(Bsys.T);no = len(Csys)
    A = np.zeros((n1+n2,n1+n2))
    B = np.zeros((n1+n2,ni));
    C = np.zeros((no,n1+n2));    
    sys_idx = np.arange(0,n1); pss_idx = np.arange(n1,n1+n2);
    in_idx = np.arange(0,ni); out_idx = np.arange(0,no)
    vr_in_idx = [strsps['vref'].item()[gen-1]-1]
    w_out_idx = [strsps['speed'].item()[gen-1]-1]
    #extract submatrices from the original systems
    #i:=speed; j:=vref
    Bj = Bsys[np.ix_(sys_idx,vr_in_idx)]
    Dj = Dsys[np.ix_(out_idx,vr_in_idx)]
    Ci = Csys[np.ix_(w_out_idx,sys_idx)]
    Di = Dsys[np.ix_(w_out_idx,in_idx)]
    #Form matrices of the new system
    A[np.ix_(sys_idx,sys_idx)] = Asys + Dpss*Bj@Ci
    A[np.ix_(sys_idx,pss_idx)] = Bj@Cpss
    A[np.ix_(pss_idx,pss_idx)] = Apss
    A[np.ix_(pss_idx,sys_idx)] = Bpss@Ci    
    B[np.ix_(sys_idx,in_idx)] = Bsys + Dpss*Bj@Di
    B[np.ix_(pss_idx,in_idx)] = Bpss@Di
    C[np.ix_(out_idx,sys_idx)] = Csys + Dpss*Dj@Ci
    C[np.ix_(out_idx,pss_idx)] = Dj@Cpss  
    D = Dsys + Dpss*Dj@Di
    sys = stsp.StateSpace(A,B,C,D) 
    return sys