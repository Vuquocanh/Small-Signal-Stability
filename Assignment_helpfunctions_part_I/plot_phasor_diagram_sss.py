# -*- coding: utf-8 -*-
"""
Plot functions for phasor diagram

@author: chhil
"""


import matplotlib.pyplot as plt

def plot_phasors(P,C,L):
    # P = Phasors with columns being x and y coordinates
    # C = Color of phasors
    # L = Labels of phasors
    
    total = len(L)
    
    # Used for setting labels position
    labels_x_minus = 0;
    labels_x_plus = 0;
    
    # Store minimum and maximum values for limits
    xmin = 0;
    xmax = 0;
    ymin = 0;
    ymax = 0;
        
    
    for i in range(0,total):
        plt.annotate('', xytext = (0, 0),  xy = (P[i,0], P[i,1]), \
                     xycoords='data', textcoords='data', \
                     arrowprops=dict(arrowstyle = '->',color = C[i], \
                                     shrinkA = 0, shrinkB = 0))
        
        # Define position for placing label        
        if(P[i,0] < 0):
            labels_x_minus += 1
            X = P[i,0]-0.2
            if(labels_x_minus == 1):
                Y = P[i,1]+0.01
            else:
                Y = P[i,1]-0.02
        else:
            labels_x_plus += 1
            X = P[i,0]+0.05
            if(labels_x_plus == 1):
                Y = P[i,1]+0.01
            else:
                Y = P[i,1]-0.01
        
        
        # Plot label
        plt.text(X,Y, L[i], color= C[i])
        
        # Update min max
        if (P[i,0] < xmin):
            xmin = P[i,0]
        if (P[i,0] > xmax):
            xmax = P[i,0]
        if (P[i,1] < ymin):
            ymin = P[i,1]
        if (P[i,1] > ymax):
            ymax = P[i,1]
    
    # Set limits of x and y axis
    plt.ylim((ymin-0.05, ymax+0.05))
    plt.xlim((xmin-0.25, xmax+0.35))
        
    return
