import numpy as np
import matplotlib.pyplot as plt
#import Liquid_Phase_O2_Analysis as lp
#from Reaction_ODE_Fitting import ODE_matrix_fit_func, reaction_string_to_matrix, reaction_string_to_numba_matrix
#from utility_functions import scientific_notation, plot_func
from scipy.integrate import odeint
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from timeit import default_timer as timer
import pprint
from numba import njit


def ODE_system_VariableF(y, t, p, cross_section, flux):
	'''
    Thedifference is in this case the flux driving the reaction
    from B--->C decreases directly as a result of the photons absorbed in A--->B
    The idea that I had was that each for each individual second that passes 1*quantum yield photons would be lost from the flux
	'''

	R1 = p[0] * y[1]
	R2 = p[1] * cross_section * flux * y[0]  
	R3 = p[2] * cross_section * flux *(1-p[1]*y[0]*t) * y[1]
	R6 = p[3] * y[2]
    
	ra = R1 - R2 + R6 
	rb = -R1 + R2 - R3
	rc = R3 - R6
	
	return [ra, rb, rc]

def ODE_explicit_rate_law_VariableF(p, initial_state, t, flux, cross_section, ravel = False):

	sol = odeint(ODE_system_VF, initial_state, t, args = (p, cross_section, flux))

	if ravel is True:
		sol = np.ravel(sol)
    
	return sol