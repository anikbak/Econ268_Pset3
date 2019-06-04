import numpy as np
from numba import njit
import inspect
import re
import mathutils


'''
Part 1: current and anticipation effects
That is dynamic effects that propagate through changes in policy holding distribution constant.    
'''


def backward_step(dinput_dict, back_step_fun, ssinput_dict, ssy_list, outcome_list, D, Pi, a_pol_i, a_grid, h=1E-4):
    # shock perturbs policies
    curlyV, da, *dy_list = numerical_diff(back_step_fun, ssinput_dict, dinput_dict, h, ssy_list)

    # which affects the distribution tomorrow
    da_pol_pi = da / (a_grid[a_pol_i + 1] - a_grid[a_pol_i])
    curlyD = forward_step_policy_shock(D, Pi.T, a_pol_i, da_pol_pi)

    # and the aggregate outcomes today
    curlyY = {name: np.vdot(D, dy) for name, dy in zip(outcome_list, [da] + dy_list)}

    return curlyV, curlyD, curlyY


def backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, V_name, D, Pi, a_pol_i, a_grid, T):
    """Iterate policy steps backward T times for a single shock."""
    # initial response to unit scalar shock
    curlyV, curlyD, curlyY = backward_step(shock, back_step_fun, ssinput_dict,
                                           ssy_list, outcome_list, D, Pi, a_pol_i, a_grid)

    # infer dimensions from this and initialize empty arrays
    curlyDs = np.empty((T,) + curlyD.shape)
    curlyYs = {k: np.empty(T) for k in curlyY.keys()}

    # fill in current effect of shock
    curlyDs[0, ...] = curlyD
    for k in curlyY.keys():
        curlyYs[k][0] = curlyY[k]

    # fill in anticipation effects
    for t in range(1, T):
        curlyV, curlyDs[t, ...], curlyY = backward_step({V_name + '_p': curlyV}, back_step_fun, ssinput_dict,
                                                        ssy_list, outcome_list, D, Pi, a_pol_i, a_grid)
        for k in curlyY.keys():
            curlyYs[k][t] = curlyY[k]

    return curlyYs, curlyDs


@njit
def forward_step_policy_shock(Dss, Pi_T, a_pol_i_ss, a_pol_pi_shock):
    """Update distribution of agents with policy function perturbed around ss."""
    Dnew = np.zeros_like(Dss)
    for s in range(Dss.shape[0]):
        for i in range(Dss.shape[1]):
            apol = a_pol_i_ss[s, i]
            dshock = a_pol_pi_shock[s, i] * Dss[s, i]
            Dnew[s, apol] -= dshock
            Dnew[s, apol + 1] += dshock
    Dnew = Pi_T @ Dnew
    return Dnew


'''
Part 2: history effects (aka prediction vectors)
That is dynamic effects that propagate through the distribution's law of motion holding policy constant.    
'''


@njit
def forward_step_transpose(D, Pi, a_pol_i, a_pol_pi):
    """Efficient implementation of D_t =  Lam_{t-1} @ D_{t-1}' using sparsity of Lam_{t-1}."""
    D = Pi @ D
    Dnew = np.empty_like(D)
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            Dnew[s, i] = api * D[s, apol] + (1 - api) * D[s, apol + 1]
    return Dnew


def forward_iteration_transpose(y_ss, Pi, a_pol_i, a_pol_pi, T):
    """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome"""
    curlyPs = np.empty((T,) + y_ss.shape)
    curlyPs[0, ...] = y_ss
    for t in range(1, T):
        curlyPs[t, ...] = forward_step_transpose(curlyPs[t-1, ...], Pi, a_pol_i, a_pol_pi)
    return curlyPs


'''
Parts 3/4: construct fake news matrix and then Jacobian.
'''


def build_F(curlyYs, curlyDs, curlyPs):
    T = curlyDs.shape[0]
    F = np.empty((T, T))
    F[0, :] = curlyYs
    F[1:, :] = curlyPs[:T - 1, ...].reshape((T - 1, -1)) @ curlyDs.reshape((T, -1)).T
    return F


def J_from_F(F):
    J = F.copy()
    for t in range(1, J.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


'''Putting it all together'''


def all_Js(back_step_fun, ss, T, shock_dict):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule
    #a_pol_i, a_pol_pi = mathutils.interpolate_coord(ss['a_grid'], ss['a'])
    a_grid=ss['a_grid']
    a_pol= ss['a']
    adiff=a_pol[:,:,np.newaxis]-a_grid[np.newaxis,np.newaxis,:]
    a_pol_i=np.argmax(-adiff*(adiff>0)-99999999*(adiff<0),axis=2)
    a_pol_i[a_pol_i==a_grid.shape[0]-1]=a_grid.shape[0]-2
    a_pol_pi=(a_pol-a_grid[a_pol_i])/(a_grid[a_pol_i+1]-a_grid[a_pol_i])

    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                    V_name, ss['D'], ss['Pi'], a_pol_i, ss['a_grid'], T)

    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[1:]):
        curlyPs[o] = forward_iteration_transpose(ssy, ss['Pi'], a_pol_i, a_pol_pi, T)

    # step 3-4: make fake news matrix and Jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
            J[o][i] = J_from_F(F)

    # report Jacobians
    return J


'''Part 5: Convenience functions.'''


def extract_info(back_step_fun, ss):
    # ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)
    """Process source code of backward iteration function.
    Parameters
    ----------
    back_step_fun : function
        backward iteration function
    ss : dict
        steady state dictionary
    Returns
    ----------
    ssinput_dict : dict
      {name: ss value} for all inputs to back_step_fun
    ssy_list : list
      steady state value of outputs of back_step_fun in same order
    outcome_list : list
      names of variables returned by back_step_fun other than V
    V_name : str
      name of backward variable
    """
    V_name, *outcome_list = re.findall('return (.*?)\n',
                                       inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')

    ssy_list = [ss[k] for k in [V_name] + outcome_list]

    input_names = inspect.getfullargspec(back_step_fun).args
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    return ssinput_dict, ssy_list, outcome_list, V_name


def numerical_diff(func, ssinputs_dict, shock_dict, h=1E-4, y_ss_list=None):
    """Differentiate function via forward difference."""
    # compute ss output if not supplied
    if y_ss_list is None:
        y_ss_list = func(**ssinputs_dict)

    # response to small shock
    shocked_inputs = {**ssinputs_dict, **{k: ssinputs_dict[k] + h * shock for k, shock in shock_dict.items()}}
    y_list = func(**shocked_inputs)

    # scale responses back up
    dy_list = [(y - y_ss) / h for y, y_ss in zip(y_list, y_ss_list)]

    return dy_list
