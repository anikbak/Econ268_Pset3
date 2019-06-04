import numpy as np
import scipy.optimize as opt
import mathutils
import het_block as het
import rec_block as rec
from rec_block import recursive


'''Part 1: Steady state'''


def backward_iterate(Va_p, Pi_p, a_grid, e_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Order of returns matters! backward_var, assets, others

    Parameters
    ----------
    Va_p : np.ndarray
        marginal value of assets tomorrow
    Pi_p : np.ndarray
        Markov transition matrix for skills tomorrow
    a_grid : np.ndarray
        asset grid
    e_grid : np.ndarray
        skill grid
    r : float
        ex-post interest rate
    w : float
        wage
    beta : float
        discount rate today
    eis : float
        elasticity of intertemporal substitution

    Returns
    ----------
    Va : np.ndarray, shape(nS, nA)
        marginal value of assets today
    a : np.ndarray, shape(nS, nA)
        asset policy today
    c : np.ndarray, shape(nS, nA)
        consumption policy today
    """
    nextvalue= (beta * Pi_p) @ Va_p
    amax=a_grid[-1]
        
    c_grid= mathutils.agrid(amax=a_grid[-1], n=a_grid.shape[0]*5)*(1+r)+0.01
    
    ###working
    disutility=B_Bar*hbar**(1+1/gamma)/(1+1/gamma)
    coh_w = (1 + r) * a_grid[np.newaxis, :] + w *hbar* e_grid[:, np.newaxis]
    a_next_w=coh_w[:,:,np.newaxis]-c_grid[np.newaxis,np.newaxis,:]
    a_next_w[a_next_w>amax]=amax
    tobeadded=np.empty_like(a_next_w)
    for i in range(e_grid.shape[0]):
        tobeadded[i,:,:]=np.interp( a_next_w[i,:,:], a_grid, nextvalue[i,:])
        
    valuebeforemax=np.log(c_grid)[np.newaxis,np.newaxis,:]-disutility+tobeadded
    valuebeforemax[a_next_w<a_lower]=-99999999
    Va_w=np.amax(valuebeforemax,axis=2)
    maxposition=np.argmax(valuebeforemax,axis=2)
    c_w=c_grid[maxposition]
    a_w=coh_w-c_w
 
    ###Not working
    coh_nw = (1 + r) * a_grid[np.newaxis, :] 
    
    tobeadded_nw=np.empty_like(a_next_w)
    a_next_nw=coh_nw[:,:,np.newaxis]-c_grid[np.newaxis,np.newaxis,:]+tobeadded_nw
    a_next_nw[a_next_nw>amax]=amax
    for i in range(e_grid.shape[0]):
        tobeadded_nw[i,:,:]=np.interp( a_next_nw[i,:,:], a_grid, nextvalue[i,:])
    valuebeforemax_nw=np.log(c_grid)[np.newaxis,np.newaxis,:]+tobeadded_nw
    valuebeforemax_nw[a_next_nw<a_lower]=-99999999
    Va_nw=np.amax(valuebeforemax_nw,axis=2)
    maxposition=np.argmax(valuebeforemax_nw,axis=2)
    c_nw=c_grid[maxposition]
    a_nw=coh_nw-c_nw
    
    ####Compare
    hh=(Va_w>Va_nw)*1
    Va=np.maximum(Va_w,Va_nw)
    c=c_w*(Va_w>Va_nw)+c_nw*(Va_w<=Va_nw)
    a=a_w*(Va_w>Va_nw)+a_nw*(Va_w<=Va_nw)
    a[a>a_grid[a_grid.shape[0]-1]]=a_grid[a_grid.shape[0]-1]
# =============================================================================
#     h=np.empty_like(Va_p)
#     Va=Va_w
#     c=c_w
#     a=a_w
# =============================================================================
    ###Compare working and nonworking 
    
    return Va, a, c, hh


def pol_ss(Pi, e_grid, a_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower, Va_seed=None, tol=1E-6, maxit=5000):
    """Find steady state policy functions."""
    if Va_seed is None:
        #coh = (1 + r) * a_grid[np.newaxis, :] + w * hbar * e_grid[:, np.newaxis]
        coh = (1 + r) * a_grid[np.newaxis, :] + hbar * e_grid[:, np.newaxis]
        Va = np.log(coh) 
    else:
        Va = Va_seed

    # iterate until convergence of a policy by tol or reach max number of iterations
    a = np.empty_like(a_grid)
    for it in range(maxit):
        Vanew, anew, c, hh = backward_iterate(Va, Pi, a_grid, e_grid, r, w, beta, eis,hbar,B_Bar,gamma,a_lower)
# =============================================================================
#         if it % 10 == 1:
#             print(np.max(np.abs(Va - Vanew)))
# =============================================================================

        if it % 10 == 1 and mathutils.within_tolerance(a, anew, tol):
            break
        Va = Vanew
        a = anew
    else:
        print(f'No convergence after {maxit} backward iterations!')

    return Va, a, c, hh


def household_ss(Pi, a_grid, e_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower,Va_seed=None, D_seed=None, pi_seed=None):
    """Solve for steady-state policies and distribution. Report results in dict."""
    # solve ha block
    Va, a, c, hh  = pol_ss(Pi, e_grid, a_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower, Va_seed)
    D = mathutils.dist_ss(a, Pi, a_grid, D_seed, pi_seed)

    # return dictionary with results and inputs
    inputs = {'Pi': Pi, 'a_grid': a_grid, 'e_grid': e_grid, 'r': r, 'w': w, 'beta': beta, 'eis': eis, 'hbar': hbar, 'B_Bar': B_Bar, 'gamma': gamma, 'a_lower': a_lower}
    results = {'D': D, 'Va': Va, 'a': a, 'c': c, 'hh': hh, 'A': np.vdot(D, a), 'C': np.vdot(D, c),'H': np.vdot(D, hh)}

    return {**inputs, **results}


def K_supply_demand(Pi, a_grid, e_grid, r, w, beta, eis,hbar,B_Bar,gamma,a_lower,alpha,delta):
    w=(1 - alpha) *  (alpha  / (r+delta)) ** (alpha / (1 - alpha))
    result=household_ss(Pi, a_grid, e_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower)
    supply=result['A']
    labor=result['H']*hbar
    demand=((result['r']+delta)/alpha)** (1/(alpha-1))*labor
    
    return supply-demand


def ks_ss(r_min=0.001, r_max=0.06, beta=0.98267, eis=1, delta=0.025, alpha=0.64, b=0.15, nA=100, amax=200,hbar=1/3,B_Bar=166.3,gamma=0.4,a_lower=0):
    """Solve steady state of full GE model. Calibrate beta to hit target for interest rate."""
    # set up grid
    a_grid = mathutils.agrid(amax=amax, n=nA)
    #L = pUE / (pUE + pEU)  # labor endowment normalized to 1
    e_grid = mathutils.markov_tauchen(rho=0.929, sigma=0.227, N=5, m=3)[0]
    Pi = mathutils.markov_tauchen(rho=0.929, sigma=0.227, N=5, m=3)[2]

    # solve for aggregates analitically
# =============================================================================
#     rk = r + delta
#     Z = (rk / alpha) ** alpha / L ** (1 - alpha)  # normalize so that Y=1
#     K = (alpha * Z / rk) ** (1 / (1 - alpha)) * L
#     Y = Z * K ** alpha * L ** (1 - alpha)
#     w = (1 - alpha) * Z * (alpha * Z / rk) ** (alpha / (1 - alpha))
# =============================================================================

    # solve for beta consistent with this
    #r_min = 0.001
    #r_max = 0.06
    sol = opt.root_scalar(lambda rr: K_supply_demand(Pi, a_grid, e_grid, rr, (1 - alpha) *  (alpha  / (rr+delta)) ** (alpha / (1 - alpha)), beta, eis, hbar,B_Bar,gamma,a_lower, alpha,delta),
                          bracket=[r_min, r_max], method='brentq')
    if sol.converged:
        r = sol.root
    else:
        raise ValueError('Steady-state solver did not converge.')
        
    w=(1 - alpha) *  (alpha  / (r+delta)) ** (alpha / (1 - alpha))
    Z= 1
    
    

    # extra evaluation to report variables
    ss = household_ss(Pi, a_grid, e_grid, r, w, beta, eis, hbar,B_Bar,gamma,a_lower)
    L= ss['H']*hbar
    K= ((ss['r']+delta)/alpha)** (1/(alpha-1))*L
    Y= K ** alpha * L ** (1 - alpha)
    mpc = mathutils.mpcs(ss['c'], ss['a_grid'], ss['r'])
    ss.update({'mpc': mpc, 'MPC': np.vdot(ss['D'], mpc),
               'w': w, 'Z': Z, 'K': K, 'L': L, 'Y': Y, 'alpha': alpha, 'delta': delta,
               'goods_mkt': Y - ss['C'] - delta * K})

    return ss


'''Part 2: linear transition dynamics'''


@recursive
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


def get_J(ss, T):
    """Compute Jacobians along computational graph: for r, w, curlyK as functions of Z and K."""

    # firm Jacobian: r and w as functions of Z and K
    J_firm = rec.all_Js(firm, ss, T, ['K', 'Z'])

    # household Jacobian: curlyK (called 'a' for assets by J_ha) as function of r and w
    J_ha = het.all_Js(backward_iterate, ss, T, {'r': {'r': 1}, 'w': {'w': 1}})

    # compose to get curlyK as function of Z and K
    J_curlyK_K = J_ha['a']['r'] @ J_firm['r']['K'] + J_ha['a']['w'] @ J_firm['w']['K']
    J_curlyK_Z = J_ha['a']['r'] @ J_firm['r']['Z'] + J_ha['a']['w'] @ J_firm['w']['Z']

    # now combine all into a single jacobian that gives r, w, curlyK as functions of Z and K
    J = {**J_firm, 'curlyK': {'K': J_curlyK_K, 'Z': J_curlyK_Z}}

    return J


def get_G(J, T):
    """Solve for equilibrium G matrices: K, r, w as functions of Z."""

    # obtain H_K, H_Z
    H_K = J['curlyK']['K'] - np.eye(T)
    H_Z = J['curlyK']['Z']

    # solve for K as function of Z
    G = {'K': -np.linalg.solve(H_K, H_Z)}  # H_K^(-1)H_Z

    # solve for r, w, Y as functions of Z too
    G['r'] = J['r']['Z'] + J['r']['K'] @ G['K']
    G['w'] = J['w']['Z'] + J['w']['K'] @ G['K']
    G['Y'] = J['Y']['Z'] + J['Y']['K'] @ G['K']

    return G


def td_linear(G, dZ, outputs=('r', 'w', 'K')):
    return {k: G[k] @ dZ for k in outputs}


'''Part 3: extend to nonlinear transitition dynamics'''


def household_td(ss, **kwargs):
    """Calculate partial equilibrium response of household to shocks to any of its inputs given in kwargs.

    Not allowed to shock transition matrix or a_grid.
    """
    # infer T from kwargs, check that all shocks have same length
    shock_lengths = [x.shape[0] for x in kwargs.values()]
    assert shock_lengths[1:] == shock_lengths[:-1], 'Shocks with different length.'
    T = shock_lengths[0]

    # ss dict only with inputs of backward_iterate
    input_names = ['Va_p', 'Pi_p', 'a_grid', 'e_grid', 'r', 'w', 'beta', 'eis']
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    # make new dict of all the ss that are not shocked
    fixed_inputs = {k: v for k, v in ssinput_dict.items() if k not in kwargs}

    # allocate empty arrays to store results
    Va_path, a_path, c_path, D_path = (np.empty((T,) + ss['a'].shape) for _ in range(4))

    # backward iteration
    for t in reversed(range(T)):
        if t == T-1:
            Va_p = ssinput_dict['Va_p']
        else:
            Va_p = Va_path[t+1, ...]

        backward_inputs = {**fixed_inputs, **{k: v[t, ...] for k, v in kwargs.items()}, 'Va_p': Va_p}  # order matters
        Va_path[t, ...], a_path[t, ...], c_path[t, ...] = backward_iterate(**backward_inputs)

    # forward iteration
    Pi_T = ss['Pi'].T.copy()
    D_path[0, ...] = ss['D']
    for t in range(T):
        a_pol_i, a_pol_pi = mathutils.interpolate_coord(ss['a_grid'], a_path[t, ...])
        if t < T-1:
            D_path[t+1, ...] = mathutils.forward_iterate(D_path[t, ...], Pi_T, a_pol_i, a_pol_pi)

    # return paths and aggregates
    return {'Va': Va_path, 'a': a_path, 'c': c_path, 'D': D_path,
            'A': np.sum(D_path*a_path, axis=(1, 2)), 'C': np.sum(D_path*c_path, axis=(1, 2))}


def td_map(ss, K, Z):
    # firm block
    r, w, Y = firm.td(ss, K=K, Z=Z)

    # ha block
    td = household_td(ss, r=r, w=w)
    td.update({'r': r, 'w': w, 'K': K, 'Z': Z, 'Y': Y})

    return td['A'] - K, td


def td_nonlinear(ss, H_K_inv, Z, tol=1E-8, maxit=30, noisy=True):
    # initialize guess at ss
    T = Z.shape[0]
    K = np.full(T, ss['K'])

    # iterate until convergence
    for it in range(maxit):
        asset_mkt, td = td_map(ss, K, Z)
        error = np.max(np.abs(asset_mkt))
        if noisy:
            print(f'Max error {error:.2E} on iteration {it}')
        if error < tol:
            break
        else:
            K -= H_K_inv @ asset_mkt
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')

    return td
