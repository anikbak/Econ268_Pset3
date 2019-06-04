import numpy as np
from scipy.stats import norm
from numba import njit, guvectorize


'''Part 1: grids and Markov chains.'''


def agrid(amax, n, amin=0, pivot=0.25):
    """Create equidistant grid in logspace between amin-pivot and amax+pivot."""
    a_grid = np.geomspace(amin + pivot, amax + pivot, n) - pivot
    a_grid[0] = amin  # make sure *exactly* equal to amin
    return a_grid


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)


def markov_tauchen(rho, sigma, N=7, m=3):
    """Tauchen method discretizing AR(1) s_t = rho*s_(t-1) + eps_t.

    Parameters
    ----------
    rho   : scalar, persistence
    sigma : scalar, unconditional sd of s_t
    N     : int, number of states in discretized Markov process
    m     : scalar, discretized s goes from approx -m*sigma to m*sigma

    Returns
    ----------
    y  : array (N), states proportional to exp(s) s.t. E[y] = 1
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    """

    # make normalized grid, start with cross-sectional sd of 1
    s = np.linspace(-m, m, N)
    ds = s[1] - s[0]
    sd_innov = np.sqrt(1 - rho ** 2)

    # standard Tauchen method to generate Pi given N and m
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sd_innov) -
                    norm.cdf(s[j] - rho * s - ds / 2, scale=sd_innov))

    # invariant distribution and scaling
    pi = stationary(Pi)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi


def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi


'''
Part 2: tools for solving Bellman equation via EGM and handle the distribution's law of motion.

Performance-sensitive, so we define custom compiled functions for interpolation, etc.
'''


# Numba's guvectorize decorator compiles functions and allows them to be broadcast by NumPy when dimensions differ.
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    yq : array (nq), empty to be filled with interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


# Numba's njit decorator does just-in-time compilation of a function.
@njit
def setmin(x, xmin):
    """Set 2-dimensional array x where each row is ascending equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i, j] < xmin:
                x[i, j] = xmin
            else:
                break


@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2."""
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """
    Efficient linear interpolation exploiting monotonicity. xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points
    xqi  : array (nq), empty to be filled with indices of lower bracketing gridpoints
    xqpi : array (nq), empty to be filled with weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi


@njit
def forward_iterate(D, Pi_T, a_pol_i, a_pol_pi):
    """Single forward iteration step to update distribution.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of Lam_{t-1}.

    Parameters
    ----------
    D        : array (S*A), beginning-of-period distribution over s_t, a_(t-1)
    Pi_T     : array (S*S), transpose Markov matrix that maps s_t to s_(t+1)
    a_pol_i  : int array (S*A), left gridpoint of asset policy
    a_pol_pi : array (S*A), weight on left gridpoint of asset policy

    Returns
    ----------
    Dnew : array (S*A), beginning-of-next-period dist s_(t+1), a_t
    """

    # first create Dnew from updating asset state
    Dnew = np.zeros_like(D)
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            d = D[s, i]
            Dnew[s, apol] += d * api
            Dnew[s, apol + 1] += d * (1 - api)

    # then use transpose Markov matrix to update income state
    Dnew = Pi_T @ Dnew

    return Dnew


def dist_ss(a_pol, Pi, a_grid, D_seed=None, pi_seed=None, tol=1E-5, maxit=500_000):
    """Iterate to find steady-state distribution."""
    if D_seed is None:
        # compute separately stationary dist of s, to start there assume a uniform distribution on assets otherwise
        pi = stationary(Pi, pi_seed)
        D = np.tile(pi[:, np.newaxis], (1, a_grid.shape[0])) / a_grid.shape[0]
    else:
        D = D_seed

    # obtain interpolated-coordinate asset policy rule
    # a_pol_i, a_pol_pi = interpolate_coord(a_grid, a_pol)
    
    adiff=a_pol[:,:,np.newaxis]-a_grid[np.newaxis,np.newaxis,:]
    a_pol_i=np.argmax(-adiff*(adiff>0)-99999999*(adiff<0),axis=2)
    a_pol_i[a_pol_i==a_grid.shape[0]-1]=a_grid.shape[0]-2
    a_pol_pi=(a_pol-a_grid[a_pol_i])/(a_grid[a_pol_i+1]-a_grid[a_pol_i])
    # to make matrix multiplication more efficient, make separate copy of Pi transpose
    Pi_T = Pi.T.copy()

    # iterate until convergence by tol, or maxit
    for it in range(maxit):
        Dnew = forward_iterate(D, Pi_T, a_pol_i, a_pol_pi)

        # only check convergence every 10 iterations for efficiency
        #if it % 50 == 0:
            #print(it)
            #print(np.max(np.abs(D-Dnew)))
        if it % 10 == 0 and within_tolerance(D, Dnew, tol):
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')

    return D


'''Random useful things'''


def mpcs(c_pol, a_grid, r):
    """Calculate MPC for exogenous income model."""
    mpc = np.empty_like(c_pol)
    mpc[:, 0] = 1                                                                          # mpc of 1 at bottom
    mpc[:, -1] = (c_pol[:, -1] - c_pol[:, -2]) / (a_grid[-1] - a_grid[-2]) / (1 + r)       # backward difference
    mpc[:, 1:-1] = (c_pol[:, 2:] - c_pol[:, :-2]) / (a_grid[2:] - a_grid[:-2]) / (1 + r)   # symmetric difference
    return mpc


def mpcs_mpes(c, wn, a, a_grid, r, eis, frisch):
    """Calculate MPC and MPE endogenous income model."""
    mpc, mpe = np.empty_like(c), np.empty_like(c)
    post_return = (1 + r) * a_grid

    # symmetric differences away from boundaries
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpe[:, 1:-1] = -(wn[:, 2:] - wn[:, 0:-2]) / (post_return[2:] - post_return[:-2])

    # asymmetric first differences at boundaries
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    mpe[:, 0] = -(wn[:, 1] - wn[:, 0]) / (post_return[1] - post_return[0])
    mpe[:, -1] = -(wn[:, -1] - wn[:, -2]) / (post_return[-1] - post_return[-2])

    # next try constrained case, implementing our formula and using mpe/mpc=1
    constrained_indices = np.nonzero(a == a_grid[0])
    mpe_mpc_ratio_constrained = (wn[constrained_indices] / c[constrained_indices] * frisch / eis)

    mpc[constrained_indices] = 1 / (1 + mpe_mpc_ratio_constrained)
    mpe[constrained_indices] = 1 - 1 / (1 + mpe_mpc_ratio_constrained)

    return mpc, mpe
