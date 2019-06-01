import numpy as np


def chain_jacobians(jacdicts, inputs, T):
    """Obtain complete Jacobian of every output in jacdicts with respect to inputs, by applying chain rule."""
    cumulative_jacdict = {i: {i: np.eye(T)} for i in inputs}
    for jacdict in jacdicts:
        cumulative_jacdict.update(compose_jacobians(cumulative_jacdict, jacdict))
    return cumulative_jacdict


def compose_jacobians(jacdict2, jacdict1):
    """Compose Jacobians via the chain rule."""
    jacdict = {}
    for output, innerjac1 in jacdict1.items():
        jacdict[output] = {}
        for middle, jac1 in innerjac1.items():
            innerjac2 = jacdict2.get(middle, {})
            for inp, jac2 in innerjac2.items():
                if inp in jacdict[output]:
                    jacdict[output][inp] += jac1 @ jac2
                else:
                    jacdict[output][inp] = jac1 @ jac2
    return jacdict


def apply_jacobians(jacdict, indict, T):
    """Apply Jacobians in jacdict to indict to obtain outputs"""
    outdict = {}
    for myout, innerjacdict in jacdict.items():
        outdict[myout] = np.zeros(T)
        for myin, jac in innerjacdict.items():
            if myin in indict:
                outdict[myout] += jac @ indict[myin]
    return outdict


def pack_jacobians(jacdict, inputs, outputs, T):
    """If we have T*T jacobians from nI inputs to nO outputs in jacdict, combine into (nO*T)*(nI*T) jacobian matrix."""
    nI, nO = len(inputs), len(outputs)

    outjac = np.empty((nO * T, nI * T))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            outjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = subdict.get(inputs[iI], np.zeros((T, T)))
    return outjac


def subtract_dict(d1, d2):
    return {k: d1.get(k, 0) - d2.get(k, 0) for k in (d1.keys() | d2.keys())}
