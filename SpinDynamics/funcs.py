from numba import jit
import numpy as np

def SingletProjection(spin_particle1, spin_particle2):
    """Calculates the singlet projection operator as 1/4 - S1.S2"""
    import qutip as qt
    return 0.25*qt.identity([2*i+1 for i in spin_particle1.spin_list]) - spin_particle1*spin_particle2

def CalculateK1(k, Ps):
    """Takes in the recombination constant and the singlet projection, 
    returns the Harberkorn operator for singlet recombination"""
    return 0.5*k*Ps

def CalculateDistances(spin_particle_list, labels):
    """Takes in a list of spin particles and a list of labels (usually the indices i.e. range(len(spin_particle_list)))
    uses a recursive approach to efficiently calculate the distances between all the particles
    
    returns a list of named tuples with the index labels and the distances between the particles at those indices"""
    from collections import namedtuple
    
    Distances = namedtuple('Distances', ['index', 'distance'])
    if len(spin_particle_list)==2:
        return [Distances(tuple(labels), spin_particle_list[0].coordinates-spin_particle_list[1].coordinates)]
    else:
        distances = CalculateDistances(spin_particle_list[1:], labels[1:])
        distances.extend([Distances((labels[0], labels[i]), spin_particle_list[0].coordinates-spin_particle_list[i].coordinates) 
         for i in range(1, len(spin_particle_list))])
        return distances


def CalculateDipolarMatrices(constant, distances):
    """Takes in the dipolar constant (this controls the units) and the distances taken from CalculateDistances
    
    returns a list of named tuples with the index and dipolar matrix"""
    from collections import namedtuple
    import numpy as np
    dip_mat = namedtuple("dipolar_matrix", ['index', 'matrix'])
    dip_mats = []
    for dist in distances:
        norm = np.linalg.norm(dist.distance)
        r_normed = dist.distance/norm
        dip_mats.append(dip_mat(dist.index, constant*(1/(norm**3)) * (3 * np.kron(r_normed, r_normed).reshape(3, 3) - np.identity(3))))
    return dip_mats

def CalculateYields(H, K1, ksc):
    """Takes in the Hamiltonian and the singlet Harberkorn operator
    returns the singlet yield"""
    import numpy as np
    H = H -1j*K1
    vals, vecs = np.linalg.eig(H)
    vecs_rev = np.linalg.inv(vecs)
    rho_0 = np.dot(vecs_rev, np.conj(vecs_rev.T))
    G = 1j*(vals[:, np.newaxis] - np.conj(vals)) + ksc
    rho_S = rho_0/G
    rho_S = np.matmul(vecs, np.matmul(rho_S, np.conj(np.transpose(vecs))))

    return 2*np.real(np.sum(np.multiply(K1, rho_S.T))) / vecs.shape[0]

