import qutip as qt
from itertools import product


class CalculateHamiltonian(object):
    
    axes = ['x', 'y', 'z']
    
    @classmethod
    def mkH1(cls, spin_particle, parameter_vector):
        """Calculates the single electron interactions (e.g. the interaction with the magnetic field)
        
        Takes in spin particle object and the vector of values to multiply by"""
        return sum([v * getattr(spin_particle, ax) for v, ax in zip(parameter_vector, cls.axes) if v!=0])
    
    @classmethod
    def mkH2(cls, spin_particle1, spin_particle2, parameter_matrix):
        """Calculates the two electron interactions by multiplying the parameter matrix 
        with the tensor product of the relevant spin operators"""
        index_calc  = {'x':0, 'y':1, 'z':2}

        return sum([parameter_matrix[index_calc[ax1], index_calc[ax2]]*
                   (getattr(spin_particle1, ax1)*getattr(spin_particle2, ax2)) 
                   	for ax1, ax2 in product(cls.axes, cls.axes)
                   		if parameter_matrix[index_calc[ax1], index_calc[ax2]] != 0])



    
    @classmethod
    def zeeman(cls, spin_particle, field):
        """Calculates the Zeeman interaction of a spin particle with an external magnetic field"""
        return cls.mkH1(spin_particle, field)
    
    @classmethod
    def dipolar(cls, spin_particle1, spin_particle2, dipolar_matrix):
        """Calculates the Dipolar interaction between 2 electrons"""
        return cls.mkH2(spin_particle1, spin_particle2, dipolar_matrix)
    
    @classmethod
    def hyperfine(cls, spin_particle1, spin_particle2, hyperfine_matrix):
        """Calculates the hyperfine interaction between 1 electron and 1 nuclear spin"""
        return cls.mkH2(spin_particle1, spin_particle2, hyperfine_matrix)

    @classmethod
    def exchange(cls, spin_particle1, spin_particle2, J):
        """Calculates the exchange interaction between 2 electrons"""
        identity = qt.identity([2*i+1 for i in spin_particle1.spin_list])
        return -J*(0.5*identity + 2*cls.mkH2(spin_particle1, spin_particle2, identity))

