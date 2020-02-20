import qutip as qt
import numpy as np

class BaseSpinParticle(object):
    """Takes in the spin of the particle, then allows the operators to be calculated on the fly.
    
    This will not store any of the spin operators however."""
    
    def __init__(self, j, coordinates=None):
        self.spin = j
        self.coordinates = np.array(coordinates)
    
    def x(self):
        return qt.jmat(self.spin, 'x')
    
    def y(self):
        return qt.jmat(self.spin, 'y')
    
    def z(self):
        return qt.jmat(self.spin, 'z')
    
    def p(self):
        return qt.jmat(self.spin, '+')
    
    def m(self):
        return qt.jmat(self.spin, '-')

    def __mul__(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __repr__(self):
        return "SpinParticle(spin: {}, coordinates: {})".format(self.spin, self.coordinates)

class LowMemSpinParticle(BaseSpinParticle):
    """If many spin particles are included and memory becomes an issue, this class can be used 
    in order to not store any of them and just calculate them on the fly.
    
    Takes in arguments of the list of spins of all the spin particles and the 
    spin_index of the current spin particle.
    
    e.g. LowMemSpinParticle([0.5, 0.5, 0.5], 0)
    Gives us a 3 particle system where the current object represents the 1st spin"""
    def __init__(self, j_list, spin_index, coordinates=None):
        super(LowMemSpinParticle, self).__init__(j_list[spin_index], coordinates)
        self.spin_list = j_list
        self.spin_index = spin_index
    
    def spinList(self):
        return [qt.identity(2*spin+1) for spin in self.spin_list]
    
    @property
    def x(self):
        spins = self.spinList()
        spins[self.spin_index] = super(LowMemSpinParticle, self).x()
        return qt.tensor(spins)
    
    @property
    def y(self):
        spins = self.spinList()
        spins[self.spin_index] = super(LowMemSpinParticle, self).y()
        return qt.tensor(spins)
    
    @property
    def z(self):
        spins = self.spinList()
        spins[self.spin_index] = super(LowMemSpinParticle, self).z()
        return qt.tensor(spins)
    
    @property
    def p(self):
        spins = self.spinList()
        spins[self.spin_index] = super(LowMemSpinParticle, self).p()
        return qt.tensor(spins)
    
    @property
    def m(self):
        spins = self.spinList()
        spins[self.spin_index] = super(LowMemSpinParticle, self).m()
        return qt.tensor(spins)
    
class HighMemSpinParticle(BaseSpinParticle):
    """This class will calculate the relevant spin operators and store them in memory,
    this is faster than calculating them on the fly, but requires more memoery.
    
    Takes in arguments of the list of spins of all the spin particles and the 
    spin_index of the current spin particle.
    
    e.g. LowMemSpinParticle([0.5, 0.5, 0.5], 0)
    Gives us a 3 particle system where the current object represents the 1st spin"""
    def __init__(self, j_list, spin_index, coordinates=None):
        super(HighMemSpinParticle, self).__init__(j_list[spin_index], coordinates)
        self.spin_list = j_list
        self.spin_index = spin_index
        spins = self.spinList()
        
        spins[self.spin_index] = super(HighMemSpinParticle, self).x()
        self.x = qt.tensor(spins)
        
        spins[self.spin_index] = super(HighMemSpinParticle, self).y()
        self.y = qt.tensor(spins)
        
        spins[self.spin_index] = super(HighMemSpinParticle, self).z()
        self.z = qt.tensor(spins)
        
        spins[self.spin_index] = super(HighMemSpinParticle, self).p()
        self.p = qt.tensor(spins)
        
        spins[self.spin_index] = super(HighMemSpinParticle, self).m()
        self.m = qt.tensor(spins)
        
    def spinList(self):
        return [qt.identity(2*spin+1) for spin in self.spin_list]
