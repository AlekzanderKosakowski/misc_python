class PhysicalConstants:
    '''
    Reference class for commonly seen physical constants in astrophysical research.
    '''

    @property
    def Msol(self):
        '''
        Mass of the Sun.
            cgs: g
            mks: kg
        '''
        return 1.9885e33 if self._unit_system == "cgs" else 1.9885e30

    @property
    def Rsol(self):
        '''
        Radius of the Sun.
            cgs: m
            mks: km
        '''
        return 6.9634e10 if self._unit_system == "cgs" else 6.9634e8

    @property
    def Lsol(self):
        '''
        Luminosity of the Sun.
            cgs: J / s
            mks: erg / s
        '''
        return 3.828e33 if self._unit_system == "cgs" else 3.828e26

    @property
    def Mjup(self):
        '''
        Mass of Jupiter.
            cgs: g
            mks: kg
        '''
        return 1.898e30 if self._unit_system == "cgs" else 1.898e27

    @property
    def Rjup(self):
        '''
        Radius of Jupiter.
            cgs: m
            mks: km
        '''
        return 6.9911e9 if self._unit_system == "cgs" else 6.9911e7

    @property
    def au(self):
        '''
        One astronomical unit.
            cgs: cm
            mks: km
        '''
        return 1.495978707e13 if self._unit_system == "cgs" else 1.495978707e11

    @property
    def pc(self):
        '''
        One parsec.
            cgs: cm
            mks: km
        '''
        return 3.085677581e18 if self._unit_system == "cgs" else 3.085677581e16


    @property
    def c(self):
        '''
        Speed of light in a vacuum.
            cgs: cm / s
            mks: km / s
        '''
        return 29979245800 if self._unit_system == "cgs" else 299792458

    @property
    def G(self):
        '''
        Newton's gravitational constant.
            cgs: cm^3 / g / s^2
            mks: m^3 / kg / s^2
        '''
        return 6.67430e-8 if self._unit_system == "cgs" else 6.67430e-11

    @property
    def h(self):
        '''
        Planck constant.
            cgs: erg / s
            mks: J / s
        '''
        return 6.62607015e-27 if self._unit_system == "cgs" else 6.62607015e-34

    @property
    def k(self):
        '''
        Boltzmann constant.
            cgs: erg / K
            mks: J / K
        '''
        return 1.380649e-16 if self._unit_system == "cgs" else 1.380649e-23

    @property
    def sigma_sb(self):
        '''
        Stefan-Boltzmann constant.
            cgs: erg / cm^2 / K^4 / s
            mks: J / m^2 / K^4 / s
        '''
        return 5.670374419e-5 if self._unit_system == "cgs" else 5.670374419e-8

    @property
    def unit_system(self):
        '''
        Unit system in use: cgs or mks
        '''
        return self._unit_system

    @unit_system.setter
    def unit_system(self, value):
        '''
        Set the unit system for physical constants: "cgs" or "mks"
        '''
        if value not in ["cgs", "mks"]:
            raise ValueError("Unit system must be 'cgs' or 'mks'")
        self._unit_system = value


    def __init__(self, *, unit_system="cgs"):
        self.unit_system = unit_system.lower()


if __name__ == "__main__":

    cgs = PhysicalConstants(unit_system="cgs")
    mks = PhysicalConstants(unit_system="mks")
    for name, value in cgs.__class__.__dict__.items():
        if isinstance(value, property):
            print(f"{name:<20s} = {getattr(cgs, name):20} | {getattr(mks, name):20} ")
