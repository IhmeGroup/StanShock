import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class TableVariable:
    '''
    Class to represent a variable in an FPV table.
    '''
    def __init__(self, name, data, Z, Q, L):
        '''
        Initialize the TableVariable object with the name of the variable and the data.
        '''
        self.name = name
        self.data = data
        self.interp = RegularGridInterpolator((Z, Q, L), data, bounds_error=False, fill_value=None)
    
    def lookup(self, Z, Q, L):
        '''
        Perform a lookup of the variable at the given Z, Q, and L values.
        '''
        return self.interp((Z, Q, L))

class FPVTable:
    '''
    Class to read an FPV table from an HDF5 file and perform lookups.
    '''
    def __init__(self, filename):
        '''
        Initialize the FPVTable object by reading the HDF5 file.
        '''
        self.filename = filename
        with h5py.File(filename, 'r') as f:
            self.P = f['Header']['Doubles']['Double_0'].attrs['Value'][0]
            self.Z = f['Coordinates']['Coor_0'][()]
            self.Q = f['Coordinates']['Coor_1'][()]
            self.L = f['Coordinates']['Coor_2'][()]
            
            var_names = [var.decode('utf-8') for var in f['Header']['Variable Names'][()]]
            data_raw = f['Data'][()]
            n_tot = self.Z.size * self.Q.size * self.L.size
            self.variables = []
            for i, var in enumerate(var_names):
                data = data_raw[i*n_tot : (i+1)*n_tot].reshape(self.Z.size, self.Q.size, self.L.size, order='C')
                self.variables.append(TableVariable(var, data, self.Z, self.Q, self.L))
    
    def L_from_C(self, Z, C):
        '''
        Compute the normalized progress variable value at the given Z and C values.
        '''
        C_min = np.zeros_like(Z)
        C_max = self.lookup('PROG', Z, 0, 1)
        L = (C - C_min) / (C_max - C_min)
        L = np.clip(L, 0, 1)
        return L
    
    def lookup(self, var, Z, Q, L):
        '''
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        '''
        for v in self.variables:
            if v.name == var:
                return v.lookup(Z, Q, L)

        raise ValueError(f'Variable {var} not found in table {self.filename}.')
    
    def lookup_all(self, Z, Q, L):
        '''
        Perform a lookup of all variables at the given Z, Q, and L values.
        '''
        return {v.name: v.lookup(Z, Q, L) for v in self.variables}
    
    def get_gamma(self, Z, Q, L, T):
        '''
        Compute the specific heat ratio at the given Z, Q, L and T values.
        '''
        gamma0 = self.lookup('GAMMA0', Z, Q, L)
        ag = self.lookup('AGAMMA', Z, Q, L)
        T0 = self.lookup('T0', Z, Q, L)
        return gamma0 + ag * (T - T0)
    
    def get_R(self, Z, Q, L):
        '''
        Compute the gas constant at the given Z, Q, and L values.
        '''
        R = self.lookup('ROM', Z, Q, L)
        return R

    def get_cp(self, Z, Q, L, T):
        '''
        Compute the specific heat at the given Z, Q, L and T values.
        '''
        R = self.get_R(Z, Q, L)
        gamma = self.get_gamma(Z, Q, L, T)
        return R * gamma / (gamma - 1)
    
    def get_cv(self, Z, Q, L):
        '''
        Compute the specific heat at constant volume at the given Z, Q, and L values.
        '''
        R = self.get_R(Z, Q, L)
        gamma = self.get_gamma(Z, Q, L, 0)
        return R / (gamma - 1)
    
    def get_mu(self, Z, Q, L, T):
        '''
        Compute the dynamic viscosity at the given Z, Q, and L values.
        '''
        mu0 = self.lookup('MU0', Z, Q, L)
        T0 = self.lookup('T0', Z, Q, L)
        amu = self.lookup('AMU', Z, Q, L)
        return mu0 * (T / T0)**amu

    def get_loc(self, Z, Q, L, T):
        '''
        Compute the thermal conductivity at the given Z, Q, and L values.
        '''
        loc0 = self.lookup('LOC0', Z, Q, L)
        T0 = self.lookup('T0', Z, Q, L)
        aloc = self.lookup('ALOC', Z, Q, L)
        return loc0 * (T / T0)**aloc