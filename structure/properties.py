import numpy as np


class opensees_constants:
    def __init__(self):
        self.FREE = 0
        self.FIXED = 1
        self.X = 1
        self.Y = 2
        self.ROTZ = 3


class MDOFCantil_Property:
    """Standard definition of multi-DoF shear building properties."""
    def __init__(self, ndof, nodal_mass, Str_Prop, Mat_Prop, rayleigh_xi=0.05, modelname="your model"):
        # Str_Prop = [bCol, dCol, lCol, 'SectionName']
        # Mat_Prop = [fy(MPa), E(MPa), strain_hardening]
        self.modelname = modelname
        self.ndof = ndof
        self.nodal_mass = nodal_mass
        self.bcolumn = Str_Prop[0]
        self.dcolumn = Str_Prop[1]
        self.Lcolumn = Str_Prop[2]
        self.section_type = Str_Prop[3]
        A = {"Rect": self.bcolumn * self.dcolumn,
             "Circ": np.pi * self.dcolumn**2 / 4}
        I = {"Rect": self.bcolumn * self.dcolumn**3 / 12,
             "Circ": np.pi * self.dcolumn**4 / 64}
        self.Acolumn = A[self.section_type]
        self.Icolumn = I[self.section_type]
        self.fy = Mat_Prop[0]
        self.Ecolumn = Mat_Prop[1]
        self.bb = Mat_Prop[2]
        self.f_yield = self.fy * self.Acolumn
        self.k_spring = 192 * self.Ecolumn * self.Icolumn / self.Lcolumn**3
        self.rayleigh_xi = rayleigh_xi
