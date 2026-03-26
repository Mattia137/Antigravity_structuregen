# config.py
# Global Parameter Block

MATERIALS = {
    'Steel': {
        'E': 29000.0,       # ksi
        'alphaT': 0.0000065, # in/in/degF
        'nu': 0.3,
        'rho': 0.283,       # pci (lb/in^3)
        'Strength': 50.0          # ksi
    },
    'Concrete': {
        'E': 3600.0,
        'alphaT': 0.0000055,
        'nu': 0.2,
        'rho': 0.0868,      # ~150 pcf
        'Strength': 4.0           # ksi
    },
    'Wood': {
        'E': 1600.0,
        'alphaT': 0.000003,
        'nu': 0.3,
        'rho': 0.020,       # ~35 pcf
        'Strength': 1.2           # ksi
    }
}

# Pre-defined cross-section dictionary
# A (in2), Iy (in4), Iz (in4), J (in4)
SECTIONS = {
    'Steel': {
        'IPE_300':  {'A': 8.34, 'Iy': 14.5, 'Iz': 201.0, 'J': 0.48},
        'HEA_200':  {'A': 8.35, 'Iy': 32.2, 'Iz': 88.7, 'J': 0.35},
        'Tubular_HSS_4x4x1/4': {'A': 3.37, 'Iy': 7.8, 'Iz': 7.8, 'J': 12.5},
        'W14x283':  {'A': 83.3, 'Iy': 1070.0, 'Iz': 3840.0, 'J': 72.8},
        'W12x50':   {'A': 14.6, 'Iy': 56.3, 'Iz': 391.0, 'J': 1.04},
        'W14x90':   {'A': 26.5, 'Iy': 362.0, 'Iz': 999.0, 'J': 4.06},
        'Core_Massive': {'A': 10240.0, 'Iy': 1400000.0, 'Iz': 1400000.0, 'J': 2000000.0}
    },
    'Concrete': {
        'Rect_16x16': {'A': 256.0, 'Iy': 5461.3, 'Iz': 5461.3, 'J': 9239.0},
        'Circ_16':    {'A': 201.0, 'Iy': 3217.0, 'Iz': 3217.0, 'J': 6434.0},
        'Core_Massive': {'A': 10240.0, 'Iy': 1400000.0, 'Iz': 1400000.0, 'J': 2000000.0},
        'Floor_Tie':    {'A': 324.0,  'Iy': 8748.0,  'Iz': 8748.0,  'J': 14820.0}
    },
    'Wood': {
        'Rect_8x8':   {'A': 64.0,  'Iy': 341.3,  'Iz': 341.3,  'J': 577.0},
        'Circ_8':     {'A': 50.2,  'Iy': 201.0,  'Iz': 201.0,  'J': 402.0},
        'Core_Massive': {'A': 10240.0, 'Iy': 1400000.0, 'Iz': 1400000.0, 'J': 2000000.0},
        'Floor_Tie':    {'A': 144.0, 'Iy': 1728.0, 'Iz': 1728.0, 'J': 2930.0}
    }
}

# Default Assignments
DEFAULTS = {
    'Steel': {'Primary': 'IPE_300', 'Secondary': 'HEA_200', 'Diagonal': 'Tubular_HSS_4x4x1/4'},
    'Concrete': {'Primary': 'Rect_16x16', 'Secondary': 'Rect_16x16', 'Diagonal': 'Circ_16'},
    'Wood': {'Primary': 'Rect_8x8', 'Secondary': 'Rect_8x8', 'Diagonal': 'Circ_8'}
}

# LABC Seismic Parameters (High Seismicity - LA Basin Class D)
SEISMIC = {
    'S_DS': 1.8,
    'S_D1': 0.8
}

CAMERA_CONFIG = {
    'eye': dict(x=1.5, y=1.5, z=1.5),
    'up': dict(x=0, y=0, z=1),
    'center': dict(x=0, y=0, z=0)
}
