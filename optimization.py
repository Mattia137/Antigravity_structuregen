import pandas as pd
import numpy as np

def evaluate_model(model):
    """
    Evaluates a single PyNite model returning the fitness metrics
    and node displacement dict to map the MIDAS style colors.
    """
    try:
        model.analyze(check_statics=False)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None, None
        
    # Extract Max Displacement and compute nodal gradients
    max_disp = 0.0
    node_disps = {}
    for name, node in model.nodes.items():
        if 'Seismic' in node.DX:
            dx = node.DX['Seismic']
            dy = node.DY['Seismic']
            dz = node.DZ['Seismic']
            disp = (dx**2 + dy**2 + dz**2)**0.5
            node_disps[name] = disp
            if disp > max_disp:
                max_disp = disp
                
    volume_in3 = 0.0
    for mname, member in model.members.items():
        L = member.L()
        if isinstance(member.section, str):
            A = model.sections[member.section].A
        else:
            A = member.section.A
        volume_in3 += L * A
        
    mat = list(model.materials.values())[0]
    rho = mat.rho
    weight_lbs = volume_in3 * rho
    
    # Generic estimations based on material type
    mat_name = mat.name
    if mat_name == 'Steel':
        carbon = weight_lbs * 0.85
        cost = weight_lbs * 2.00
    elif mat_name == 'Concrete':
        carbon = weight_lbs * 0.15
        cost = weight_lbs * 0.50
    else: # Wood
        carbon = weight_lbs * -0.5 # Carbon sequestration
        cost = weight_lbs * 1.50
        
    metrics = {
        'Carbon_kgCO2e': round(carbon, 2),
        'Cost_USD': round(cost, 2),
        'Volume_in3': round(volume_in3, 2),
        'Max_Disp_in': round(max_disp, 4)
    }
    
    return metrics, node_disps
