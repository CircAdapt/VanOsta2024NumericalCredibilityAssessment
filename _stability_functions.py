# -*- coding: utf-8 -*-
""" Functions for stability """
import numpy as np
import circadapt 
n_par = 25


def setX(model, X):
    model_ref = circadapt.VanOsta2024()
    
    model.set('Model.Peri.TriSeg.wLv.pLv0.Sf_act', (1+X[0])*60000, float)
    model.set('Model.Peri.TriSeg.wSv.pSv0.Sf_act', (1+X[1])*60000, float)
    model.set('Model.Peri.TriSeg.wRv.pRv0.Sf_act', (1+X[2])*60000, float)
    model.set('Model.Peri.TriSeg.wLv.pLv0.k1', (5+X[3]*10), float)
    model.set('Model.Peri.TriSeg.wSv.pSv0.k1', (5+X[4]*10), float)
    model.set('Model.Peri.TriSeg.wRv.pRv0.k1', (5+X[5]*10), float)
    model.set('Model.Peri.TriSeg.wLv.pLv0.Sf_pas', (100+X[6]*2400), float)
    model.set('Model.Peri.TriSeg.wSv.pSv0.Sf_pas', (100+X[7]*2400), float)
    model.set('Model.Peri.TriSeg.wRv.pRv0.Sf_pas', (100+X[8]*2400), float)
    model.set('Model.Peri.TriSeg.wLv.pLv0.dt', (X[9]*0.05), float)
    model.set('Model.Peri.TriSeg.wSv.pSv0.dt', (X[10]*0.05), float)
    model.set('Model.Peri.TriSeg.wRv.pRv0.dt', (X[11]*0.05), float)
    
    model['Patch']['V_wall'][2:] = (0.8 + 0.4*X[12:15]) * model_ref['Patch']['V_wall'][2:]
    model['Patch']['Am_ref'][2:] = (0.8 + 0.4*X[15:18]) * model_ref['Patch']['Am_ref'][2:]

    model['Valve']['A_leak']['LaLv'] = np.max([1e-12, X[18]-0.75])*0.5*model_ref['Valve']['A_open']['LaLv']
    model['Valve']['A_leak']['LvSyArt'] = np.max([1e-12, X[19]-0.75])*0.5*model_ref['Valve']['A_open']['LvSyArt']
    model['Valve']['A_open']['LaLv'] = np.min([1, X[20]+0.75]) * model_ref['Valve']['A_open']['LaLv']
    model['Valve']['A_open']['LvSyArt'] = np.min([1, X[21]+0.75]) * model_ref['Valve']['A_open']['LvSyArt']

    t_cycle = 60 / (60 + 40 * X[22])
    t_cycle = 1e-3 * np.round(t_cycle*1e3)
    model.set('Model.t_cycle', t_cycle, float)
    model.set('Model.PFC.p0', 12200 * (0.8+0.4*X[23]), float)
    model.set('Model.PFC.q0', (4/60000 + 6/60000*X[24]), float)

# def getX(model):
#     X = np.ndarray(n_par)

#     X[0] = model.get('Model.Peri.TriSeg.wLv.pLv0.Sf_act', float) / 60000 - 1
#     X[1] = model.get('Model.Peri.TriSeg.wSv.pSv0.Sf_act', float) / 60000 - 1
#     X[2] = model.get('Model.Peri.TriSeg.wRv.pRv0.Sf_act', float) / 60000 - 1
#     X[3] = (model.get('Model.Peri.TriSeg.wLv.pLv0.k1', float) - 5) / 25
#     X[4] = (model.get('Model.Peri.TriSeg.wSv.pSv0.k1', float) - 5) / 25
#     X[5] = (model.get('Model.Peri.TriSeg.wRv.pRv0.k1', float) - 5) / 25
#     X[6] = (model.get('Model.Peri.TriSeg.wLv.pLv0.Sf_pas', float) - 15000) / 45000
#     X[7] = (model.get('Model.Peri.TriSeg.wSv.pSv0.Sf_pas', float) - 15000) / 45000
#     X[8] = (model.get('Model.Peri.TriSeg.wRv.pRv0.Sf_pas', float) - 15000) / 45000
#     X[9] = model.get('Model.Peri.TriSeg.wLv.pLv0.dt', float) / 0.1
#     X[10] = model.get('Model.Peri.TriSeg.wSv.pSv0.dt', float) / 0.1
#     X[11] = model.get('Model.Peri.TriSeg.wRv.pRv0.dt', float) / 0.1

#     X[12] = model.get('Model.Peri.LaLv.A_leak', float) / model.get('Model.Peri.LaLv.A_open', float) / 0.5 + 0.5 # , (np.max([1e-6, X[9]-0.5])*0.5*model.get('Model.Peri.LaLv.AOpen')))
#     X[13] = model.get('Model.Peri.RaRv.A_leak', float) / model.get('Model.Peri.RaRv.A_open', float) / 0.5 + 0.5 #, (np.max([1e-6, X[10]-0.5])*0.5*model.get('Model.Peri.RaRv.AOpen')))

#     X[14] = model.get('Model.t_cycle', float) - 0.5
#     X[15] = (model.get('Model.PFC.p0', float)/12200 - 0.8) / 0.4
#     X[16] = (model.get('Model.PFC.q0', float)* 60000 - 4 ) / 6



#     return X



#%% Helper functions
def calc_Am_ef_Xm(model, Vm, Y):
    Am_ref = model['Patch']['Am_ref'][2:5]
    SignVm = np.sign(Vm)
    Vm = np.abs(Vm)
    Ym3 = np.array([1, 1, 1]) * Y
    V = (3 / np.pi) * Vm
    Q = (V + np.sqrt(V**2 + Ym3**6))**(1/3)
    Xm = SignVm * (Q - Ym3**2 / Q)

    Am = np.pi * (Xm**2 + Ym3**2)
    ef = 1 / 2 * np.log(Am / Am_ref)
    return Am, ef, Xm


def calc_T_DADT_Am0(model, Vm, Y, Am, Sf, dSf_dEf):
    # to tension
    V_wall = model['Patch']['V_wall'][2:5]
    TDSf = 0.5 * V_wall / Am
    T = TDSf * Sf
    dA_dT = Am / (0.5 * TDSf * dSf_dEf)
    Am0 = Am - T * dA_dT

    return T, dA_dT, Am0

def calc_Sf_dEf(model, ef):
    ls = model['Patch']['l_s'][1:, 2:5]
    dl_s_pas = model['Patch']['dl_s_pas'][2:5]
    k1 = model['Patch']['k1'][2:5]
    Sf_act = model['Patch']['Sf_act'][2:5]
    Sf_pas = model['Patch']['Sf_pas'][2:5]
    fac_Sf_tit = model['Patch']['fac_Sf_tit'][2:5]

    ls = 2 * np.exp(ef)
    yTit = (ls/1.8)**(4/dl_s_pas)
    Sf = Sf_pas * ((ls/1.8)**k1-1) + (yTit - 1) * (fac_Sf_tit * Sf_act)
    dSf_dEf = (ls/1.8)**k1 * (Sf_pas * k1) + yTit * ((fac_Sf_tit * Sf_act) * (4/dl_s_pas))

    model_C = model['Patch']['C'][1:, 2:5]
    model_lsi = model['Patch']['l_si'][1:, 2:5]
    L = (model_lsi/1.51)-1
    SfIso = (model_C * L) * (1.51 * Sf_act)
    Sf += SfIso * (ls - model_lsi) / 0.04
    dSf_dEf += SfIso * ls / 0.04
    return Sf, dSf_dEf

def calc_total_enegy_from_V_Y(model, idx):
    V_wall = model['Patch']['V_wall'][2:5]
    VL = model['Cavity']['V']['cLv']
    VR = model['Cavity']['V']['cRv']
    V = model['TriSeg']['V']
    Y = model['TriSeg']['Y']


    Vm = np.array([
        -VL -1/2*V_wall[0] - 1/2*V_wall[1]+V,
        V,
        VR + 1/2*V_wall[1]+1/2*V_wall[2] + V,
        ]).T

    Am, ef, Xm = calc_Am_ef_Xm(Vm, Y)
    Sf, dSf_dEf = calc_Sf_dEf(ef, idx)
    T, dA_dT, Am0 = calc_T_DADT_Am0(Vm, Y, Am, Sf, dSf_dEf)
    E = 1 / 2 * dA_dT * (Am-Am0)**2
    return np.sum(E)

def calc_Tx_Ty_TriSeg(model):
    V_wall = model['Patch']['V_wall'][2:5]
    VL = model['Cavity']['V'][1:, ['cLv']] + 1/2*V_wall[0] + 1/2*V_wall[1]
    VR = model['Cavity']['V'][1:, ['cRv']] + 1/2*V_wall[1] + 1/2*V_wall[2]
    V = model['TriSeg']['V'][1:]
    Y = model['TriSeg']['Y'][1:]


    Vm = np.array([
        V - VL,
        V,
        VR + V,
        ]).T
    Vm = Vm[0]

    Am, ef, Xm = calc_Am_ef_Xm(model, Vm, Y)
    Sf, dSf_dEf = calc_Sf_dEf(model, ef)
    T, dA_dT, Am0 = calc_T_DADT_Am0(model, Vm, Y, Am, Sf, dSf_dEf)
    sinalpha = 2*Xm*Y / (Xm**2 + Y**2)
    cosalpha = (-Xm**2+Y**2)/ (Xm**2 + Y**2)
    Tx = T * sinalpha
    Ty = T * cosalpha
    return Tx, Ty


# %% Get Y
n_out = 16
def getY(model):
    y = np.ndarray(n_out)

    # LV info
    y[0] = 1e6*np.max(model.get('Model.Peri.TriSeg.cLv.V', list))
    y[1] = 1e6*np.min(model.get('Model.Peri.TriSeg.cLv.V', list))
    y[2] = 1/133*np.max(model.get('Model.Peri.TriSeg.cLv.p', list))
    y[3] = 1/133*np.min(model.get('Model.Peri.TriSeg.cLv.p', list))

    # pressure flow control
    y[4] = model.get('Model.PFC.fac_pfc', float)
    y[5] = model.get('Model.PFC.error_q_q0', float)
    y[6] = model.get('Model.PFC.error_q_std', float)

    y[7] = 1e6*(model.get('Model.Peri.La.V', list)[0] + model.get('Model.Peri.Ra.V', list)[0] +
                model.get('Model.Peri.TriSeg.cLv.V', list)[0] + model.get('Model.Peri.TriSeg.cRv.V', list)[0] +
                model.get('Model.SyArt.V', list)[0] + model.get('Model.SyVen.V', list)[0] +
                model.get('Model.PuArt.V', list)[0] + model.get('Model.PuVen.V', list)[0])
    y[8] = 1/133*np.mean(model.get('Model.SyArt.p', list))
    y[9] = np.mean(model.get('Model.Peri.SyVenRa.q', list)) * 60000

    # ENERGY
    stress = model['Patch']['Sf'][1:, :]
    strain = model['Patch']['Ef'][1:, :]
    V_wall = model['Patch']['V_wall'][:]

    Energy_local = V_wall * np.sum((stress[:-1, :]+stress[1:, :])/2*np.diff(strain, axis=0), axis=0)


    V = model['Cavity']['V'][1:, ['La', 'Ra', 'cLv', 'cRv']]
    p = np.array([1, 1, -1, 1]) * model['Wall']['p_trans'][1:, ['wLa', 'wRa', 'wLv', 'wRv']]

    Energy_global = np.sum((p[:-1, :]+p[1:, :])/2*np.diff(V, axis=0), axis=0)

    # print(np.sum(Energy_local), np.sum(Energy_global))
    # print(np.sum(Energy_local) / np.sum(Energy_global))

    y[10] = np.sum(Energy_local)
    y[11] = np.sum(Energy_global)

    Tx, Ty = calc_Tx_Ty_TriSeg(model)

    y[12] = np.max(np.sum(Tx, axis=1))
    y[13] = np.max(np.sum(Ty, axis=1))
    y[14] = np.max(np.sum(np.abs(Tx), axis=1))
    y[15] = np.max(np.sum(np.abs(Ty), axis=1))

    signals = [
        model['Solver']['t'],
        model['Cavity']['V'][:, 'cLv'] * 1e6,
        model['Cavity']['p'][:, 'cLv'] / 133,
        ]

    return y, np.array(signals)

output_names = [
    '$max V_{lv}$ [mL]', '$min V_{lv}$ [mL]', 'Lv pmax [mmHg]', 'LV pmin [mmHg]',
    'FacpControl', 'Errq', 'ErrSy',  'Vtot', 'MAP [mmHg]', 'Venous Return [L/min]',
    'Local energy', 'Global energy',
    'Sum Tx', 'Sum Ty', 'Sum abs Tx', 'Sum abs Ty',
    ]
output_ylim = [[0,2], [0, 4000], None, None, [0, 350], [0, 200], [0, 20], [0, 20]]
