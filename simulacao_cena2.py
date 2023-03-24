# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:46:19 2022

@author: kemer

Simulação com um material adicionando a impedancia de sup
    
"""

import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL, fem_load

#%%

#path_to_geo = "G:\Meu Drive\TCC\codes\geo\mini-camarareverb.iges"
path_to_geo = "G:\Meu Drive\TCC\Simulacao_computacional\codes\geo\minicamara_matSup.iges"

AP = fd.AirProperties(c0 = 343)
fmin = 170
fmax = 1000
AC = fd.AlgControls(AP,fmin,fmax,1)

#%%

S = fd.Source("spherical")
#S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.coord = np.array([[1.222,0.15,0.2]])
S.q = np.array([0.001])

R = fd.Receiver()
#R.coord = np.array([[0.33,0.23,0.21],[0.38,0.93,0.21],[0.92,0.95,0.21],[0.8,0.6,0.015]]) 
R.coord = np.array([[1.057,0.695,0.282],[0.425,0.925,0.235],
                    [0.409,0.406,0.235],[0.336,0.677,0.282],
                    [0.685,0.965,0.235],[0.661,0.30,0.235]])
#%% Caracterizar superficies

BC = fd.BC(AC,AP)
BC.delany(6,RF=10000, d=0.01, model='miki')

from sea.sea.materials import Material as mat
  
sup = mat(octave_bands_statistical_alpha = [0.0246, 0.0369, 0.0347, 0.0388, 0.0581, 0.0515], octave_bands = [125, 250, 500, 1000, 2000, 4000], freq_vec=AC.freq)
sup.impedance_from_alpha(absorber_type="hard")
sup_surface_impedance = sup.surface_impedance  
BC.normalized_admittance([2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18],sup.admittance)

#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = fmax,num_freq=6,scale=1000,order=1,load_method='meshio')
obj2 = fd.FEM3D(grid,S,R,AP,AC,BC)
obj2.plot_problem(renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='png')

#%%
#enable_p()

obj2.compute()
#%%
#Pmin=None, frequencies=[60], Pmax=None
obj2.pressure_field(frequencies = 400,renderer='browser',axis=['xy','yz'],saveFig=False,camera_angles=['diagonal_front'],extension='pdf')

#%%
obj2.evaluate(R,True)

plt.style.use('seaborn-notebook')
plt.figure(figsize=(12,6))

if len(obj2.R.coord)==1:
  plt.semilogx(obj2.freq, p2SPL(obj2.pR), linestyle='-', label=f'R | {obj2.R.coord[0]}m')
else:
  for i in range(len(obj2.R.coord)):
    plt.semilogx(obj2.freq, p2SPL(obj2.pR[:,i]), linestyle='-', label=f'R{i} | {obj2.R.coord[i,:]}m')

#if len(obj.R.coord) > 1:
  #pR_med = np.mean(obj.pR,axis=1)
  #plt.semilogx(obj.freq, p2SPL(pR_med), linestyle='--', label='Average', linewidth=4)
plt.title('Com impedancia de superficie')
plt.grid(linestyle = '--', which='both')
plt.legend(loc='best')
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
#plt.xticks([20,40,60,80,100,120,160,200],[20,40,60,80,100,120,160,200]);
#plt.xticks([100,125,160,200,250,315,400,500,630,1000,1250],[100,125,160,200,250,315,400,500,630,1000,1250]);
#plt.ylim(40,160)
plt.tight_layout()
plt.show()

#%% Salva simu 90500 elementos, 1201 7.75min
import pickle
# =============================================================================
with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\dados-pickle\obj_Minicamara2_170_1000.pkl", "wb") as arquivo:
    pickle.dump(obj2, arquivo)
#   
# =============================================================================
#%% Carrega simu 
import pickle
with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\dados-pickle\obj_Minicamara2_100_1300.pkl","rb") as arquivo:
  obj2 = pickle.load(arquivo)

#%% Resposta Impulsiva
domain = fd.Domain(100, 1300,2)
domain.alpha = 0.1
ir = domain.compute_impulse_response(obj2.pR[:,0], view=True, irr_filters=False)


