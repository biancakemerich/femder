# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:46:19 2022

@author: kemer


"""

import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL
#%%

path_to_geo = r'C:\Users\kemer\Documents\MNAV\caixa_meteoroFEM.iges'

AP = fd.AirProperties(c0 = 343)
fmin = 300
fmax = 800
AC = fd.AlgControls(AP,fmin,fmax,1)

S = fd.Source("spherical")
S.coord = np.array([[0,0.15,0]])
S.q = np.array([0.001])
R = None
grid = fd.GridImport3D(AP, path_to_geo, S, R, fmax, num_freq=6, scale=1000, 
                    order=1, plot=False, load_method='meshio')

#%%
obj = fd.FEM3D(grid,S,R,AP,AC)
obj.plot_problem(renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='png')

#%%
#enable_plotly_in_cell()

obj.compute()
#%%
#Pmin=None, frequencies=[60], Pmax=None
obj.pressure_field(frequencies = 1000,renderer='browser',axis=['xy','yz'],saveFig=False,camera_angles=['diagonal_front'],extension='pdf')
#%%
obj.evaluate(R,True);
#%%
plt.style.use('seaborn-notebook')
plt.figure(figsize=(12,6))

if len(obj.R.coord)==1:
  plt.semilogx(obj.freq, p2SPL(obj.pR), linestyle='-', label=f'R | {obj.R.coord[0]}m')
else:
  for i in range(len(obj.R.coord)):
      plt.semilogx(obj.freq, p2SPL(obj.pR[:,i]), linestyle='-', label=f'R{i} | {obj.R.coord[i,:]}m')

#if len(obj.R.coord) > 1:
  #pR_med = np.mean(obj.pR,axis=1)
  #plt.semilogx(obj.freq, p2SPL(pR_med), linestyle='--', label='Average', linewidth=4)
plt.title('SEM MATERIAL')
plt.grid(linestyle = '--', which='both')
plt.legend(loc='best')
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
plt.ylim(40,160)
#plt.xticks([20,40,60,80,100,120,160,200],[20,40,60,80,100,120,160,200]);
#plt.xticks([100,125,160,200,250,315,400,500,630,1000,1250],[100,125,160,200,250,315,400,500,630,1000,1250]);
#plt.ylim(40,160)
plt.tight_layout()
plt.show()

#%% Resposta Impulsiva
domain = fd.Domain(200, 1000,1)
domain.alpha = 0.1
ir = domain.compute_impulse_response(obj.pR[:,0], view=True, irr_filters=False)

#%% Salvar simu 90536 elem e 8.06 min

with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\dados-pickle\obj_Minicamara1_170_2300_0_4.pkl", "wb") as arquivo:
    pickle.dump(obj, arquivo)

#%% Carrega simu
import pickle
with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\obj_Minicamara1_100_1300_05.pkl","rb") as arquivo:
  obj = pickle.load(arquivo)

  
  
  