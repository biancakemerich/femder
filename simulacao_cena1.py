# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:46:19 2022

@author: kemer

Simulação sem material 
() Obter a FRF 
() Comparar FRF do medido com o simulado

"""

import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL
#%%

path_to_geo = "G:\Meu Drive\TCC\Simulacao_computacional\codes\geo\mini-camarareverb.iges"

AP = fd.AirProperties(c0 = 343)
fmax = 600#2300
AC = fd.AlgControls(AP,200,fmax,1)

#%%

S = fd.Source("spherical")
#S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
#S.coord = np.array([[1.15,0.19,0.52]])
S.coord = np.array([[1.222,0.15,0.2]])
S.q = np.array([0.001])

R = fd.Receiver()
#R.coord = np.array([[0.33,0.23,0.21],[0.38,0.93,0.21],[0.92,0.95,0.21]]) 
R.coord = np.array([[1.057,0.695,0.282],[0.425,0.925,0.235],
                    [0.409,0.406,0.235],[0.336,0.677,0.282],
                    [0.685,0.965,0.235],[0.661,0.30,0.235]])
 
#%% Caracterizar superficies

from sea.materials import Material as mat

#with open(r"G:\Meu Drive\TCC\codes\dados-pickle\admitance_minicamara1.pkl","rb") as arquivo:
  #sup_admittance = pickle.load(arquivo)
  
sup = mat(octave_bands_statistical_alpha = [0.0246, 0.0369, 0.0347, 0.0388, 0.0581, 0.0515], octave_bands = [125, 250, 500, 1000, 2000, 4000], freq_vec=AC.freq)
sup.impedance_from_alpha(absorber_type="hard")
sup_admittance = sup.admittance
sup_surface_impedance = sup.surface_impedance  

#%%

BC = fd.BC(AC,AP) #[2,3,4,5,6,7]
BC.normalized_admittance(list(np.arange(2,8,1)),sup_admittance)

#%%
# Zs = BC.mu[6]
# z_ar = AP.c0*AP.rho0

# plt.semilogx(AC.freq, abs(Zs), label='Real')
# #plt.xlim(20, 10000)
# #plt.ylim(0, 100)
# plt.xlabel('Frequência [Hz]')
# plt.ylabel('Impedância de superficie')
# plt.legend()
# plt.grid(True,which="both")
# #plt.savefig('L_fixo.png')
# plt.show()

# plt.semilogx(AC.freq, np.angle(Zs,deg=True), label='Complexo')
# #plt.xlim(20, 10000)
# #plt.ylim(0, 100)
# plt.xlabel('Frequência [Hz]')
# plt.ylabel('Impedância de superficie')
# plt.legend()
# plt.grid(True,which="both")
# #plt.savefig('L_fixo.png')
# plt.show()

# # Coeficiente de reflexão e absorção
# Reflexao = (Zs - 1) / (Zs + 1)
# Absorcao = 1 - (np.abs(Reflexao) ** 2)  # 1 - |R|²
# plt.title("Gráfico de Coef. de Absorção")
# plt.semilogx(AC.freq, 100*Absorcao)
# #plt.xlim(20, 10000)
# #plt.ylim(0, 100)
# plt.xlabel('Frequência [Hz]')
# plt.ylabel('Coef. de Absorção [%]')
# plt.legend()
# plt.grid(True,which="both")
# #plt.savefig('L_fixo.png')
# plt.show()
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = fmax,num_freq=6,scale=1000,order=1,load_method='meshio')
#%%
obj = fd.FEM3D(grid,S,R,AP,AC,BC)
obj.plot_problem(renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='png')

#%%
#enable_plotly_in_cell()

obj.compute()
#%%
#Pmin=None, frequencies=[60], Pmax=None
obj.pressure_field(frequencies = 300,renderer='browser',axis=['xy','yz'],saveFig=False,camera_angles=['diagonal_front'],extension='pdf')
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

  
  
  