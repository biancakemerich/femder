# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:46:19 2022

@author: kemer

Simulação com o volume do material
em determinados ponto dentro ou na superficie do material a pressão é 0
"""
import sys
sys.path.append(r'C:\Users\kemer\femder')
import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from femder.FEM_3D import  p2SPL, fem_load
#%%

path_to_geo = "G:\Meu Drive\TCC\Simulacao_computacional\codes\geo\minicamra_cmat.geo"
#path_to_geo = "G:/Meu Drive/TCC/codes/geo/mini_camara_cmaterial.iges"

AP = fd.AirProperties(c0 = 343)
fmin = 170
fmax = 300
AC = fd.AlgControls(AP,fmin,fmax,1)

#%%

S = fd.Source("spherical")
#S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.coord = np.array([[1.222,0.15,0.2]])
S.q = np.array([0.001])

R = fd.Receiver()
#R.coord = np.array([[0.33,0.23,0.21],[0.38,0.93,0.21],[0.92,0.95,0.21]]) 
R.coord = np.array([[1.057,0.695,0.282],[0.425,0.925,0.235],
                    [0.409,0.406,0.235],[0.336,0.677,0.282],
                    [0.685,0.965,0.235],[0.661,0.30,0.235]])
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = fmax,num_freq=6,scale=1000,order=1,load_method='meshio')
BC = fd.BC(AC,AP);
#BC.normalized_admittance([2,3,4,5,7,8,9,10,11,12],0.02)
BC.delany(1,RF=10000, model='miki')
#BC.delany([6,13,14,15,16,17,18,38],RF=10000, model='miki')
#%%

obj3 = fd.FEM3D(grid,S,R,AP,AC,BC)
obj3.plot_problem(renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='png')

#%% plot campo sonoro
#Pmin=None, frequencies=[60], Pmax=None
obj3.compute()
#obj3.pressure_field(frequencies = 400, renderer='browser',axis=['xy','yz'],saveFig=False,camera_angles=['floorplan'],extension='pdf')
obj3.pressure_field(frequencies = 400,renderer='browser',axis=['xy','yz'],saveFig=False,camera_angles=['diagonal_front'],extension='pdf')

#%%campo sonoro nos pontos receptores ao longo da frequencia
obj3.evaluate(R,True)


plt.style.use('seaborn-notebook')
plt.figure(figsize=(12,6))

if len(obj3.R.coord)==1:
  plt.semilogx(obj3.freq, p2SPL(obj3.pR), linestyle='-', label=f'R | {obj3.R.coord[0]}m')
else:
  for i in range(len(obj3.R.coord)):
    plt.semilogx(obj3.freq, p2SPL(obj3.pR[:,i]), linestyle='-', label=f'R{i} | {obj3.R.coord[i,:]}m')

plt.title('Com fluido equivalente resistividade 80000 Ns/m^4 e 2 pontos entre as superficie do material')
plt.grid(linestyle = '--', which='both')
plt.legend(loc='best')
plt.xlabel('Frequencia [Hz]')
plt.ylabel('NPS [dB]')
#plt.xticks([20,40,60,80,100,120,160,200],[20,40,60,80,100,120,160,200]);
#plt.xticks([100,125,160,200,250,315,400,500,630,1000,1250],[100,125,160,200,250,315,400,500,630,1000,1250]);
#plt.ylim(40,160)
#plt.savefig("G:\Meu Drive\TCC\Simulacao_computacional\grafico_ptsentresuperficiemat_resalta.pdf")
plt.tight_layout()
plt.show()


#%% Salva simu 90500 elementos, 1201 7.75min
import pickle
# with open("G:\Meu Drive\TCC\Simulacao_computacional\simu_fluidoeq_1k_altares.pkl", "wb") as arquivo:
#     pickle.dump(obj3, arquivo)
# #%%
# import pickle
with open("G:\Meu Drive\TCC\Simulacao_computacional\simu_fluidoeq_1k_altares.pkl","rb") as arquivo:
  obj3 = pickle.load(arquivo)

# plt.style.use('seaborn-notebook')
# plt.figure(figsize=(12,6))
# plt.semilogx(obj3.freq, p2SPL(obj3.pR), linestyle='-', label='Fluido equivalente')
# plt.semilogx(obj2.freq, p2SPL(obj2.pR), linestyle='-', label='Impedância prescrita')
# plt.title('Comparação entre as simulações com resistividade ao fluxo de 80000 Ns/m^4')
# plt.grid(linestyle = '--', which='both')
# plt.legend(loc='best')
# plt.xlabel('Frequência [Hz]')
# plt.ylabel('NPS [dB]')
# plt.savefig('comparacaoSimu_altares.pdf')
# plt.tight_layout()
# plt.show()




