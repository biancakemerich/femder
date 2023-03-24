# -*- coding: utf-8 -*-

"""
Created on Sun Jul 31 18:46:19 2022

@author: kemer
( ) Aplicar o boolean fragments nas duas faces do material - rever a geometria no autocad

"""
import sys
sys.path.append(r'C:\Users\kemer\femder')
import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL

#%%

path_to_geo = "C:/Users/kemer/Documents/MNAV/Trabalho2-FEM3D/tubo_impedancia.geo"

AP = fd.AirProperties(c0 = 343)
fmax = 4000
AC = fd.AlgControls(AP,100,fmax,1)

#%%

S = fd.Source("spherical")
#S.coord = np.array([[-1,2.25,1.2],[1,2.25,1.2]])
S.coord = np.array([[0.73,0,0]])
S.q = np.array([0.001])

R = fd.Receiver()
R.coord = np.array([[0.087,0,0], # Simulando a medição do tubo
                    [0.067,0,0]]) 
# R.coord = np.array([[0.017,0,0], # Pontos próximos entre os diferentes meios
#                     [0.027,0,0]]) 
# R.coord = np.array([[0.1,0,0]])

#%% Caracterizar superficies

# sup = mat(octave_bands_statistical_alpha = [0.0246, 0.0369, 0.0347, 0.0388, 0.0581, 0.0515], octave_bands = [125, 250, 500, 1000, 2000, 4000], freq_vec=AC.freq)
# sup.impedance_from_alpha(absorber_type="hard")
# sup_admittance = sup.admittance
# sup_surface_impedance = sup.surface_impedance
  
  
#%%

BC = fd.BC(AC,AP) #[2,3,4,5,6,7]
BC.delany(1,RF=25000, model='miki');
#%% Plote impedancia e absorção
# Zs = BC.mu[6]
z_ar = AP.c0*AP.rho0
d=0.025 # Espessura do material
w= AC.freq*2*np.pi
Zs = (-1j*BC.rhoc*BC.cc)/np.tan((w/BC.cc)*d) 

plt.semilogx(AC.freq, abs(Zs), label='Real')
#plt.xlim(20, 10000)
#plt.ylim(0, 100)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Impedância de superficie')
plt.legend()
plt.grid(True,which="both")
#plt.savefig('L_fixo.png')
plt.show()

plt.semilogx(AC.freq, np.angle(Zs,deg=True), label='Complexo')
#plt.xlim(20, 10000)
#plt.ylim(0, 100)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Impedância de superficie')
plt.legend()
plt.grid(True,which="both")
#plt.savefig('L_fixo.png')
plt.show()

# Coeficiente de reflexão e absorção
Reflexao = (Zs - z_ar) / (Zs + z_ar)
Absorcao = 1 - (np.abs(Reflexao) ** 2)  # 1 - |R|²
plt.title("Gráfico de Coef. de Absorção")
plt.semilogx(AC.freq, 100*Absorcao)
#plt.xlim(20, 10000)
#plt.ylim(0, 100)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Coef. de Absorção [%]')
plt.legend()
plt.grid(True,which="both")
#plt.savefig('L_fixo.png')
plt.show()
#%%
grid = fd.GridImport3D(AP,path_to_geo,S,R,fmax = fmax,num_freq=6,scale=1000,order=1,load_method='meshio')
obj = fd.FEM3D(grid,S,R,AP,AC,BC)
obj.plot_problem(renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='png')
#%%
#enable_plotly_in_cell()

obj.compute()
#%%
#Pmin=None, frequencies=[60], Pmax=None
#obj.pressure_field(frequencies = 200,renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='pdf')
#%%
obj.evaluate(R,True);

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
#plt.title('SEM MATERIAL')
plt.grid(linestyle = '--', which='both')
plt.legend(loc='best')
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB]')
#plt.ylim(40,160)
#plt.xticks([20,40,60,80,100,120,160,200],[20,40,60,80,100,120,160,200]);
#plt.xticks([100,125,160,200,250,315,400,500,630,1000,1250],[100,125,160,200,250,315,400,500,630,1000,1250]);
#plt.ylim(40,160)
plt.tight_layout()
plt.show()

#%% Salvar simu 90536 elem e 8.06 min

# with open("G:\Meu Drive\TCC\Simulacao_computacional\simulacao_tuboImpedancia.pkl", "wb") as arquivo:
#   pickle.dump(obj, arquivo)

#%% Carrega simu

with open("G:\Meu Drive\TCC\Simulacao_computacional\simulacao_tuboImpedancia.pkl","rb") as arquivo:
  obj = pickle.load(arquivo)

#%% Comparação coef de absorção analitico e simulado
obj.evaluate(R,True);
p2 = obj.pR[:,1]
p1 = obj.pR[:,0]
H = p2/p1
diam = 0.0277 #diametro do tubo de impedancia
k_comp = (w/AC.c0)#-1j*(1.94*10**(-2))*(np.sqrt(AC.freq)/(AC.c0*diam))
s = R.coord[0,0]-R.coord[1,0]
reflexao_simu = ( (H - np.exp(-1j*k_comp*s)) / (np.exp(1j*k_comp*s) - H) ) * (np.exp(2j*k_comp*R.coord[0,0]))
absorcao_simu = 1 - np.abs(reflexao_simu**2)

plt.title("Coeficiente de absorção")
plt.semilogx(AC.freq, 100*Absorcao, label='Analítico')
plt.semilogx(AC.freq, 100*absorcao_simu, label='Numérico')
#plt.xlim(20, 10000)
#plt.ylim(0, 100)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Coef. de Absorção [%]')
plt.legend()
plt.grid(True,which="both")
#plt.savefig('G:\Meu Drive\TCC\Simulacao_computacional\coefAbs_tuboImp_analitico_exp.pdf')
plt.show()
  