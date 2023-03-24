# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:18:17 2023

@author: kemer
"""

import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
#%% Funções de carregamento arquivo .mat
def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes 
    todict is called to change them to nested dictionaries
    """
    for key in dict:
       if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
           dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj._dict_[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#%% Carregamento arquivo .mat (RI medidas) e .pkl (RI simuladas)

medicao = loadmat('G:\Meu Drive\TCC\Experimental\Medicao_tcc2\Dados_medicao\H_comMat.mat')
#%%
import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append(r'C:/Users/kemer/femder/pyMKL')

with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\dados-pickle\obj_Minicamara2_170_1000.pkl","rb") as arquivo:
  obj = pickle.load(arquivo)

#obj.pressure_field(frequencies = 200,renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='pdf')
  
# Resposta Impulsiva
R = fd.Receiver()
R.coord = np.array([[1.057,0.695,0.282],[0.425,0.925,0.235],
                    [0.409,0.406,0.235],[0.336,0.677,0.282],
                    [0.685,0.965,0.235],[0.661,0.30,0.235]])
obj.evaluate(R,True);

#%% RI Simulada
import domain
domain = domain.Domain(170, 1000, 1, fs=51200)
domain.alpha = 0.1
ir = domain.compute_impulse_response(obj.pR[:,0], view=True, irr_filters=False)

#%% Conversão de RIs como um SignalObj

import pytta
import numpy as np
txAmostragem = 51200 # [Hz]
fftDegree = 19
T = (2**fftDegree-1)/txAmostragem # [s]
# Cria vetor no tempo de um sinal aleatório
#meuRuido = np.random.randn(txAmostragem*T)
# SignalObj com sinal provido pelo usuário
meuSignalObj1 = pytta.SignalObj(medicao['H_comMat'][:(txAmostragem*1),0], 'time', txAmostragem) #medido
#meuSignalObj1.plot_time_dB()
#meuSignalObj1.plot_freq()
# Cria sinal senoidal via módulo de geração
meuSignalObj2 = pytta.SignalObj(ir, 'time', txAmostragem) #simulado
    
#%% Filtrando a RI em bandas de oitava
if __name__ == "__main__":    
    
    octFiltParams = {
        'order': 2,  # Second order SOS butterworth filter design
        'nthOct': 3,  # 1/nthOct bands of frequency octaves, between minFreq and maxFreq
        'samplingRate': 51200,  # Frequency of sampling
        'minFreq': 1e2,  # Minimum frequency of filter coverage
        'maxFreq': 1000,  # Maximum frequency of filter coverage
        'refFreq': 500,  # Reference central frequency for octave band divisions
        'base': 10,  # Calculate band cut with base 10 values: 10**(0.3/nthOct)
    }

    myFilt = pytta.generate.filter(**octFiltParams)  # OctFilter object

    # Apply filters to signal, gets list of filtered signals per channel
    myListFSignal1 = myFilt.filter(meuSignalObj1)
    # Filter visualization
    myListFSignal1[0].plot_time()
    myListFSignal1[0].plot_freq()
    
    myListFSignal2 = myFilt.filter(meuSignalObj2)
    myListFSignal2[0].plot_time()
    myListFSignal2[0].plot_freq()

#%% plot primeira comparação

for n in range(0,len(myListFSignal1[0].channels)):
    RI_dB_500band_exp = 10*np.log10(abs(myListFSignal1[0][n].timeSignal/max(myListFSignal1[0][n].timeSignal)))
    RI_dB_500band_simu = 10*np.log10(abs(myListFSignal2[0][n].timeSignal/max(myListFSignal2[0][n].timeSignal)))
    plt.figure(figsize = (11, 5))
    plt.plot(myListFSignal1[0][n].timeVector,RI_dB_500band_exp, linewidth = 1, color = 'b', label = 'exp')
    plt.plot(myListFSignal2[0][n].timeVector, RI_dB_500band_simu, linewidth = 1, color = 'r', alpha = 0.7,label = 'simu')
    plt.grid(linestyle = '--', which='both')
    plt.title(f'Frequência central: {myFilt.center[n]}')
    plt.ylabel(r'$h(t) (dB)$')
    plt.xlabel('Tempo (s)')
    plt.legend(loc='best')
    #plt.xlim((0, 1));
    plt.xlim((0, myListFSignal2[0][n].timeVector[-1]));
    #plt.ylim((-100, 0.5));
    plt.grid(True,which="both")
    plt.savefig(f'G:\Meu Drive\TCC\Simulacao_computacional\RI_dB_banda{myFilt.center[n]}_comMat.pdf')
    plt.show()

#%%
for n in range(0,len(obj.pR[0])):
    RI_dB_500band_exp = medicao['R_comMatfreq'][:,n]/max(abs(medicao['R_comMatfreq'][:,n])) 
    RI_dB_500band_simu = 10*np.log10(obj.pR[:,n]/max(abs(obj.pR[:,n])))
    plt.figure(figsize = (11, 5))
    plt.semilogx(medicao['freqVector'][2641:26401],10*np.log10(RI_dB_500band_exp[2641:26401]), linewidth = 1, color = 'b', label = 'exp')
    plt.plot(obj.freq, RI_dB_500band_simu, linewidth = 1, color = 'r', alpha = 0.7,label = 'simu')
    plt.grid(linestyle = '--', which='both')
    plt.title(f'Com impedância de superficie recuperada dos coefs. de absorção calculados da primeira medição na Posição {n+1}')
    plt.ylabel(r'$NPS (dB)$')
    plt.xlabel('Frequência (Hz)')
    plt.legend(loc='best')
    #plt.xlim((0, 1));
    plt.xlim((200, 2000));
    plt.grid(True,which="both")
    plt.savefig(f'G:\Meu Drive\TCC\PDFs_resultados\Pontos_med_simu0{n+1}.pdf')
    plt.show()
 
#%%
arg1200 = np.argwhere(meuSignalObj1.freqVector == 1000)[0,0]
arg100 = np.argwhere(meuSignalObj1.freqVector == 200)[0,0]
H_exp = 10*np.log10(meuSignalObj1.freqSignal[arg100:arg1200]/max(abs(meuSignalObj1.freqSignal[arg100:arg1200])))
H_simu = 10*np.log10(meuSignalObj2.freqSignal/max(abs(meuSignalObj2.freqSignal)))
plt.figure(figsize = (11, 5))
plt.plot(meuSignalObj1.freqVector[arg100:arg1200],H_exp, linewidth = 1, color = 'b', label = 'exp')
plt.plot(meuSignalObj2.freqVector, H_simu, linewidth = 1, color = 'r', alpha = 0.7,label = 'simu')
plt.grid(linestyle = '--', which='both')
plt.title('Comparação das FRFs no ponto 1')
plt.ylabel(r'$H(j$\omega$) (dB)$')
plt.xlabel('Tempo (s)')
plt.legend(loc='best')
#plt.xlim((0, 1));
plt.xlim((meuSignalObj1.freqVector[arg100:arg1200][0], meuSignalObj1.freqVector[arg100:arg1200][-1]));
plt.ylim((-30, 0.2));
plt.grid(True,which="both")
#plt.savefig('G:\Meu Drive\TCC\Simulacao_computacional\EDCband4.pdf')
plt.show()