# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:53:28 2022

@author: kemer
"""

import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL


with open("G:\Meu Drive\TCC\Simulacao_computacional\codes\dados-pickle\obj_Minicamara1_200_1000_05.pkl","rb") as arquivo:
  obj = pickle.load(arquivo)

#obj.pressure_field(frequencies = 200,renderer='browser',saveFig=False,camera_angles=['diagonal_front'],extension='pdf')
  
#%% Resposta Impulsiva
R = fd.Receiver()
R.coord = np.array([[1.057,0.695,0.282],[0.425,0.925,0.235],
                    [0.409,0.406,0.235],[0.336,0.677,0.282],
                    [0.685,0.965,0.235],[0.661,0.30,0.235]])
obj.evaluate(R,True);
domain = fd.Domain(200, 1000,2)
domain.alpha = 0.1
ir = domain.compute_impulse_response(obj.pR[:,0], view=True, irr_filters=False)