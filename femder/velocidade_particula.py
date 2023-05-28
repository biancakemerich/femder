import femder as fd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from femder.FEM_3D import  p2SPL

# import pickle
# with open("G:\\Meu Drive\\TCC\\Simulacao_computacional\\minicamara-90mm\\minicamara_comMat\\espessura2_5cm\\fluido_eq\\obj_150_500_0_5.pkl","rb") as arquivo:
#   obj1 = pickle.load(arquivo)

# with open("G:\\Meu Drive\\TCC\\Simulacao_computacional\\minicamara-90mm\\minicamara_comMat\\espessura2_5cm\\fluido_eq\\obj_500_900_0_5.pkl","rb") as arquivo:
#   obj2 = pickle.load(arquivo)

# with open("G:\\Meu Drive\\TCC\\Simulacao_computacional\\minicamara-90mm\\minicamara_comMat\\espessura2_5cm\\fluido_eq\\obj_900_2k_0_5.pkl","rb") as arquivo:
#   obj3 = pickle.load(arquivo)


def impedancia_u(tel,coord_u,obj):
  
  qsi1=0; qsi2=0; qsi3=0; # primeiro nó do elemento padrao (TET10)
    #função de forma T10 - Ziekwineski OKOK - CORRIGIDO e verificado COM Figura "tetra_Zienkiewicz_TET10_.png" (*)
  N = np.array([[(1-qsi1-qsi2-qsi3)*(2*(1-qsi1-qsi2-qsi3)-1)],
               [qsi1*(2*qsi1-1)],
               [qsi2*(2*qsi2-1)],
               [qsi3*(2*qsi3-1)],
               [4*qsi1*(1-qsi1-qsi2-qsi3)],
               [4*qsi1*qsi2],
               [4*qsi2*(1-qsi1-qsi2-qsi3)],
               [4*qsi3*(1-qsi1-qsi2-qsi3)],
               [4*qsi3*qsi1],
               [4*qsi2*qsi3]])
  # derivada da função de forma T10 - Ziekwineski OKOK - refeita, de acordo COM Figura "tetra_Zienkiewicz_TET10_.png"
  GNi= np.array([[4*qsi1 + 4*qsi2 + 4*qsi3 - 3, 4*qsi1 - 1, 0, 0, 4 - 4*qsi2 - 4*qsi3 - 8*qsi1, 4*qsi2, -4*qsi2, -4*qsi3, 4*qsi3, 0],
               [4*qsi1 + 4*qsi2 + 4*qsi3 - 3, 0, 4*qsi2 - 1, 0, -4*qsi1, 4*qsi1, 4 - 8*qsi2 - 4*qsi3 - 4*qsi1, -4*qsi3, 0, 4*qsi3],
               [4*qsi1 + 4*qsi2 + 4*qsi3 - 3, 0, 0, 4*qsi3 - 1, -4*qsi1, 0, -4*qsi2, 4 - 4*qsi2 - 8*qsi3 - 4*qsi1, 4*qsi1, 4*qsi2]])
  
  if coord_u.shape[0]>1:
    Zs_u = np.zeros((coord_u.shape[0],len(obj.freq)))
    for l in range(coord_u.shape[0]):
      coord_el_med=np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [0.5,0,0],
                       [0.5,0.5,0],
                       [0,0.5,0],
                       [0,0,0.5],
                       [0.5,0,0.5],
                       [0,0.5,0.5]])
      coord_el_med=coord_el_med-np.array([qsi1,qsi2,qsi3])  # translação necessária no espaço
      coord_el_med=coord_el_med*tel+coord_u[l,:]#coord_mat[0,:] # translação necessária no espaço com elemento redimensionado
      R = fd.Receiver()
      R.coord = coord_el_med
      obj.evaluate(R);
      ## Jacobiano e derivada da pressão
      Ja = GNi@coord_el_med
      B = (np.linalg.inv(Ja)@GNi) # compute the B matrix - B é o gradiente
      p_total_nodais = obj.pR.T # PRESSÃO NOS n nós do elemento de volume
      delPmed=B@p_total_nodais# para todas as frequencias fazer um for, ver como fazer operando matrizes
      pmed=p_total_nodais*N # calculada na coordenada qsi1 qsi2 qsi3   # 
      dir = np.array([0,0,-1]).reshape((1,3))
      delPmed1 = dir@delPmed
      u=(delPmed1)/(1j*obj.w*obj.rho0) # convençãao exp(-j\omega t)
      Zs_u[l,:]=pmed[0]/(u) # IMPEDANCIA
  else:
    coord_el_med=np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [0.5,0,0],
                       [0.5,0.5,0],
                       [0,0.5,0],
                       [0,0,0.5],
                       [0.5,0,0.5],
                       [0,0.5,0.5]])
    coord_el_med=coord_el_med-np.array([qsi1,qsi2,qsi3])  # translação necessária no espaço
    coord_el_med=coord_el_med*tel+coord_u#coord_mat[0,:] # translação necessária no espaço com elemento redimensionado
    R = fd.Receiver()
    R.coord = coord_el_med
    obj.evaluate(R);
    # Jacobiano e derivada da pressão
    Ja = GNi@coord_el_med
    B = (np.linalg.inv(Ja)@GNi) # compute the B matrix - B é o gradiente
    p_total_nodais = obj.pR.T # PRESSÃO NOS n nós do elemento de volume
    delPmed=B@p_total_nodais# para todas as frequencias fazer um for, ver como fazer operando matrizes
    pmed=p_total_nodais*N # calculada na coordenada qsi1 qsi2 qsi3   # 
    dir = np.array([0,0,-1]).reshape((1,3))
    delPmed1 = dir@delPmed
    u=(delPmed1)/(1j*obj.w*obj.rho0) # convençãao exp(-j\omega t)
    Zs_u=pmed[0]/(u) # IMPEDANCIA

  return Zs_u