# 制御器Aと制御器Bの比較．
# モデル予測制御(MPC)のみ．
# 品質は2種類．
# 制約条件を，行列(E)を用いた1つの不等式で表記している．最適化問題を解く際にfor文を使用．

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Problem data.
# np.random.seed(3)
delta_n, gamma_n = 5, 10
delta_t = 1



#==================================================================================#
#==================================================================================#
#=========================== potatoのMLDモデルのパラメータ ===========================#
#==================================================================================#
#==================================================================================#

A = np.array([[-4.21270271e-04+1], [-2.36178784e-04+1], [-3.22164500e-04+1], [-2.52650737e-04+1], [-5.55708636e-04+1]])
B = np.array([[-4.84453301e-05], [-3.14845920e-05], [-5.36849855e-05], [-1.67910305e-05], [1.35079377e-05]])
C = np.array([[6.47483981e-03], [3.51970772e-03], [5.42080197e-03], [4.06639287e-03], [8.64274190e-03]])
D = np.array([-3.29723012e-01, -3.06409958e-01, -3.63390858e-01, -3.42315514e-01, -3.13246241e-01])
Q = np.array([[0.02627317], [-0.00111771], [0.01880834], [-0.01876554], [-0.04472564], [-0.03862955], [-0.05427779], [0.02798923], [-0.00845947], [-0.01385068]])
T = np.array([[-0.07538652], [0.04318842], [-0.04245705], [-0.05633143], [0.05458913], [0.0601666], [-0.03055113], [-0.01195942], [-0.00280735], [0.02021543]])
R = np.array([[-0.16798629], [-0.24951434], [-0.34649457], [0.24132598], [0.2253457], [0.20345188], [0.14436165], [-0.23591518], [0.32142254], [0.1428877]])
S = np.array([8.4387248, 3.37998739, 16.72282594, 24.14807339, 9.62861827, 1.78944618, 51.78167187, -6.80356885, -7.37417206, -0.34880202])

gmin = np.array([899.47084533937, 899.5771749904001, 899.49317965545, 899.54725329965, 899.4496129264676])
gmax = np.array([1099.8086670060725, 1099.7595073502, 1099.7824410081375, 1099.7614584645626, 1099.90058712121])
hmin = np.array([-6.490075750000003, -9.676540410000001, -12.00376721, -6.153670209999998, -17.797203979999995, -18.053687420000003, -12.758386630000004, -7.61302995, -7.879117860000002, -5.738675770000001])
hmax = np.array([11.568330099999997, 7.845144190000001, 15.34147409, 14.693912240000003, 7.160122770000001, 4.400759780000007, 8.244456869999993, 13.618288250000004, 14.775424989999998, 6.824546480000001])
eps = np.array([1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05])

hmin_hat_first = np.zeros((gamma_n, delta_n))
hmin_hat_second = np.array([[hmin[0],0,0,0,0,0,0,0,0,0],
                            [0,hmin[1],0,0,0,0,0,0,0,0],
                            [0,0,hmin[2],0,0,0,0,0,0,0],
                            [0,0,0,hmin[3],0,0,0,0,0,0],
                            [0,0,0,0,hmin[4],0,0,0,0,0],
                            [0,0,0,0,0,hmin[5],0,0,0,0],
                            [0,0,0,0,0,0,hmin[6],0,0,0],
                            [0,0,0,0,0,0,0,hmin[7],0,0],
                            [0,0,0,0,0,0,0,0,hmin[8],0],
                            [0,0,0,0,0,0,0,0,0,hmin[9]]])
hmin_hat = np.concatenate([hmin_hat_first,hmin_hat_second], 1)

hmax_hat_first = np.zeros((gamma_n, delta_n))
hmax_hat_second = np.array([[hmax[0],0,0,0,0,0,0,0,0,0],
                     [0,hmax[1],0,0,0,0,0,0,0,0],
                     [0,0,hmax[2],0,0,0,0,0,0,0],
                     [0,0,0,hmax[3],0,0,0,0,0,0],
                     [0,0,0,0,hmax[4],0,0,0,0,0],
                     [0,0,0,0,0,hmax[5],0,0,0,0],
                     [0,0,0,0,0,0,hmax[6],0,0,0],
                     [0,0,0,0,0,0,0,hmax[7],0,0],
                     [0,0,0,0,0,0,0,0,hmax[8],0],
                     [0,0,0,0,0,0,0,0,0,hmax[9]]])
hmax_hat = np.concatenate([hmax_hat_first,hmax_hat_second], 1)

eps_hat_first = np.zeros((gamma_n, delta_n))
eps_hat_second = np.array([[eps[0],0,0,0,0,0,0,0,0,0],
                     [0,eps[1],0,0,0,0,0,0,0,0],
                     [0,0,eps[2],0,0,0,0,0,0,0],
                     [0,0,0,eps[3],0,0,0,0,0,0],
                     [0,0,0,0,eps[4],0,0,0,0,0],
                     [0,0,0,0,0,eps[5],0,0,0,0],
                     [0,0,0,0,0,0,eps[6],0,0,0],
                     [0,0,0,0,0,0,0,eps[7],0,0],
                     [0,0,0,0,0,0,0,0,eps[8],0],
                     [0,0,0,0,0,0,0,0,0,eps[9]]])
eps_hat = np.concatenate([eps_hat_first,eps_hat_second], 1)

gmin_hat_first = np.array([[gmin[0],0,0,0,0],[0,gmin[1],0,0,0],[0,0,gmin[2],0,0],[0,0,0,gmin[3],0],[0,0,0,0,gmin[4]]])
gmin_hat_second = np.zeros((delta_n, gamma_n))
gmin_hat = np.concatenate([gmin_hat_first,gmin_hat_second], 1)

gmax_hat_first = np.array([[gmax[0],0,0,0,0],[0,gmax[1],0,0,0],[0,0,gmax[2],0,0],[0,0,0,gmax[3],0],[0,0,0,0,gmax[4]]])
gmax_hat_second = np.zeros((delta_n, gamma_n))
gmax_hat = np.concatenate([gmax_hat_first,gmax_hat_second], 1)

Gamma1_first = np.array([[1,1,1,1,1],[-1,-1,-1,-1,-1]])
Gamma1_second = np.zeros((2, gamma_n))
Gamma1 = np.concatenate([Gamma1_first,Gamma1_second], 1)

Eps1 = np.array([1,-1])

Gamma2_first = np.array([[1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,0,1],
                   [0,0,0,0,1],
                   [0,0,0,0,1],
                   [0,0,0,0,1]])
Gamma2_second = np.array([[-1,0,0,0,0,0,0,0,0,0],
                   [0,-1,0,0,0,0,0,0,0,0],
                   [0,0,-1,0,0,0,0,0,0,0],
                   [0,0,0,-1,0,0,0,0,0,0],
                   [1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,-1,0,0,0,0,0],
                   [0,0,0,0,0,-1,0,0,0,0],
                   [0,0,0,0,0,0,-1,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,-1,0,0],
                   [0,0,0,0,0,0,0,0,-1,0],
                   [0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,0,-1],
                   [0,0,0,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,1]])
Gamma2 = np.concatenate([Gamma2_first,Gamma2_second], 1)
Eps2 = np.array([0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1])
One = np.ones((1,delta_n))
I = np.eye(delta_n)
Zero = np.zeros(delta_n)
# zero = np.stack(([Zero]*(time-1)), axis = 0)
# beta = np.concatenate([zero,One], 0).flatten()

E1 = np.concatenate([np.zeros((2,1)),-Q,Q,-Q,Q,np.zeros((20+delta_n+delta_n,1)),A,-A], 0)
E2 = np.concatenate([np.zeros((2,1)),-T,T,-T,T,np.zeros((20+delta_n+delta_n,1)),B,-B], 0)
E3 = np.concatenate([np.zeros((2,1)),-R,R,-R,R,np.zeros((20+delta_n+delta_n,1)),C,-C], 0)
E4 = np.concatenate([np.zeros((2+gamma_n+gamma_n+gamma_n+gamma_n+20,delta_n)),-I,I,-I,I], 0)
E5 = np.concatenate([Gamma1,np.zeros((gamma_n+gamma_n,delta_n+gamma_n)),-hmin_hat,-hmax_hat-eps_hat,Gamma2,gmin_hat,-gmax_hat,gmax_hat,-gmin_hat], 0)
E6 = np.concatenate([Eps1,-hmin+S,hmax-S,-hmin+S,-eps-S,Eps2,np.zeros(delta_n),np.zeros(delta_n),gmax-D,-gmin+D], 0)






#==================================================================================#
#==================================================================================#
#=========================== tomatoのMLDモデルのパラメータ ===========================#
#==================================================================================#
#==================================================================================#

A_H = np.array([-0.10280675, -0.01900384, -1.0640578, -0.2409164, -0.55442332])
A_H = np.array([[A_H[0]+delta_t, -A_H[0], A_H[1]+delta_t, -A_H[1], A_H[2]+delta_t, -A_H[2], A_H[3]+delta_t, -A_H[3], A_H[4]+delta_t, -A_H[4]]]).T
B_H = np.array([-0.07503288, -0.01304328, -0.89262608, -0.25439021, -0.2968465])
B_H = np.array([[B_H[0], -B_H[0]+delta_t, B_H[1], -B_H[1]+delta_t, B_H[2], -B_H[2]+delta_t, B_H[3], -B_H[3]+delta_t, B_H[4], -B_H[4]+delta_t]]).T
C_H = np.array([-1.1250703,  -0.21467077, -12.75212815,  -3.70708767,  -7.70798855])
C_H = np.array([[C_H[0], -C_H[0], C_H[1], -C_H[1], C_H[2], -C_H[2], C_H[3], -C_H[3], C_H[4], -C_H[4]]]).T
D_H = np.array([328.0856564, 61.06533784, 3847.73057284, 1094.81508891, 2286.72706302])
D_H =np.array([D_H[0], -D_H[0], D_H[1], -D_H[1], D_H[2], -D_H[2], D_H[3], -D_H[3], D_H[4], -D_H[4]])
Q_H = np.array([[0.07264316, -0.11880306, -0.09235975, -0.21013151,  0.02206956,  0.02047818, 0.05606904,  0.1741325, 0.08299591, -0.10074457]]).T
T_H = np.array([[0.08162662, -0.0247667, -0.04874518, -0.1643359,  -0.01855246, -0.12330535, -0.0595362,  0.15179147,  0.02938379, -0.11320038]]).T
R_H = np.array([[1.14902717, -0.93113606, -1.03643109, -2.01372202, -0.3875179, -1.31042227, -0.74833271,  1.57561027,  0.91628172, -1.10607559]]).T
S_H = np.array([-336.11696047,  280.33474471,  308.28579863, 609.3326419,   112.83704085, 383.89718463,  217.84698594, -483.75776015, -277.27304113, 337.39388709])

AB_H = np.concatenate((A_H,B_H),axis=1)
QT_H = np.concatenate((Q_H,T_H),axis=1)

gmin_H = np.array([24.4941931, 54.57877618, 37.2518247, 61.20693758, -27.84293788, -142.49832866, 1.63325765, 22.39157496, -15.28702432, -41.51283018])
gmax_H = np.array([67.17256275, 99.65446797, 62.29588216, 85.14535556, 245.49832866, 170.74293788, 96.47327228, 125.40189511, 153.82538279, 148.87447171])
hmin_H = np.array([-8.65717136, -6.5978507, -10.27971023, -17.1206635, -3.20056881, -15.61299624,  -7.5641566,  -29.16526147, -17.26848351, -7.60950382])
hmax_H = np.array([17.39251969, 14.97842177, 13.30538893, 30.66790753, 5.36353966, 13.36624483, 9.70552816,  8.87035104,  3.35005729, 18.76837683])
eps = np.array([1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05])

hmin_hat_first = np.zeros((gamma_n, delta_n))
hmin_H_hat_second = np.array([[hmin_H[0],0,0,0,0,0,0,0,0,0],
                            [0,hmin_H[1],0,0,0,0,0,0,0,0],
                            [0,0,hmin_H[2],0,0,0,0,0,0,0],
                            [0,0,0,hmin_H[3],0,0,0,0,0,0],
                            [0,0,0,0,hmin_H[4],0,0,0,0,0],
                            [0,0,0,0,0,hmin_H[5],0,0,0,0],
                            [0,0,0,0,0,0,hmin_H[6],0,0,0],
                            [0,0,0,0,0,0,0,hmin_H[7],0,0],
                            [0,0,0,0,0,0,0,0,hmin_H[8],0],
                            [0,0,0,0,0,0,0,0,0,hmin_H[9]]])
hmin_H_hat = np.concatenate([hmin_hat_first,hmin_H_hat_second], 1)

hmax_hat_first = np.zeros((gamma_n, delta_n))
hmax_H_hat_second = np.array([[hmax_H[0],0,0,0,0,0,0,0,0,0],
                     [0,hmax_H[1],0,0,0,0,0,0,0,0],
                     [0,0,hmax_H[2],0,0,0,0,0,0,0],
                     [0,0,0,hmax_H[3],0,0,0,0,0,0],
                     [0,0,0,0,hmax_H[4],0,0,0,0,0],
                     [0,0,0,0,0,hmax_H[5],0,0,0,0],
                     [0,0,0,0,0,0,hmax_H[6],0,0,0],
                     [0,0,0,0,0,0,0,hmax_H[7],0,0],
                     [0,0,0,0,0,0,0,0,hmax_H[8],0],
                     [0,0,0,0,0,0,0,0,0,hmax_H[9]]])
hmax_H_hat = np.concatenate([hmax_hat_first,hmax_H_hat_second], 1)

eps_hat_first = np.zeros((gamma_n, delta_n))
eps_hat_second = np.array([[eps[0],0,0,0,0,0,0,0,0,0],
                     [0,eps[1],0,0,0,0,0,0,0,0],
                     [0,0,eps[2],0,0,0,0,0,0,0],
                     [0,0,0,eps[3],0,0,0,0,0,0],
                     [0,0,0,0,eps[4],0,0,0,0,0],
                     [0,0,0,0,0,eps[5],0,0,0,0],
                     [0,0,0,0,0,0,eps[6],0,0,0],
                     [0,0,0,0,0,0,0,eps[7],0,0],
                     [0,0,0,0,0,0,0,0,eps[8],0],
                     [0,0,0,0,0,0,0,0,0,eps[9]]])
eps_hat = np.concatenate([eps_hat_first,eps_hat_second], 1)

gmin_H_hat_first = np.array([[gmin_H[0],0,0,0,0],
                            [gmin_H[1],0,0,0,0],
                            [0,gmin_H[2],0,0,0],
                            [0,gmin_H[3],0,0,0],
                            [0,0,gmin_H[4],0,0],
                            [0,0,gmin_H[5],0,0],
                            [0,0,0,gmin_H[6],0],
                            [0,0,0,gmin_H[7],0],
                            [0,0,0,0,gmin_H[8]],
                            [0,0,0,0,gmin_H[9]]])
gmin_hat_second = np.zeros((2*delta_n, gamma_n))
gmin_H_hat = np.concatenate([gmin_H_hat_first,gmin_hat_second], 1)

gmax_H_hat_first = np.array([[gmax_H[0],0,0,0,0],
                            [gmax_H[1],0,0,0,0],
                            [0,gmax_H[2],0,0,0],
                            [0,gmax_H[3],0,0,0],
                            [0,0,gmax_H[4],0,0],
                            [0,0,gmax_H[5],0,0],
                            [0,0,0,gmax_H[6],0],
                            [0,0,0,gmax_H[7],0],
                            [0,0,0,0,gmax_H[8]],
                            [0,0,0,0,gmax_H[9]]])
gmax_hat_second = np.zeros((2*delta_n, gamma_n))
gmax_H_hat = np.concatenate([gmax_H_hat_first,gmax_hat_second], 1)

Gamma1_first = np.array([[1,1,1,1,1],[-1,-1,-1,-1,-1]])
Gamma1_second = np.zeros((2, gamma_n))
Gamma1 = np.concatenate([Gamma1_first,Gamma1_second], 1)

Eps1 = np.array([1,-1])

Gamma2_first = np.array([[1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,1,0,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,1,0],
                   [0,0,0,0,1],
                   [0,0,0,0,1],
                   [0,0,0,0,1],
                   [0,0,0,0,1]])
Gamma2_second = np.array([[-1,0,0,0,0,0,0,0,0,0],
                   [0,-1,0,0,0,0,0,0,0,0],
                   [0,0,-1,0,0,0,0,0,0,0],
                   [0,0,0,-1,0,0,0,0,0,0],
                   [1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,-1,0,0,0,0,0],
                   [0,0,0,0,0,-1,0,0,0,0],
                   [0,0,0,0,0,0,-1,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,-1,0,0],
                   [0,0,0,0,0,0,0,0,-1,0],
                   [0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,0,-1],
                   [0,0,0,1,0,0,0,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,1]])
Gamma2 = np.concatenate([Gamma2_first,Gamma2_second], 1)
Eps2 = np.array([0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1])
One_H = np.concatenate((np.eye(2),np.eye(2),np.eye(2),np.eye(2),np.eye(2)),axis=1)
I = np.eye(2*delta_n)

E1_H = np.concatenate([np.zeros((2,2)),-QT_H,QT_H,-QT_H,QT_H,np.zeros((20+2*delta_n+2*delta_n,2)),AB_H,-AB_H], 0)
E3_H = np.concatenate([np.zeros((2,1)),-R_H,R_H,-R_H,R_H,np.zeros((20+2*delta_n+2*delta_n,1)),C_H,-C_H], 0)
E4_H = np.concatenate([np.zeros((2+gamma_n+gamma_n+gamma_n+gamma_n+20,2*delta_n)),-I,I,-I,I], 0)
E5_H = np.concatenate([Gamma1,np.zeros((gamma_n+gamma_n,delta_n+gamma_n)),-hmin_H_hat,-hmax_H_hat-eps_hat,Gamma2,gmin_H_hat,-gmax_H_hat,gmax_H_hat,-gmin_H_hat], 0)
E6_H = np.concatenate([Eps1,-hmin_H+S_H,hmax_H-S_H,-hmin_H+S_H,-eps-S_H,Eps2,np.zeros(2*delta_n),np.zeros(2*delta_n),gmax_H-D_H,-gmin_H+D_H], 0)



#=============================================================================#
#=============================================================================#
#=========================== Construct the problem ===========================#
#=============================================================================#
#=============================================================================#

# s：モードの数，time：制御を行う最終時刻，N：予測ステップ数
tm, tf = 7, 25
s, N = 5, 5
time = tf+N


# トマトモデルのパラメータ
H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314
H_min, H_max = 42, H_0      # H_min:トマト(q2)の下限値
Enz_min, Enz_max = 61, 80

# ポテトモデルのパラメータ
a = 1.00
b = -1.59
E = 0.045


#   外部の温度と湿度
Ta_min, Ta_max = 278, 298
Rh_min, Rh_max = 30, 95
T0 = np.random.uniform(Ta_min,Ta_max,(1,time))
Rh0 = np.random.uniform(Rh_min,Rh_max,(1,time))


#### 品質の設定 ####
# q_2:H(t), q_3:Enz(t) とする
ta0, rh0 = 280, 50
q_10, q_20, q_30 = 1100, H_0, Enz_min       # 品質の初期値
q1_min, q2_min = 900, H_min                 # 品質の下限値
desire = 0.95                                # 所望係数
qf_1, qf_2 = q_10*desire, q_20*desire       # 品質の所望値
w1, w2, w3, w4 = 0.5, 1, 900, 800           # 重み係数


def k_rate(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def H(t,T):
    return H_plusinf + (H_minusinf-H_plusinf)/(1+np.exp((k_rate(T)*t)*(H_minusinf-H_plusinf))*(H_minusinf-H_0)/(H_0-H_plusinf))

Enz_0 = Enz_min
t_span = [0.0,time]
t_eval = list(range(tf))

def fun2(t,X,T):
    H,Enz = X
    return [-k_rate(T)*H*Enz, k_rate(T)*H*Enz]

def fun1(t,X,T,Rh):
    q = X
    return ((-a * np.exp(b*Rh/100) * np.exp(-E/(Rg*T)))/1000) * q




#=================================================================================================#
#=================================================================================================#
#=========================== 制御器Aのモデル予測制御（MPC of controller_A） ===========================#
#=================================================================================================#
#=================================================================================================#
print("MPC of Controller_A start !") 

q_1star, q_2star, Ta_star, Rh_star = np.zeros(time+1), np.zeros((2,time+1)), np.zeros(time), np.zeros(time)
q_1star[0], q_2star[0,0], q_2star[1,0] = q_10, q_20, q_30
for j in range(time):
    if j <= tf:
        cost_mpc = 0        #cost : コスト関数．最小化問題の目的関数
        constr_mpc = []     #constr : 最小化問題の制約条件
        t = 0
        q_1, q_2 = cp.Variable((1,N+1)), cp.Variable((2,N+1))
        Ta, Rh = cp.Variable((1,N)), cp.Variable((1,N))
        z_1, z_2 = cp.Variable((delta_n, N)), cp.Variable((2*delta_n, N))
        delta_1, delta_2 = cp.Variable((delta_n+gamma_n, N), integer=True), cp.Variable((delta_n+gamma_n, N), integer=True)
        for k in range(j,j+N):
            # q, Ta, Rh = cp.Variable((1,j+N+1)), cp.Variable((1,j+N)), cp.Variable((1,j+N))
            cost_mpc += w1*cp.square(Ta[:,t]-T0[:,k]) + w2*cp.square(Rh[:,t]-Rh0[:,k])
            constr_mpc += [q_1[:,t+1] == One@z_1[:,t],
            q_2[:,t+1] == One_H@z_2[:,t],
            E1@q_1[:,t] + E2@Ta[:,t] + E3@Rh[:,t] + E4@z_1[:,t] + E5@delta_1[:,t] <= E6,
            E1_H@q_2[:,t] + E3_H@Ta[:,t] + E4_H@z_2[:,t] + E5_H@delta_2[:,t] <= E6_H
            ]
            t = t+1
        cost_mpc += w3*cp.square(q_1star[j] - q_1[:,N])
        if j <= tm:
            cost_mpc += w4*cp.square(q_2star[0,j] - q_2[0,N])
            constr_mpc += [q_2[0,N] >= q2_min]
        constr_mpc += [q_1[:,0] == q_1star[j], q_2[0,0] == q_2star[0,j], q_2[1,0] == q_2star[1,j]]
        constr_mpc += [q_1[:,N] >= q1_min]
        constr_mpc += [Ta <= Ta_max, Ta >= Ta_min, Rh <= Rh_max, Rh >= Rh_min]
        obj_mpc = cp.Minimize(cost_mpc)
        prob_mpc = cp.Problem(obj_mpc, constr_mpc)
        prob_mpc.solve(solver=cp.CPLEX, verbose=False)
        Ta_star[j] = Ta[:,0].value
        Rh_star[j] = Rh[:,0].value
        # q_1star[j+1], q_2star[:,j+1] = q_1[:,1].value, q_2[:,1].value
        #   得られた制御入力(Ta,Rh)を，制御対象(非線形関数)に代入する．
        init_q1 = [q_1star[j]]
        sol_q1 = solve_ivp(fun1,t_span,init_q1,method='RK45',t_eval=t_eval,args=[Ta_star[j],Rh_star[j]])
        # print("sol_q1.y:\n", sol_q1.y)
        q_1star[j+1] = sol_q1.y[0,1]
        init_q2 = [q_2star[0,j]-H_plusinf, q_2star[1,j]]
        sol_q2 = solve_ivp(fun2,t_span,init_q2,method='RK45',t_eval=t_eval,args=[Ta_star[j]])
        q_2star[0,j+1], q_2star[1,j+1] = sol_q2.y[0,1]+H_plusinf, sol_q2.y[1,1]
        # q_1star[j+1] = fun(Ta_star[j],Rh_star[j],q_star[j])
        # print("j=",j)
        # print("q_1star(j+1):",q_1star[j+1])
        # print("q_2star(j+1):",q_2star[:,j+1])
print("MPC of Controller_A finished !")




#=================================================================================================#
#=================================================================================================#
#=========================== 制御器Bのモデル予測制御（MPC of controller_B） ===========================#
#=================================================================================================#
#=================================================================================================#
print("MPC of Controller_B start !") 

q_1star_B, q_2star_B, Ta_star_B, Rh_star_B = np.zeros(time+1), np.zeros((2,time+1)), np.zeros(time), np.zeros(time)
q_1star_B[0], q_2star_B[0,0], q_2star_B[1,0] = q_10, q_20, q_30
for j in range(time):
    if j <= tf:
        cost_mpc = 0        #cost : コスト関数．最小化問題の目的関数
        constr_mpc = []     #constr : 最小化問題の制約条件
        t = 0
        q_1, q_2 = cp.Variable((1,N+1)), cp.Variable((2,N+1))
        Ta, Rh = cp.Variable((1,N)), cp.Variable((1,N))
        z_1, z_2 = cp.Variable((delta_n, N)), cp.Variable((2*delta_n, N))
        delta_1, delta_2 = cp.Variable((delta_n+gamma_n, N), integer=True), cp.Variable((delta_n+gamma_n, N), integer=True)
        for k in range(j,j+N):
            # q, Ta, Rh = cp.Variable((1,j+N+1)), cp.Variable((1,j+N)), cp.Variable((1,j+N))
            cost_mpc += w1*cp.square(Ta[:,t]-T0[:,k]) + w2*cp.square(Rh[:,t]-Rh0[:,k])
            constr_mpc += [q_1[:,t+1] == One@z_1[:,t],
            q_2[:,t+1] == One_H@z_2[:,t],
            E1@q_1[:,t] + E2@Ta[:,t] + E3@Rh[:,t] + E4@z_1[:,t] + E5@delta_1[:,t] <= E6,
            E1_H@q_2[:,t] + E3_H@Ta[:,t] + E4_H@z_2[:,t] + E5_H@delta_2[:,t] <= E6_H
            ]
            t = t+1
        cost_mpc += w3*cp.square(q_1[:,N] - desire*q_1star[j])
        if j <= tm:
            cost_mpc += w4*cp.square(q_2[0,N] - desire*q_2star_B[0,j])
            constr_mpc += [q_2[0,N] >= q2_min]
        constr_mpc += [q_1[:,0] == q_1star[j], q_2[0,0] == q_2star[0,j], q_2[1,0] == q_2star[1,j]]
        constr_mpc += [q_1[:,N] >= q1_min]
        constr_mpc += [Ta <= Ta_max, Ta >= Ta_min, Rh <= Rh_max, Rh >= Rh_min]
        obj_mpc = cp.Minimize(cost_mpc)
        prob_mpc = cp.Problem(obj_mpc, constr_mpc)
        prob_mpc.solve(solver=cp.CPLEX, verbose=False)
        Ta_star_B[j] = Ta[:,0].value
        Rh_star_B[j] = Rh[:,0].value
        # q_1star[j+1], q_2star[:,j+1] = q_1[:,1].value, q_2[:,1].value
        #   得られた制御入力(Ta,Rh)を，制御対象(非線形関数)に代入する．
        init_q1 = [q_1star_B[j]]
        sol_q1 = solve_ivp(fun1,t_span,init_q1,method='RK45',t_eval=t_eval,args=[Ta_star_B[j],Rh_star_B[j]])
        # print("sol_q1.y:\n", sol_q1.y)
        q_1star_B[j+1] = sol_q1.y[0,1]
        init_q2 = [q_2star_B[0,j]-H_plusinf, q_2star_B[1,j]]
        sol_q2 = solve_ivp(fun2,t_span,init_q2,method='RK45',t_eval=t_eval,args=[Ta_star_B[j]])
        q_2star_B[0,j+1], q_2star_B[1,j+1] = sol_q2.y[0,1]+H_plusinf, sol_q2.y[1,1]
        # print("j=",j)
        # print("q_1star(j+1):",q_1star_B[j+1])
        # print("q_2star(j+1):",q_2star_B[:,j+1])
print("MPC of Controller_B finished !")





#=====================================================================#
#=====================================================================#
#=========================== Plot results. ===========================#
#=====================================================================#
#=====================================================================#

sns.set()
sns.set_style("whitegrid")
fig1 = plt.figure(figsize=(6,4))
fig2 = plt.figure(figsize=(6,4))
fig3 = plt.figure(figsize=(6,4))
fig4 = plt.figure(figsize=(6,4))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

ax1.plot(range(tf), q_1star[0:tf], label="$q_{1}(k)$(Controller A MPC)")
ax1.plot(range(tf), q_1star_B[0:tf], label="$q_{1}(k)$(Controller B OC)")
ax1.set_ylabel("quality 1$[g]$",fontsize=12)
ax1.set_xlabel("$k[days]$",fontsize=12)
ax1.legend(loc='best')

ax2.plot(range(tm), q_2star[0,0:tm], label="$q_{2}(k)$(Controller AMPC)")
ax2.plot(range(tm), q_2star_B[0,0:tm], label="$q_{2}(k)$(Controller B OC)")
ax2.set_ylabel("quality 2$[{}^\circ]$",fontsize=12)
ax2.set_xlabel("$k[days]$",fontsize=12)
ax2.legend(loc='best')

ax3.step(range(tf), Ta_star[0:tf], where='post', label="$T_{a}(k)$(Controller A MPC)", marker="o")
ax3.step(range(tf), Ta_star_B[0,0:tf], where='post', label="$T_{a}(k)$(Controller B OC)", marker="o")
ax3.plot(range(tf), T0[0,0:tf], label="$T_{aout}(k)$", linestyle="dashed")
ax3.set_ylabel("$T_a[K]$",fontsize=12)
ax3.set_xlabel("$k[days]$",fontsize=12)
ax3.legend(loc='best')

ax4.step(range(tf), Rh_star[0:tf], where='post', label="$R_{h}(k)$(Controller A MPC)", marker="o")
ax4.step(range(tf), Rh_star_B[0,0:tf], where='post', label="$R_{h}(k)$(Controller B OC)", marker="o")
ax4.plot(range(tf), Rh0[0,0:tf], label="$R_{hout}(k)$", linestyle="dashed")
ax4.set_ylabel("$R_h[\%]$",fontsize=12)
ax4.set_xlabel("$k[days]$",fontsize=12)
ax4.legend(loc='best')
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
# fig1.savefig("sim2_q1.png")
# fig2.savefig("sim2_q2.png")
# fig3.savefig("sim2_imput1.png")
# fig4.savefig("sim2_imput2.png")
plt.show()