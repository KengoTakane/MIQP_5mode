import cvxpy as cp
import numpy as np
from sympy import exp
import cplex
import docplex
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

#制約条件を，行列を用いた1つの不等式で表記している．最適化問題を解く際にfor文を使用．

# Problem data.
# np.random.seed(3)
s, time = 5, 10
delta_n, gamma_n = 5, 10
qf = 1000
Ta_max = 298
Ta_min = 278
Rh_max = 95
Rh_min = 30
# T0 = np.random.randint(Ta_min,Ta_max,(time,1))
# Rh0 = np.random.randint(Rh_min,Rh_max,(time,1))
T0 = np.random.randint(Ta_min,Ta_max,(1,time))
Rh0 = np.random.randint(Rh_min,Rh_max,(1,time))


################################################################################
#############################potatoのパラメータ###################################
################################################################################

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
                   [0,0,0,1,0,0,0,1,0,0],
                   [0,0,0,0,0,0,1,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,1]])
Gamma2 = np.concatenate([Gamma2_first,Gamma2_second], 1)
Eps2 = np.array([0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1])
One = np.ones((1,delta_n))
I = np.eye(delta_n)
Zero = np.zeros(delta_n)
zero = np.stack(([Zero]*(time-1)), axis = 0)
beta = np.concatenate([zero,One], 0).flatten()

E1 = np.concatenate([np.zeros((2,1)),-Q,Q,-Q,Q,np.zeros((20+delta_n+delta_n,1)),A,-A], 0)
E2 = np.concatenate([np.zeros((2,1)),-T,T,-T,T,np.zeros((20+delta_n+delta_n,1)),B,-B], 0)
E3 = np.concatenate([np.zeros((2,1)),-R,R,-R,R,np.zeros((20+delta_n+delta_n,1)),C,-C], 0)
E4 = np.concatenate([np.zeros((2+gamma_n+gamma_n+gamma_n+gamma_n+20,delta_n)),-I,I,-I,I], 0)
E5 = np.concatenate([Gamma1,np.zeros((gamma_n+gamma_n,delta_n+gamma_n)),-hmin_hat,-hmax_hat-eps_hat,Gamma2,gmin_hat,-gmax_hat,gmax_hat,-gmin_hat], 0)
E6 = np.concatenate([Eps1,-hmin+S,hmax-S,-hmin+S,-eps-S,Eps2,np.zeros(delta_n),np.zeros(delta_n),gmax-D,-gmin+D], 0)








################################################################################
############################# dHのパラメータ ###################################
################################################################################

A_H = np.array([[-4.21270271e-04+1], [-2.36178784e-04+1], [-3.22164500e-04+1], [-2.52650737e-04+1], [-5.55708636e-04+1]])
B_H = np.array([[-4.84453301e-05], [-3.14845920e-05], [-5.36849855e-05], [-1.67910305e-05], [1.35079377e-05]])
C_H = np.array([[6.47483981e-03], [3.51970772e-03], [5.42080197e-03], [4.06639287e-03], [8.64274190e-03]])
D_H = np.array([-3.29723012e-01, -3.06409958e-01, -3.63390858e-01, -3.42315514e-01, -3.13246241e-01])
Q_H = np.array([[0.02627317], [-0.00111771], [0.01880834], [-0.01876554], [-0.04472564], [-0.03862955], [-0.05427779], [0.02798923], [-0.00845947], [-0.01385068]])
T_H = np.array([[-0.07538652], [0.04318842], [-0.04245705], [-0.05633143], [0.05458913], [0.0601666], [-0.03055113], [-0.01195942], [-0.00280735], [0.02021543]])
R_H = np.array([[-0.16798629], [-0.24951434], [-0.34649457], [0.24132598], [0.2253457], [0.20345188], [0.14436165], [-0.23591518], [0.32142254], [0.1428877]])
S_H = np.array([8.4387248, 3.37998739, 16.72282594, 24.14807339, 9.62861827, 1.78944618, 51.78167187, -6.80356885, -7.37417206, -0.34880202])

gmin_H = np.array([899.47084533937, 899.5771749904001, 899.49317965545, 899.54725329965, 899.4496129264676])
gmax_H = np.array([1099.8086670060725, 1099.7595073502, 1099.7824410081375, 1099.7614584645626, 1099.90058712121])
hmin_H = np.array([-6.490075750000003, -9.676540410000001, -12.00376721, -6.153670209999998, -17.797203979999995, -18.053687420000003, -12.758386630000004, -7.61302995, -7.879117860000002, -5.738675770000001])
hmax_H = np.array([11.568330099999997, 7.845144190000001, 15.34147409, 14.693912240000003, 7.160122770000001, 4.400759780000007, 8.244456869999993, 13.618288250000004, 14.775424989999998, 6.824546480000001])
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

gmin_H_hat_first = np.array([[gmin_H[0],0,0,0,0],[0,gmin_H[1],0,0,0],[0,0,gmin_H[2],0,0],[0,0,0,gmin_H[3],0],[0,0,0,0,gmin_H[4]]])
gmin_hat_second = np.zeros((delta_n, gamma_n))
gmin_H_hat = np.concatenate([gmin_H_hat_first,gmin_hat_second], 1)

gmax_H_hat_first = np.array([[gmax_H[0],0,0,0,0],[0,gmax_H[1],0,0,0],[0,0,gmax_H[2],0,0],[0,0,0,gmax_H[3],0],[0,0,0,0,gmax_H[4]]])
gmax_hat_second = np.zeros((delta_n, gamma_n))
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
                   [0,0,0,1,0,0,0,1,0,0],
                   [0,0,0,0,0,0,1,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,1]])
Gamma2 = np.concatenate([Gamma2_first,Gamma2_second], 1)
Eps2 = np.array([0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1])
One = np.ones((1,delta_n))
I = np.eye(delta_n)
Zero = np.zeros(delta_n)
zero = np.stack(([Zero]*(time-1)), axis = 0)
beta = np.concatenate([zero,One], 0).flatten()

E1_H = np.concatenate([np.zeros((2,1)),-Q_H,Q_H,-Q_H,Q_H,np.zeros((20+delta_n+delta_n,1)),A_H,-A_H], 0)
E2_H = np.concatenate([np.zeros((2,1)),-T_H,T_H,-T_H,T_H,np.zeros((20+delta_n+delta_n,1)),B_H,-B_H], 0)
E3_H = np.concatenate([np.zeros((2,1)),-R_H,R_H,-R_H,R_H,np.zeros((20+delta_n+delta_n,1)),C_H,-C_H], 0)
E4_H = np.concatenate([np.zeros((2+gamma_n+gamma_n+gamma_n+gamma_n+20,delta_n)),-I,I,-I,I], 0)
E5_H = np.concatenate([Gamma1,np.zeros((gamma_n+gamma_n,delta_n+gamma_n)),-hmin_H_hat,-hmax_H_hat-eps_hat,Gamma2,gmin_H_hat,-gmax_H_hat,gmax_H_hat,-gmin_H_hat], 0)
E6_H = np.concatenate([Eps1,-hmin_H+S_H,hmax_H-S_H,-hmin_H+S_H,-eps-S_H,Eps2,np.zeros(delta_n),np.zeros(delta_n),gmax_H-D_H,-gmin_H+D_H], 0)






################################################################################
############################# dEnzのパラメータ ###################################
################################################################################

A_Enz = np.array([[-4.21270271e-04+1], [-2.36178784e-04+1], [-3.22164500e-04+1], [-2.52650737e-04+1], [-5.55708636e-04+1]])
B_Enz = np.array([[-4.84453301e-05], [-3.14845920e-05], [-5.36849855e-05], [-1.67910305e-05], [1.35079377e-05]])
C_Enz = np.array([[6.47483981e-03], [3.51970772e-03], [5.42080197e-03], [4.06639287e-03], [8.64274190e-03]])
D_Enz = np.array([-3.29723012e-01, -3.06409958e-01, -3.63390858e-01, -3.42315514e-01, -3.13246241e-01])
Q_Enz = np.array([[0.02627317], [-0.00111771], [0.01880834], [-0.01876554], [-0.04472564], [-0.03862955], [-0.05427779], [0.02798923], [-0.00845947], [-0.01385068]])
T_Enz = np.array([[-0.07538652], [0.04318842], [-0.04245705], [-0.05633143], [0.05458913], [0.0601666], [-0.03055113], [-0.01195942], [-0.00280735], [0.02021543]])
R_Enz = np.array([[-0.16798629], [-0.24951434], [-0.34649457], [0.24132598], [0.2253457], [0.20345188], [0.14436165], [-0.23591518], [0.32142254], [0.1428877]])
S_Enz = np.array([8.4387248, 3.37998739, 16.72282594, 24.14807339, 9.62861827, 1.78944618, 51.78167187, -6.80356885, -7.37417206, -0.34880202])

gmin_Enz = np.array([899.47084533937, 899.5771749904001, 899.49317965545, 899.54725329965, 899.4496129264676])
gmax_Enz = np.array([1099.8086670060725, 1099.7595073502, 1099.7824410081375, 1099.7614584645626, 1099.90058712121])
hmin_Enz = np.array([-6.490075750000003, -9.676540410000001, -12.00376721, -6.153670209999998, -17.797203979999995, -18.053687420000003, -12.758386630000004, -7.61302995, -7.879117860000002, -5.738675770000001])
hmax_Enz = np.array([11.568330099999997, 7.845144190000001, 15.34147409, 14.693912240000003, 7.160122770000001, 4.400759780000007, 8.244456869999993, 13.618288250000004, 14.775424989999998, 6.824546480000001])
eps = np.array([1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05])

hmin_hat_first = np.zeros((gamma_n, delta_n))
hmin_Enz_hat_second = np.array([[hmin_Enz[0],0,0,0,0,0,0,0,0,0],
                            [0,hmin_Enz[1],0,0,0,0,0,0,0,0],
                            [0,0,hmin_Enz[2],0,0,0,0,0,0,0],
                            [0,0,0,hmin_Enz[3],0,0,0,0,0,0],
                            [0,0,0,0,hmin_Enz[4],0,0,0,0,0],
                            [0,0,0,0,0,hmin_Enz[5],0,0,0,0],
                            [0,0,0,0,0,0,hmin_Enz[6],0,0,0],
                            [0,0,0,0,0,0,0,hmin_Enz[7],0,0],
                            [0,0,0,0,0,0,0,0,hmin_Enz[8],0],
                            [0,0,0,0,0,0,0,0,0,hmin_Enz[9]]])
hmin_Enz_hat = np.concatenate([hmin_hat_first,hmin_Enz_hat_second], 1)

hmax_hat_first = np.zeros((gamma_n, delta_n))
hmax_Enz_hat_second = np.array([[hmax_Enz[0],0,0,0,0,0,0,0,0,0],
                     [0,hmax_Enz[1],0,0,0,0,0,0,0,0],
                     [0,0,hmax_Enz[2],0,0,0,0,0,0,0],
                     [0,0,0,hmax_Enz[3],0,0,0,0,0,0],
                     [0,0,0,0,hmax_Enz[4],0,0,0,0,0],
                     [0,0,0,0,0,hmax_Enz[5],0,0,0,0],
                     [0,0,0,0,0,0,hmax_Enz[6],0,0,0],
                     [0,0,0,0,0,0,0,hmax_Enz[7],0,0],
                     [0,0,0,0,0,0,0,0,hmax_Enz[8],0],
                     [0,0,0,0,0,0,0,0,0,hmax_Enz[9]]])
hmax_Enz_hat = np.concatenate([hmax_hat_first,hmax_Enz_hat_second], 1)

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

gmin_Enz_hat_first = np.array([[gmin_Enz[0],0,0,0,0],[0,gmin_Enz[1],0,0,0],[0,0,gmin_Enz[2],0,0],[0,0,0,gmin_Enz[3],0],[0,0,0,0,gmin_Enz[4]]])
gmin_hat_second = np.zeros((delta_n, gamma_n))
gmin_Enz_hat = np.concatenate([gmin_Enz_hat_first,gmin_hat_second], 1)

gmax_Enz_hat_first = np.array([[gmax_Enz[0],0,0,0,0],[0,gmax_Enz[1],0,0,0],[0,0,gmax_Enz[2],0,0],[0,0,0,gmax_Enz[3],0],[0,0,0,0,gmax_Enz[4]]])
gmax_hat_second = np.zeros((delta_n, gamma_n))
gmax_Enz_hat = np.concatenate([gmax_Enz_hat_first,gmax_hat_second], 1)

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
                   [0,0,0,1,0,0,0,1,0,0],
                   [0,0,0,0,0,0,1,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,1]])
Gamma2 = np.concatenate([Gamma2_first,Gamma2_second], 1)
Eps2 = np.array([0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,1])
One = np.ones((1,delta_n))
I = np.eye(delta_n)
Zero = np.zeros(delta_n)
zero = np.stack(([Zero]*(time-1)), axis = 0)
beta = np.concatenate([zero,One], 0).flatten()

E1_Enz = np.concatenate([np.zeros((2,1)),-Q_Enz,Q_Enz,-Q_Enz,Q_Enz,np.zeros((20+delta_n+delta_n,1)),A_Enz,-A_Enz], 0)
E2_Enz = np.concatenate([np.zeros((2,1)),-T_Enz,T_Enz,-T_Enz,T_Enz,np.zeros((20+delta_n+delta_n,1)),B_Enz,-B_Enz], 0)
E3_Enz = np.concatenate([np.zeros((2,1)),-R_Enz,R_Enz,-R_Enz,R_Enz,np.zeros((20+delta_n+delta_n,1)),C_Enz,-C_Enz], 0)
E4_Enz = np.concatenate([np.zeros((2+gamma_n+gamma_n+gamma_n+gamma_n+20,delta_n)),-I,I,-I,I], 0)
E5_Enz = np.concatenate([Gamma1,np.zeros((gamma_n+gamma_n,delta_n+gamma_n)),-hmin_Enz_hat,-hmax_Enz_hat-eps_hat,Gamma2,gmin_Enz_hat,-gmax_Enz_hat,gmax_Enz_hat,-gmin_Enz_hat], 0)
E6_Enz = np.concatenate([Eps1,-hmin_Enz+S_Enz,hmax_Enz-S_Enz,-hmin_Enz+S_Enz,-eps-S_Enz,Eps2,np.zeros(delta_n),np.zeros(delta_n),gmax_Enz-D_Enz,-gmin_Enz+D_Enz], 0)






# Construct the problem.
tm = 5

H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314

H_min, H_max = 42, H_0
Enz_min, Enz_max = 61, 80

def k_rate(T):
    return k_ref*exp((E_a/Rg)*(1/T_ref-1/T))

def H(t,T):
    return H_plusinf + (H_minusinf-H_plusinf)/(1+exp(((k_ref*exp((E_a/Rg)*(1/T_ref-1/T)))*t)*(H_minusinf-H_plusinf))*(H_minusinf-H_0)/(H_0-H_plusinf))


#### q_2:H(t), q_3:Enz(t) とする ####
q_1, q_2, q_3 = cp.Variable((1,time+1)), cp.Variable((1,time+1)), cp.Variable((1,time+1))
Ta, Rh = cp.Variable((1,time)), cp.Variable((1,time))
z_1, z_2, z_3 = cp.Variable((delta_n, time)), cp.Variable((delta_n, time)), cp.Variable((delta_n, time))
delta_1, delta_2, delta_3 = cp.Variable((delta_n+gamma_n, time), integer=True), cp.Variable((delta_n+gamma_n, time), integer=True), cp.Variable((delta_n+gamma_n, time), integer=True)
ta0, rh0 = 280, 50
q_10, q_20, q_30 = 1100, H_0, Enz_min
w1, w2, w3, w4 = 0.5, 0.5, 900, 10


a = (k_rate(Ta_min)-k_rate(Ta_max))/(Ta_max-Ta_min)
b = k_rate(Ta_min)-a*Ta_min


cost = 0
constr = []
for k in range(time):
    cost += w1*cp.square(Ta[:,k]-T0[:,k]) + w2*cp.square(Rh[:,k]-Rh0[:,k])
    constr += [q_1[:,k+1] == One@z_1[:,k],
    q_2[:,k+1] == One@z_2[:,k],
    q_3[:,k+1] == One@z_3[:,k],
    E1@q_1[:,k] + E2@Ta[:,k] + E3@Rh[:,k] + E4@z_1[:,k] + E5@delta_1[:,k] <= E6,
    E1_H@q_2[:,k] + E2_H@q_3[:,k] + E3_H@Ta[:,k] + E4_H@z_2[:,k] + E5_H@delta_2[:,k] <= E6_H,
    E1_Enz@q_2[:,k] + E2_Enz@q_3[:,k] + E3_Enz@Ta[:,k] + E4_Enz@z_3[:,k] + E5_Enz@delta_3[:,k] <= E6_Enz,
    ]
cost += w3*cp.square(q_10 - q_1[:,time])
cost += w4*cp.abs(q_20 - q_2[:,tm])
constr += [q_1[:,0] == q_10, q_2[:,0] == q_20, q_3[:,0] == q_30]
constr += [Ta <= Ta_max, Ta >= Ta_min, Rh <= Rh_max, Rh >= Rh_min]



objective = cp.Minimize(cost)
prob = cp.Problem(objective, constr)
prob.solve(solver=cp.CPLEX, verbose=True)
print("Status: ", prob.status)
print("The optimal value is\n", prob.value)
print('q:\n', q_1.value)
print('Ta:\n', Ta.value)
print('Rh:\n', Rh.value)
print('delta_1:\n', delta_1.value)
print('delta_2:\n', delta_2.value)
print('delta_3:\n', delta_3.value)
