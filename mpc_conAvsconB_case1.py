# 制御器Aと制御器Bの比較．
# モデル予測制御(MPC)のみ．
# 品質は1種類．
# 制約条件を，行列(E)を用いた1つの不等式で表記している．最適化問題を解く際にfor文を使用．

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py

#============================ Problem data.==============================================================#
# np.random.seed(3)
delta_n, gamma_n = 5, 10
#============================ Problem data.==============================================================#


#=========================== potatoのMLDモデルのパラメータ ================================================#
with h5py.File('potato_mld_param.hdf5', mode='r') as f:
    f.keys()
    One = f['One'][()]
    E1 = f['E1'][()]
    E2 = f['E2'][()]
    E3 = f['E3'][()]
    E4 = f['E4'][()]
    E5 = f['E5'][()]
    E6 = f['E6'][()]
    pass
#=========================== potatoのMLDモデルのパラメータ ================================================#

#=========================== tomatoのMLDモデルのパラメータ ================================================#
with h5py.File('tomato_mld_param.hdf5', mode='r') as f:
    f.keys()
    One_H = f['One'][()]
    E1_H = f['E1'][()]
    E3_H = f['E3'][()]
    E4_H = f['E4'][()]
    E5_H = f['E5'][()]
    E6_H = f['E6'][()]
    pass
#=========================== tomatoのMLDモデルのパラメータ ================================================#




#========================= Construct the problem.=========================================================#
s, time, N = 5, 22, 5   # s：モードの数，time：制御を行う最終時刻，N：予測ステップ数
tm, tf = 10, 15

H_plusinf, H_minusinf = 43.1, 124
H_0, Enz_0 = 62.9, 61   # トマト品質の初期値

abs_T = 273.15
Ta_min, Ta_max = 5+abs_T, 28+abs_T
Rh_min, Rh_max = 30, 95
H_min, H_max = 44-H_plusinf, H_0
Enz_min, Enz_max = Enz_0, 80

T0 = np.random.uniform(Ta_min,Ta_max,(1,time))  # 外部温度
Rh0 = np.random.uniform(Rh_min,Rh_max,(1,time)) # 外部湿度

ta0, rh0 = 280, 50  # 制御入力の初期値
q_10, q_20, q_30 = 1100, H_0-H_plusinf, Enz_0   # 品質の初期値, q_2:H(t), q_3:Enz(t)
q1_min, q2_min = 1000, H_min      #　品質の下限値
w1, w2, w3, w4 = 0.5, 0.5, 900, 900   # 重み係数. w1:Ta, w2:Rh, w3:q1, w4:q2

def k_rate(T):      #速度定数式
    k_ref = 0.0019
    E_a = 170.604
    T_ref = 288.15
    Rg = 0.008314
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def H(t,T):     # トマトの微分方程式の解
    return H_plusinf + (H_minusinf-H_plusinf)/(1+np.exp((k_rate(T)*t)*(H_minusinf-H_plusinf))*(H_minusinf-H_0)/(H_0-H_plusinf))

def fun2(t,X,T):    # トマトの微分方程式
    H,Enz = X
    return [-k_rate(T)*H*Enz, k_rate(T)*H*Enz]

def fun1(t,X,T,Rh): # ポテトの微分方程式
    q = X
    a = 1.00
    b = -1.59
    Ea = 0.045
    Rg = 0.008314
    return ((-a * np.exp(b*Rh/100) * np.exp(-Ea/(Rg*T)))/1000) * q

t_span = [0.0,time]
t_eval = list(range(tf))
#========================= Construct the problem.=========================================================#






#======================= MPC of Controller A ===============================================================#
print("MPC of A start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
            # q_2[:,t+1] == One_H@z_2[:,t],
            E1@q_1[:,t] + E2@Ta[:,t] + E3@Rh[:,t] + E4@z_1[:,t] + E5@delta_1[:,t] <= E6,
            # E1_H@q_2[:,t] + E3_H@Ta[:,t] + E4_H@z_2[:,t] + E5_H@delta_2[:,t] <= E6_H
            ]
            t = t+1
        cost_mpc += w3*cp.square(q_1star[j] - q_1[:,N])
        # if j <= tm:
            # cost_mpc += w4*cp.square(q_2star[0,j] - q_2[0,N])
            # constr_mpc += [q_2[0,N] >= q2_min]
        constr_mpc += [q_1[:,0] == q_1star[j]]
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
        # q_1star[j+1] = fun(Ta_star[j],Rh_star[j],q_star[j])
        print("j=",j)
        # print("q_1star(j+1):",q_1star[j+1])
print("MPC of A finished !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#======================= MPC of Controller A ================================================================#





#======================= MPC of Controller B ================================================================#
print("MPC of B start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
q_1star_B, q_2star_B, Ta_star_B, Rh_star_B = np.zeros(time+1), np.zeros((2,time+1)), np.zeros(time), np.zeros(time)
q_1star_B[0], q_2star_B[0,0], q_2star_B[1,0] = q_10, q_20, q_30
desire = 0.9
qf_1, qf_2 = desire*q_10, desire*q_20
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
            # q_2[:,t+1] == One_H@z_2[:,t],
            E1@q_1[:,t] + E2@Ta[:,t] + E3@Rh[:,t] + E4@z_1[:,t] + E5@delta_1[:,t] <= E6,
            # E1_H@q_2[:,t] + E3_H@Ta[:,t] + E4_H@z_2[:,t] + E5_H@delta_2[:,t] <= E6_H
            ]
            t = t+1
        # cost_mpc += w3*cp.square(q_1[:,N] - desire*q_1star[j])
        cost_mpc += w3*cp.square(q_1[:,N] - qf_1)
        # if j <= tm:
            # cost_mpc += w4*cp.square(q_2[0,N] - desire*q_2star_B[0,j])
            # constr_mpc += [q_2[0,N] >= q2_min]
        constr_mpc += [q_1[:,0] == q_1star[j]]
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
        print("j=",j)
        # print("q_1star(j+1):",q_1star_B[j+1])
print("MPC of B finished !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#======================= MPC of Controller B ================================================================#





#=========================== Plot results. ===================================================================#
sns.set()
sns.set_style("whitegrid")
fig1 = plt.figure(figsize=(6,4))
fig3 = plt.figure(figsize=(6,4))
fig4 = plt.figure(figsize=(6,4))
ax1 = fig1.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

ax1.plot(range(tf), q_1star[0:tf], label="$q_{1}(k)$(MPC of A)")
ax1.plot(range(tf), q_1star_B[0:tf], label="$q_{1}(k)$(MPC of B)")
ax1.plot(range(tf), np.array([qf_1]*tf), label="$qf_{1}$")
ax1.plot(range(tf), np.array([q1_min]*tf), label="$q1_{min}$")
ax1.set_ylabel("quality 1$[g]$",fontsize=12)
ax1.set_xlabel("$k[days]$",fontsize=12)
ax1.legend(loc='best')


ax3.step(range(tf), Ta_star[0:tf], where='post', label="$T_{a}(k)$(MPC of A)", marker="o")
ax3.step(range(tf), Ta_star_B[0:tf], where='post', label="$T_{a}(k)$(MPC of B)", marker="o")
ax3.plot(range(tf), T0[0,0:tf], label="$T_{aout}(k)$", linestyle="dashed")
ax3.set_ylabel("$T_a[K]$",fontsize=12)
ax3.set_xlabel("$k[days]$",fontsize=12)
ax3.legend(loc='best')

ax4.step(range(tf), Rh_star[0:tf], where='post', label="$R_{h}(k)$(MPC of A)", marker="o")
ax4.step(range(tf), Rh_star_B[0:tf], where='post', label="$R_{h}(k)$(MPC of B)", marker="o")
ax4.plot(range(tf), Rh0[0,0:tf], label="$R_{hout}(k)$", linestyle="dashed")
ax4.set_ylabel("$R_h[\%]$",fontsize=12)
ax4.set_xlabel("$k[days]$",fontsize=12)
ax4.legend(loc='best')
fig1.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig1.savefig("sim1_q1.png")
fig3.savefig("sim1_imput1.png")
fig4.savefig("sim1_imput2.png")
plt.show()
#=========================== Plot results. ===================================================================#