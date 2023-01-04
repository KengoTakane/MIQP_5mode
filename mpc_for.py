import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# モデル予測制御
# 制約条件を，行列を用いた1つの不等式で表記している．最適化問題を解く際にfor文を使用．

# Problem data.
# np.random.seed(3)

delta_n, gamma_n = 5, 10


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


""" 
# 対角行列は使ってない．
E1_bar = block_diag(*([E1] * time))
E2_bar = block_diag(*([E2] * time))
E3_bar = block_diag(*([E3] * time))
E4_bar = block_diag(*([E4] * time))
E5_bar = block_diag(*([E5] * time))
E6_bar = np.stack(([E6]*time), axis = 0).flatten()
E6_bar = E6_bar[:, np.newaxis]
# print('E6_bar shape:', E6_bar.shape)
"""


# Construct the problem.

# s：モードの数，time：制御を行う最終時刻，N：予測ステップ数
s, time, N = 5, 25, 5
tm, tf = 9, 17

# ポテトモデルのパラメータ
a = 1.00
b = -1.59
E = 0.045
Rg = 0.008314

t_span = [0.0,time]
t_eval = list(range(tf))
def fun1(t,X,T,Rh):
    q = X
    return ((-a * np.exp(b*Rh/100) * np.exp(-E/(Rg*T)))/1000) * q

Ta_min, Ta_max = 278, 298
Rh_min, Rh_max = 30, 95
qf = 1000

T0 = np.random.uniform(Ta_min,Ta_max,(1,time))
Rh0 = np.random.uniform(Rh_min,Rh_max,(1,time))

q0, ta0, rh0 = 1100, 280, 50
w1, w2, w3 = 0.5, 0.8, 80000

q_star, Ta_star, Rh_star = np.zeros(time+1), np.zeros(time), np.zeros(time)
q_star[0] = q0
# print("Ta_star:\n", Ta_star)



for j in range(time):
    if j <= tf:
        cost = 0
        constr = []
        t = 0
        q, Ta, Rh = cp.Variable((1,N+1)), cp.Variable((1,N)), cp.Variable((1,N))
        z = cp.Variable((delta_n, N))
        delta = cp.Variable((delta_n+gamma_n, N), integer=True)
        for k in range(j,j+N):
            # q, Ta, Rh = cp.Variable((1,j+N+1)), cp.Variable((1,j+N)), cp.Variable((1,j+N))
            cost += w1*cp.square(Ta[:,t]-T0[:,k]) + w2*cp.square(Rh[:,t]-Rh0[:,k])
            constr += [q[:,t+1] == One@z[:,t],
            E1@q[:,t] + E2@Ta[:,t] + E3@Rh[:,t] + E4@z[:,t] + E5@delta[:,t] <= E6]
            t = t+1
        cost += w3*cp.square(q_star[j] - q[:,N])
        constr += [q[:,0] == q_star[j]]
        constr += [Ta <= Ta_max, Ta >= Ta_min, Rh <= Rh_max, Rh >= Rh_min]
        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constr)
        prob.solve(solver=cp.CPLEX, verbose=False)
        Ta_star[j] = Ta[:,0].value
        Rh_star[j] = Rh[:,0].value
        # q_star[j+1] = fun1(Ta_star[j],Rh_star[j],q_star[j])
        init_q1 = [q_star[j]]
        sol_q1 = solve_ivp(fun1,t_span,init_q1,method='RK45',t_eval=t_eval,args=[Ta_star[j],Rh_star[j]])
        q_star[j+1] = sol_q1.y[0,1]
        print("j=",j)
        print("q_star(j+1):",q_star[j+1])



# Plot results.
fig1 = plt.figure(figsize=(6,4))
fig3 = plt.figure(figsize=(6,4))
fig4 = plt.figure(figsize=(6,4))
ax1 = fig1.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

ax1.plot(range(tf),q_star[0:tf])
ax1.set_ylabel("quality 1$[g]$",fontsize=12)
ax1.set_xlabel("$k[days]$",fontsize=12)

ax3.step(range(tf), Ta_star[0:tf], where='post', label="$T_{a}(k)$", marker="o")
ax3.plot(range(tf), T0[0,0:tf], label="$T_{aout}(k)$", linestyle="dashed")
ax3.set_ylabel("$T_a[K]$",fontsize=12)
ax3.set_xlabel("$k[days]$",fontsize=12)
ax3.legend(loc='best')

ax4.step(range(tf), Rh_star[0:tf], where='post', label="$R_{h}(k)$", marker="o")
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


"""
f = plt.figure()

# Plot (u_t)_1.
ax = f.add_subplot(311)
plt.step(range(time),Ta_star, where='post',label="$T_{a}(k)$", marker="o")
plt.plot(range(time),T0[0,0:time], label="$T_{a0}(k)$", linestyle="dashed")
plt.ylabel(r"$T_a$", fontsize=16)
plt.yticks(np.linspace(Ta_min, Ta_max, 3))
plt.xticks(np.linspace(0, time, 5))
plt.legend()

# Plot (u_t)_2.
plt.subplot(3,1,2)
plt.plot(range(time),Rh_star, label="$R_{h}(k)$", marker="o")
plt.plot(range(time),Rh0[0,0:time], label="$R_{h0}(k)$", linestyle="dashed")
plt.ylabel(r"$R_h$", fontsize=16)
plt.yticks(np.linspace(Rh_min, Rh_max, 3))
plt.xticks(np.linspace(0, time, 5))
plt.legend()

# Plot (x_t)_1.
plt.subplot(3,1,3)
x1 = q_star
plt.plot(range(time+1),x1)
plt.ylabel(r"$quality$", fontsize=16)
plt.yticks([1092, (1090+1105)/2, 1103])
plt.ylim([1092, 1103])
plt.xticks(np.linspace(0, time, 5))
plt.xlabel(r"$t$", fontsize=16)

plt.tight_layout()
plt.show()
"""