import numpy as np
import math as m
from matplotlib import pyplot as plt

def ro_a(delt, L_x, L_y, L_xy, m_x, m_y):
    #return 0
    return 1/(max(8*L_x/m_x, 8*L_y/m_y, 4*L_xy/(delt*m_x), 4*L_xy*delt/m_y))

def ro_b(delt, L_x, L_y, L_xy, m_x, m_y, m_xy):
    #return 0
    if m_xy == 0:
        return 0
    return 1/(max(8*L_x/m_x, 512*L_x*L_y/(m_xy**2), 4*L_xy/(delt*m_x), 256*L_x*L_xy*delt/(m_xy**2), 256*L_y*L_xy/(delt*m_xy**2), 128*L_xy**2/(m_xy**2)))

def ro_c(delt, L_x, L_y, L_xy, m_x, m_y, m_yx):
    #return 0
    if m_yx == 0:
        return 0
    return 1/(max(8*L_y/m_y, 512*L_x*L_y/(m_yx**2), 4*L_xy*delt/m_y, 256*L_x*L_xy*delt/(m_yx**2), 256*L_y*L_xy/(delt*m_yx**2), 128*L_xy**2/(m_yx**2)))

def ro_d(delt, L_x, L_y, L_xy, m_x, m_y, m_yx, m_xy):
    #return 0
    if m_xy == 0 or m_yx == 0:
        return 0
    return 1/(max(512*L_x*L_y/min(m_xy**2,m_yx**2), 256*L_x*L_xy*delt/min(m_xy**2,m_yx**2), 256*L_y*L_xy/(min(m_xy**2,m_yx**2)*delt), 128*L_xy**2/min(m_xy**2,m_yx**2)))

# def grad_Fx(x, y, A, mu_x):
#     return mu_x*x + A.T@y

# def grad_Fy(x, y, A, mu_y):
#     return A@x - mu_y*y

def grad_Fx(x, y, A, mu_x, B):
    return B.T@B@x + A.T@y

def grad_Fy(x, y, A, mu_y, B):
    return A@x - B.T@B@y


np.random.seed(1)
size_r = 5
size_c = 5
A = np.random.random((size_r, size_c))

np.fill_diagonal(A, np.sum(np.abs(A), axis=1))
A = np.ones((size_r, size_c))

eig1 = np.real(np.linalg.eig(A@A.T)[0])
eig2 = np.real(np.linalg.eig(A.T@A)[0])

eig1 = eig1[eig1 >= 0.]
eig2 = eig2[eig2 >= 0.]
mu_xy, mu_yx = m.sqrt(min(eig1)), m.sqrt(min(eig2))
mu_xy = mu_yx = 0
#mu_x = L_x = 2
#mu_y = L_y = 2

B = np.diag(np.arange(1, size_c+1))
mu_x = mu_y = np.linalg.norm(B.T@B, -2)
L_x = L_y = np.linalg.norm(B.T@B,  2)

delta = m.sqrt(mu_y/mu_x)
L_xy = m.sqrt(min(eig1))

if L_xy == 0:
    eta_x = 1/(8*L_x)
    eta_y = 1/(8*L_y)
else:
    eta_x = min(1/(8*L_x), delta/(4*L_xy))
    eta_y = min(1/(8*L_y), 1/(4*L_xy*delta))


theta = 1 - max(ro_a(delta, L_x, L_y, L_xy, mu_x, mu_y), ro_b(delta, L_x, L_y, L_xy, mu_x, mu_y, mu_xy), ro_c(delta, L_x, L_y, L_xy, mu_x, mu_y, mu_yx), ro_d(delta, L_x, L_y, L_xy, mu_x, mu_y, mu_yx, mu_xy))
#theta_min = max(8*L_x/mu_x, 8*L_y/mu_y, 4*L_xy/(m.sqrt(mu_x*mu_y)))

kk = []
oo = []
for i in range(-1, -10, -1):
    eps = 10**(i)
    k = 0
    x = np.random.random((size_c, 1))
    y = np.random.random((size_r, 1))
    erx = np.linalg.norm(x)**2
    ery = np.linalg.norm(y)**2
    PSY0 = (1/eta_x)*(np.linalg.norm(x))**2 + (1/eta_y)*(np.linalg.norm(y))**2
    C = PSY0*max(4*eta_x/3, eta_y)
    y_cur = y
    x_cur = x
    x_pre = x
    y_pre = y
    #O1 = 1/(1 - theta)
    O1 = max(8*L_x/mu_x, 8*L_y/mu_y, 4*L_xy/(m.sqrt(mu_x*mu_y)))
    while(max(erx, ery) > eps):
        x_next = x_cur - eta_x*grad_Fx(x_cur, y_cur, A, mu_x, B) - eta_x*theta*(A.T@(y_cur - y_pre))
        y_next = y_cur + eta_y*grad_Fy(x_next, y_cur, A, mu_y, B)
        x_pre = x_cur
        y_pre = y_cur
        x_cur = x_next
        y_cur = y_next
        erx = (np.linalg.norm(x_next))**2
        ery = (np.linalg.norm(y_next))**2
        k += 1
        o = m.ceil(O1*m.log(C/eps))
    kk.append(k)
    oo.append(o)
print(kk)
print(oo)

t = np.arange(-9, 0, 1)
plt.grid(True)
#plt.plot(t, oo[::-1], label='теоретическая оценка 2') 
plt.plot(t, O1*(m.log(C) - t*m.log(10)), label='теоретическая оценка 2')
plt.plot(t, kk[::-1], label='алгоритм 2')
plt.xlabel('log10(eps)')
plt.ylabel('число итераций')
plt.title(f'случайная матрица {size_r}x{size_c},  mu_x={mu_x},  mu_y={mu_y}')
plt.legend()
plt.show()
