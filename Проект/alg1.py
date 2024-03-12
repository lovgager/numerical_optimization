import numpy as np
import math as m
from matplotlib import pyplot as plt

def Ta(Lx, m_x, Ly, m_y, Lxy):
    return max(m.sqrt(Lx/m_x), m.sqrt(Ly/m_y), Lxy/m.sqrt(m_x*m_y))

def Tb(Lx, Ly, Lxy, m_xy, m_x):
    return max(m.sqrt(Lx*Ly)/m_xy, Lxy/m_xy, m.sqrt(Lx/m_x), Lxy**2/m_xy**2)

def Tc(Lx, Ly, Lxy, m_yx, m_y):
    return max(m.sqrt(Lx*Ly)/m_yx, Lxy/m_yx, m.sqrt(Ly/m_y), Lxy**2/m_yx**2)

def Td(Lx, Ly, Lxy, m_yx, m_xy):
    return max(m.sqrt(Lx*Ly)*Lxy/(m_xy*m_yx), Lxy**2/m_yx**2, Lxy**2/m_xy**2)


def ro_a(delt, s_x, s_y, L_x, L_y, L_xy, m_x, m_y):
    #return 0
    return 1/(max(4*(m_x + L_x*s_x)/m_x, 2/s_x, 4*(m_y + L_y*s_y)/m_y, 2/s_y, 4*L_xy/(m_x*delt), 4*L_xy*delt/m_y))

def ro_b(delt, s_x, s_y, L_x, L_y, L_xy, m_x, m_y, m_xy):
    #return 0
    if m_xy == 0:
        return 0
    return 1/(max(4*(m_x + L_x*s_x)/m_x, 2/s_x, 8*L_x*(m_y + L_y*s_y)/(m_xy**2), 2/s_y, 2*L_xy**2/(m_xy**2), 8*L_x*L_xy*delt/(m_xy**2), 4*L_xy/(m_x*delt)))

def ro_c(delt, s_x, s_y, L_x, L_y, L_xy, m_x, m_y, m_yx):
    #return 0
    if m_yx == 0:
        return 0
    return 1/(max(4*(m_y + L_y*s_y)/m_y, 2/s_y, 8*L_y*(m_x + L_x*s_x)/(m_yx**2), 2/s_x, 2*L_xy**2/(m_yx**2), 8*L_y*L_xy/(m_yx**2*delt), 4*L_xy*delt/m_x))

def ro_d(delt, s_x, s_y, L_x, L_y, L_xy, m_x, m_y, m_yx, m_xy):
    #return 0
    if m_xy == 0:
        return 0
    if m_yx == 0:
        return 0
    return 1/(max(8*L_y*(m_x + L_x*s_x)/(m_yx**2), 2/s_x, 8*L_x*(m_y + L_y*s_y)/(m_xy**2), 2/s_y, 8*L_y*L_xy/(m_yx**2*delt), 8*L_x*L_xy*delt/(m_xy**2), 2*L_xy**2/(m_yx**2), 2*L_xy**2/(m_xy**2)))

# def grad_f(x, mu_x):
#     return mu_x*x

# def grad_g(y, mu_y):
#     return mu_y*y

def grad_f(x, B):
    return B.T@B@x

def grad_g(y, B):
    return B.T@B@y

np.random.seed(1)
size_r = 5
size_c = 5
A = np.random.random((size_r, size_c)) # случайная матрица, обычно близка к вырожденной

#np.fill_diagonal(A, np.sum(np.abs(A), axis=1)) # это гарантированно невырожденная матрица
#A = np.ones((size_r, size_c)) # это гарантированно вырожденная матрица

eig = np.real(np.linalg.eig(A@A.T)[0])
eg1 = eig[eig > 0.]
mu_xy = mu_yx = m.sqrt(min(eg1))
mu_xy = mu_yx= 0
#mu_x = L_x = 2
#mu_y = L_y = 2

B = np.diag(np.arange(1, size_c+1))
mu_x = mu_y = np.linalg.norm(B.T@B, -2)
L_x = L_y = np.linalg.norm(B.T@B,  2)

delta = m.sqrt(mu_y/mu_x)
L_xy = m.sqrt(max(eg1))

sigma_x = m.sqrt(mu_x/(2*L_x))
sigma_y = m.sqrt(mu_y/(2*L_y))
tau_x = 1/(1/sigma_x + 0.5)
tau_y = 1/(1/sigma_y + 0.5)
eta_x = min(1/(4*(mu_x + L_x*sigma_x)), delta/(4*L_xy))
eta_y = min(1/(4*(mu_y + L_y*sigma_y)), 1/(4*delta*L_xy))
a_x = mu_x
a_y = mu_y
b_x = min(1/(2*L_y), 1/(2*eta_x*L_xy**2))
b_y = min(1/(2*L_x), 1/(2*eta_x*L_xy**2))

theta = 1 - max(ro_a(delta, sigma_x, sigma_y, L_x, L_y, L_xy, mu_x, mu_y), ro_b(delta, sigma_x, sigma_y, L_x, L_y, L_xy, mu_x, mu_y, mu_xy), ro_c(delta, sigma_x, sigma_y, L_x, L_y, L_xy, mu_x, mu_y, mu_yx), ro_d(delta, sigma_x, sigma_y, L_x, L_y, L_xy, mu_x, mu_y, mu_yx, mu_xy))

kk = []
oo = []
for i in range(-1, -10, -1):
    eps = 10**(i)
    k = 0
    x = np.random.random((size_c, 1))
    y = np.random.random((size_r, 1))
    x, y = A.T@y, A@x
    PSY0 = (1/eta_x + 2/sigma_x)*(np.linalg.norm(x))**2 + (1/eta_y + 2/sigma_y)*(np.linalg.norm(y))**2
    C = PSY0*max(4*eta_x/3, eta_y)
    y0 = y
    #O1 = 1/(1 - theta)
    O1 = 4+4*Ta(L_x, mu_x, L_y, mu_y, L_xy) 
    erx = np.linalg.norm(x)**2
    ery = np.linalg.norm(y)**2
    xf = x
    yf = y
    while(max(erx, ery) > eps):
        ym = y + theta*(y - y0)
        xg = tau_x*x + (1 - tau_x)*xf
        yg = tau_y*y + (1 - tau_y)*yf
        x0 = x
        y0 = y
        x = x0 + eta_x*a_x*(xg - x) - eta_x*b_x*(A.T@(A@x - grad_g(yg, B))) \
            - eta_x*(grad_f(xg, B) + A.T@ym)
        y = y0 + eta_y*a_y*(yg - y) - eta_y*b_y*(A@(A.T@y - grad_f(xg, B))) \
            - eta_y*(grad_g(yg, B) - A@x)
        xf = xg + sigma_x*(x - x0)
        yf = yg + sigma_y*(y - y0)
        erx = np.linalg.norm(x)**2
        ery = np.linalg.norm(y)**2
        k += 1
        o = m.ceil(O1*m.log(C/eps)) 
    kk.append(k)
    oo.append(o)
    
print(kk)
print(oo)

t = np.arange(-9, 0, 1)
plt.grid(True)
#plt.plot(t, oo[::-1], label='теоретическая оценка 1')
plt.plot(t, O1*(m.log(C) - t*m.log(10)), label='теоретическая оценка 1')
plt.plot(t, kk[::-1], label='алгоритм 1')
plt.xlabel('log10(eps)')
plt.ylabel('число итераций')
plt.title(f'случайная матрица {size_r}x{size_c},  mu_x={mu_x},  mu_y={mu_y}')
plt.legend()
plt.show()