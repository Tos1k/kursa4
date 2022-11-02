import math
from cmath import *
import os
import imageio
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy import Quaternion,symbols

lam = symbols("lambda0:4")
q1 = Quaternion(0.2, 1, 2, 3);
q2 = Quaternion(0.5, 1, 2, 3);
q3 = Quaternion(0.8, 1, 2, 3);

psi_0=5
alpha_2 = 10
#alpha_2 = 0
#alpha_2 = 1
c=psi_0/alpha_2

def k_1(scal): # k_1 при \nu_0(0)
    k1 = (math.log((1-scal)/(1+scal)))/2
    return k1

print("k_1",k_1(0.2))
def alpha_1(scal_2, k1):  #\alpha_1 при \nu_1,2,3 (0)
    a1 = (scal_2 - (1/(1+e**(2*k1)))+(e**(2*k1)/(1+e**(2*k1))))/e**(-math.log(1+e**(2*k1)))
    return a1

def nu_0(k1, t): # \nu_0
    nu0 = (e**(2*c*t)-e**(2*k1))/(e**(2*c*t)+e**(2*k1))
    return nu0


def nu_other(k1, t, a1):
    nuOth = a1*e**(c*(t-math.log(e**(2*c*t)+e**(2*k1))/c))+(e**(2*c*t))/((e**(2*c*t)+e**(2*k1)) - (e**(2*k1))/(e**(2*c*t)+e**(2*k1)))
    return nuOth

nu_0_l = []
nu_oth_l = []


def nu_other_list(scal, scal_2):
    for i in range (0, 5, 1):
       nu_oth_l.append( nu_other(k_1(scal), i, alpha_1(scal_2, k_1(scal))))
    return nu_oth_l


def nu_0_list(scal):
    for i in range (0, 5, 1):
       nu_0_l.append(nu_0(k_1(scal), i))
    return nu_0_l

#график для \nu_0
f1=plt.figure(1)
plt.grid('both')
#t=np.linspace(0, 5, 1)
#=[0,1,2,3,4]
t=np.arange(0, 5, 1)
#print(nu_0_list(0.2))
fig = plt.subplots()
#plt.subplot(211)
plt.grid()
#plt.ylabel('nu_0')
#plt.xlabel('t')
#plt.subplot(211)
#plt.ylabel('nu_0')
#plt.xlabel('t')
#plt.plot(t, nu_0_list(0.5), 'r', linewidth=5)
#plt.plot(t, nu_other_list(0.2,1), 'b', linewidth=5)
plt.subplot(211)
plt.ylabel('nu_2')
plt.xlabel('t')
plt.plot(t, nu_other_list(0.5, 2), 'r', linewidth=5)
nu_oth_l=[]
plt.subplot(212)
plt.ylabel('nu_3')
plt.xlabel('t')
plt.plot(t, nu_other_list(0.5, 3), 'b', linewidth=5)
plt.show()


