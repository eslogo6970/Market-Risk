#//////////////////////////////////////////////
#///////////// Extreme value Theory////////////
#//////////////////////////////////////////////


# Distribution maximum

# -*- coding: utf-8 -*-

#importing libraries 
import pandas as pd 
import numpy as np
import scipy.interpolate as inter


#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold
LP=data['L'].dropna() #supress the NaN of the L&P

#LP=LP.tail(1500)

#Compute the inverse of the empirical CDF, i.e. the quantile function
p = np.linspace(0,1,len(LP))                  
LossSorted = sorted(LP)
ppF = inter.interp1d(p,LossSorted)

n=len(LP)

#Compute the samples
N=10**4 #size of the samples
SampleU=np.random.uniform(0,1,size=N) #ff

SampleU=SampleU**(1/n)

#Compute the distribution of the Maximum
SampleMax=ppF(SampleU)


#/////////////

#3.1a

# -*- coding: utf-8 -*-

#importing libraries 
import pandas as pd 
import numpy as np
from scipy.stats import norm

#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1)
L=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1)

mL=L.mean()
sL=L.std()
p=0.95
xsi1=0.2
q1=mL-sL/xsi1*(1-(-np.log(p))**(-xsi1))
print('The 95% quantile of the maximum of the loss for Xsi=',xsi1,'is',q1)

xsi2=0.3
q2=mL-sL/xsi2*(1-(-np.log(p))**(-xsi2))
print('The 95% quantile of the maximum of the loss for Xsi=',xsi2,'is',q2)

q3=mL-sL*np.log((-np.log(p)))
print('The 95% quantile of the maximum of the loss for Xsi=0 is',q3)

z95=norm.ppf(0.95)

VaR95=mL+sL*z95

print('The 95% parametric VaR assume gaussianity is',VaR95)


#/////////////

#3.1b

# -*- coding: utf-8 -*-

#importing libraries 
import scipy.optimize as opt
import pandas as pd 
import numpy as np
import scipy.interpolate as inter
import math as mt


#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold
LP=data['L'].dropna() #supress the NaN of the L&P

LP=LP.tail(1500)
print(LP) #print the L&P on the screen to chek and compare with Excel

#Compute the inverse of the empirical CDF, i.e. the quantile function
p = np.linspace(0,1,len(LP))                  
LossSorted = sorted(LP)
ppF = inter.interp1d(p,LossSorted)

#Compute the samples
N=10**3 #size of the samples
SampleU=np.random.uniform(0,1,size=N) #ff

SampleU=SampleU**(1/1500)

#Compute the distribution of the Maximum
SampleMax=ppF(SampleU)


def func(x):
    dumb=0
    for i in range(1, N+1):
        dumb=dumb+mt.log(x[1])+(x[2]+1)/x[2]*mt.log(1+x[2]*(SampleMax[i-1]-x[0])/x[1])+(1+x[2]*(SampleMax[i-1]-x[0])/x[1])**(-1/x[2])
    return dumb


muInf=np.nanmin(SampleU)
xInf=np.array([0,0,0])
xSup=np.array([np.inf,np.inf,0.35])

x0=[10,10,0.1]

bounds=opt.Bounds(lb=xInf, ub=xSup)

res=opt.minimize(func, x0, bounds=bounds)

muGEV=res.x[0]
sigGEV=res.x[1]
XsiGEV=res.x[2]

p90=0.9
q90=muGEV-sigGEV/XsiGEV*(1-(-mt.log(p90))**(-XsiGEV))
p95=0.95
q95=muGEV-sigGEV/XsiGEV*(1-(-mt.log(p95))**(-XsiGEV))
p99=0.99
q99=muGEV-sigGEV/XsiGEV*(1-(-mt.log(p99))**(-XsiGEV))







#/////////////

#3.1c

# -*- coding: utf-8 -*-

#importing libraries 
import scipy.optimize as opt
import pandas as pd 
import numpy as np
import scipy.interpolate as inter
import math as mt


#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold
LP=data['L'].dropna() #supress the NaN of the L&P

LP=LP.tail(1500)
print(LP) #print the L&P on the screen to chek and compare with Excel

#Compute the inverse of the empirical CDF, i.e. the quantile function
p = np.linspace(0,1,len(LP))                  
LossSorted = sorted(LP)
ppF = inter.interp1d(p,LossSorted)

n=len(LP)

#Compute the samples
N=10**3 #size of the samples
SampleU=np.random.uniform(0,1,size=N) #ff

SampleU=SampleU**(1/n)

#Compute the distribution of the Maximum
SampleMax=ppF(SampleU)

MaxSorted=sorted(SampleMax)

#size=len(LP)
mV = np.arange(1,N+1)

yData=np.log(-np.log(mV/(1+N)))

def func(x, mu, sig, xsi):

    return -1/xsi * np.log(1+xsi*(x-mu)/sig)

muInf=np.nanmin(SampleU)

xInf=np.array([0,0,0])

xSup=np.array([np.inf,np.inf,0.35])

popt,pcov = opt.curve_fit(func, MaxSorted, yData,bounds=(xInf, xSup))
perr = np.sqrt(np.diag(pcov))

#Computation of various quantile of the maximum of the loss
muGEV=popt[0]
sigGEV=popt[1]
Xsi=popt[2]

p90=0.9
q90=muGEV-sigGEV/Xsi*(1-(-mt.log(p90))**(-Xsi))
p95=0.95
q95=muGEV-sigGEV/Xsi*(1-(-mt.log(p95))**(-Xsi))
p99=0.99
q99=muGEV-sigGEV/Xsi*(1-(-mt.log(p99))**(-Xsi))










#/////////////

#3.1d

# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame


#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold

#Arange data in descending order
data = data.sort_values(by=['L'], ascending=False)
data= data.reset_index(drop=True)


k=10
Hill10=np.log(data['L'][0])
for i in range(1,k):
    Hill10=Hill10+np.log(data['L'][i])

Hill10=1/k*Hill10-np.log(data['L'][k+1])

k=11
Hill11=np.log(data['L'][0])
for i in range(1,k):
    Hill11=Hill11+np.log(data['L'][i])

Hill11=1/k*Hill11-np.log(data['L'][k+1])

k=12
Hill12=np.log(data['L'][0])
for i in range(1,k):
    Hill12=Hill12+np.log(data['L'][i])

Hill12=1/k*Hill12-np.log(data['L'][k+1])

k=13
Hill13=np.log(data['L'][0])
for i in range(1,k):
    Hill13=Hill13+np.log(data['L'][i])

Hill13=1/k*Hill13-np.log(data['L'][k+1])

k=14
Hill14=np.log(data['L'][0])
for i in range(1,k):
    Hill14=Hill14+np.log(data['L'][i])

Hill14=1/k*Hill14-np.log(data['L'][k+1])

k=15
Hill15=np.log(data['L'][0])
for i in range(1,k):
    Hill15=Hill15+np.log(data['L'][i])

Hill15=1/k*Hill15-np.log(data['L'][k+1])

HillVec=np.array([Hill10, Hill11, Hill12,Hill13, Hill14, Hill15])

Hill=HillVec.mean()

print('The Hill estimate of Xsi is',Hill)


k=2
Hill2=np.log(data['L'][0])
for i in range(1,k):
    Hill2=Hill2+np.log(data['L'][i])

Hill2=1/k*Hill2-np.log(data['L'][k+1])








#/////////////

#3.1e

# -*- coding: utf-8 -*-


#importing libraries 
import scipy.special as spa
import pandas as pd 
import numpy as np
import scipy.interpolate as inter
import math as mt


#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold
LP=data['L'].dropna() #supress the NaN of the L&P
LP=LP.tail(1500)
print(LP) #print the L&P on the screen to chek and compare with Excel

n=len(LP)

#Compute the inverse of the empirical CDF, i.e. the quantile function
p = np.linspace(0,1,len(LP))                  
LossSorted = sorted(LP)
ppF = inter.interp1d(p,LossSorted)

#Compute the samples
N=10**3 #size of the samples
SampleU=np.random.uniform(0,1,size=N) #ff

SampleU=SampleU**(1/n)

#Compute the distribution of the VaR
SampleLoss=ppF(SampleU)
SampleLoss2=SampleLoss**2

mu1=SampleLoss.mean()
mu2=SampleLoss2.mean()

sig2=mu2-mu1**2

#Computation of the coefficients
Xsi=0.35
A1=(spa.gamma(1-Xsi)-1)/Xsi
A2=(spa.gamma(1-2*Xsi)-spa.gamma(1-Xsi)**2)/Xsi**2
sig2GEV=sig2/A2
sigGEV=sig2GEV**0.5
muGEV=mu1-sigGEV*A1

#Computation of various quantile of the maximum of the loss
p90=0.9
q90=muGEV-sigGEV/Xsi*(1-(-mt.log(p90))**(-Xsi))
p95=0.95
q95=muGEV-sigGEV/Xsi*(1-(-mt.log(p95))**(-Xsi))
p99=0.99
q99=muGEV-sigGEV/Xsi*(1-(-mt.log(p99))**(-Xsi))



#///////////

## 3.2a

#importing libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#importe file
data=pd.read_excel('GoldandOilData.xlsx',sheet_name='Price') #read the data into a panda DataFrame

#Compute L&P
data['L']=data['Gold'].diff(periods=-1)+10*data['Oil'].diff(periods=-1) #compute the L&P of gold
LP=data['L'].dropna() #supress the NaN of the L&P


def meanExcessFunc(u):
    num=LP-u
    num=num[num>0]
    den=len(num)
    num=np.sum(num)
    return num/den

N =1000

u=np.linspace(0,40,N)

meanExcessVec=np.zeros(N)

for i in range(0, N):
    meanExcessVec[i]=meanExcessFunc(u[i])
#v=meanExcess(u)

plt.scatter(u,meanExcessVec)

uThreshold1=25
uThreshold2=35
u=u[u>uThreshold1]
u=u[u<uThreshold2]
N=len(u)
meanExcessVec=np.zeros(N)

for i in range(0, N):
    meanExcessVec[i]=meanExcessFunc(u[i])


u = sm.add_constant(u)
reg=sm.OLS(meanExcessVec,u)
res = reg.fit()
param=res.params
xsi=param[1]/(1+param[1])
beta=param[0]*(1-xsi)

n=len(LP)
nu=len(LP[LP>uThreshold2])

alpha=0.95
alphaVaR= uThreshold2+beta/xsi*((n/nu*(1-alpha))**(-xsi)-1)

