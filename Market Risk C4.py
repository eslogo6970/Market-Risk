#//////////////// 4a

# -*- coding: utf-8 -*-
#importing libraries 
import scipy.stats as sp 
import math as mt

N=20 #realized number of exceptions
p=0.05 #forecasted percentage of exceptions
T=250 #number of trading days during the périod
NFor = p*T # forecasted number of exceptions 
SDTest = (p*(1-p)*T)**0.5# standard deviation of the test

z=(N-NFor)/SDTest

#Compute the values of zalpha
z90=sp.norm.ppf(0.90)
z95=sp.norm.ppf(0.95)
z99=sp.norm.ppf(0.99)

print('One sided Z statistics at 90% confidence level', z90)
print('One sided Z statistics at 95% confidence level', z95)
print('One sided Z statistics at 99% confidence level', z99)

print('The value taken by the test statistics is', z)

interNmax=NFor+z99*SDTest
Nmax=mt.floor(interNmax)-1

print('The maximum number of realized losses to be statistically not significant at 99% is', Nmax)



#//////////////// 4b

# -*- coding: utf-8 -*-

#importing libraries 
import scipy.stats as sp 
p1=1-sp.binom.cdf(3,250,0.01)
print('The type 1 error is', p1)





#/////////////// 4c

# -*- coding: utf-8 -*-
#importing libraries 
import scipy.stats as sp 

p2=sp.binom.cdf(3,250,0.03)
print('The type 2 error is', p2)




#////////////// 4d

# -*- coding: utf-8 -*-
#importing libraries 
import scipy.stats as sp 
import math


#compute the cut-off value
cutOff95=sp.chi2.ppf(q=0.95,df=1)
cutOff99=sp.chi2.ppf(q=0.99,df=1)

print('The cut-off value at 95% for a Xhi2 with one degree of freedom is',cutOff95)
print('The cut-off value at 99% for a Xhi2 with one degree of freedom is',cutOff99)

#LR test for JPMorgan
N=20
alpha=0.95
p=1-alpha
T=250
LRNum=(N/T)**N*(1-N/T)**(T-N)
LRDen=p**N*(1-p)**(T-N)
LR1=2*math.log(LRNum/LRDen)

#print('The cut-off statistics at 95% is',cutOff95)
#print('The cut-off statistics at 99% is',cutOff99)
print('The log likehood ratio is',LR1)





#///////////// 4e

# -*- coding: utf-8 -*-

#importing libraries 
import numpy as np
import scipy.stats as sp 
import pandas as pd 

#compute the cut-off value
cutOff95=sp.chi2.ppf(q=0.95,df=1)
#No rejection region
size=100
mV = np.arange(1,size+1)
alpha=0.95
p=1-alpha
T=252
LRNum=(mV/T)**mV*(1-mV/T)**(T-mV)
LRDen=p**mV*(1-p)**(T-mV)
LR1=2*np.log(LRNum/LRDen)
LR1Fr=pd.DataFrame(data=LR1,columns=['Result'])
NoRejReg1 = LR1Fr.index[LR1Fr['Result'] <= cutOff95]

#No rejection region
size=100
mV = np.arange(1,size+1)
alpha=0.95
p=1-alpha
T=510
LRNum=(mV/T)**mV*(1-mV/T)**(T-mV)
LRDen=p**mV*(1-p)**(T-mV)
LR2=2*np.log(LRNum/LRDen)
LR2Fr=pd.DataFrame(data=LR2,columns=['Result'])
NoRejReg2 = LR2Fr.index[LR2Fr['Result'] <= cutOff95]

#No rejection region
size=100
mV = np.arange(1,size+1)
alpha=0.95
p=1-alpha
T=1000
LRNum=(mV/T)**mV*(1-mV/T)**(T-mV)
LRDen=p**mV*(1-p)**(T-mV)
LR3=2*np.log(LRNum/LRDen)
LR3Fr=pd.DataFrame(data=LR3,columns=['Result'])
NoRejReg3 = LR3Fr.index[LR3Fr['Result'] <= cutOff95]