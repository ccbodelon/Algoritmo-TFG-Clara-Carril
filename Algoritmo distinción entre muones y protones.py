#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot as ur
import pandas
import matplotlib.pyplot as plt
import numpy as np


def get_renames(df, prefix, sufix):
    renames = {}
    for key in df.keys():
        renames[key] = key.replace(prefix, '').replace(sufix, '')
    return renames


root = 'TFG Clara'
signal = ur.open('monitoring-jpsimumu-central.root:Hlt1_MCTruthMuonID/monitor_tree').arrays(library='pd')
background = ur.open('monitoring-bigb.root:Hlt1_MCTruthMuonID/monitor_tree').arrays(library='pd')


renames = get_renames(signal, 'Hlt1_MCTruthMuonID__tupling_', '_t')
print(renames)
data = map(lambda x: x.rename(columns=renames), [signal, background])
# So nos quedamos con aqueles eventos que pasan IsMuon (muon hits asociadas a traza)
data = map(lambda x: x.query('ismuon==True'), data)
signal, background = data


# In[2]:


tx_scifi=signal['tx_scifi']
ty_scifi=signal['ty_scifi']

x0=signal['x0']
x1=signal['x1']
x2=signal['x2']
x3=signal['x3']
y0=signal['y0']
y1=signal['y1']
y2=signal['y2']
y3=signal['y3']
z0=signal['z0']
z1=signal['z1']
z2=signal['z2']
z3=signal['z3']

ax_scifi=signal['ax_scifi']
ay_scifi=signal['ay_scifi']

dx0=signal['dx0']
dx1=signal['dx1']
dx2=signal['dx2']
dx3=signal['dx3']
dy0=signal['dy0']
dy1=signal['dy1']
dy2=signal['dy2']
dy3=signal['dy3']

pt=signal['pt']


# In[3]:


def delta(t_scifi,z,a_scifi,x):
    term=(t_scifi*z)+a_scifi-x
    return np.array(term)

def vect(ini):
    return np.array(ini)

def funchi(x0,x1,x2,x3,z0,z1,z2,z3,delta0,delta1,delta2,delta3,d0,d1,d2,d3,zi,deltazi,beta,p):
    x=np.array([x0,x1,x2,x3])
    z=np.array([z0,z1,z2,z3])
    delta=np.array([[delta0],[delta1],[delta2],[delta3]])
    pad=np.array([d0,d1,d2,d3])
    vect=[]
    plus=0
    for i in range(len(x)):
        if x[i]==-99999:
            vect.append(i-plus)
            plus+=1
    for i in range(len(vect)):
        x=np.delete(x,vect[i]) 
        z=np.delete(z,vect[i])
        delta=np.delete(delta,vect[i],axis=0)
        pad=np.delete(pad,vect[i])
    sigmaMS=[]
    for i in range(len(zi)):
        term=(13.6*np.sqrt(deltazi[i]))/(beta*p[i])
        sigmaMS.append(term)
    n=len(x)
    V=np.zeros([n,n])  
    for i in range(n): #cogemos una fila
        for j in range(n): #cogemos una columna
            suma=0
            for k in range(len(zi)):
                if zi[k]<z[i] and zi[k]<z[j]:
                    term=(z[i]-zi[k])*(z[j]-zi[k])*(sigmaMS[k]**2)
                else:
                    term=0
                suma+=term
            V[i,j]=suma
            if j==i:
                V[i,i]+=((pad[i])/np.sqrt(12))**2
    mult=np.dot(np.transpose(delta),np.linalg.inv(V))
    chicuadradocorr=np.dot(mult,delta)  
    chicuadradocorrnorm=chicuadradocorr/n
    return chicuadradocorrnorm
def funchisinms(x0,x1,x2,x3,z0,z1,z2,z3,delta0,delta1,delta2,delta3,d0,d1,d2,d3):
    x=np.array([x0,x1,x2,x3])
    z=np.array([z0,z1,z2,z3])
    delta=np.array([[delta0],[delta1],[delta2],[delta3]])
    pad=np.array([d0,d1,d2,d3])
    vect=[]
    plus=0
    for i in range(len(x)):
        if x[i]==-99999:
            vect.append(i-plus)
            plus+=1
    for i in range(len(vect)):
        x=np.delete(x,vect[i]) 
        z=np.delete(z,vect[i])
        delta=np.delete(delta,vect[i],axis=0)
        pad=np.delete(pad,vect[i])
    n=len(x)
    V=np.zeros([n,n])  
    for i in range(n): #cogemos una fila
        for j in range(n): #cogemos una columna
            if j==i:
                V[i,i]=((pad[i])/np.sqrt(12))**2
    mult=np.dot(np.transpose(delta),np.linalg.inv(V))
    chicuadradocorr=np.dot(mult,delta)  
    chicuadradocorrnorm=chicuadradocorr/n
    return chicuadradocorrnorm


# In[4]:


deltax0=delta(tx_scifi,z0,ax_scifi,x0)
deltax1=delta(tx_scifi,z1,ax_scifi,x1)
deltax2=delta(tx_scifi,z2,ax_scifi,x2)
deltax3=delta(tx_scifi,z3,ax_scifi,x3)

deltay0=delta(ty_scifi,z0,ay_scifi,y0)
deltay1=delta(ty_scifi,z1,ay_scifi,y1)
deltay2=delta(ty_scifi,z2,ay_scifi,y2)
deltay3=delta(ty_scifi,z3,ay_scifi,y3)

x0=vect(x0)
x1=vect(x1)
x2=vect(x2)
x3=vect(x3)
y0=vect(y0)
y1=vect(y1)
y2=vect(y2)
y3=vect(y3)

z0=vect(z0)
z1=vect(z1)
z2=vect(z2)
z3=vect(z3)

dx0=vect(dx0)
dx1=vect(dx1)
dx2=vect(dx2)
dx3=vect(dx3)
dy0=vect(dy0)
dy1=vect(dy1)
dy2=vect(dy2)
dy3=vect(dy3)

pt=vect(pt)


# In[5]:


qop=signal['qop']
qop=np.array(qop)
p=[]
p=abs(1/qop)


# In[6]:


zi=[12.8,14.3,15.8,17.1,18.3]
deltazi=[25,53,47.5,47.5,47.5] #esto en realidad es el delta zi entre X0
beta=1.0

chi=[]

for m in range(len(tx_scifi)):
    chicuadradocorrnormx=funchi(x0[m],x1[m],x2[m],x3[m],z0[m],z1[m],z2[m],z3[m],deltax0[m],deltax1[m],deltax2[m],deltax3[m],dx0[m],dx1[m],dx2[m],dx3[m],zi,deltazi,beta,p)
    chicuadradocorrnormy=funchi(y0[m],y1[m],y2[m],y3[m],z0[m],z1[m],z2[m],z3[m],deltay0[m],deltay1[m],deltay2[m],deltay3[m],dy0[m],dy1[m],dy2[m],dy3[m],zi,deltazi,beta,p)
    chicuadradonorm=chicuadradocorrnormx+chicuadradocorrnormy
    chicuadradonorm=chicuadradonorm[0,0]
    chi.append(chicuadradonorm)
    
chisinms=[]

for m in range(len(tx_scifi)):
    chicuadradocorrnormx=funchisinms(x0[m],x1[m],x2[m],x3[m],z0[m],z1[m],z2[m],z3[m],deltax0[m],deltax1[m],deltax2[m],deltax3[m],dx0[m],dx1[m],dx2[m],dx3[m])
    chicuadradocorrnormy=funchisinms(y0[m],y1[m],y2[m],y3[m],z0[m],z1[m],z2[m],z3[m],deltay0[m],deltay1[m],deltay2[m],deltay3[m],dy0[m],dy1[m],dy2[m],dy3[m])
    chicuadradonorm=chicuadradocorrnormx+chicuadradocorrnormy
    chicuadradonorm=chicuadradonorm[0,0]
    chisinms.append(chicuadradonorm)


# In[7]:


tx_scifi=background['tx_scifi']
ty_scifi=background['ty_scifi']

ax_scifi=background['ax_scifi']
ay_scifi=background['ay_scifi']

x0=background['x0']
x1=background['x1']
x2=background['x2']
x3=background['x3']
z0=background['z0']
z1=background['z1']
z2=background['z2']
z3=background['z3']
y0=background['y0']
y1=background['y1']
y2=background['y2']
y3=background['y3']
dx0=background['dx0']
dx1=background['dx1']
dx2=background['dx2']
dx3=background['dx3']
dy0=background['dy0']
dy1=background['dy1']
dy2=background['dy2']
dy3=background['dy3']


# In[8]:


deltax0=delta(tx_scifi,z0,ax_scifi,x0)
deltax1=delta(tx_scifi,z1,ax_scifi,x1)
deltax2=delta(tx_scifi,z2,ax_scifi,x2)
deltax3=delta(tx_scifi,z3,ax_scifi,x3)

deltay0=delta(ty_scifi,z0,ay_scifi,y0)
deltay1=delta(ty_scifi,z1,ay_scifi,y1)
deltay2=delta(ty_scifi,z2,ay_scifi,y2)
deltay3=delta(ty_scifi,z3,ay_scifi,y3)

qop=background['qop']
qop=np.array(qop)
p=[]
p=abs(1/qop)

x0=vect(x0)
x1=vect(x1)
x2=vect(x2)
x3=vect(x3)
y0=vect(y0)
y1=vect(y1)
y2=vect(y2)
y3=vect(y3)
z0=vect(z0)
z1=vect(z1)
z2=vect(z2)
z3=vect(z3)
dx0=vect(dx0)
dx1=vect(dx1)
dx2=vect(dx2)
dx3=vect(dx3)
dy0=vect(dy0)
dy1=vect(dy1)
dy2=vect(dy2)
dy3=vect(dy3)


# In[9]:


chib=[]

for m in range(len(tx_scifi)):
    chicuadradocorrnormx=funchi(x0[m],x1[m],x2[m],x3[m],z0[m],z1[m],z2[m],z3[m],deltax0[m],deltax1[m],deltax2[m],deltax3[m],dx0[m],dx1[m],dx2[m],dx3[m],zi,deltazi,beta,p)
    chicuadradocorrnormy=funchi(y0[m],y1[m],y2[m],y3[m],z0[m],z1[m],z2[m],z3[m],deltay0[m],deltay1[m],deltay2[m],deltay3[m],dy0[m],dy1[m],dy2[m],dy3[m],zi,deltazi,beta,p)
    chicuadradonorm=chicuadradocorrnormx+chicuadradocorrnormy
    chicuadradonorm=chicuadradonorm[0,0]
    chib.append(chicuadradonorm)
    
chibsinms=[]

for m in range(len(tx_scifi)):
    chicuadradocorrnormx=funchisinms(x0[m],x1[m],x2[m],x3[m],z0[m],z1[m],z2[m],z3[m],deltax0[m],deltax1[m],deltax2[m],deltax3[m],dx0[m],dx1[m],dx2[m],dx3[m])
    chicuadradocorrnormy=funchisinms(y0[m],y1[m],y2[m],y3[m],z0[m],z1[m],z2[m],z3[m],deltay0[m],deltay1[m],deltay2[m],deltay3[m],dy0[m],dy1[m],dy2[m],dy3[m])
    chicuadradonorm=chicuadradocorrnormx+chicuadradocorrnormy
    chicuadradonorm=chicuadradonorm[0,0]
    chibsinms.append(chicuadradonorm)    


# In[10]:


plt.hist(np.log10(chi),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[11]:


plt.hist(np.log10(chisinms),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[12]:


from sklearn.metrics import roc_curve, auc

y_true=np.concatenate([np.ones(len(chi)),np.zeros(len(chib))])
y_scores=np.concatenate([chi,chib])

y_true_sinms=np.concatenate([np.ones(len(chisinms)),np.zeros(len(chibsinms))])
y_scores_sinms=np.concatenate([chisinms,chibsinms])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.title('ROC curve')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)

