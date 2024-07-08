#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot as ur
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


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

def subdiv(pos,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3):
    x01=[];x11=[];x21=[];x31=[]
    y01=[];y11=[];y21=[];y31=[]
    z01=[];z11=[];z21=[];z31=[]
    deltax01=[];deltax11=[];deltax21=[];deltax31=[]
    deltay01=[];deltay11=[];deltay21=[];deltay31=[]
    dx01=[];dx11=[];dx21=[];dx31=[]
    dy01=[];dy11=[];dy21=[];dy31=[]
    pt1=[]
    for i in range(len(pos)):
        k=pos[i]
        x01.append(x0[k])
        x11.append(x1[k])
        x21.append(x2[k])
        x31.append(x3[k])
        y01.append(y0[k])
        y11.append(y1[k])
        y21.append(y2[k])
        y31.append(y3[k])
        z01.append(z0[k])
        z11.append(z1[k])
        z21.append(z2[k])
        z31.append(z3[k])
        deltax01.append(deltax0[k])
        deltax11.append(deltax1[k])
        deltax21.append(deltax2[k])
        deltax31.append(deltax3[k])
        deltay01.append(deltay0[k])
        deltay11.append(deltay1[k])
        deltay21.append(deltay2[k])
        deltay31.append(deltay3[k])
        dx01.append(dx0[k])
        dx11.append(dx1[k])
        dx21.append(dx2[k])
        dx31.append(dx3[k])
        dy01.append(dy0[k])
        dy11.append(dy1[k])
        dy21.append(dy2[k])
        dy31.append(dy3[k])
    return x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31

def clase(p,pos,pt):
    pta=[];ptb=[];ptc=[];ptd=[]
    pa=[];pb=[];pc=[];pd=[]
    posa=[];posb=[];posc=[];posd=[]
    for i in range(len(pt)):
        if 500<=pt[i]<800:
            pta.append(pt[i])
            pa.append(p[i])
            posa.append(i)
        if 800<=pt[i]<1200:
            ptb.append(pt[i])
            pb.append(p[i])
            posb.append(i)
        if 1200<=pt[i]<2000:
            ptc.append(pt[i])
            pc.append(p[i])
            posc.append(i)
        if pt[i]>=2000:
            ptd.append(pt[i])
            pd.append(p[i])
            posd.append(i)
    return pta,pa,posa,ptb,pb,posb,ptc,pc,posc,ptd,pd,posd


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

p1=[];p2=[];p3=[]
pt1=[];pt2=[];pt3=[]
pos1=[];pos2=[];pos3=[]
for i in range(len(p)):
    if 3000<=p[i]<6000:
        p1.append(p[i])
        pos1.append(i)
        pt1.append(pt[i])
    if 6000<=p[i]<10000:
        p2.append(p[i])
        pos2.append(i)
        pt2.append(pt[i])
    if p[i]>=10000:
        p3.append(p[i])
        pos3.append(i)
        pt3.append(pt[i])


# In[6]:


pt1a,p1a,pos1a,pt1b,p1b,pos1b,pt1c,p1c,pos1c,pt1d,p1d,pos1d=clase(p1,pos1,pt1)
pt2a,p2a,pos2a,pt2b,p2b,pos2b,pt2c,p2c,pos2c,pt2d,p2d,pos2d=clase(p2,pos2,pt2)
pt3a,p3a,pos3a,pt3b,p3b,pos3b,pt3c,p3c,pos3c,pt3d,p3d,pos3d=clase(p3,pos3,pt3)

x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31=subdiv(pos1,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x01a,x11a,x21a,x31a,y01a,y11a,y21a,y31a,z01a,z11a,z21a,z31a,deltax01a,deltax11a,deltax21a,deltax31a,deltay01a,deltay11a,deltay21a,deltay31a,dx01a,dx11a,dx21a,dx31a,dy01a,dy11a,dy21a,dy31a=subdiv(pos1a,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01b,x11b,x21b,x31b,y01b,y11b,y21b,y31b,z01b,z11b,z21b,z31b,deltax01b,deltax11b,deltax21b,deltax31b,deltay01b,deltay11b,deltay21b,deltay31b,dx01b,dx11b,dx21b,dx31b,dy01b,dy11b,dy21b,dy31b=subdiv(pos1b,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01c,x11c,x21c,x31c,y01c,y11c,y21c,y31c,z01c,z11c,z21c,z31c,deltax01c,deltax11c,deltax21c,deltax31c,deltay01c,deltay11c,deltay21c,deltay31c,dx01c,dx11c,dx21c,dx31c,dy01c,dy11c,dy21c,dy31c=subdiv(pos1c,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01d,x11d,x21d,x31d,y01d,y11d,y21d,y31d,z01d,z11d,z21d,z31d,deltax01d,deltax11d,deltax21d,deltax31d,deltay01d,deltay11d,deltay21d,deltay31d,dx01d,dx11d,dx21d,dx31d,dy01d,dy11d,dy21d,dy31d=subdiv(pos1d,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)

x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32=subdiv(pos2,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x02a,x12a,x22a,x32a,y02a,y12a,y22a,y32a,z02a,z12a,z22a,z32a,deltax02a,deltax12a,deltax22a,deltax32a,deltay02a,deltay12a,deltay22a,deltay32a,dx02a,dx12a,dx22a,dx32a,dy02a,dy12a,dy22a,dy32a=subdiv(pos2a,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02b,x12b,x22b,x32b,y02b,y12b,y22b,y32b,z02b,z12b,z22b,z32b,deltax02b,deltax12b,deltax22b,deltax32b,deltay02b,deltay12b,deltay22b,deltay32b,dx02b,dx12b,dx22b,dx32b,dy02b,dy12b,dy22b,dy32b=subdiv(pos2b,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02c,x12c,x22c,x32c,y02c,y12c,y22c,y32c,z02c,z12c,z22c,z32c,deltax02c,deltax12c,deltax22c,deltax32c,deltay02c,deltay12c,deltay22c,deltay32c,dx02c,dx12c,dx22c,dx32c,dy02c,dy12c,dy22c,dy32c=subdiv(pos2c,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02d,x12d,x22d,x32d,y02d,y12d,y22d,y32d,z02d,z12d,z22d,z32d,deltax02d,deltax12d,deltax22d,deltax32d,deltay02d,deltay12d,deltay22d,deltay32d,dx02d,dx12d,dx22d,dx32d,dy02d,dy12d,dy22d,dy32d=subdiv(pos2d,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)

x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33=subdiv(pos3,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x03a,x13a,x23a,x33a,y03a,y13a,y23a,y33a,z03a,z13a,z23a,z33a,deltax03a,deltax13a,deltax23a,deltax33a,deltay03a,deltay13a,deltay23a,deltay33a,dx03a,dx13a,dx23a,dx33a,dy03a,dy13a,dy23a,dy33a=subdiv(pos3a,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03b,x13b,x23b,x33b,y03b,y13b,y23b,y33b,z03b,z13b,z23b,z33b,deltax03b,deltax13b,deltax23b,deltax33b,deltay03b,deltay13b,deltay23b,deltay33b,dx03b,dx13b,dx23b,dx33b,dy03b,dy13b,dy23b,dy33b=subdiv(pos3b,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03c,x13c,x23c,x33c,y03c,y13c,y23c,y33c,z03c,z13c,z23c,z33c,deltax03c,deltax13c,deltax23c,deltax33c,deltay03c,deltay13c,deltay23c,deltay33c,dx03c,dx13c,dx23c,dx33c,dy03c,dy13c,dy23c,dy33c=subdiv(pos3c,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03d,x13d,x23d,x33d,y03d,y13d,y23d,y33d,z03d,z13d,z23d,z33d,deltax03d,deltax13d,deltax23d,deltax33d,deltay03d,deltay13d,deltay23d,deltay33d,dx03d,dx13d,dx23d,dx33d,dy03d,dy13d,dy23d,dy33d=subdiv(pos3d,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)


# In[7]:


zi=[12.8,14.3,15.8,17.1,18.3]
deltazi=[25,53,47.5,47.5,47.5] #esto en realidad es el delta zi entre X0
beta=1.0


# In[8]:


#p1 y pta

#para 3000<p<6000 y 500<pt<800

chi1a=[]

for m in range(len(x01a)):
    chicuadradocorrnormx1a=funchi(x01a[m],x11a[m],x21a[m],x31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltax01a[m],deltax11a[m],deltax21a[m],deltax31a[m],dx01a[m],dx11a[m],dx21a[m],dx31a[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1a=funchi(y01a[m],y11a[m],y21a[m],y31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltay01a[m],deltay11a[m],deltay21a[m],deltay31a[m],dy01a[m],dy11a[m],dy21a[m],dy31a[m],zi,deltazi,beta,p1)
    chicuadradonorm1a=chicuadradocorrnormx1a+chicuadradocorrnormy1a
    chicuadradonorm1a=chicuadradonorm1a[0,0]
    chi1a.append(chicuadradonorm1a)
    
chisinms1a=[]

for m in range(len(x01a)):
    chicuadradocorrnormx1a=funchisinms(x01a[m],x11a[m],x21a[m],x31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltax01a[m],deltax11a[m],deltax21a[m],deltax31a[m],dx01a[m],dx11a[m],dx21a[m],dx31a[m])
    chicuadradocorrnormy1a=funchisinms(y01a[m],y11a[m],y21a[m],y31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltay01a[m],deltay11a[m],deltay21a[m],deltay31a[m],dy01a[m],dy11a[m],dy21a[m],dy31a[m])
    chicuadradonorm1a=chicuadradocorrnormx1a+chicuadradocorrnormy1a
    chicuadradonorm1a=chicuadradonorm1a[0,0]
    chisinms1a.append(chicuadradonorm1a)


# In[9]:


#p1 y ptb

#para 3000<p<6000 y 800<pt<1200

chi1b=[]

for m in range(len(x01b)):
    chicuadradocorrnormx1b=funchi(x01b[m],x11b[m],x21b[m],x31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltax01b[m],deltax11b[m],deltax21b[m],deltax31b[m],dx01b[m],dx11b[m],dx21b[m],dx31b[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1b=funchi(y01b[m],y11b[m],y21b[m],y31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltay01b[m],deltay11b[m],deltay21b[m],deltay31b[m],dy01b[m],dy11b[m],dy21b[m],dy31b[m],zi,deltazi,beta,p1)
    chicuadradonorm1b=chicuadradocorrnormx1b+chicuadradocorrnormy1b
    chicuadradonorm1b=chicuadradonorm1b[0,0]
    chi1b.append(chicuadradonorm1b)
    
chisinms1b=[]

for m in range(len(x01b)):
    chicuadradocorrnormx1b=funchisinms(x01b[m],x11b[m],x21b[m],x31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltax01b[m],deltax11b[m],deltax21b[m],deltax31b[m],dx01b[m],dx11b[m],dx21b[m],dx31b[m])
    chicuadradocorrnormy1b=funchisinms(y01b[m],y11b[m],y21b[m],y31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltay01b[m],deltay11b[m],deltay21b[m],deltay31b[m],dy01b[m],dy11b[m],dy21b[m],dy31b[m])
    chicuadradonorm1b=chicuadradocorrnormx1b+chicuadradocorrnormy1b
    chicuadradonorm1b=chicuadradonorm1b[0,0]
    chisinms1b.append(chicuadradonorm1b)


# In[10]:


#p1 y ptc

#para 3000<p<6000 y 1200<pt<2000

chi1c=[]

for m in range(len(x01c)):
    chicuadradocorrnormx1c=funchi(x01c[m],x11c[m],x21c[m],x31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltax01c[m],deltax11c[m],deltax21c[m],deltax31c[m],dx01c[m],dx11c[m],dx21c[m],dx31c[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1c=funchi(y01c[m],y11c[m],y21c[m],y31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltay01c[m],deltay11c[m],deltay21c[m],deltay31c[m],dy01c[m],dy11c[m],dy21c[m],dy31c[m],zi,deltazi,beta,p1)
    chicuadradonorm1c=chicuadradocorrnormx1c+chicuadradocorrnormy1c
    chicuadradonorm1c=chicuadradonorm1c[0,0]
    chi1c.append(chicuadradonorm1c)
    
chisinms1c=[]

for m in range(len(x01c)):
    chicuadradocorrnormx1c=funchisinms(x01c[m],x11c[m],x21c[m],x31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltax01c[m],deltax11c[m],deltax21c[m],deltax31c[m],dx01c[m],dx11c[m],dx21c[m],dx31c[m])
    chicuadradocorrnormy1c=funchisinms(y01c[m],y11c[m],y21c[m],y31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltay01c[m],deltay11c[m],deltay21c[m],deltay31c[m],dy01c[m],dy11c[m],dy21c[m],dy31c[m])
    chicuadradonorm1c=chicuadradocorrnormx1c+chicuadradocorrnormy1c
    chicuadradonorm1c=chicuadradonorm1c[0,0]
    chisinms1c.append(chicuadradonorm1c)


# In[11]:


#p1 y ptd

#para 3000<p<6000 y pt>2000

chi1d=[]

for m in range(len(x01d)):
    chicuadradocorrnormx1d=funchi(x01d[m],x11d[m],x21d[m],x31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltax01d[m],deltax11d[m],deltax21d[m],deltax31d[m],dx01d[m],dx11d[m],dx21d[m],dx31d[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1d=funchi(y01d[m],y11d[m],y21d[m],y31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltay01d[m],deltay11d[m],deltay21d[m],deltay31d[m],dy01d[m],dy11d[m],dy21d[m],dy31d[m],zi,deltazi,beta,p1)
    chicuadradonorm1d=chicuadradocorrnormx1d+chicuadradocorrnormy1d
    chicuadradonorm1d=chicuadradonorm1d[0,0]
    chi1d.append(chicuadradonorm1d)
    
chisinms1d=[]

for m in range(len(x01d)):
    chicuadradocorrnormx1d=funchisinms(x01d[m],x11d[m],x21d[m],x31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltax01d[m],deltax11d[m],deltax21d[m],deltax31d[m],dx01d[m],dx11d[m],dx21d[m],dx31d[m])
    chicuadradocorrnormy1d=funchisinms(y01d[m],y11d[m],y21d[m],y31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltay01d[m],deltay11d[m],deltay21d[m],deltay31d[m],dy01d[m],dy11d[m],dy21d[m],dy31d[m])
    chicuadradonorm1d=chicuadradocorrnormx1d+chicuadradocorrnormy1d
    chicuadradonorm1d=chicuadradonorm1d[0,0]
    chisinms1d.append(chicuadradonorm1d)


# In[12]:


#p2 y pta

#para 6000<p<10000 y 500<pt<800

chi2a=[]

for m in range(len(x02a)):
    chicuadradocorrnormx2a=funchi(x02a[m],x12a[m],x22a[m],x32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltax02a[m],deltax12a[m],deltax22a[m],deltax32a[m],dx02a[m],dx12a[m],dx22a[m],dx32a[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2a=funchi(y02a[m],y12a[m],y22a[m],y32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltay02a[m],deltay12a[m],deltay22a[m],deltay32a[m],dy02a[m],dy12a[m],dy22a[m],dy32a[m],zi,deltazi,beta,p2)
    chicuadradonorm2a=chicuadradocorrnormx2a+chicuadradocorrnormy2a
    chicuadradonorm2a=chicuadradonorm2a[0,0]
    chi2a.append(chicuadradonorm2a)
    
chisinms2a=[]

for m in range(len(x02a)):
    chicuadradocorrnormx2a=funchisinms(x02a[m],x12a[m],x22a[m],x32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltax02a[m],deltax12a[m],deltax22a[m],deltax32a[m],dx02a[m],dx12a[m],dx22a[m],dx32a[m])
    chicuadradocorrnormy2a=funchisinms(y02a[m],y12a[m],y22a[m],y32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltay02a[m],deltay12a[m],deltay22a[m],deltay32a[m],dy02a[m],dy12a[m],dy22a[m],dy32a[m])
    chicuadradonorm2a=chicuadradocorrnormx2a+chicuadradocorrnormy2a
    chicuadradonorm2a=chicuadradonorm2a[0,0]
    chisinms2a.append(chicuadradonorm2a)    


# In[13]:


#p2 y ptb

#para 6000<p<10000 y 800<pt<1200

chi2b=[]

for m in range(len(x02b)):
    chicuadradocorrnormx2b=funchi(x02b[m],x12b[m],x22b[m],x32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltax02b[m],deltax12b[m],deltax22b[m],deltax32b[m],dx02b[m],dx12b[m],dx22b[m],dx32b[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2b=funchi(y02b[m],y12b[m],y22b[m],y32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltay02b[m],deltay12b[m],deltay22b[m],deltay32b[m],dy02b[m],dy12b[m],dy22b[m],dy32b[m],zi,deltazi,beta,p2)
    chicuadradonorm2b=chicuadradocorrnormx2b+chicuadradocorrnormy2b
    chicuadradonorm2b=chicuadradonorm2b[0,0]
    chi2b.append(chicuadradonorm2b)
    
chisinms2b=[]

for m in range(len(x02b)):
    chicuadradocorrnormx2b=funchisinms(x02b[m],x12b[m],x22b[m],x32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltax02b[m],deltax12b[m],deltax22b[m],deltax32b[m],dx02b[m],dx12b[m],dx22b[m],dx32b[m])
    chicuadradocorrnormy2b=funchisinms(y02b[m],y12b[m],y22b[m],y32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltay02b[m],deltay12b[m],deltay22b[m],deltay32b[m],dy02b[m],dy12b[m],dy22b[m],dy32b[m])
    chicuadradonorm2b=chicuadradocorrnormx2b+chicuadradocorrnormy2b
    chicuadradonorm2b=chicuadradonorm2b[0,0]
    chisinms2b.append(chicuadradonorm2b)    


# In[14]:


#p2 y ptc

#para 6000<p<10000 y 1200<pt<2000

chi2c=[]

for m in range(len(x02c)):
    chicuadradocorrnormx2c=funchi(x02c[m],x12c[m],x22c[m],x32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltax02c[m],deltax12c[m],deltax22c[m],deltax32c[m],dx02c[m],dx12c[m],dx22c[m],dx32c[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2c=funchi(y02c[m],y12c[m],y22c[m],y32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltay02c[m],deltay12c[m],deltay22c[m],deltay32c[m],dy02c[m],dy12c[m],dy22c[m],dy32c[m],zi,deltazi,beta,p2)
    chicuadradonorm2c=chicuadradocorrnormx2c+chicuadradocorrnormy2c
    chicuadradonorm2c=chicuadradonorm2c[0,0]
    chi2c.append(chicuadradonorm2c)
    
chisinms2c=[]

for m in range(len(x02c)):
    chicuadradocorrnormx2c=funchisinms(x02c[m],x12c[m],x22c[m],x32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltax02c[m],deltax12c[m],deltax22c[m],deltax32c[m],dx02c[m],dx12c[m],dx22c[m],dx32c[m])
    chicuadradocorrnormy2c=funchisinms(y02c[m],y12c[m],y22c[m],y32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltay02c[m],deltay12c[m],deltay22c[m],deltay32c[m],dy02c[m],dy12c[m],dy22c[m],dy32c[m])
    chicuadradonorm2c=chicuadradocorrnormx2c+chicuadradocorrnormy2c
    chicuadradonorm2c=chicuadradonorm2c[0,0]
    chisinms2c.append(chicuadradonorm2c)    


# In[15]:


#p2 y ptd

#para 6000<p<10000 y pt>2000

chi2d=[]

for m in range(len(x02d)):
    chicuadradocorrnormx2d=funchi(x02d[m],x12d[m],x22d[m],x32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltax02d[m],deltax12d[m],deltax22d[m],deltax32d[m],dx02d[m],dx12d[m],dx22d[m],dx32d[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2d=funchi(y02d[m],y12d[m],y22d[m],y32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltay02d[m],deltay12d[m],deltay22d[m],deltay32d[m],dy02d[m],dy12d[m],dy22d[m],dy32d[m],zi,deltazi,beta,p2)
    chicuadradonorm2d=chicuadradocorrnormx2d+chicuadradocorrnormy2d
    chicuadradonorm2d=chicuadradonorm2d[0,0]
    chi2d.append(chicuadradonorm2d)
    
chisinms2d=[]

for m in range(len(x02d)):
    chicuadradocorrnormx2d=funchisinms(x02d[m],x12d[m],x22d[m],x32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltax02d[m],deltax12d[m],deltax22d[m],deltax32d[m],dx02d[m],dx12d[m],dx22d[m],dx32d[m])
    chicuadradocorrnormy2d=funchisinms(y02d[m],y12d[m],y22d[m],y32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltay02d[m],deltay12d[m],deltay22d[m],deltay32d[m],dy02d[m],dy12d[m],dy22d[m],dy32d[m])
    chicuadradonorm2d=chicuadradocorrnormx2d+chicuadradocorrnormy2d
    chicuadradonorm2d=chicuadradonorm2d[0,0]
    chisinms2d.append(chicuadradonorm2d)    


# In[16]:


#p3 y pta

#para p>10000 y 500<pt<800

chi3a=[]

for m in range(len(x03a)):
    chicuadradocorrnormx3a=funchi(x03a[m],x13a[m],x23a[m],x33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltax03a[m],deltax13a[m],deltax23a[m],deltax33a[m],dx03a[m],dx13a[m],dx23a[m],dx33a[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3a=funchi(y03a[m],y13a[m],y23a[m],y33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltay03a[m],deltay13a[m],deltay23a[m],deltay33a[m],dy03a[m],dy13a[m],dy23a[m],dy33a[m],zi,deltazi,beta,p3)
    chicuadradonorm3a=chicuadradocorrnormx3a+chicuadradocorrnormy3a
    chicuadradonorm3a=chicuadradonorm3a[0,0]
    chi3a.append(chicuadradonorm3a)
    
chisinms3a=[]

for m in range(len(x03a)):
    chicuadradocorrnormx3a=funchisinms(x03a[m],x13a[m],x23a[m],x33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltax03a[m],deltax13a[m],deltax23a[m],deltax33a[m],dx03a[m],dx13a[m],dx23a[m],dx33a[m])
    chicuadradocorrnormy3a=funchisinms(y03a[m],y13a[m],y23a[m],y33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltay03a[m],deltay13a[m],deltay23a[m],deltay33a[m],dy03a[m],dy13a[m],dy23a[m],dy33a[m])
    chicuadradonorm3a=chicuadradocorrnormx3a+chicuadradocorrnormy3a
    chicuadradonorm3a=chicuadradonorm3a[0,0]
    chisinms3a.append(chicuadradonorm3a)    


# In[17]:


#p3 y ptb

#para p>10000 y 800<pt<1200

chi3b=[]

for m in range(len(x03b)):
    chicuadradocorrnormx3b=funchi(x03b[m],x13b[m],x23b[m],x33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltax03b[m],deltax13b[m],deltax23b[m],deltax33b[m],dx03b[m],dx13b[m],dx23b[m],dx33b[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3b=funchi(y03b[m],y13b[m],y23b[m],y33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltay03b[m],deltay13b[m],deltay23b[m],deltay33b[m],dy03b[m],dy13b[m],dy23b[m],dy33b[m],zi,deltazi,beta,p3)
    chicuadradonorm3b=chicuadradocorrnormx3b+chicuadradocorrnormy3b
    chicuadradonorm3b=chicuadradonorm3b[0,0]
    chi3b.append(chicuadradonorm3b)
    
chisinms3b=[]
chisinmssinnorm3b=[]

for m in range(len(x03b)):
    chicuadradocorrnormx3b=funchisinms(x03b[m],x13b[m],x23b[m],x33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltax03b[m],deltax13b[m],deltax23b[m],deltax33b[m],dx03b[m],dx13b[m],dx23b[m],dx33b[m])
    chicuadradocorrnormy3b=funchisinms(y03b[m],y13b[m],y23b[m],y33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltay03b[m],deltay13b[m],deltay23b[m],deltay33b[m],dy03b[m],dy13b[m],dy23b[m],dy33b[m])
    chicuadradonorm3b=chicuadradocorrnormx3b+chicuadradocorrnormy3b
    chicuadradonorm3b=chicuadradonorm3b[0,0]
    chisinms3b.append(chicuadradonorm3b)    


# In[18]:


#p3 y ptc

#para p>10000 y 1200<pt<2000

chi3c=[]

for m in range(len(x03c)):
    chicuadradocorrnormx3c=funchi(x03c[m],x13c[m],x23c[m],x33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltax03c[m],deltax13c[m],deltax23c[m],deltax33c[m],dx03c[m],dx13c[m],dx23c[m],dx33c[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3c=funchi(y03c[m],y13c[m],y23c[m],y33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltay03c[m],deltay13c[m],deltay23c[m],deltay33c[m],dy03c[m],dy13c[m],dy23c[m],dy33c[m],zi,deltazi,beta,p3)
    chicuadradonorm3c=chicuadradocorrnormx3c+chicuadradocorrnormy3c
    chicuadradonorm3c=chicuadradonorm3c[0,0]
    chi3c.append(chicuadradonorm3c)
    
chisinms3c=[]

for m in range(len(x03c)):
    chicuadradocorrnormx3c=funchisinms(x03c[m],x13c[m],x23c[m],x33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltax03c[m],deltax13c[m],deltax23c[m],deltax33c[m],dx03c[m],dx13c[m],dx23c[m],dx33c[m])
    chicuadradocorrnormy3c=funchisinms(y03c[m],y13c[m],y23c[m],y33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltay03c[m],deltay13c[m],deltay23c[m],deltay33c[m],dy03c[m],dy13c[m],dy23c[m],dy33c[m])
    chicuadradonorm3c=chicuadradocorrnormx3c+chicuadradocorrnormy3c
    chicuadradonorm3c=chicuadradonorm3c[0,0]
    chisinms3c.append(chicuadradonorm3c)    


# In[19]:


#p3 y ptd

#para p>10000 y pt>2000

chi3d=[]

for m in range(len(x03d)):
    chicuadradocorrnormx3d=funchi(x03d[m],x13d[m],x23d[m],x33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltax03d[m],deltax13d[m],deltax23d[m],deltax33d[m],dx03d[m],dx13d[m],dx23d[m],dx33d[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3d=funchi(y03d[m],y13d[m],y23d[m],y33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltay03d[m],deltay13d[m],deltay23d[m],deltay33d[m],dy03d[m],dy13d[m],dy23d[m],dy33d[m],zi,deltazi,beta,p3)
    chicuadradonorm3d=chicuadradocorrnormx3d+chicuadradocorrnormy3d
    chicuadradonorm3d=chicuadradonorm3d[0,0]
    chi3d.append(chicuadradonorm3d)
    
chisinms3d=[]

for m in range(len(x03d)):
    chicuadradocorrnormx3d=funchisinms(x03d[m],x13d[m],x23d[m],x33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltax03d[m],deltax13d[m],deltax23d[m],deltax33d[m],dx03d[m],dx13d[m],dx23d[m],dx33d[m])
    chicuadradocorrnormy3d=funchisinms(y03d[m],y13d[m],y23d[m],y33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltay03d[m],deltay13d[m],deltay23d[m],deltay33d[m],dy03d[m],dy13d[m],dy23d[m],dy33d[m])
    chicuadradonorm3d=chicuadradocorrnormx3d+chicuadradocorrnormy3d
    chicuadradonorm3d=chicuadradonorm3d[0,0]
    chisinms3d.append(chicuadradonorm3d)    


# In[20]:


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


# In[21]:


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


# In[22]:


p1=[];p2=[];p3=[]
pt1=[];pt2=[];pt3=[]
pos1=[];pos2=[];pos3=[]
for i in range(len(p)):
    if 3000<=p[i]<6000:
        p1.append(p[i])
        pos1.append(i)
        pt1.append(pt[i])
    if 6000<=p[i]<10000:
        p2.append(p[i])
        pos2.append(i)
        pt2.append(pt[i])
    if p[i]>=10000:
        p3.append(p[i])
        pos3.append(i)
        pt3.append(pt[i])


# In[23]:


pt1a,p1a,pos1a,pt1b,p1b,pos1b,pt1c,p1c,pos1c,pt1d,p1d,pos1d=clase(p1,pos1,pt1)
pt2a,p2a,pos2a,pt2b,p2b,pos2b,pt2c,p2c,pos2c,pt2d,p2d,pos2d=clase(p2,pos2,pt2)
pt3a,p3a,pos3a,pt3b,p3b,pos3b,pt3c,p3c,pos3c,pt3d,p3d,pos3d=clase(p3,pos3,pt3)

x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31=subdiv(pos1,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x01a,x11a,x21a,x31a,y01a,y11a,y21a,y31a,z01a,z11a,z21a,z31a,deltax01a,deltax11a,deltax21a,deltax31a,deltay01a,deltay11a,deltay21a,deltay31a,dx01a,dx11a,dx21a,dx31a,dy01a,dy11a,dy21a,dy31a=subdiv(pos1a,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01b,x11b,x21b,x31b,y01b,y11b,y21b,y31b,z01b,z11b,z21b,z31b,deltax01b,deltax11b,deltax21b,deltax31b,deltay01b,deltay11b,deltay21b,deltay31b,dx01b,dx11b,dx21b,dx31b,dy01b,dy11b,dy21b,dy31b=subdiv(pos1b,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01c,x11c,x21c,x31c,y01c,y11c,y21c,y31c,z01c,z11c,z21c,z31c,deltax01c,deltax11c,deltax21c,deltax31c,deltay01c,deltay11c,deltay21c,deltay31c,dx01c,dx11c,dx21c,dx31c,dy01c,dy11c,dy21c,dy31c=subdiv(pos1c,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)
x01d,x11d,x21d,x31d,y01d,y11d,y21d,y31d,z01d,z11d,z21d,z31d,deltax01d,deltax11d,deltax21d,deltax31d,deltay01d,deltay11d,deltay21d,deltay31d,dx01d,dx11d,dx21d,dx31d,dy01d,dy11d,dy21d,dy31d=subdiv(pos1d,x01,x11,x21,x31,y01,y11,y21,y31,z01,z11,z21,z31,deltax01,deltax11,deltax21,deltax31,deltay01,deltay11,deltay21,deltay31,dx01,dx11,dx21,dx31,dy01,dy11,dy21,dy31)

x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32=subdiv(pos2,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x02a,x12a,x22a,x32a,y02a,y12a,y22a,y32a,z02a,z12a,z22a,z32a,deltax02a,deltax12a,deltax22a,deltax32a,deltay02a,deltay12a,deltay22a,deltay32a,dx02a,dx12a,dx22a,dx32a,dy02a,dy12a,dy22a,dy32a=subdiv(pos2a,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02b,x12b,x22b,x32b,y02b,y12b,y22b,y32b,z02b,z12b,z22b,z32b,deltax02b,deltax12b,deltax22b,deltax32b,deltay02b,deltay12b,deltay22b,deltay32b,dx02b,dx12b,dx22b,dx32b,dy02b,dy12b,dy22b,dy32b=subdiv(pos2b,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02c,x12c,x22c,x32c,y02c,y12c,y22c,y32c,z02c,z12c,z22c,z32c,deltax02c,deltax12c,deltax22c,deltax32c,deltay02c,deltay12c,deltay22c,deltay32c,dx02c,dx12c,dx22c,dx32c,dy02c,dy12c,dy22c,dy32c=subdiv(pos2c,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)
x02d,x12d,x22d,x32d,y02d,y12d,y22d,y32d,z02d,z12d,z22d,z32d,deltax02d,deltax12d,deltax22d,deltax32d,deltay02d,deltay12d,deltay22d,deltay32d,dx02d,dx12d,dx22d,dx32d,dy02d,dy12d,dy22d,dy32d=subdiv(pos2d,x02,x12,x22,x32,y02,y12,y22,y32,z02,z12,z22,z32,deltax02,deltax12,deltax22,deltax32,deltay02,deltay12,deltay22,deltay32,dx02,dx12,dx22,dx32,dy02,dy12,dy22,dy32)

x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33=subdiv(pos3,x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3,deltax0,deltax1,deltax2,deltax3,deltay0,deltay1,deltay2,deltay3,dx0,dx1,dx2,dx3,dy0,dy1,dy2,dy3)
x03a,x13a,x23a,x33a,y03a,y13a,y23a,y33a,z03a,z13a,z23a,z33a,deltax03a,deltax13a,deltax23a,deltax33a,deltay03a,deltay13a,deltay23a,deltay33a,dx03a,dx13a,dx23a,dx33a,dy03a,dy13a,dy23a,dy33a=subdiv(pos3a,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03b,x13b,x23b,x33b,y03b,y13b,y23b,y33b,z03b,z13b,z23b,z33b,deltax03b,deltax13b,deltax23b,deltax33b,deltay03b,deltay13b,deltay23b,deltay33b,dx03b,dx13b,dx23b,dx33b,dy03b,dy13b,dy23b,dy33b=subdiv(pos3b,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03c,x13c,x23c,x33c,y03c,y13c,y23c,y33c,z03c,z13c,z23c,z33c,deltax03c,deltax13c,deltax23c,deltax33c,deltay03c,deltay13c,deltay23c,deltay33c,dx03c,dx13c,dx23c,dx33c,dy03c,dy13c,dy23c,dy33c=subdiv(pos3c,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)
x03d,x13d,x23d,x33d,y03d,y13d,y23d,y33d,z03d,z13d,z23d,z33d,deltax03d,deltax13d,deltax23d,deltax33d,deltay03d,deltay13d,deltay23d,deltay33d,dx03d,dx13d,dx23d,dx33d,dy03d,dy13d,dy23d,dy33d=subdiv(pos3d,x03,x13,x23,x33,y03,y13,y23,y33,z03,z13,z23,z33,deltax03,deltax13,deltax23,deltax33,deltay03,deltay13,deltay23,deltay33,dx03,dx13,dx23,dx33,dy03,dy13,dy23,dy33)


# In[24]:


#p1 y pta

#para 3000<p<6000 y 500<pt<800

chib1a=[]

for m in range(len(x01a)):
    chicuadradocorrnormx1a=funchi(x01a[m],x11a[m],x21a[m],x31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltax01a[m],deltax11a[m],deltax21a[m],deltax31a[m],dx01a[m],dx11a[m],dx21a[m],dx31a[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1a=funchi(y01a[m],y11a[m],y21a[m],y31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltay01a[m],deltay11a[m],deltay21a[m],deltay31a[m],dy01a[m],dy11a[m],dy21a[m],dy31a[m],zi,deltazi,beta,p1)
    chicuadradonorm1a=chicuadradocorrnormx1a+chicuadradocorrnormy1a
    chicuadradonorm1a=chicuadradonorm1a[0,0]
    chib1a.append(chicuadradonorm1a)
    
chibsinms1a=[]

for m in range(len(x01a)):
    chicuadradocorrnormx1a=funchisinms(x01a[m],x11a[m],x21a[m],x31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltax01a[m],deltax11a[m],deltax21a[m],deltax31a[m],dx01a[m],dx11a[m],dx21a[m],dx31a[m])
    chicuadradocorrnormy1a=funchisinms(y01a[m],y11a[m],y21a[m],y31a[m],z01a[m],z11a[m],z21a[m],z31a[m],deltay01a[m],deltay11a[m],deltay21a[m],deltay31a[m],dy01a[m],dy11a[m],dy21a[m],dy31a[m])
    chicuadradonorm1a=chicuadradocorrnormx1a+chicuadradocorrnormy1a
    chicuadradonorm1a=chicuadradonorm1a[0,0]
    chibsinms1a.append(chicuadradonorm1a)


# In[25]:


#p1 y ptb

#para 3000<p<6000 y 800<pt<1200

chib1b=[]

for m in range(len(x01b)):
    chicuadradocorrnormx1b=funchi(x01b[m],x11b[m],x21b[m],x31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltax01b[m],deltax11b[m],deltax21b[m],deltax31b[m],dx01b[m],dx11b[m],dx21b[m],dx31b[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1b=funchi(y01b[m],y11b[m],y21b[m],y31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltay01b[m],deltay11b[m],deltay21b[m],deltay31b[m],dy01b[m],dy11b[m],dy21b[m],dy31b[m],zi,deltazi,beta,p1)
    chicuadradonorm1b=chicuadradocorrnormx1b+chicuadradocorrnormy1b
    chicuadradonorm1b=chicuadradonorm1b[0,0]
    chib1b.append(chicuadradonorm1b)
    
chibsinms1b=[]

for m in range(len(x01b)):
    chicuadradocorrnormx1b=funchisinms(x01b[m],x11b[m],x21b[m],x31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltax01b[m],deltax11b[m],deltax21b[m],deltax31b[m],dx01b[m],dx11b[m],dx21b[m],dx31b[m])
    chicuadradocorrnormy1b=funchisinms(y01b[m],y11b[m],y21b[m],y31b[m],z01b[m],z11b[m],z21b[m],z31b[m],deltay01b[m],deltay11b[m],deltay21b[m],deltay31b[m],dy01b[m],dy11b[m],dy21b[m],dy31b[m])
    chicuadradonorm1b=chicuadradocorrnormx1b+chicuadradocorrnormy1b
    chicuadradonorm1b=chicuadradonorm1b[0,0]
    chibsinms1b.append(chicuadradonorm1b)


# In[26]:


#p1 y ptc

#para 3000<p<6000 y 1200<pt<2000

chib1c=[]

for m in range(len(x01c)):
    chicuadradocorrnormx1c=funchi(x01c[m],x11c[m],x21c[m],x31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltax01c[m],deltax11c[m],deltax21c[m],deltax31c[m],dx01c[m],dx11c[m],dx21c[m],dx31c[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1c=funchi(y01c[m],y11c[m],y21c[m],y31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltay01c[m],deltay11c[m],deltay21c[m],deltay31c[m],dy01c[m],dy11c[m],dy21c[m],dy31c[m],zi,deltazi,beta,p1)
    chicuadradonorm1c=chicuadradocorrnormx1c+chicuadradocorrnormy1c
    chicuadradonorm1c=chicuadradonorm1c[0,0]
    chib1c.append(chicuadradonorm1c)
    
chibsinms1c=[]

for m in range(len(x01c)):
    chicuadradocorrnormx1c=funchisinms(x01c[m],x11c[m],x21c[m],x31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltax01c[m],deltax11c[m],deltax21c[m],deltax31c[m],dx01c[m],dx11c[m],dx21c[m],dx31c[m])
    chicuadradocorrnormy1c=funchisinms(y01c[m],y11c[m],y21c[m],y31c[m],z01c[m],z11c[m],z21c[m],z31c[m],deltay01c[m],deltay11c[m],deltay21c[m],deltay31c[m],dy01c[m],dy11c[m],dy21c[m],dy31c[m])
    chicuadradonorm1c=chicuadradocorrnormx1c+chicuadradocorrnormy1c
    chicuadradonorm1c=chicuadradonorm1c[0,0]
    chibsinms1c.append(chicuadradonorm1c)


# In[27]:


#p1 y ptd

#para 3000<p<6000 y pt>2000

chib1d=[]

for m in range(len(x01d)):
    chicuadradocorrnormx1d=funchi(x01d[m],x11d[m],x21d[m],x31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltax01d[m],deltax11d[m],deltax21d[m],deltax31d[m],dx01d[m],dx11d[m],dx21d[m],dx31d[m],zi,deltazi,beta,p1)
    chicuadradocorrnormy1d=funchi(y01d[m],y11d[m],y21d[m],y31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltay01d[m],deltay11d[m],deltay21d[m],deltay31d[m],dy01d[m],dy11d[m],dy21d[m],dy31d[m],zi,deltazi,beta,p1)
    chicuadradonorm1d=chicuadradocorrnormx1d+chicuadradocorrnormy1d
    chicuadradonorm1d=chicuadradonorm1d[0,0]
    chib1d.append(chicuadradonorm1d)
    
chibsinms1d=[]

for m in range(len(x01d)):
    chicuadradocorrnormx1d=funchisinms(x01d[m],x11d[m],x21d[m],x31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltax01d[m],deltax11d[m],deltax21d[m],deltax31d[m],dx01d[m],dx11d[m],dx21d[m],dx31d[m])
    chicuadradocorrnormy1d=funchisinms(y01d[m],y11d[m],y21d[m],y31d[m],z01d[m],z11d[m],z21d[m],z31d[m],deltay01d[m],deltay11d[m],deltay21d[m],deltay31d[m],dy01d[m],dy11d[m],dy21d[m],dy31d[m])
    chicuadradonorm1d=chicuadradocorrnormx1d+chicuadradocorrnormy1d
    chicuadradonorm1d=chicuadradonorm1d[0,0]
    chibsinms1d.append(chicuadradonorm1d)


# In[28]:


#p2 y pta

#para 6000<p<10000 y 500<pt<800

chib2a=[]

for m in range(len(x02a)):
    chicuadradocorrnormx2a=funchi(x02a[m],x12a[m],x22a[m],x32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltax02a[m],deltax12a[m],deltax22a[m],deltax32a[m],dx02a[m],dx12a[m],dx22a[m],dx32a[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2a=funchi(y02a[m],y12a[m],y22a[m],y32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltay02a[m],deltay12a[m],deltay22a[m],deltay32a[m],dy02a[m],dy12a[m],dy22a[m],dy32a[m],zi,deltazi,beta,p2)
    chicuadradonorm2a=chicuadradocorrnormx2a+chicuadradocorrnormy2a
    chicuadradonorm2a=chicuadradonorm2a[0,0]
    chib2a.append(chicuadradonorm2a)
    
chibsinms2a=[]

for m in range(len(x02a)):
    chicuadradocorrnormx2a=funchisinms(x02a[m],x12a[m],x22a[m],x32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltax02a[m],deltax12a[m],deltax22a[m],deltax32a[m],dx02a[m],dx12a[m],dx22a[m],dx32a[m])
    chicuadradocorrnormy2a=funchisinms(y02a[m],y12a[m],y22a[m],y32a[m],z02a[m],z12a[m],z22a[m],z32a[m],deltay02a[m],deltay12a[m],deltay22a[m],deltay32a[m],dy02a[m],dy12a[m],dy22a[m],dy32a[m])
    chicuadradonorm2a=chicuadradocorrnormx2a+chicuadradocorrnormy2a
    chicuadradonorm2a=chicuadradonorm2a[0,0]
    chibsinms2a.append(chicuadradonorm2a)    


# In[29]:


#p2 y ptb

#para 6000<p<10000 y 800<pt<1200

chib2b=[]

for m in range(len(x02b)):
    chicuadradocorrnormx2b=funchi(x02b[m],x12b[m],x22b[m],x32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltax02b[m],deltax12b[m],deltax22b[m],deltax32b[m],dx02b[m],dx12b[m],dx22b[m],dx32b[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2b=funchi(y02b[m],y12b[m],y22b[m],y32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltay02b[m],deltay12b[m],deltay22b[m],deltay32b[m],dy02b[m],dy12b[m],dy22b[m],dy32b[m],zi,deltazi,beta,p2)
    chicuadradonorm2b=chicuadradocorrnormx2b+chicuadradocorrnormy2b
    chicuadradonorm2b=chicuadradonorm2b[0,0]
    chib2b.append(chicuadradonorm2b)
    
chibsinms2b=[]

for m in range(len(x02b)):
    chicuadradocorrnormx2b=funchisinms(x02b[m],x12b[m],x22b[m],x32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltax02b[m],deltax12b[m],deltax22b[m],deltax32b[m],dx02b[m],dx12b[m],dx22b[m],dx32b[m])
    chicuadradocorrnormy2b=funchisinms(y02b[m],y12b[m],y22b[m],y32b[m],z02b[m],z12b[m],z22b[m],z32b[m],deltay02b[m],deltay12b[m],deltay22b[m],deltay32b[m],dy02b[m],dy12b[m],dy22b[m],dy32b[m])
    chicuadradonorm2b=chicuadradocorrnormx2b+chicuadradocorrnormy2b
    chicuadradonorm2b=chicuadradonorm2b[0,0]
    chibsinms2b.append(chicuadradonorm2b)


# In[30]:


#p2 y ptc

#para 6000<p<10000 y 1200<pt<2000

chib2c=[]

for m in range(len(x02c)):
    chicuadradocorrnormx2c=funchi(x02c[m],x12c[m],x22c[m],x32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltax02c[m],deltax12c[m],deltax22c[m],deltax32c[m],dx02c[m],dx12c[m],dx22c[m],dx32c[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2c=funchi(y02c[m],y12c[m],y22c[m],y32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltay02c[m],deltay12c[m],deltay22c[m],deltay32c[m],dy02c[m],dy12c[m],dy22c[m],dy32c[m],zi,deltazi,beta,p2)
    chicuadradonorm2c=chicuadradocorrnormx2c+chicuadradocorrnormy2c
    chicuadradonorm2c=chicuadradonorm2c[0,0]
    chib2c.append(chicuadradonorm2c)
    
chibsinms2c=[]

for m in range(len(x02c)):
    chicuadradocorrnormx2c=funchisinms(x02c[m],x12c[m],x22c[m],x32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltax02c[m],deltax12c[m],deltax22c[m],deltax32c[m],dx02c[m],dx12c[m],dx22c[m],dx32c[m])
    chicuadradocorrnormy2c=funchisinms(y02c[m],y12c[m],y22c[m],y32c[m],z02c[m],z12c[m],z22c[m],z32c[m],deltay02c[m],deltay12c[m],deltay22c[m],deltay32c[m],dy02c[m],dy12c[m],dy22c[m],dy32c[m])
    chicuadradonorm2c=chicuadradocorrnormx2c+chicuadradocorrnormy2c
    chicuadradonorm2c=chicuadradonorm2c[0,0]
    chibsinms2c.append(chicuadradonorm2c)    


# In[31]:


#p2 y ptd

#para 6000<p<10000 y pt>2000

chib2d=[]

for m in range(len(x02d)):
    chicuadradocorrnormx2d=funchi(x02d[m],x12d[m],x22d[m],x32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltax02d[m],deltax12d[m],deltax22d[m],deltax32d[m],dx02d[m],dx12d[m],dx22d[m],dx32d[m],zi,deltazi,beta,p2)
    chicuadradocorrnormy2d=funchi(y02d[m],y12d[m],y22d[m],y32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltay02d[m],deltay12d[m],deltay22d[m],deltay32d[m],dy02d[m],dy12d[m],dy22d[m],dy32d[m],zi,deltazi,beta,p2)
    chicuadradonorm2d=chicuadradocorrnormx2d+chicuadradocorrnormy2d
    chicuadradonorm2d=chicuadradonorm2d[0,0]
    chib2d.append(chicuadradonorm2d)
    
chibsinms2d=[]

for m in range(len(x02d)):
    chicuadradocorrnormx2d=funchisinms(x02d[m],x12d[m],x22d[m],x32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltax02d[m],deltax12d[m],deltax22d[m],deltax32d[m],dx02d[m],dx12d[m],dx22d[m],dx32d[m])
    chicuadradocorrnormy2d=funchisinms(y02d[m],y12d[m],y22d[m],y32d[m],z02d[m],z12d[m],z22d[m],z32d[m],deltay02d[m],deltay12d[m],deltay22d[m],deltay32d[m],dy02d[m],dy12d[m],dy22d[m],dy32d[m])
    chicuadradonorm2d=chicuadradocorrnormx2d+chicuadradocorrnormy2d
    chicuadradonorm2d=chicuadradonorm2d[0,0]
    chibsinms2d.append(chicuadradonorm2d)    


# In[32]:


#p3 y pta

#para p>10000 y 500<pt<800

chib3a=[]

for m in range(len(x03a)):
    chicuadradocorrnormx3a=funchi(x03a[m],x13a[m],x23a[m],x33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltax03a[m],deltax13a[m],deltax23a[m],deltax33a[m],dx03a[m],dx13a[m],dx23a[m],dx33a[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3a=funchi(y03a[m],y13a[m],y23a[m],y33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltay03a[m],deltay13a[m],deltay23a[m],deltay33a[m],dy03a[m],dy13a[m],dy23a[m],dy33a[m],zi,deltazi,beta,p3)
    chicuadradonorm3a=chicuadradocorrnormx3a+chicuadradocorrnormy3a
    chicuadradonorm3a=chicuadradonorm3a[0,0]
    chib3a.append(chicuadradonorm3a)
    
chibsinms3a=[]

for m in range(len(x03a)):
    chicuadradocorrnormx3a=funchisinms(x03a[m],x13a[m],x23a[m],x33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltax03a[m],deltax13a[m],deltax23a[m],deltax33a[m],dx03a[m],dx13a[m],dx23a[m],dx33a[m])
    chicuadradocorrnormy3a=funchisinms(y03a[m],y13a[m],y23a[m],y33a[m],z03a[m],z13a[m],z23a[m],z33a[m],deltay03a[m],deltay13a[m],deltay23a[m],deltay33a[m],dy03a[m],dy13a[m],dy23a[m],dy33a[m])
    chicuadradonorm3a=chicuadradocorrnormx3a+chicuadradocorrnormy3a
    chicuadradonorm3a=chicuadradonorm3a[0,0]
    chibsinms3a.append(chicuadradonorm3a)    


# In[33]:


#p3 y ptb

#para p>10000 y 800<pt<1200

chib3b=[]

for m in range(len(x03b)):
    chicuadradocorrnormx3b=funchi(x03b[m],x13b[m],x23b[m],x33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltax03b[m],deltax13b[m],deltax23b[m],deltax33b[m],dx03b[m],dx13b[m],dx23b[m],dx33b[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3b=funchi(y03b[m],y13b[m],y23b[m],y33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltay03b[m],deltay13b[m],deltay23b[m],deltay33b[m],dy03b[m],dy13b[m],dy23b[m],dy33b[m],zi,deltazi,beta,p3)
    chicuadradonorm3b=chicuadradocorrnormx3b+chicuadradocorrnormy3b
    chicuadradonorm3b=chicuadradonorm3b[0,0]
    chib3b.append(chicuadradonorm3b)
    
chibsinms3b=[]

for m in range(len(x03b)):
    chicuadradocorrnormx3b=funchisinms(x03b[m],x13b[m],x23b[m],x33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltax03b[m],deltax13b[m],deltax23b[m],deltax33b[m],dx03b[m],dx13b[m],dx23b[m],dx33b[m])
    chicuadradocorrnormy3b=funchisinms(y03b[m],y13b[m],y23b[m],y33b[m],z03b[m],z13b[m],z23b[m],z33b[m],deltay03b[m],deltay13b[m],deltay23b[m],deltay33b[m],dy03b[m],dy13b[m],dy23b[m],dy33b[m])
    chicuadradonorm3b=chicuadradocorrnormx3b+chicuadradocorrnormy3b
    chicuadradonorm3b=chicuadradonorm3b[0,0]
    chibsinms3b.append(chicuadradonorm3b)


# In[34]:


#p3 y ptc

#para p>10000 y 1200<pt<2000

chib3c=[]

for m in range(len(x03c)):
    chicuadradocorrnormx3c=funchi(x03c[m],x13c[m],x23c[m],x33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltax03c[m],deltax13c[m],deltax23c[m],deltax33c[m],dx03c[m],dx13c[m],dx23c[m],dx33c[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3c=funchi(y03c[m],y13c[m],y23c[m],y33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltay03c[m],deltay13c[m],deltay23c[m],deltay33c[m],dy03c[m],dy13c[m],dy23c[m],dy33c[m],zi,deltazi,beta,p3)
    chicuadradonorm3c=chicuadradocorrnormx3c+chicuadradocorrnormy3c
    chicuadradonorm3c=chicuadradonorm3c[0,0]
    chib3c.append(chicuadradonorm3c)
    
chibsinms3c=[]

for m in range(len(x03c)):
    chicuadradocorrnormx3c=funchisinms(x03c[m],x13c[m],x23c[m],x33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltax03c[m],deltax13c[m],deltax23c[m],deltax33c[m],dx03c[m],dx13c[m],dx23c[m],dx33c[m])
    chicuadradocorrnormy3c=funchisinms(y03c[m],y13c[m],y23c[m],y33c[m],z03c[m],z13c[m],z23c[m],z33c[m],deltay03c[m],deltay13c[m],deltay23c[m],deltay33c[m],dy03c[m],dy13c[m],dy23c[m],dy33c[m])
    chicuadradonorm3c=chicuadradocorrnormx3c+chicuadradocorrnormy3c
    chicuadradonorm3c=chicuadradonorm3c[0,0]
    chibsinms3c.append(chicuadradonorm3c)    


# In[35]:


#p3 y ptd

#para p>10000 y pt>2000

chib3d=[]

for m in range(len(x03d)):
    chicuadradocorrnormx3d=funchi(x03d[m],x13d[m],x23d[m],x33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltax03d[m],deltax13d[m],deltax23d[m],deltax33d[m],dx03d[m],dx13d[m],dx23d[m],dx33d[m],zi,deltazi,beta,p3)
    chicuadradocorrnormy3d=funchi(y03d[m],y13d[m],y23d[m],y33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltay03d[m],deltay13d[m],deltay23d[m],deltay33d[m],dy03d[m],dy13d[m],dy23d[m],dy33d[m],zi,deltazi,beta,p3)
    chicuadradonorm3d=chicuadradocorrnormx3d+chicuadradocorrnormy3d
    chicuadradonorm3d=chicuadradonorm3d[0,0]
    chib3d.append(chicuadradonorm3d)
    
chibsinms3d=[]

for m in range(len(x03d)):
    chicuadradocorrnormx3d=funchisinms(x03d[m],x13d[m],x23d[m],x33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltax03d[m],deltax13d[m],deltax23d[m],deltax33d[m],dx03d[m],dx13d[m],dx23d[m],dx33d[m])
    chicuadradocorrnormy3d=funchisinms(y03d[m],y13d[m],y23d[m],y33d[m],z03d[m],z13d[m],z23d[m],z33d[m],deltay03d[m],deltay13d[m],deltay23d[m],deltay33d[m],dy03d[m],dy13d[m],dy23d[m],dy33d[m])
    chicuadradonorm3d=chicuadradocorrnormx3d+chicuadradocorrnormy3d
    chicuadradonorm3d=chicuadradonorm3d[0,0]
    chibsinms3d.append(chicuadradonorm3d)    


# In[36]:


#p1 y pta

#para 3000<p<6000 y 500<pt<800

#con MS

plt.hist(np.log10(chi1a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib1a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[37]:


#sin MS

plt.hist(np.log10(chisinms1a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms1a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[38]:


#p1 y ptb

#para 3000<p<6000 y 800<pt<1200

#con MS

plt.hist(np.log10(chi1b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib1b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[39]:


#sin MS

plt.hist(np.log10(chisinms1b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms1b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[40]:


#p1 y ptc

#para 3000<p<6000 y 1200<pt<2000

#con MS

plt.hist(np.log10(chi1c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib1c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[41]:


#sin MS

plt.hist(np.log10(chisinms1c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms1c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[42]:


#p2 y pta

#para 6000<p<10000 y 500<p

#con MS

plt.hist(np.log10(chi2a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib2a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[43]:


#sin MS


plt.hist(np.log10(chisinms2a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms2a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[44]:


#p2 y ptb

#para 6000<p<10000 y 800<pt<1200

#con MS

plt.hist(np.log10(chi2b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib2b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[45]:


#sin MS

plt.hist(np.log10(chisinms2b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms2b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[46]:


#p2 y ptc

#para 6000<p<10000 y 1200<pt<2000

#con MS

plt.hist(np.log10(chi2c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib2c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[47]:


#sin MS

plt.hist(np.log10(chisinms2c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms2c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[48]:


#p2 y ptd

#para 6000<p<10000 y pt>2000

#con MS

plt.hist(np.log10(chi2d),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib2d), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[49]:


#sin MS

plt.hist(np.log10(chisinms2c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms2c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[50]:


#p3 y pta

#para p>10000 y 500<pt<800

#con MS

plt.hist(np.log10(chi3a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib3a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[51]:


#sin MS

plt.hist(np.log10(chisinms3a),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms3a), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[52]:


#p3 y ptb

#para p>10000 y 800<pt<1200

#con MS

plt.hist(np.log10(chi3b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib3b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[53]:


#sin MS

plt.hist(np.log10(chisinms3b),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms3b), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[54]:


#p3 y ptc

#para p>10000 y 1200<pt<2000

#con MS

plt.hist(np.log10(chi3c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib3c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[55]:


#sin MS

plt.hist(np.log10(chisinms3c),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms3c), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[56]:


#p3 y ptd

#para p>10000 y pt>2000

#con MS

plt.hist(np.log10(chi3d),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chib3d), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[57]:


#sin MS

plt.hist(np.log10(chisinms3d),bins=50,range=(-2,5),color='g',density=True,edgecolor='white',alpha=0.7,label='$\mu$')
plt.hist(np.log10(chibsinms3d), bins=50, range=(-2,5), color='r',density=True,edgecolor='white',alpha=0.5,label='p')
plt.legend(loc='upper right')
plt.xlabel('$log_{10}$($\chi^2_{CORR}/ndof$)')
plt.ylabel('Probability density')


# In[58]:


#p1 y pta

#para 3000<p<6000 y 500<pt<800

y_true=np.concatenate([np.ones(len(chi1a)),np.zeros(len(chib1a))])
y_scores=np.concatenate([chi1a,chib1a])

y_true_sinms=np.concatenate([np.ones(len(chisinms1a)),np.zeros(len(chibsinms1a))])
y_scores_sinms=np.concatenate([chisinms1a,chibsinms1a])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms,1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[59]:


#p1 y ptb

#para 3000<p<6000 y 800<pt<1200

y_true=np.concatenate([np.ones(len(chi1b)),np.zeros(len(chib1b))])
y_scores=np.concatenate([chi1b,chib1b])

y_true_sinms=np.concatenate([np.ones(len(chisinms1b)),np.zeros(len(chibsinms1b))])
y_scores_sinms=np.concatenate([chisinms1b,chibsinms1b])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[60]:


#p1 y ptc

#para 3000<p<6000 y 1200<pt<2000

y_true=np.concatenate([np.ones(len(chi1c)),np.zeros(len(chib1c))])
y_scores=np.concatenate([chi1c,chib1c])

y_true_sinms=np.concatenate([np.ones(len(chisinms1c)),np.zeros(len(chibsinms1c))])
y_scores_sinms=np.concatenate([chisinms1c,chibsinms1c])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[61]:


#p2 y pta

#para 6000<p<10000 y 500<p<800

y_true=np.concatenate([np.ones(len(chi2a)),np.zeros(len(chib2a))])
y_scores=np.concatenate([chi2a,chib2a])

y_true_sinms=np.concatenate([np.ones(len(chisinms2a)),np.zeros(len(chibsinms2a))])
y_scores_sinms=np.concatenate([chisinms2a,chibsinms2a])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[62]:


#p2 y ptb

#para 6000<p<10000 y 800<pt<1200

y_true=np.concatenate([np.ones(len(chi2b)),np.zeros(len(chib2b))])
y_scores=np.concatenate([chi2b,chib2b])

y_true_sinms=np.concatenate([np.ones(len(chisinms2b)),np.zeros(len(chibsinms2b))])
y_scores_sinms=np.concatenate([chisinms2b,chibsinms2b])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[63]:


#p2 y ptc

#para 6000<p<10000 y 1200<pt<2000

y_true=np.concatenate([np.ones(len(chi2c)),np.zeros(len(chib2c))])
y_scores=np.concatenate([chi2c,chib2c])

y_true_sinms=np.concatenate([np.ones(len(chisinms2c)),np.zeros(len(chibsinms2c))])
y_scores_sinms=np.concatenate([chisinms2c,chibsinms2c])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[64]:


#p2 y ptd

#para 6000<p<10000 y pt>2000

y_true=np.concatenate([np.ones(len(chi2d)),np.zeros(len(chib2d))])
y_scores=np.concatenate([chi2d,chib2d])

y_true_sinms=np.concatenate([np.ones(len(chisinms2d)),np.zeros(len(chibsinms2d))])
y_scores_sinms=np.concatenate([chisinms2d,chibsinms2d])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[65]:


#p3 y pta

#para p>10000 y 500<pt<800

y_true=np.concatenate([np.ones(len(chi3a)),np.zeros(len(chib3a))])
y_scores=np.concatenate([chi3a,chib3a])

y_true_sinms=np.concatenate([np.ones(len(chisinms3a)),np.zeros(len(chibsinms3a))])
y_scores_sinms=np.concatenate([chisinms3a,chibsinms3a])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[66]:


#p3 y ptb

#para p>10000 y 800<pt<1200

y_true=np.concatenate([np.ones(len(chi3b)),np.zeros(len(chib3b))])
y_scores=np.concatenate([chi3b,chib3b])

y_true_sinms=np.concatenate([np.ones(len(chisinms3b)),np.zeros(len(chibsinms3b))])
y_scores_sinms=np.concatenate([chisinms3b,chibsinms3b])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[67]:


#p3 y ptc

#para p>10000 y 1200<pt<2000

y_true=np.concatenate([np.ones(len(chi3c)),np.zeros(len(chib3c))])
y_scores=np.concatenate([chi3c,chib3c])

y_true_sinms=np.concatenate([np.ones(len(chisinms3c)),np.zeros(len(chibsinms3c))])
y_scores_sinms=np.concatenate([chisinms3c,chibsinms3c])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)


# In[68]:


#p3 y ptd

#para p>10000 y pt>2000

y_true=np.concatenate([np.ones(len(chi3d)),np.zeros(len(chib3d))])
y_scores=np.concatenate([chi3d,chib3d])

y_true_sinms=np.concatenate([np.ones(len(chisinms3d)),np.zeros(len(chibsinms3d))])
y_scores_sinms=np.concatenate([chisinms3d,chibsinms3d])

fpr, tpr, thresholds = roc_curve(y_true,y_scores)
roc_auc = auc(fpr, 1-tpr)

fprsinms, tprsinms, thresholdssinms = roc_curve(y_true_sinms,y_scores_sinms)
roc_auc_sinms = auc(fprsinms, 1-tprsinms)

plt.plot(fpr,1-tpr, color='b',label='con MS')
plt.plot(fprsinms,1-tprsinms,color='g',label='sin MS')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.legend(loc="lower left")
plt.show()
print('auc con ms=', roc_auc)
print('auc sin ms=', roc_auc_sinms)

