#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import warnings
import timeit as dt
import pandas as pd
from sklearn.preprocessing import normalize
from utils import divide
# add robustness where repeated data is present
class DA:
    def __init__(self,K,NORM='L2',TOL=1e-4,MAX_ITER=300,GROWTH_RATE=1.05,
               PURTURB_RATIO=0.01,BETA_INIT=1e-4,BETA_TOL=1e-6,verbos=0,NORMALIZE=False):
        self.K=K;self.TOL=TOL;self.MAX_ITER=MAX_ITER;self.BETA_INIT=BETA_INIT
        self.GROWTH_RATE=GROWTH_RATE;self.PURTURB_RATIO=PURTURB_RATIO;self.BETA_TOL=BETA_TOL
        self.CONSTRAINED=False;self.VERBOS=verbos;self.NORM=NORM;self.NORMALIZE=NORMALIZE
        print('DA model was created successfully')
    def fit(self,X,**kwargs):
        m, n = np.shape(X)
        l = np.sum(X * X, axis = 0, keepdims = True)
        l_sqrt = np.sqrt(l)
        X_norm = divide (X.copy(), l_sqrt)
        self.d,self.n=X.shape
        if self.NORMALIZE:
            self.X=X_norm.copy()
        else:
            self.X=X.copy()        
        self.XU=X.copy()
        if (self.X<0).any() and self.NORM=='KL':
            raise Exception('Your input matrix contains negative values. Try using another norm')
        self.Data_points=np.repeat(self.X[:,:,np.newaxis],self.K,axis=2)
        if 'Px' in kwargs:
            self.Px=kwargs['Px']
        else:
            self.Px=np.array([1 for i in range(self.n)])/self.n
        #self.Y=np.repeat(np.mean(self.X,axis=1)[:,np.newaxis],self.K,axis=1);
        self.Y=self.X.copy()[:,np.random.randint(10, size=self.K)]
        if 'Lambdas' in kwargs:
            self.CONSTRAINED=True
            self.Lambdas=kwargs['Lambdas']
            if np.sum(self.Lambdas) != 1.0:
                warnings.warn('Lambdas should sum up to one')
        else:
            self.Lambdas=np.array([1 for i in range(self.K)])/self.K
        self.Ethas=self.Lambdas
        print('Class DA fitted on DATA')
    def classify(self):
        start=dt.default_timer()
        #P_old=np.random.rand(self.K,self.n)
        Y_old=np.random.rand(self.d,self.K)
        Beta=self.BETA_INIT
        START_OK=0
        while True: #beta loop
            
            counter=1
            while True: #y p etha loop
                Cluster_points=np.repeat(self.Y[:,np.newaxis,:],self.n,axis=1)+1e-6
                if self.NORM=='KL':
                    d=np.log(Cluster_points/(self.Data_points+1e-6))
                    #d[np.where(d==-np.inf)]=0
                    D=d*Cluster_points-Cluster_points+self.Data_points
                elif self.NORM=='L2':
                    D=(self.Data_points-Cluster_points)**2
                else:
                    raise Exception("Wrong Norm!")
                D=np.sum(D,axis=0).T
                counter2=1
                if self.CONSTRAINED:
                    while True:
                        p=np.exp(-D*Beta)
                        if (np.count_nonzero(np.abs(p-1)<1e-5)/(self.K*self.n))<0.9 and not START_OK:
                            
                            START_OK=1
                        p=p*np.repeat(self.Ethas[:,np.newaxis],self.n,axis=1)
                        J=np.where(p.sum(axis=0)==0)[0]
                        I=np.argmin(D[:,J],axis=0)
                        p[I,J]=[1 for i in range(len(J))]
                        P=p/p.sum(axis=0)
                        Py=P@self.Px
                        self.Ethas=(self.Ethas*self.Lambdas)/(Py+1e-6)
                        if np.linalg.norm(P-P_old)/np.linalg.norm(P_old)<self.TOL:
                            break
                        if counter2>self.MAX_ITER:
                            warnings.warn("MAX ITER REACHED: ETHAS LOOP")
                            break
                        P_old=P
                        counter2+=1
                    self.Ethas=self.Ethas/self.Ethas.sum()
                    self.Y=((self.X*self.Px)@P.T)/(Py+1e-6)
                    PURTURB=self.PURTURB_RATIO*np.random.rand(self.d,self.K)*self.Y
                    self.Y=self.Y+PURTURB
                    if np.linalg.norm(P-P_old)/np.linalg.norm(P_old)<self.TOL:
                        break
                    if counter>self.MAX_ITER:
                        warnings.warn("MAX ITER REACHED: Y LOOP")
                        break
                    P_old=P
                    counter=counter+1
                else:
                    p=np.exp(-D*Beta)
                    if (np.count_nonzero(np.abs(p-1)<1e-5)/(self.K*self.n))<0.9 and not START_OK:
                        
                        START_OK=1
                    J=np.where(p.sum(axis=0)==0)[0]
                    I=np.argmin(D[:,J],axis=0)
                    p[I,J]=[1 for i in range(len(J))]
                    P=p/p.sum(axis=0)
                    Py=P@self.Px
                    self.Ethas=self.Ethas/self.Ethas.sum()
                    if self.NORM=='KL':
                        self.Y=np.exp(((np.log(self.X+1e-6)*self.Px)@P.T)/(Py+1e-6))
                    elif self.NORM=='L2':
                        self.Y=((self.X*self.Px)@P.T)/(Py+1e-6)
                    else:
                        raise Exception("Wrong Norm!")
                    if np.linalg.norm(self.Y-Y_old)/np.linalg.norm(Y_old)<self.TOL:
                        break
                    if counter>self.MAX_ITER:
                        warnings.warn("MAX ITER REACHED: Y LOOP")
                        break
                    Y_old=self.Y
                    counter=counter+1
            com=(np.count_nonzero(np.abs(P-1)<1e-5)/self.n)
            if (1-com)<self.BETA_TOL:
            #if Beta>self.BETA_TOL:
                time=dt.default_timer()-start
                print(f"Beta Max reached: {Beta} completeness:{com} time:{time}")
                break
            Beta=Beta*self.GROWTH_RATE
            PURTURB=self.PURTURB_RATIO*(np.random.rand(self.d,self.K)-0.5)*(np.max(self.Y)-np.min(self.Y))
            self.Y=self.Y+PURTURB
            if self.VERBOS:
                print(f'Beta: {Beta} completeness:{com}')
        self.P=np.round(P)
        w=np.zeros_like(P)
        for i in range(self.n):
            id=np.where(P[:,i]==1)[0]
            norm1 = np.linalg.norm(self.Y[:,id])
            if norm1>0:
                w[id,i]=(self.XU[:,i].T @ self.Y[:,id])/norm1/norm1
        self.P=w    
        return self.Y,self.P
    def plot(self,size=(12,10),random_color=False):
        plt.figure(figsize=size)
        plt.scatter(self.X[0,:],self.X[1,:],marker='.');plt.grid()
        plt.scatter(self.Y[0,:],self.Y[1,:],marker='*',c='red',linewidths=2)
    def return_cost(self):
        return np.linalg.norm(self.XU-(self.Y@self.P),'fro')/np.linalg.norm(self.XU,'fro')
    
    
    
if __name__=='__main__':
    X=[]
    for i in range(100):
        X.append([0+(-1)**(int(np.random.rand()))*np.random.rand(),4+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([1.5+(-1)**(int(np.random.rand()))*np.random.rand(),2+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-1.5+(-1)**(int(np.random.rand()))*np.random.rand(),2+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-3+(-1)**(int(np.random.rand()))*np.random.rand(),-3+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-4.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([-1.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([3+(-1)**(int(np.random.rand()))*np.random.rand(),-3+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([4.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
        X.append([1.5+(-1)**(int(np.random.rand()))*np.random.rand(),-5+(-1)**(int(np.random.rand()))*np.random.rand()])
    X=(np.vstack(X).T+10)/10
    clus_num=9
    model=DA(clus_num,NORM='KL',GROWTH_RATE=1.5,BETA_INIT=1e-1,BETA_TOL=1e-16,PURTURB_RATIO=0.2,verbos=0)
    model.fit(X);Y,P=model.classify();model.plot()
    print(model.return_cost())