#!/usr/bin/env python
# coding: utf-8

#___________________In Coding We Believe___________________
#___________________Salar Basiri___________________________
#___________________Deterministic Annealing________________

#___________________Importing Libraries___________________________
import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib as matlib
import warnings
import timeit as dt
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
import plotly.express as px

def divide (N, P):
    out = np.copy(N)
    np.divide(out, P, where = np.logical_not(np.isclose(P, 0)), out = out)
    return out

def flatten(t):
    '''
    this finction is used for the purpose of flattening the Betas
    and data points to be ploted in the animation
    '''
    return [item for sublist in t for item in sublist]

def bifurcate(x,y,p,px,py,beta,purturb,patience=0.5):
    '''
    this finction decides whether any codevectore needs to split.
    it chooses the codevectore with biggest covarience eigenvalue,
    and if beta>1/(2*lambda_max) it splits the codevector in the 
    direction of the corrosponding eigenvector.
    '''
    status=False #whether any codevetor needs to split or not
    xx=np.repeat(x[:,:,np.newaxis],y.shape[1],axis=2)
    yy=np.repeat(y[:,np.newaxis,:],x.shape[1],axis=1)
    #pxy=p/np.repeat(py[:,np.newaxis],x.shape[1],axis=1)*np.repeat(px[np.newaxis,:],y.shape[1],axis=0)
    pxy=p*np.repeat(px[np.newaxis,:],y.shape[1],axis=0)
    pp=np.repeat(pxy.T[np.newaxis,:,:],x.shape[0],axis=0)
    z=xx-yy
    out=np.moveaxis(z.T,1,2)@(pp*z).T
    eigs=[eigh(out[i,:,:],subset_by_index=(x.shape[0]-1,x.shape[0]-1)) for i in range(out.shape[0])]
    l=[beta-patience/eig[0] for eig in eigs]
    i=np.argmax(l)
    if l[i]<=0:
        return y,py,False
    direction=eigs[i][1]
    PURTURB=purturb*direction
    u=y[:,i][:,None]+PURTURB
    #y[:,i]=y[:,i]-PURTURB[:,0]
    y_h=np.insert(y,i,u[:,0],axis=1)
    py_d=py.copy()
    py_d[i]=py_d[i]/2
    py_h=np.insert(py_d,i,py_d[i])
    if y_h.shape!=y.shape:
        #print(f"\nDevision occured: from {y.shape[1]} to {y_h.shape[1]} at {beta}\n")
        status=True
    return y_h,py_h,status


class DA:
    def __init__(self,K,NORM='L2',TOL=1e-4,MAX_ITER=1000,GROWTH_RATE=1.05,
              PURTURB_RATIO=0.01,PATIENCE=0.5,BETA_FINAL=None,verbos=0,NORMALIZE=False):
        '''
         K: number of clusters Norm: the type of norm applied as the distance measure
         TOL: The relative tolerence applied as the stopping criteria
         Max_ITER: maximum number of iterations applied as the stopping criteria
         PURTURB_RATIO: the magnitude of purturbation after each split
        '''
        self.K=K;self.TOL=TOL;self.MAX_ITER=MAX_ITER;
        self.GROWTH_RATE=GROWTH_RATE;self.PURTURB_RATIO=PURTURB_RATIO;self.BETA_FINAL=BETA_FINAL
        self.CONSTRAINED=False;self.VERBOS=verbos;self.NORM=NORM;self.NORMALIZE=NORMALIZE
        self.PATIENCE=PATIENCE

    #___________________Fitting the model on data___________________

    def fit(self,X,**kwargs):
        '''
         **kwargs:
         px: data points probability array
         lambdas: for the constrained codevectors probability array
        '''
        flag=False
        m, self.n = np.shape(X)
        self.d=m
        l = np.sum(X * X, axis = 0, keepdims = True);l_sqrt = np.sqrt(l);
        weights=l[0]
        weights=weights/weights.sum()
        if self.NORMALIZE:
            self.X = divide (X, l_sqrt)
        else:
            self.X=X;
               
        if 'Px' in kwargs:
            if kwargs['Px']=='auto':
                self.Px=weights
                flag=True
            else:
                self.Px=kwargs['Px']
        else:
            self.Px=np.array([1 for i in range(self.n)])/self.n
        if not (flag == self.NORMALIZE):
            print('!!--Normalization without auto weighting on Px--!!')
        self.Y=np.repeat((self.X@self.Px).reshape(m,1),1,axis=1)
        self.Py=np.array([1]) #all the points belong to this cluster
        if 'Lambdas' in kwargs:
            self.CONSTRAINED=True
            self.Lambdas=kwargs['Lambdas']
            if np.sum(self.Lambdas) != 1.0:
                warnings.warn('Lambdas should sum up to one')
        else:
            self.Lambdas=np.array([1 for i in range(self.K)])/self.K
        self.Ethas=self.Lambdas
        #self.BETA_INIT=0.5/(2*np.max(np.abs(eigh((X*self.Px)@X.T, eigvals_only=True))))
        self.BETA_INIT=0.49/eigh((self.X*self.Px)@self.X.T,eigvals_only=True,subset_by_index=(m-1,m-1))[0]
        
    

    def classify(self):
  
        y_list=[] # a list to save codevectors over betas
        beta_list=[] # list of all betas
        y_list.append(self.Y);beta_list.append(0)
        start=dt.default_timer()
        k_n=1 # at first we just have one codevector
        Y_old=np.random.rand(self.d,k_n*2)*1e6;P_old=np.random.rand(2,2);
        beta_devide=[] #list to store critical betas
        cost_l=[] #list to store all costs
        Beta=self.BETA_INIT
        if self.VERBOS:
            print(f'Classification started at Beta:{Beta}')
        while True: #beta loop
            counter=1
            while True: #y p etha loop
                self.Data_points=np.repeat(self.X[:,:,np.newaxis],self.Y.shape[1],axis=2)
                Cluster_points=np.repeat(self.Y[:,np.newaxis,:],self.n,axis=1)
                if self.NORM=='KL':
                    d=np.log(Cluster_points/(self.Data_points+1e-6))
                    D=d*Cluster_points-Cluster_points+self.Data_points
                elif self.NORM=='L2':
                    D=(self.Data_points-Cluster_points)**2
                else:
                    raise Exception("Wrong Norm!")
                D=np.sum(D,axis=0).T
                D=D-np.min(D,axis=0)
                delta=np.exp(-D*Beta)
                counter2=1
                if self.CONSTRAINED:
                    print('under construction!')
                else:  
                    while True:  
                        p=(delta.T*self.Py).T
                        P=p/p.sum(axis=0)
                        self.Py=P@self.Px
                        if P.shape==P_old.shape:
                            if np.linalg.norm(P-P_old)/np.linalg.norm(P_old)<self.TOL:
                                break
                        if counter>self.MAX_ITER:
                            warnings.warn("MAX ITER REACHED: py LOOP")
                            break
                        P_old=P
                        counter2=counter2+1
                    if self.NORM=='KL':
                        self.Y=np.exp(((np.log(self.X+1e-6)*self.Px)@P.T)/(self.Py+1e-6))
                    elif self.NORM=='L2':
                        self.Y=((self.X*self.Px)@P.T)/(self.Py)
                    else:
                        raise Exception("Wrong Norm!")
                    
                    if self.Y.shape==Y_old.shape:
                        if np.linalg.norm(self.Y-Y_old)/np.linalg.norm(Y_old)<self.TOL:
                            break
                    if counter>self.MAX_ITER:
                        warnings.warn("MAX ITER REACHED: Y LOOP")
                        break
                    Y_old=self.Y
                    counter=counter+1
            
            com=(np.count_nonzero(np.abs(P-1)==0)/self.n)
            beta_list.append(Beta)
            y_list.append(self.Y)
            cost=np.linalg.norm(self.X-(self.Y@P))/np.linalg.norm(self.X)
            cost_l.append(cost)
            if (not (self.BETA_FINAL is None)) and Beta<1e100:
              if Beta>self.BETA_FINAL and k_n==self.K:
                time=dt.default_timer()-start
                print(f"Beta Max reached: {Beta} completeness:{com} time:{time}")
                Beta=1e100
            else:
              if (1-com)<1e-18 and k_n==self.K:
              #if not np.count_nonzero([list(i).count(1.0)-1.0 for i in np.split(P,P.shape[1],axis=1)]) and k_n==self.K:
                time=dt.default_timer()-start
                print(f"stopping criteria met at Beta: {Beta} cost:{cost} time:{time}")
                break
            
            if self.VERBOS:
                print(f'Beta: {Beta} cost:{cost}')
            
            
            Beta=Beta*self.GROWTH_RATE
            if k_n<self.K:
                self.Y,self.Py,status=bifurcate(self.X,self.Y,P,self.Px,self.Py,Beta,self.PURTURB_RATIO,self.PATIENCE)
                if self.VERBOS:
                    print(f"\nDevision occured: to {self.Y.shape[1]} at {Beta}\n")
                k_n=self.Y.shape[1]
                if status:
                  beta_devide.append(Beta)
                
            
        self.y_list=y_list;self.cost_list=cost_l
        self.beta_list=beta_list
        self.beta_devide=beta_devide
        self.P=np.round(P)
        return self.Y,self.P
    def plot(self,size=(12,10)):
        plt.figure(figsize=size)
        plt.scatter(self.X[0,:],self.X[1,:],marker='.',color='black');plt.grid()
        plt.scatter(self.Y[0,:],self.Y[1,:],marker='*',c='red',linewidths=2)
    def return_cost(self):
        return np.linalg.norm(self.X-(self.Y@self.P),'fro')/np.linalg.norm(self.X,'fro')
    def pie_chart(self):
        labels=['Y_'+str(i) for i in range(self.Py.shape[0])]
        explode=np.zeros_like(self.Py)
        explode[np.argmin(self.Py)]=0.5
        fig1, ax1 = plt.subplots()
        ax1.pie(self.Py, explode=explode,labels=labels,autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title("Cluster Masses %")
        plt.show()
    def animation(self):
        xx=np.array(flatten([list(i[0,:]) for i in self.y_list]))
        yy=np.array(flatten([list(i[1,:]) for i in self.y_list]))
        Betas=[]
        for i , y in enumerate(self.y_list):
            for j in range(y.shape[1]):
                Betas.append(self.beta_list[i])
        df = px.data.gapminder()
        fig=px.scatter(x=xx, y=yy,animation_frame=Betas,
                   log_x=False)
        fig.show()
    def mesh_plot(self,size=(12,10)):
      plt.figure(figsize=size)
      for i in range(self.K):
          J=np.where(1-self.P[i,:]<=1e-5)
          plt.scatter(self.X[0,J],self.X[1,J],color=np.random.rand(3))
    def plot_criticals(self,log=False,size=(6,4)):
        plt.figure(figsize=size)
        if log:
            plt.scatter(range(1,self.K),np.log(self.beta_devide));plt.grid();plt.xticks(np.arange(1, self.K, 1.0))
            plt.ylabel('log(Beta)')
        else:
            plt.scatter(range(1,self.K),(self.beta_devide));plt.grid();plt.xticks(np.arange(1, self.K, 1.0))
            plt.ylabel('Beta')
        plt.title('Critical Betas over number of clusters')
        plt.xlabel('Cluster Number')

    def return_true_number(self):
      return np.argmax([(self.beta_devide[i+1]/self.beta_devide[i]) for i in range(len(self.beta_devide)-1)])+2
