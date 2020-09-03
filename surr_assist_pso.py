import multiprocessing
import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt
import math
import os

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F

class model(nn.Module):

  def __init__(self , in_size , h_size):
    super().__init__()

    self.fc1=nn.Linear(in_features=in_size, out_features=h_size)
    self.fc2=nn.Linear(in_features=h_size, out_features=int(h_size/2))
    self.out=nn.Linear(in_features=int(h_size/2), out_features=1)
  
  def forward(self, t):

    t=self.fc1(t)
    t=F.relu(t)
    t=F.dropout(t , p=0.5)
    t=self.fc2(t)
    t=F.relu(t)
    t=F.dropout(t , p=0.5)
    t=self.out(t)
    t=torch.sigmoid(t)

    return t

class dataset_class(data.Dataset):
    def __init__(self, particle , fitness ):
        self.particle = particle
        self.fitness = fitness

    def __getitem__(self, index):

        return self.particle[index], self.fitness[index] 

    def __len__(self):
        return (self.particle.shape[0])


class Surrogate():
    def __init__(self, in_size , h_size):
        
        self.in_size = in_size
        self.h_size = h_size

        self.surrogate_model = model(in_size , h_size)


    def trainer(self, particle , fitness , batch_size ):
            
        dataset = dataset_class(particle , fitness)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size= batch_size,drop_last=True, shuffle=True )

        criterion = nn.MSELoss()
        optimizer= optim.Adam(self.surrogate_model.parameters(),lr=0.001,betas=(0.5, 0.999))

        self.surrogate_model.train()
        print("Training start")
        start = time.time()
        num_epoch=20

        for epoch in range(num_epoch):

            for i, data in enumerate(dataloader, 0):

                self.surrogate_model.zero_grad()

                particle , fitness = data
                particle , fitness = particle.type(torch.FloatTensor) , fitness.type(torch.FloatTensor)
                model_out = self.surrogate_model(particle)

                err = criterion(model_out, fitness.reshape(-1,1))
                
                err.backward()

                optimizer.step()

        torch.save(self.surrogate_model.state_dict(), "/home/yash/Desktop/unsw/model" )
        
        
        out = self.surrogate_model(particle)
        error = criterion(out , fitness.reshape(-1,1))
        print("Training done in",(time.time()-start)/60,"mins Error=",error)
        # save error

    def predict(self,particle):
        
        self.surrogate_model.eval()
        particle = torch.tensor(particle).type(torch.FloatTensor)
        return self.surrogate_model(particle)


class surrogate_EA(multiprocessing.Process):
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals,swap_interval,parameter_queue,signal_main,event,island_id, model , normaliser_list , surrogate_parameter_queues,surrogate_start,surrogate_resume,surrogate_interval,surrogate_prob,compare_surrogate,path,exp):
        
        multiprocessing.Process.__init__(self)
        
        # Island variables
        self.parameter_queue = parameter_queue
        self.signal_main = signal_main
        self.event =  event
        self.island_id = island_id
        self.swap_interval = swap_interval

        # PSO variables
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.num_evals = num_evals
        
        self.position = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0] 
        self.velocity = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0]
        
        self.problem = problem
        self.p_best = copy.copy(self.position)
        self.p_best_score = np.zeros((self.pop_size,))
        for i in range(self.pop_size):
            self.p_best_score[i] = self.evaluate(self.position[i])

        self.exp = exp
        
        self.g_best_score = 999999 #initialize
        self.g_best = copy.copy(self.p_best[0]) #initialize

        for i in range(self.pop_size)  :
            if self.p_best_score[i]<self.g_best_score:
                self.g_best_score = copy.copy(self.p_best_score[i])
                self.g_best = copy.copy(self.position[i])

        # Surrogate Variables
        self.surrogate_parameter_queue = surrogate_parameter_queues
        self.ev22 = surrogate_start
        self.ev1 = surrogate_resume
        self.surrogate_interval = surrogate_interval
        self.surrogate_prob = surrogate_prob
        self.compare_surrogate = compare_surrogate
        self.path = path
        self.surg_fit_list = [np.zeros((int(self.num_evals/10) ,3)) for k in range(self.pop_size)]
        self.index_list = [0 for k in range(self.pop_size)]

        self.model = model
        normaliser_list.append(self.g_best_score)
        self.normaliser_list = normaliser_list
        
    def evaluate(self,position):
        pop_size = position.reshape(-1,self.dim).shape[0]
        problem = self.problem
        
        if self.problem == 1: # rosenbrock
            
            if pop_size==1:
                fit=0
                # print('1',position)
                for j in range(self.dim -1):
                    fit += (100.0*(position[j]**2 - position[j+1])**2 + (position[j]-1.0)**2)      
        
        elif problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*pop_size)
            if pop_size==1:
                fit =((position**2)*matrix[0])
                fit = np.sum(fit)
            else:
                fit =((position**2)*matrix)
                fit = np.sum(fit , axis=1)
        

        elif self.problem ==3:  # rastrigin's function
            fit = [10*self.dim for i in range(pop_size)]
            if pop_size==1:
                fit = fit + np.sum( position**2-10*np.cos(2*math.pi*position) ) 
            else:   
                fit = fit + np.sum(position**2-10*np.cos(2*math.pi*position) , axis=1 ) 
   
        elif self.problem ==4: #styblinski-tang function
            if pop_size==1:
                fit = np.sum( (position**4 - 16*(position**2) + 5*position)/2 ) 
            else:
                fit = np.sum( (position**4 - 16*(position**2) + 5*position)/2 , axis=1  ) 

        return fit


    def run(self):
        if not os.path.exists('%s/surrogate/%s%s%s%s' % (self.path ,'problem=',self.problem,'dim=',self.dim)):
            os.mkdir('%s/surrogate/%s%s%s%s' % (self.path ,'problem=',self.problem,'dim=',self.dim))
            
        if not os.path.exists('%s/surrogate/%s%s%s%s/%s' % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp )):
            os.mkdir('%s/surrogate/%s%s%s%s/%s' % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ))
            
        if not os.path.exists('%s/surrogate/%s%s%s%s/%s/%s' % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id)):
            os.mkdir('%s/surrogate/%s%s%s%s/%s/%s' % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id))

        trainset_empty = True
        is_true_fit = True
        surr_train_set = np.zeros((10000, self.dim+1))

        self.surrogate_metric = np.nan

        self.evals=0

        self.a=0
        self.b=0
        self.c=0
        self.d=0

        count_real = 0
        idx = 0 
        start = 0
        print("SIZE", len(self.normaliser_list))
        normaliser = max(self.normaliser_list)
        print(normaliser)
        score_list = []
        eval_list = []
        self.sp = []
        self.sp1 = []
        while self.evals<self.num_evals:
        
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            w=0.729       # constant inertia weight (how much to weigh the previous velocity)
            c1=1.4        # cognative constant
            c2=1.4        # social constant
            
            score_list.append(self.g_best_score)
            eval_list.append(self.evals)
            if self.evals!=0:            
                print("G_BEST_SCORE" ,self.g_best_score ,"on", self.island_id,"island", self.evals , (time.time()-start )/60 , ts+tt+tu+tc ,"surr" , ts ,"surr p",tsp/60,"true"  , tt ,"t update", tu ,"t chain" , tc ,"extra evals",self.a )

            start = time.time()

            # self.event.clear()

            # Update rule
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(self.g_best-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity 

            score = np.full((self.pop_size,1), 0 )

            # Evaluation
            ts,tt,tu,tc = 0,0,0,0
            tsp = 0
            for i in range(self.pop_size):

                w_proposal = self.position[i]
                
                ku = random.uniform(0,1)
                if ku<self.surrogate_prob and self.evals >= self.surrogate_interval+1 :

                    ts1 = time.time()
                    is_true_fit = False

                    tsp1 = time.time()
                    surrogate_pred = self.model.predict( w_proposal.reshape(-1,self.dim) )
                    surrogate_pred = surrogate_pred.detach().numpy()[0][0]
                    self.sp.append(surrogate_pred)
                    
                    tsp+=time.time()-tsp1

                    surr_mov_ave = ((self.surg_fit_list[i])[self.index_list[i],2] + (self.surg_fit_list[i])[self.index_list[i] - 1,2]+ (self.surg_fit_list[i])[self.index_list[i] - 2,2])/3
                    surr_proposal = (surrogate_pred*0.5 + surr_mov_ave*0.5)

                    self.sp1.append(surr_proposal)

                    if self.compare_surrogate is True:
                        fitness_proposal_true = self.evaluate(w_proposal)
                    else:
                        fitness_proposal_true = 0

                    (self.surg_fit_list[i])[self.index_list[i]+1,0] =  fitness_proposal_true/normaliser
                    (self.surg_fit_list[i])[self.index_list[i]+1,1]= surr_proposal
                    (self.surg_fit_list[i])[self.index_list[i]+1,2] = surr_mov_ave
                    ts = ts+ (time.time() - ts1)/60
                else:
                    tt1 = time.time()
                    is_true_fit = True
                    trainset_empty = False
                    (self.surg_fit_list[i])[self.index_list[i]+1,1] =  np.nan
                    surr_proposal = self.evaluate(w_proposal)/normaliser
                    fitness_arr = np.array([surr_proposal])
                    X, Y = w_proposal,fitness_arr
                    X = X.reshape(1, X.shape[0])
                    Y = Y.reshape(1, Y.shape[0])
                    param_train = np.concatenate([X, Y],axis=1)
                    (self.surg_fit_list[i])[self.index_list[i]+1,0] = surr_proposal
                    (self.surg_fit_list[i])[self.index_list[i]+1,2] = surr_proposal
                    surr_train_set[count_real, :] = param_train
                    count_real = count_real +1
                    tt = tt+ (time.time() - tt1)/60

                error = surr_proposal*normaliser
                score[i] = surr_proposal*normaliser
                
                tu1 = time.time()                
                if error < self.p_best_score[i]:
                    if is_true_fit==False :
                        self.a+=1

                        actual_err = self.evaluate(w_proposal)
                        
                        if actual_err<self.p_best_score[i] :
                            self.b+=1
                            self.p_best_score[i] = actual_err
                            self.p_best[i] = copy.copy(self.position[i])
                        
                    else:
                        self.p_best_score[i] = error
                        self.p_best[i] = copy.copy(self.position[i])

                if error < self.g_best_score:
                    
                    if is_true_fit==False :
                        self.c+=1
                        if actual_err<self.g_best_score:
                            self.d+=1
                            self.g_best_score = actual_err
                            self.g_best = copy.copy(self.position[i])   

                    else:        
                        self.g_best_score = error
                        self.g_best = copy.copy(self.position[i])   

                self.index_list[i] += 1  
                tu = tu+ (time.time()-tu1)/60

            #SWAPPING PREP
            
            if self.evals%self.swap_interval==0 and self.evals!=0 :
                tc1 = time.time()
                a = np.concatenate([self.position , score] , axis=1)
                a = a[a[:,1].argsort()][:,:-1]
                param = a[:(int(0.2*self.pop_size))]
                self.parameter_queue.put(param)
                
                surr_train = surr_train_set[0:count_real, :]

                self.surrogate_parameter_queue.put(surr_train)
                
                self.signal_main.set()
                self.event.clear()
                self.event.wait()

                result =  self.parameter_queue.get()
                best_swarm_pos = result 
                self.position[(int(0.8*self.pop_size)):] = best_swarm_pos.copy()


                trainset_empty = True 
                count_real = 0      
                tc= tc + (time.time() -tc1)/60

                self.model.surrogate_model.load_state_dict(torch.load("/home/yash/Desktop/unsw/model"))

            self.evals += self.pop_size


        
        with open(("%s/surrogate/sp.txt" % (self.path)),'ab') as outfile:
           np.savetxt(outfile, self.sp)
           
        with open(("%s/surrogate/sp1.txt" % (self.path)),'ab') as outfile:
           np.savetxt(outfile, self.sp1)

        # os.makedirs("%s/surrogate/%s%s%s%s/%s/%s" % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id))
        
    #    a = np.zeros((1,3))
    #    for i in range(self.pop_size):
    #        a  = np.concatenate((self.surg_fit_list[i] , a))
    #    b = a[~np.isnan(a).any(axis=1)]
    #    c = b[~np.all(b == 0, axis=1)]
    #    rmse = np.sqrt(np.mean((c[:,0]-c[:,1])**2))

    #    with open(("%s/surrogate/%s%s%s%s/%s/%s/surr_metric.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id)),'ab') as outfile:
    #        np.savetxt(outfile, [rmse])
       
    #    with open(("%s/surrogate/%s%s%s%s/%s/%s/surrogate_perf.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id)),'ab') as outfile:
    #        np.savetxt(outfile, c)

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/score_list.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, score_list)

        with open(("%s/surrogate/%s%s%s%s/%s/score_list.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, self.exp )),'ab') as outfile:
            np.savetxt(outfile, [self.g_best_score])

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/extra_evals.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, [self.a , self.b])

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/extra_evals.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, [self.c , self.d])


        # print("Best particle:", self.g_best , 'of island', self.island_id)
 
        # plt.plot(eval_list , score_list , label = self.island_id)
        # plt.show()

        # print("Island: {} chain dead!".format(self.island_id))
        # self.signal_main.set()
        return


class distributed_surrogate_EA:
    def __init__(self,pop_size,dim,bounds,problem,num_evals,num_islands,interval,compare_surrogate,path,exp):
        
        self.pop_size = pop_size
        self.dim=dim
        self.num_evals = num_evals
        self.bounds= bounds
        self.num_islands = num_islands
        self.problem = problem
        self.islands = [] 
        self.island_numevals = int(self.num_evals/self.num_islands)

        # create queues for transfer of parameters between process islands running in parallel 
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_islands)]
        self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
        self.event = [multiprocessing.Event() for i in range (self.num_islands)]

        self.swap_interval = interval*self.pop_size

        # Surrogate Variables
        self.surrogate_interval = interval*self.pop_size
        self.surrogate_prob = 0.5
        self.ev1 = [multiprocessing.Event() for i in range(self.num_islands)]
        self.ev2 = [multiprocessing.Event() for i in range(self.num_islands)]
        self.surrogate_parameter_queues = [multiprocessing.Queue() for i in range(self.num_islands)]
        self.surrchain_queue = multiprocessing.JoinableQueue()
        self.compare_surrogate = compare_surrogate
        self.path = path
        self.folder = path
        self.total_swap_proposals = 0
        self.num_swaps = 0
        self.exp =exp

        self.model = Surrogate(self.dim , int(0.8*self.dim))
        self.normaliser_list = []

    def initialize_islands(self):
        for i in range(self.num_islands):
            self.islands.append(surrogate_EA(self.pop_size,self.dim,self.bounds,self.problem,self.island_numevals,self.swap_interval,self.parameter_queue[i],self.wait_island[i],self.event[i],i,self.model,self.normaliser_list,self.surrogate_parameter_queues[i],self.ev2[i],self.ev1[i],self.surrogate_interval,self.surrogate_prob,self.compare_surrogate,self.path,self.exp))

    def swap_procedure(self, parameter_queue_1, parameter_queue_2): 

            param1 = parameter_queue_1.get()
            param2 = parameter_queue_2.get()
            u = np.random.uniform(0,1)
            self.swap_proposal=1
            swapped = False
            if u < self.swap_proposal:   
                param_temp =  param1
                param1 = param2
                param2 = param_temp
                swapped = True 
            else:
                swapped = False 
            return param1, param2 ,swapped


    def evolve_islands(self): 
        
        self.initialize_islands()

        for j in range(0,self.num_islands):        
            self.wait_island[j].clear()
            self.event[j].clear()
            self.islands[j].start()
        surr_train_time = 0
        swaps_appected_main =0
        total_swaps_main =0
        a = int(self.island_numevals/self.swap_interval) 

        if a == self.island_numevals/self.swap_interval :
            a = a - 1


        for i in range(a):   #3000/1000 = 3 , 3-1=2
            # print(i)

            count = 0
            for index in range(self.num_islands):
                if not self.islands[index].is_alive():
                    count+=1
                    self.wait_island[index].set() 

            if count == self.num_islands:
                break

            timeout_count = 0
            for index in range(0,self.num_islands): 
                flag = self.wait_island[index].wait()

                if flag: 
                    timeout_count += 1
            
            if timeout_count != self.num_islands: 
                continue
            
            if timeout_count== self.num_islands :
                
                for index in range(0,self.num_islands-1):
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                all_param =   np.empty((1,self.dim+1))
                
                for index in range(0,self.num_islands):
                    queue_surr=  self.surrogate_parameter_queues[index] 
                    surr_data = queue_surr.get() 
                    all_param =   np.concatenate([all_param,surr_data],axis=0)

                data_train = all_param[1:,:]
                start = time.time()

                print("surrogate training started")
                self.model.trainer(data_train[:,:-1] , data_train[:,-1] , 50)
                
                end = time.time()
                surr_train_time+= (end-start)/60
                
    

            for index in range (self.num_islands):
                    self.event[index].set()
                    self.wait_island[index].clear() 

        
        for index in range(0,self.num_islands):
            self.islands[index].join()

        for i in range(0,self.num_islands):
            self.parameter_queue[i].close()
            self.parameter_queue[i].join_thread()
            self.surrogate_parameter_queues[i].close()
            self.surrogate_parameter_queues[i].join_thread()

        print("Surrogate Train time:",surr_train_time,"Mins")

if __name__ == "__main__":

    for problem in [1]:
        for dim in [30]:
            for exp in range(1):
                if dim==30:
                    num_evals=100000
                else:
                    num_evals=200000

                start = time.time()
                a = distributed_surrogate_EA(pop_size=100,dim=dim,bounds=[-5,5],problem=problem,num_evals=num_evals,num_islands=8,interval=10,compare_surrogate=False,path='/home/yash/Desktop/unsw/surr',exp=exp)
                a.evolve_islands()
                min = (time.time()-start)/60
                with open(("/home/yash/Desktop/unsw/surr/surrogate/%s%s%s%s/%s/time.txt" % ('problem=',problem,'dim=',dim, exp )),'ab') as outfile:
                    np.savetxt(outfile, [min])

                print('Time Taken= ',(time.time()-start)/60 ,"Minutes")
