import multiprocessing
import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt
import math
import os

class EA(multiprocessing.Process):
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals,swap_interval,island_swap,parameter_queue,signal_main,event,island_id,exp):
        
        multiprocessing.Process.__init__(self)
        
        # Island variables
        self.parameter_queue = parameter_queue
        self.signal_main = signal_main
        self.event =  event
        self.island_id = island_id
        self.swap_interval = swap_interval
        self.island_swap = island_swap
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
        
        self.g_best_score = 999999 #initialize
        self.g_best = self.p_best[0]  #initialize

        for i in range(self.pop_size)  :
            if self.p_best_score[i]<self.g_best_score:
                self.g_best_score = self.p_best_score[i]
                self.g_best = copy.copy(self.position[i])
        self.exp = exp


    def evaluate(self,position):
        pop_size = position.reshape(-1,self.dim).shape[0]
        problem = self.problem
        
        if self.problem == 1: # rosenbrock
            
            if pop_size==1:
                fit=0
                for j in range(self.dim -1):
                    fit += (100.0*(position[j]**2 - position[j+1])**2 + (position[j]-1.0)**2)
        
        elif problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*pop_size) # [1,2,3,4,5,6,7,8,9,10]
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

        score_list = []
        eval_list = []
        self.evals = 0
        while self.evals<self.num_evals:
        
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            w=0.729       # constant inertia weight (how much to weigh the previous velocity)
            c1=1.4        # cognative constant
            c2=1.4        # social constant
            
            score_list.append(self.g_best_score)
            eval_list.append(self.evals)
            # print("G_BEST_SCORE" ,self.g_best_score ,"on", self.island_id,"island", self.evals )

            # self.event.clear()

            # Update rule
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(self.g_best-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity 

            score = np.full((self.pop_size,1), 0 )

            # Evaluation
            for i in range(self.pop_size):
               
                w_proposal = self.position[i]

                time.sleep(0.003)

                fit = self.evaluate(w_proposal)
                self.evals+=1
                
                error = fit
                score[i] = fit

                if error < self.p_best_score[i]:
                  self.p_best_score[i] = error
                  self.p_best[i] = copy.copy(self.position[i])

                if error < self.g_best_score:
                  self.g_best_score = error
                  self.g_best = copy.copy(self.position[i])   


            if self.evals % self.swap_interval == 0 : # interprocess (island) communication for exchange of neighbouring best_swarm_pos
                a = np.concatenate([self.position , score] , axis=1)
                a = a[a[:,1].argsort()][:,:-1]
                param = a[:(int(0.2*self.pop_size))]
                self.parameter_queue.put(param)
                
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result =  self.parameter_queue.get()
                best_swarm_pos = result 
                self.position[(int(0.8*self.pop_size)):] = best_swarm_pos.copy()

        os.makedirs("%s/surrogate/%s%s%s%s/%s/%s" % ('/home/yash/Desktop/unsw/surr' ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id))
        
        # # with open(("%s/surrogate/%s%s%s%s/%s/%s/surrogate_perf.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        # #   if self.use_surrogate:
        # #         np.savetxt(outfile, self.surg_fit_list[self.island_id])

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/score_list.txt" % ('C:/Users/admin/Desktop/unsw/surr' ,'problem=',self.problem,'dim=',self.dim, self.exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, score_list)

        with open(("%s/surrogate/%s%s%s%s/%s/%s/score.txt" % ('/home/yash/Desktop/unsw/surr' ,'problem=',self.problem,'dim=',self.dim, self.exp ,self.island_id)),'ab') as outfile:
          np.savetxt(outfile, [self.g_best_score])

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/extra_evals.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, [self.a , self.b])

        # with open(("%s/surrogate/%s%s%s%s/%s/%s/extra_evals.txt" % (self.path ,'problem=',self.problem,'dim=',self.dim, exp , self.island_id)),'ab') as outfile:
        #   np.savetxt(outfile, [self.c , self.d])
        
        self.signal_main.set()

class distributed_EA:
    def __init__(self,pop_size,dim,bounds,problem,num_evals,num_islands,interval,island_swap,exp):
        
        self.pop_size = pop_size
        self.dim=dim
        self.num_evals = num_evals
        self.bounds= bounds
        self.num_islands = num_islands
        self.problem = problem
        self.islands = [] 
        self.island_numevals = int(self.num_evals/self.num_islands) 
        self.island_swap = island_swap

        # create queues for transfer of parameters between process islands running in parallel 
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_islands)]
        self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
        self.event = [multiprocessing.Event() for i in range (self.num_islands)]

        self.swap_interval = interval*self.pop_size
        self.exp = exp

    def initialize_islands(self):
        for i in range(self.num_islands):
            self.islands.append(EA(self.pop_size,self.dim,self.bounds,self.problem,self.island_numevals,self.swap_interval,self.island_swap,self.parameter_queue[i],self.wait_island[i],self.event[i],i,self.exp))


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

        swaps_appected_main =0
        total_swaps_main =0
        
        for i in range(int(self.island_numevals/self.swap_interval)):
            count = 0
            for index in range(self.num_islands):
                if not self.islands[index].is_alive():
                    count+=1
                    self.wait_island[index].set() 

            if count == self.num_islands:
                print("breaking")
                break 

            timeout_count = 0
            
            for index in range(0,self.num_islands): 
                flag = self.wait_island[index].wait()
                
                if flag: 
                    timeout_count += 1

                    
            if timeout_count != self.num_islands: 
                print("continuing")
                continue

            for index in range(0,self.num_islands-1): 
                # print('1')
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index+1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_appected_main += 1
                    total_swaps_main += 1
            for index in range (self.num_islands):
                    self.event[index].set()
                    self.wait_island[index].clear() 


        for index in range(0,self.num_islands):
            self.islands[index].join()

        for i in range(0,self.num_islands):
            self.parameter_queue[i].close()
            self.parameter_queue[i].join_thread()

if __name__ == "__main__":


    for problem in [1,2,3]:
        for dim in [30,50]:
            for exp in range(1):
                if dim==30:
                    num_evals=100000
                else:
                    num_evals=200000

                start = time.time()
                a = distributed_EA(pop_size=100,dim=dim,bounds=[-5,5],problem=problem,num_evals=num_evals,num_islands=8,interval=1,island_swap=True,exp=exp)
                a.evolve_islands()
                min = (time.time()-start)/60
                
                with open(("/home/yash/Desktop/unsw/surr/surrogate/%s%s%s%s/%s/time.txt" % ('problem=',problem,'dim=',dim, exp )),'ab') as outfile:
                    np.savetxt(outfile, [min])
    
                print('Time Taken= ',(time.time()-start)/60 ,"Minutes")
