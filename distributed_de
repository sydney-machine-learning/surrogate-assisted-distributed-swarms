import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt 
import multiprocessing

class EA(multiprocessing.Process):
    
    def __init__(self,pop_size,dim,problem,num_evals,bounds,mut,cross_p,swap_interval,parameter_queue,signal_main,event,island_id ):
        multiprocessing.Process.__init__(self)


        self.parameter_queue = parameter_queue
        self.signal_main = signal_main
        self.event =  event
        self.island_id = island_id
        self.swap_interval = swap_interval

        self.pop_size = pop_size
        self.dim = dim
        self.num_evals = num_evals
        self.problem = problem
        self.bounds = bounds
        self.pos = np.random.rand(pop_size ,dim)
        self.position = (bounds[1] - bounds[0])*self.pos + bounds[0]
        self.mut = mut
        self.cross_p = cross_p

    def evaluate(self,position):
        pop_size = position.reshape(-1,self.dim).shape[0]
        problem = self.problem
        if problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*pop_size) # [1,2,3,4,5,6,7,8,9,10]
            if pop_size==1:
                fit =((position**2)*matrix[0])
                fit = np.sum(fit)
            else:
                fit =((position**2)*matrix)
                fit = np.sum(fit , axis=1)
            return fit

    def run(self):
    
        evals = 0
        fitness = self.evaluate(self.position)
        for i in range(int(self.num_evals/self.pop_size)):
            for j in range(self.pop_size):

                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = self.pos[np.random.choice(idxs, 3, replace = False)]
                mutant = np.clip(a + self.mut * (b - c), 0, 1)
                cross_points = np.random.rand(self.dim) < self.cross_p
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pos[j])
                trial_denorm = self.bounds[0] + trial*(self.bounds[1]-self.bounds[0])
                f = self.evaluate(trial_denorm)
                if f < fitness[j]:
                    self.pos[j] = trial
                    fitness[j] = f
            
            evals += self.pop_size
            
            best_score = np.min(fitness)

            print("After",evals,"number of evals best score=",best_score)
            
            if (evals % self.swap_interval == 0 ): # interprocess (island) communication for exchange of neighbouring best_swarm_pos
                param = best_score
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result =  self.parameter_queue.get()
                best_score = result 
                self.position[0] = best_score.copy()
                

class distributed_EA:
    def __init__(self,pop_size,dim,bounds,problem,num_evals,mut,cross_p,num_islands):
        
        self.pop_size = pop_size
        self.dim=dim
        self.num_evals = num_evals
        self.bounds= bounds
        self.num_islands = num_islands
        self.problem = problem
        self.mut = mut
        self.cross_p = cross_p
        self.islands = [] 
        self.island_numevals = int(self.num_evals/self.num_islands) 

        # create queues for transfer of parameters between process islands running in parallel 
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_islands)]
        self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
        self.event = [multiprocessing.Event() for i in range (self.num_islands)]


        self.swap_interval = pop_size #means 1 iteration

    def initialize_islands(self):
        for i in range(self.num_islands):
            self.islands.append(EA(self.pop_size,self.dim,self.problem,self.island_numevals,self.bounds,self.mut,self.cross_p,self.swap_interval,self.parameter_queue[i],self.wait_island[i],self.event[i],i))

    
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
        #SWAP PROCEDURE

        swaps_appected_main =0
        total_swaps_main =0
        for i in range(int(self.island_numevals/self.swap_interval)):
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

            for index in range(0,self.num_islands-1): 
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
        # self.island_queue.join()

if __name__ == "__main__":
    start= time.time()
    
    a = distributed_EA(10,50,[-5,5],2,50000,0.8,0.7,3)  #pop_size,dim,bounds,problem,num_evals,mut,cross_p,num_islands
    a.evolve_islands()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
