import multiprocessing
import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt


class EA(multiprocessing.Process):
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals,swap_interval,parameter_queue,signal_main,event,island_id):
        
        multiprocessing.Process.__init__(self)
        
        self.parameter_queue = parameter_queue
        self.signal_main = signal_main
        self.event =  event
        self.island_id = island_id

        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        
        self.position = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0] 
        self.velocity = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0]

        self.p_best = copy.copy(self.position)
        self.g_best = sys.float_info.max
        self.num_evals = num_evals

        self.problem = problem
            
        self.swap_interval = swap_interval

        self.p_best_score = self.evaluate()

    def evaluate(self):
        swarm = self.position
        problem = self.problem
        if problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*self.pop_size)
            fit =((swarm**2)*matrix)
            fit = np.sum(fit , axis=1)
            return fit

    def run(self):
        evals=0
        
        while evals<self.num_evals:
            score = self.evaluate()

            for i in range(self.pop_size):

                if score[i]<self.p_best_score[i]:
                    self.p_best[i]=copy.copy(self.position[i])
                    self.p_best_score[i] = score[i]

                if score[i]<self.g_best:
                    self.g_best = score[i]
                    g_best_pos = copy.copy(self.position[i])
        
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            w=0.729       # constant inertia weight (how much to weigh the previous velocity)
            c1=1.4        # cognative constant
            c2=1.4        # social constant

            
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(g_best_pos-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity

            print("Best Score:",self.g_best)
           
            if (evals % self.swap_interval == 0 ): # interprocess (island) communication for exchange of neighbouring best_swarm_pos
                param = g_best_pos
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                print('2')
                # print(evals)
                # if evals!=100000:
                self.event.wait()
                
                print('1')
                result =  self.parameter_queue.get()
                g_best_pos = result 
                self.position[0] = g_best_pos.copy()

            evals+=self.pop_size
            print("Evals=",evals)
   
   
class distributed_EA:
    def __init__(self,pop_size,dim,bounds,problem,num_evals,num_islands):
        
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
        # self.island_queue = multiprocessing.JoinableQueue()	
        self.wait_island = [multiprocessing.Event() for i in range (self.num_islands)]
        self.event = [multiprocessing.Event() for i in range (self.num_islands)]


        self.swap_interval = pop_size #means 1 iteration

    def initialize_islands(self):
        for i in range(self.num_islands):
            self.islands.append(EA(self.pop_size,self.dim,self.bounds,self.problem,self.island_numevals,self.swap_interval,self.parameter_queue[i],self.wait_island[i],self.event[i],i))

    
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
    a = distributed_EA(50,50,[-5,5],2,500000,3) #contraint : pop_size%(max_evals/islands)==0
    a.evolve_islands()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
