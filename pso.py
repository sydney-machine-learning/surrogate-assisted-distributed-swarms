import numpy as np
import copy
import random
import time
import math

class EA():
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals,exp):
        
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
        # print(pop_size)
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
            
            # score_list.append(self.g_best_score)
            # eval_list.append(self.evals)
            # print("G_BEST_SCORE" ,self.g_best_score ,"after", self.evals )

            # Update rule
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(self.g_best-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity 

            score = np.full((self.pop_size,1), 0 )

            # Evaluation
            for i in range(self.pop_size):

                time.sleep(0.003)
                w_proposal = self.position[i]
                
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
        
        with open(("/home/yash/Desktop/unsw/surr/surrogate/score.txt" ),'ab') as outfile:
            np.savetxt(outfile, [self.g_best_score])
    

if __name__ == "__main__":


    for problem in [1,2,3]:
        for dim in [30,50]:
            for exp in range(1):
                if dim==30:
                    num_evals=100000
                else:
                    num_evals=200000

                start = time.time()
                a = EA(pop_size=100,dim=dim,bounds=[-5,5],problem=problem,num_evals=num_evals,exp=exp)
                a.run()
                min = (time.time()-start)/60
                
                with open(("/home/yash/Desktop/unsw/surr/surrogate/time.txt") ,'ab') as outfile:
                    np.savetxt(outfile, [min])
    
                print('Time Taken= ',(time.time()-start)/60 ,"Minutes")
