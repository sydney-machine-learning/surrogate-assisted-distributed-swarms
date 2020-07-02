import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt 

class EA:
    
    def __init__(self,pop_size,dim,problem,num_evals,bounds,mut,cross_p ):
        
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
        evals=0
        score_list = []
        eval_list = []
        evals = 0
        fitness = self.evaluate(self.position)
        for i in range(int(self.num_evals/self.pop_size)):
            eval_list.append(evals)
            best_score = np.min(fitness)
            score_list.append(best_score)
            print("After",evals,"number of evals best score=",best_score)
            for j in range(self.pop_size):

                idxs = [idx for idx in range(self.pop_size) if idx != j] #[0,1,2,4,5,6...]
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
            evals+=self.pop_size
            if i==int(self.num_evals/self.pop_size)-1 :
                print("After",evals,"number of evals best score=",np.min(fitness))

        plt.plot(eval_list,score_list)
        plt.show()


if __name__ == "__main__":
    start= time.time()

    a = EA(pop_size=20,dim=10,problem=2,num_evals=10000,bounds=[-5,5],mut=0.8,cross_p=0.7)
    a.run()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
