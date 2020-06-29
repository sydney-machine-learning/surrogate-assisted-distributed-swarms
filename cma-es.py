import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt


class EA:
    
    def __init__(self,pop_size,dim,problem,num_evals,bounds,mu ):

        self.pop_size = pop_size
        self.dim = dim
        self.mu = mu 
        self.num_evals = num_evals
        self.problem = problem
        self.bounds = bounds
        self.position = (bounds[1] - bounds[0])*np.random.rand(pop_size ,dim) + bounds[0]
        
        self.mean = None
        self.covar= None


    def evaluate(self):
        
        problem = self.problem
        if problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*self.pop_size)
            fit =((self.position**2)*matrix)
            fit = np.sum(fit , axis=1)
            return fit

    def run(self):
        
        evals=0
        score_list = []
        eval_list = []
        while evals<self.num_evals:

            score = self.evaluate()
            score= score.reshape(-1,1)
            metric = np.concatenate((self.position,score),axis=1)
            metric =metric[metric[:,-1].argsort()]
            score = metric[:,-1]
    
            self.position = metric[:,:-1]
            print("Best Score:", score[0])
            # print("At:", self.position[0])
            self.mean = np.mean(self.position[:int(self.mu*self.pop_size)],axis=0)
            self.covar = np.cov(self.position[:int(self.mu*self.pop_size)].T)

            self.position = np.random.multivariate_normal(self.mean,self.covar, self.pop_size)

            evals+=self.pop_size
            score_list.append(score[0])
            
            eval_list.append(evals)
            if len(score_list)>100:
                if np.all(score_list[-20:][0]==score_list[-10:]):
                    print("Stopping due to probable saturation at evals:",evals)
                    break

        plt.plot(eval_list,score_list)
        plt.show()



if __name__ == "__main__":
    start= time.time()

    a = EA(pop_size=1000,dim=50,problem=2,num_evals=500000,bounds=[-5,5],mu=0.5)
    a.run()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
