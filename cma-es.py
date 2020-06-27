import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt
#now i have cma-es
#required = pop_size, dim , bounds , problem , num_evals , init_mean , init_covar , mu 
#to compute = new_mean , new_covar 

class EA:
    
    def __init__(self,pop_size,dim,problem,num_evals, mean, covar , mu ):

        self.pop_size = pop_size
        self.dim = dim
        self.mean = mean
        self.covar = covar
        self.mu = mu 
        self.num_evals = num_evals
        self.problem = problem
        
        self.position = np.random.multivariate_normal(self.mean,self.covar , self.pop_size)

    def evaluate(self):
        
        problem = self.problem
        if problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*self.pop_size)
            fit =((self.position**2)*matrix)
            fit = np.sum(fit , axis=1)
            return fit

    def run(self):
        # ab hoga asla
        evals=0
        score_list = []
        eval_list = []
        while evals<self.num_evals:

            score = self.evaluate()
            score= score.reshape(-1,1)
            metric = np.concatenate((self.position,score),axis=1)
            metric =metric[metric[:,-1].argsort()]
            score = metric[:,-1]
            # print(score[:5])
            self.position = metric[:,:-1]
            print("Best Score:", score[0])
            print("At:", self.position[0])
            self.mean = np.mean(self.position[:int(self.mu*self.pop_size)],axis=0)
            self.covar = np.cov(self.position[:int(self.mu*self.pop_size)].T)
            # print(self.mean)
            # print(self.covar)
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
    
    a = EA(args) #put agrs there
    a.run()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
