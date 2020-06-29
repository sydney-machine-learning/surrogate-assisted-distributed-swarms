import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt 

class EA:
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals):

        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds

        self.position = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0]
        self.velocity = (bounds[1] - bounds[0]) * np.random.rand(pop_size,dim)  + bounds[0]
    
        self.p_best = copy.copy(self.position)
        self.g_best = sys.float_info.max
        self.num_evals = num_evals

        self.problem = problem

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
        # ab hoga asla
        evals=0
        score_list = []
        eval_list = []
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
            # print("At:",g_best_pos)

            evals+=self.pop_size
            score_list.append(self.g_best)
            eval_list.append(evals)
            if len(score_list)>100:
                if np.all(score_list[-20:][0]==score_list[-10:]):
                    print("Stopping due to probable saturation at evals:",evals)
                    break

        plt.plot(eval_list,score_list)
        plt.show()


if __name__ == "__main__":
    start= time.time()
    a = EA(1000,50,[-5,5],2,500000)
    a.run()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
