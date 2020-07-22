import multiprocessing
import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class surrogate: #General Class for surrogate models for predicting likelihood(here the fitness) given the weights

    def __init__(self, model, X, Y, min_X, max_X, min_Y , max_Y, path, save_surrogate_data, model_topology):

        self.path = path + '/surrogate'
        indices = np.where(Y==np.inf)[0]
        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices, axis=0)
        self.model_signature = 0.0
        self.X = X
        self.Y = Y
        self.min_Y = min_Y
        self.max_Y = max_Y
        self.min_X = min_X
        self.max_X = max_X

        self.model_topology = model_topology

        self.save_surrogate_data =  save_surrogate_data

        if model=="gp":
            self.model_id = 1
        elif model == "nn":
            self.model_id = 2
        elif model == "krnn": # keras nn
            self.model_id = 3
            self.krnn = Sequential()
        else:
            print("Invalid Model!")

    # This function is ignored
    def normalize(self, X):
        maxer = np.zeros((1,X.shape[1]))
        miner = np.ones((1,X.shape[1]))

        for i in range(X.shape[1]):
            maxer[0,i] = max(X[:,i])
            miner[0,i] = min(X[:,i])
            X[:,i] = (X[:,i] - min(X[:,i]))/(max(X[:,i]) - min(X[:,i]))
        return X, maxer, miner

    def create_model(self):
        krnn = Sequential()

        if self.model_topology == 1:
            krnn.add(Dense(64, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(16, kernel_initializer='uniform', activation='relu'))  #16

        if self.model_topology == 2:
            krnn.add(Dense(120, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(40, kernel_initializer='uniform', activation='relu'))  #16

        if self.model_topology == 3:
            krnn.add(Dense(200, input_dim=self.X.shape[1], kernel_initializer='uniform', activation ='relu')) #64
            krnn.add(Dense(50, kernel_initializer='uniform', activation='relu'))  #16

        krnn.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))
        return krnn

    def train(self, model_signature):
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.10, random_state=42)

        X_train = self.X
        X_test = self.X
        y_train = self.Y
        y_test =  self.Y #train_test_split(self.X, self.Y, test_size=0.10, random_state=42)

        self.model_signature = model_signature


        if self.model_id is 3:
            if self.model_signature==1.0:
                self.krnn = self.create_model()
            else:
                while True:
                    try:
                        # You can see two options to initialize model now. If you uncomment the first line then the model id loaded at every time with stored weights. On the other hand if you uncomment the second line a new model will be created every time without the knowledge from previous training. This is basically the third scheme we talked about for surrogate experiments.
                        # To implement the second scheme you need to combine the data from each training.

                        self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%(model_signature-1))
                        #self.krnn = self.create_model()
                        break
                    except EnvironmentError as e:
                        # pass
                        # # print(e.errno)
                        # time.sleep(1)
                        print ('ERROR in loading latest surrogate model, loading previous one in TRAIN')

            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
            train_log = self.krnn.fit(X_train, y_train.ravel(), batch_size=50, epochs=500, validation_split=0.1, verbose=1, callbacks=[early_stopping])

            scores = self.krnn.evaluate(X_test, y_test.ravel(), verbose = 1)
            # print("%s: %.5f" % (self.krnn.metrics_names[1], scores[1]))

            self.krnn.save(self.path+'/model_krnn_%s_.h5' %self.model_signature)
            # print("Saved model to disk  ", self.model_signature)
 

            results = np.array([scores[1]])
            # print(results, 'train-metrics')


            with open(('%s/train_metrics.txt' % (self.path)),'ab') as outfile:
                np.savetxt(outfile, results)

            if self.save_surrogate_data is True:
                with open(('%s/learnsurrogate_data/X_train.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, X_train)
                with open(('%s/learnsurrogate_data/Y_train.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, y_train)
                with open(('%s/learnsurrogate_data/X_test.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, X_test)
                with open(('%s/learnsurrogate_data/Y_test.csv' % (self.path)),'ab') as outfile:
                    np.savetxt(outfile, y_test)

    def predict(self, X_load, initialized):


        if self.model_id == 3:

            if initialized == False:
                model_sign = np.loadtxt(self.path+'/model_signature.txt')
                self.model_signature = model_sign
                while True:
                    try:
                        self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%self.model_signature)
                        # # print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%self.model_signature)
                        break
                    except EnvironmentError as e:
                        print(e)
                        # pass

                self.krnn.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
                krnn_prediction =-1.0
                prediction = -1.0

            else:
                krnn_prediction = self.krnn.predict(X_load)[0]
                # print('nn_out',self.krnn.predict(X_load))
                prediction = krnn_prediction#*(self.max_Y[0,0]-self.min_Y[0,0]) + self.min_Y[0,0]
            
            return prediction, krnn_prediction


class surrogate_EA(multiprocessing.Process):
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals,swap_interval,parameter_queue,signal_main,event,island_id,surrogate_parameter_queues,surrogate_start,surrogate_resume,surrogate_interval,surrogate_prob,save_surrogatedata,use_surrogate,compare_surrogate,surrogate_topology,path):
        
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
        self.p_best_score = self.evaluate(self.position)
        
        self.g_best_score = 999999 #initialize
        self.g_best = self.p_best[0]  #initialize

        for i in range(self.pop_size)  :
            if self.p_best_score[i]<self.g_best_score:
                self.g_best_score = self.p_best_score[i]
                self.g_best = copy.copy(self.position[i])

        # Surrogate Variables
        self.surrogate_parameter_queue = surrogate_parameter_queues
        self.surrogate_start = surrogate_start
        self.surrogate_resume = surrogate_resume
        self.surrogate_interval = surrogate_interval
        self.surrogate_prob = surrogate_prob
        self.save_surrogate_data = save_surrogatedata
        self.use_surrogate = use_surrogate
        self.compare_surrogate = compare_surrogate
        self.surrogate_topology = surrogate_topology
        
        self.path = path
        

    def evaluate(self,position):
        pop_size = position.reshape(-1,self.dim).shape[0]
        problem = self.problem
        
        if self.problem == 1: # rosenbrock
            
            if pop_size==1:
                fit=0
                # print('1',position)
                for j in range(self.dim -1):
                    fit += (100.0*(position[j]**2 - position[j+1]**2)**2 + (position[j]-1.0)**2)

            else:    
                fit = np.zeros((pop_size,self.dim))
                # print(position)
                for i in range(pop_size):
                    for j in range(self.dim -1):
                        fit[i] = (100.0*(position[i,j]**2 - position[i,j+1]**2)**2 + (position[i,j]-1.0)**2)

                fit = np.sum(fit,axis=1)    
        
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

        surrogate_model = None 
        surrogate_counter = 0
        trainset_empty = True
        is_true_fit = True
        surg_fit_list = np.zeros((self.num_evals * 10,3))
        surr_train_set = np.zeros((1000, self.dim+1))
        local_model_signature = 0.0
        self.surrogate_init = 0.0
        evals=0
        
        count_real = 0
        idx = 0

        score_list = []
        eval_list = []
        while evals<self.num_evals:
        
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            w=0.729       # constant inertia weight (how much to weigh the previous velocity)
            c1=1.4        # cognative constant
            c2=1.4        # social constant
            
            score_list.append(self.g_best_score)
            eval_list.append(evals)
            print("G_BEST_SCORE" ,self.g_best_score ,"on", self.island_id,"island" )

            # self.event.clear()

            # Update rule
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(self.g_best-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity
            

            # Evaluation
            for i in range(self.pop_size):
                
                surrogate_X = self.g_best
                best_surr_fit = self.g_best_score
                surrogate_Y = np.array([best_surr_fit])
                # proposed best parameters after the evaluation
                w_proposal = self.position[i]
                ku = random.uniform(0,1)
                if ku<self.surrogate_prob and evals >= self.surrogate_interval+1:
                    
                    is_true_fit = False

                    # Create the model when there was no previously assigned model for surrogate
                    if surrogate_model == None:
                        # Load the text saved before in the training surrogate func. in manager process 
                        surrogate_model = surrogate("krnn",surrogate_X.copy(),surrogate_Y.copy(), self.minx, self.maxx, self.minY, self.maxY, self.path, self.save_surrogate_data, self.surrogate_topology)
                        surrogate_pred, nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]),False)
                        #surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)

                    # Getting the initial predictions if the surrogate model has yet not been initialized     
                    elif self.surrogate_init == 0.0:
                        surrogate_pred,  nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), False)

                    # Getting the predictions if surrogate model is already initialized    
                    else:
                        surrogate_pred,  nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), True)
                       
                    surr_mov_ave = (surg_fit_list[idx,2] + surg_fit_list[idx-1,2]+ surg_fit_list[idx-2,2])/3
                    #surr_proposal = (surrogate_pred * 0.5) + (  surr_mov_ave * 0.5)
                    surr_proposal = surrogate_pred


                    if self.compare_surrogate is True:
                        fitness_proposal_true = self.evaluate(w_proposal)
                    else:
                        fitness_proposal_true = 0
                    surrogate_counter += 1
                    surg_fit_list[idx+1,0] =  fitness_proposal_true
                    surg_fit_list[idx+1,1] = surr_proposal
                    surg_fit_list[idx+1,2] = surr_mov_ave
                else:
                    is_true_fit = True
                    trainset_empty = False
                    surg_fit_list[idx+1,1] =  np.nan
                    surr_proposal = self.evaluate(w_proposal)
                    fitness_arr = np.array([surr_proposal])
                    X, Y = w_proposal,fitness_arr
                    X = X.reshape(1, X.shape[0])
                    Y = Y.reshape(1, Y.shape[0])
                    param_train = np.concatenate([X, Y],axis=1)
                    #surr_train_set = np.vstack((surr_train_set, param_train))
                    surg_fit_list[idx+1,0] = surr_proposal
                    surg_fit_list[idx+1,2] = surr_proposal

                    surr_train_set[count_real, :] = param_train
                    count_real = count_real +1
                
                error = surr_proposal
                # print(i,error)
                if error < self.p_best_score[i]:
                    self.p_best_score[i] = error
                    self.p_best = copy.copy(self.position[i])

                if error < self.g_best_score:
                    self.g_best_score = error
                    self.g_best = copy.copy(self.position[i])   

                idx += 1    
                       

           
            
            #SWAPPING PREP
            """
            if (evals % self.swap_interval == 0 ): # interprocess (island) communication for exchange of neighbouring best_swarm_pos
                param = best_swarm_pos
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result =  self.parameter_queue.get()
                best_swarm_pos = result 
                swarm[0].position = best_swarm_pos.copy()
            """
            if evals % self.surrogate_interval == 0 and evals != 0:
                #print("\n\nSample:{}\n\n".format(i))
                # add parameters to the swap param queue and surrogate params queue
                #self.parameter_queue.put(param)

                surr_train = surr_train_set[0:count_real, :]
                print("Total Data Collected in island_id:",self.island_id,":",count_real)

                #self.surrogate_parameter_queue.put(all_param)

                self.surrogate_parameter_queue.put(surr_train)
                # Pause the chain execution and signal main process
                self.signal_main.set()
                # Wait for the main process to complete the swap and surrogate training
                self.event.clear()
                self.event.wait()
                
                model_sign = np.loadtxt(self.path+'/surrogate/model_signature.txt')
                self.model_signature = model_sign
                #print("model_signature updated")

                if self.model_signature==1.0:
                    # # print 'min ', self.minY, ' max ', self.maxY
                    dummy_X = np.zeros((1,1))
                    dummy_Y = np.zeros((1,1))
                    surrogate_model = surrogate("krnn", dummy_X, dummy_Y, 0, 0, 0, 0, self.path, self.save_surrogate_data, self.surrogate_topology )

                    local_model_signature = local_model_signature +1  

                # Initialize the surrogate
                self.surrogate_init,  nn_predict  = surrogate_model.predict(self.g_best.reshape(1,self.g_best.shape[0]), False)
                #del surr_train_set
                trainset_empty = True 
                #np.savetxt(self.folder+'/surrogate/traindata_'+ str(self.island_id) +'_'+str(local_model_signature)    +'_.txt', surr_train_set)
                count_real = 0      



            # epoch += 1
            evals += self.pop_size
      
        # FINAL RESULTS
    
        print("Best particle:", self.g_best , 'of island', self.island_id)

        plt.plot(eval_list , score_list , label = self.island_id)
        plt.show()

        print("Island: {} chain dead!".format(self.island_id))
        self.signal_main.set()

class distributed_surrogate_EA:
    def __init__(self,pop_size,dim,bounds,problem,num_evals,num_islands,surrogate_topology,use_surrogate,compare_surrogate,save_surrogate_data,path):
        
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

        self.swap_interval = pop_size #means 1 iteration

        # Surrogate Variables
        self.surrogate_interval = self.pop_size
        self.surrogate_prob = 0.5
        self.surrogate_resume = [multiprocessing.Event() for i in range(self.num_islands)]
        self.surrogate_start = [multiprocessing.Event() for i in range(self.num_islands)]
        self.surrogate_parameter_queues = [multiprocessing.Queue() for i in range(self.num_islands)]
        self.surrchain_queue = multiprocessing.JoinableQueue()
        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1))
        self.model_signature = 0.0
        self.use_surrogate = use_surrogate
        self.surrogate_topology = surrogate_topology
        self.save_surrogate_data =  save_surrogate_data
        self.compare_surrogate = compare_surrogate
        self.path = path
        self.folder = path
        self.total_swap_proposals = 0
        self.num_swaps = 0

    def initialize_islands(self):
        for i in range(self.num_islands):
            self.islands.append(surrogate_EA(self.pop_size,self.dim,self.bounds,self.problem,self.island_numevals,self.swap_interval,self.parameter_queue[i],self.wait_island[i],self.event[i],i,self.surrogate_parameter_queues[i],self.surrogate_start[i],self.surrogate_resume[i],self.surrogate_interval,self.surrogate_prob,self.save_surrogate_data,self.use_surrogate,self.compare_surrogate,self.surrogate_topology,self.path))


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


    def surrogate_trainer(self,params): 

            X = params[:,:self.dim]
            Y = params[:,self.dim].reshape(X.shape[0],1)
            self.model_signature += 1.0

            np.savetxt(self.folder+'/surrogate/model_signature.txt', [self.model_signature])
            indices = np.where(Y==np.inf)[0]
            X = np.delete(X, indices, axis=0)
            Y = np.delete(Y,indices, axis=0)
            surrogate_model = surrogate("krnn", X , Y , 0 , 0 , 0 , 0 , self.folder, self.save_surrogate_data, self.surrogate_topology )
            surrogate_model.train(self.model_signature)        


    def evolve_islands(self): 
        
        self.initialize_islands()
        swap_proposal = np.ones(self.num_islands-1)

        for j in range(0,self.num_islands):        
            self.wait_island[j].clear()
            self.event[j].clear()
            self.islands[j].start()
        #SWAP PROCEDURE

        swaps_appected_main =0
        total_swaps_main =0
        #for i in range(int(self.island_numevals/self.swap_interval)):
        for i in range(int(self.island_numevals/self.surrogate_interval) - 1):
            count = 0
            # checking if the processes are still alive
            for index in range(self.num_islands):
                if not self.islands[index].is_alive():
                    count+=1
                    self.wait_island[index].set() 

            if count == self.num_islands:
                break

            print("Waiting for the swapping signals.")
            timeout_count = 0
            for index in range(0,self.num_islands): 
                flag = self.wait_island[index].wait()
                if flag: 
                    timeout_count += 1
            # If signals from all the islands are not received then skip the swap and continue the loop.
            """
            if timeout_count != self.num_islands: 
                print("Skipping the swap")
                continue
            """ 
            if timeout_count == self.num_islands:
                """ 
                for index in range(0,self.num_islands-1): 
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                """
                all_param =   np.empty((1,self.dim+1))
                for index in range(0,self.num_islands):
                    print('starting surrogate')
                    queue_surr=  self.surrogate_parameter_queues[index] 
                    surr_data = queue_surr.get() 
                    #print("Shape of surr_data:",surr_data.shape)
                    #print("all_param.shape:",all_param.shape)
                    all_param =   np.concatenate([all_param,surr_data],axis=0) 
                print("Shape of all_param Collected :",all_param.shape)
                data_train = all_param[1:,:]  
                print("Shape of Data Collected :",data_train.shape)
                self.surrogate_trainer(data_train) 

                for index in range (self.num_islands):
                        self.event[index].set()
                        self.wait_island[index].clear()

            elif timeout_count == 0:
                break
            else:
                print("Skipping the swap")             

            
            
        for index in range(0,self.num_islands):
            self.islands[index].join()

        for i in range(0,self.num_islands):
            #self.parameter_queue[i].close()
            #self.parameter_queue[i].join_thread()
            self.surrogate_parameter_queues[i].close()
            self.surrogate_parameter_queues[i].join_thread()












if __name__ == "__main__":
    start = time.time()
    a = distributed_surrogate_EA(pop_size=100,dim=50,bounds=[-5,5],problem=2,num_evals=4500,num_islands=3,surrogate_topology=1,use_surrogate=True,compare_surrogate=False,save_surrogate_data=False,path='C:\\Users\\admin\\Desktop\\unsw\\surr')
    a.evolve_islands()
    print('Time Taken= ',(time.time()-start)/60 ,"Minutes")
