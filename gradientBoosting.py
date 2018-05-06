import numpy as np
from scipy.optimize import minimize_scalar
import decisionTree
import time

class GradientBoosting:
    def loss_fun(self, loss_function):
        if loss_function == 'mse':
            return lambda y,yi : (y-yi)
        raise "Unknown loss function"
        
    def __init__(self, iteration, learning_rate=0.1, loss_function='mse',
                max_depth_tree=2, min_size=None):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.loss_function = self.loss_fun(loss_function)
        self.max_depth_tree = max_depth_tree
        self.min_size = min_size
        self.models = []
        self.gammas = []
        
    def _find_gamma(self, label, y_prev, y_predict):
        func = lambda x : sum((label - (y_prev - x*y_predict))**2)
        res = minimize_scalar(func)
        print("GAMMA : " + str(res))
        return res.x
        
    def fit(self, dataset, label):
        label = np.asarray(label)
        
        yi = label
        y_prev = 0
        
        for i in range(self.iteration):
            begin = time.time()
            model = decisionTree.DecisionTree(self.max_depth_tree, self.min_size, 'sse')
            model.fit(dataset, yi)
            
            y_predict = model.predict(dataset)
            y_prev = y_prev + y_predict
                                    
            yi = label - self.learning_rate * y_prev
            
            self.models.append(model)
#             self.gammas.append(gamma)
            print("Iteration " + str(i+1) + ": " + str(time.time() - begin) + " s")
            
    def predict(self, data):
        return (sum(self.learning_rate * self.models[i].predict(data) for i in range(len(self.models))))
