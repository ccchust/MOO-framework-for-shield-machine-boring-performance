import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def calculate_vaf(y_true, y_pred):
    return (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100

# Define the standard test function 
def fun(F, X):
    if F == 'F1':
        O = np.sum(X * X)
    elif F == 'F2':
        O = np.sum(np.abs(X)) + np.prod(np.abs(X))
    elif F == 'F3':
        O = 0
        for i in range(len(X)):
            O = O + np.square(np.sum(X[0:i+1]))
    elif F == 'F4':
        O = np.max(np.abs(X))
    elif F == 'F5':
        X_len = len(X)
        O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    elif F == 'F6':
        O = np.sum(np.square(np.abs(X + 0.5)))
    return O

# Boundary constraint function
def Bounds(s, Lb, Ub):
    temp = s.copy()
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]
    return temp

# SSA implementation for hyperparameter optimization
def SSA(pop, M, lb, ub, dim, func):
    P_percent = 0.2
    pNum = round(pop * P_percent)  
    X = np.zeros((pop, dim))  
    fit = np.zeros((pop, 1))  

    # Initialize population
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i, 0] = func(X[i, :])

    pFit = fit.copy()  
    pX = X.copy()  
    fMin = np.min(fit[:, 0])  
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :].copy()
    Convergence_curve = np.zeros((1, M))

    # Optimization loop
    for t in range(M):
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :].copy()

        r2 = np.random.rand(1)  
        if r2 < 0.8:
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i + 1) / (r1 * M))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])
        else:
            for i in range(pNum):
                Q = np.random.normal(0, 1, size=(1, dim))
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + Q
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])

        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :].copy()

        for ii in range(pop - pNum):
            i = ii + pNum
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
            if i > pop / 2:
                Q = np.random.normal(0, 1, size=(1, dim))
                X[sortIndex[0, i], :] = Q * np.exp((worse - pX[sortIndex[0, i], :]) / np.square(i + 1))
            else:
                X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX), 1 / (A.T @ A)) * np.ones((1, dim))
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = func(X[sortIndex[0, i], :])

        arrc = np.arange(pop)
        c = np.random.permutation(arrc)
        b = sortIndex[0, c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX + np.random.normal(0, 1, size=(1, dim)) * np.abs(pX[sortIndex[0, b[j]], :] - bestX)
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 1e-50)
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0, b[j]], 0] = func(X[sortIndex[0, b[j]], :])

        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :].copy()
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :].copy()
        Convergence_curve[0, t] = fMin

    return fMin, bestX, Convergence_curve

def fitness_function(params, X_train, y_train, X_test, y_test):
    learning_rate, num_leaves = params
    lgb_model = lgb.LGBMRegressor(
        learning_rate=learning_rate,
        num_leaves=int(num_leaves),
        max_depth=2,
        reg_alpha=0.5,
        reg_lambda=5,
        n_estimators=200,
        min_child_samples=10,
        boosting_type='gbdt'
    )
    lgb_model.fit(X_train, y_train.ravel())
    y_pred = lgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

X = pd.read_excel('path')
y = pd.read_excel('path')

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, train_size=0.7
)

pop = 20
M = 10
dim = 2
lb = np.array([[0.01, 20]])  
ub = np.array([[0.3, 100]]) 

best_mse, best_params, convergence_curve = SSA(pop, M, lb, ub, dim, 
    lambda params: fitness_function(params, X_train, y_train, X_test, y_test)
)
optimal_lr, optimal_leaves = best_params[0], int(best_params[1])

final_model = lgb.LGBMRegressor(
    learning_rate=optimal_lr,
    num_leaves=optimal_leaves,
    max_depth=2,
    reg_alpha=0.5,
    reg_lambda=5,
    n_estimators=200,
    min_child_samples=10,
    boosting_type='gbdt'
)
final_model.fit(X_train, y_train.ravel())

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

y_train_orig = scaler_y.inverse_transform(y_train)
y_test_orig = scaler_y.inverse_transform(y_test)
y_train_pred_orig = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))

class MultiObjectiveProblem(Problem):
    def __init__(self, model, scaler_X, feature_ranges):

        xl = [fr[0] for fr in feature_ranges]
        xu = [fr[1] for fr in feature_ranges]
        
        super().__init__(n_var=7, 
                        n_obj=2, 
                        n_constr=0,
                        xl=np.array(xl),
                        xu=np.array(xu))
        
        self.model = model
        self.scaler_X = scaler_X
        self.feature_ranges = feature_ranges

    def _evaluate(self, X, out, *args, **kwargs):
        X_normalized = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_normalized)

        y_pred_scaled1= y_pred_scaled.reshape(-1, 1) 
      
        GS = scaler_y.inverse_transform(y_pred_scaled1) 
        
        FPI = (X[:, 0] * X[:, 1]) / X[:, 2]
        FPI = FPI.reshape(-1, 1)
        
        out["F"] = np.hstack([GS, FPI])

def configure_nsga2():
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.7, eta=15),
        mutation=PM(prob=0.7, eta=20),
        eliminate_duplicates=True
    )
    return algorithm

feature_ranges = [
    (),   
    (),     
    (),            
    (),     
    (),       
    (),
    ()     
]

if __name__ == "__main__":

    problem = MultiObjectiveProblem(final_model, scaler_X, feature_ranges)
    
    algorithm = configure_nsga2()
    
    res = minimize(problem,
                   algorithm,
                   ("n_gen", 100),
                   seed=42,
                   verbose=True)
    
    F = res.F
    X = res.X
    
    plt.figure(figsize=(10, 6))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Pareto Optimal Front")
    plt.xlabel("GS (Predicted)", fontsize=12)
    plt.ylabel("FPI (f1*f2/f3)", fontsize=12)
    plt.grid(True)
    plt.show()
    
    pd.DataFrame(np.hstack([X, F]), 
                 columns=[f'f{i+1}' for i in range(7)] + ['GS', 'FPI']
                ).to_excel("optimization_results.xlsx", index=False)
    