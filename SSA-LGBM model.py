import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i, 0] = func(X[i, :])

    pFit = fit.copy()  
    pX = X.copy()  
    fMin = np.min(fit[:, 0]) 
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :].copy()
    Convergence_curve = np.zeros((1, M))

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

#%%
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
M = 50
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

def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    vaf = calculate_vaf(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "VAF": vaf
    }

train_metrics = evaluate_metrics(y_train_orig, y_train_pred_orig)
test_metrics = evaluate_metrics(y_test_orig, y_test_pred_orig)

# ==================== 结果输出 ====================
print("\n========= 优化结果 =========")
print(f"最优参数: learning_rate={optimal_lr:.4f}, num_leaves={optimal_leaves}")
print(f"SSA优化最佳MSE: {best_mse:.4f}")

print("\n========= 最终模型评估 =========")
print("训练集指标:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n测试集指标:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(convergence_curve[0], 'b-')
plt.title('SSA Convergence Curve'), plt.xlabel('Iteration'), plt.ylabel('RMSE')
plt.grid(True), plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_test_pred_orig, c='blue', alpha=0.6, label='Test Set')
plt.scatter(y_train_orig, y_train_pred_orig, c='green', alpha=0.4, label='Training Set')
plt.plot([min(y_test_orig), max(y_test_orig)], 
         [min(y_test_orig), max(y_test_orig)], 'r--', lw=2)
plt.title("Actual vs Predicted Values"), plt.xlabel("Actual"), plt.ylabel("Predicted")
plt.legend(), plt.grid(True)
plt.savefig('path')
plt.show()
