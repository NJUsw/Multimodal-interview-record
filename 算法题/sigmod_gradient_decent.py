import numpy as np

# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义损失函数的梯度
def gradient(x):
    sig_x = sigmoid(x)
    return 2 * (sig_x - 0.4) * sig_x * (1 - sig_x)

# 梯度下降法
def gradient_descent(lr=0.1, max_iter=10000, tol=1e-6):
    x = 0  # 初始值
    for i in range(max_iter):
        grad = gradient(x)
        x_new = x - lr * grad
        
        # 检查损失是否足够小
        if abs(x_new - x) < tol:
            print(f"Converged at iteration {i}")
            break
        
        x = x_new
    
    return x

# 求解 x 使得 sigmoid(x) = 0.4
x_solution = gradient_descent(lr=0.1)
print(f"Solved x: {x_solution}")
print(f"sigmoid(x) at solution: {sigmoid(x_solution)}")
