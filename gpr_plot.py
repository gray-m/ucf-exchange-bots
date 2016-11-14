from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt
import numpy as np

def plot_gpr(gpr):
    # assuming that X is increasing
    X = gpr.X_train_.reshape(-1, 1)
    y = gpr.y_train_.reshape(-1, 1)

    x_fill = np.linspace(X[0, 0], X[-1, 0], 1000).reshape(-1, 1)
    y_pred, sigma = gpr.predict(x_fill, return_std=True)
    
    y_pred = y_pred.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    fig = plt.figure()
    plt.scatter(X, y, color='r', label='observations')
    plt.plot(x_fill, y_pred, 'b-', label='prediction')
    upper, lower = y_pred + 1.96*sigma, y_pred - 1.96*sigma
    plt.fill_between(x_fill.squeeze(), upper.squeeze(), lower.squeeze(), color='r', alpha='0.2')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    f = lambda x: x*np.sin(x)+2
    X = np.atleast_2d([0.3, 1.2, 2.5,  4., 6.2])
    obs = f(X)
    gpr = GaussianProcessRegressor(kernel=kernels.Matern(nu=2.5))
    gpr.fit(X.reshape(-1, 1), obs.reshape(-1, 1))
    plot_gpr(gpr)
