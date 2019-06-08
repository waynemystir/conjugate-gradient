import numpy as np

innprd = lambda u, v: u.flatten().dot( v.flatten() )

default_residual_period = 50
tolerance = 1e-11

def gradient_descent(A, b, x, alpha = 0.1, epsilon=tolerance, residual_period=default_residual_period, max_iters=30000):
    x = x.reshape(-1, 1)
    b = b.reshape(-1, 1)
    steps=np.asarray(x)
    i = 0
    r = b - A.dot(x)
    delta = innprd(r, r)
    delta0 = delta
    while i < max_iters and delta > epsilon**2 * delta0:
        q = A.dot(r)
        x = x + alpha * r
        if i % residual_period == 0:
            r = b - A.dot(x)
        else:
            r = r - alpha * q
        delta = innprd(r, r)
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def steepest_descent(A, b, x, epsilon=tolerance, residual_period=default_residual_period, max_iters=10000):
    x = x.reshape(-1, 1)
    b = b.reshape(-1, 1)
    steps=np.asarray(x)
    i = 0
    r = b - A.dot(x)
    delta = innprd(r, r)
    delta0 = delta
    while i < max_iters and delta > epsilon**2 * delta0:
        q = A.dot(r)
        alpha = delta / innprd(r, q)
        x = x + alpha * r
        if i % residual_period == 0:
            r = b - A.dot(x)
        else:
            r = r - alpha * q
        delta = innprd(r, r)
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def conjugate_gradient(A, b, x, epsilon=tolerance, residual_period=default_residual_period, max_iters=1000):
    x = x.reshape(-1, 1)
    b = b.reshape(-1, 1)
    steps=np.asarray(x)
    i = 0
    r = b - A.dot(x)
    d = r.copy()
    delta_new = innprd(r, r)
    delta_0 = delta_new
    while i < max_iters and delta_new > epsilon**2 * delta_0:
        q = A.dot(d)
        alpha = delta_new / innprd(d, q)
        x = x + alpha * d
        if i % residual_period == 0:
            r = b - A.dot(x)
        else:
            r = r - alpha * q
        delta_old = delta_new
        delta_new = innprd(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        i = i + 1
        steps = np.append(steps, np.asarray(x), axis=1)
    return steps

def test1():
    A = np.array([3.,2,2,6]).reshape(2,2)
    b = np.array([2.0, -8.0]).reshape(2,1)
    c = 0
    x0 = np.array([[-2.0],[-1.0]])
    x = np.array( conjugate_gradient(A, b, x0) )[:,-1].reshape(2,)

    print("x0={} x={}".format(x0, x))
    return

def main():
    test1()
    return

if __name__ == "__main__":
    main()
