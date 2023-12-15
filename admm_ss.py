import numpy as np


def admm_ss(D, W, b, mu, max_iter):
    """
    :param D: matrix of green's functions from monopole-source pos to input mic pos
    :param W: matrix of plane waves from outer region to center, direction k_l
    :param b: fft of the input signal(s) at a specific frequency bin. (column vector)
    :param mu: weight of the plane waves as reverb 0-1
    :param max_iter: number of iterations before stopping
    :return: x: weights of monopoles at curr freq bin
             u: weights of plane waves at curr freq bin
    """

    # Initialization
    eps1 = 1e-4
    eps2 = 1e-4
    
    M, N = np.shape(D)
    A, L = np.shape(W)  # A == M
    B, T = np.shape(b)  # B == M

    # print(M, N, A, L, B, T)

    num_it = 0
    x = np.zeros((N, T))
    u = np.zeros((L, T))
    theta = np.zeros((M, T))

    # Compute nw, nd
    nd = 1.02 * np.max(np.linalg.eigvalsh(D.T @ D))
    nw = 1.02 * np.max(np.linalg.eigvalsh(W.T @ W))

    # print(nd, nw)  # >> 0

    # Adaptive parameter rho
    rho_max = 10e10
    alpha_0 = 1.9

    rho = M * eps2

    for k in range(1, max_iter + 1):
        # print(rho)  # >> 0

        x_old = x.copy()
        # Update x
        gx = x - rho / nd * D.T @ (theta + 1 / rho * (D @ x + W @ u - b))
        x = soft(gx, rho / nd)
        #if np.linalg.norm(gx) > 1:
            # print(np.linalg.norm(gx))
        # print(rho)
        x = x / 100  # contains amplitude escalation
        # print(np.max(x))

        u_old = u.copy()
        # Update u
        gu = u - rho / nw * W.T @ (theta + 1 / rho * (D @ x + W @ u - b))
        u = soft(gu, mu * rho / nw)
        u = u / 100  # contains amplitude escalation

        # Update theta
        theta = theta + 1 / rho * (D @ x + W @ u - b)

        # Update parameter rho
        if (rho * np.max([np.sqrt(nd) * np.linalg.norm(x - x_old, 'fro'),
                          np.sqrt(nw) * np.linalg.norm(u - u_old, 'fro')]) / np.linalg.norm(b, 'fro') < eps2):
            alpha = alpha_0
        else:
            alpha = 1  # .5  # 1

        rho = min(rho_max, alpha * rho)
        # rho = min(rho_max, alpha * rho + 10e-2)

        num_it = k
        # print(num_it)

        # Stop conditions
        # if (np.linalg.norm(D @ x + W @ u - b, 'fro') / np.linalg.norm(b, 'fro') <= eps1) and \
        #         (np.max([np.sqrt(nd) * np.linalg.norm(x - x_old, 'fro'),
        #                  np.sqrt(nw) * np.linalg.norm(u - u_old, 'fro')]) <= (eps2 * rho * np.linalg.norm((b, 'fro')))):
        if (np.linalg.norm(D @ x + W @ u - b, 'fro') / np.linalg.norm(b, 'fro') <= eps1) and \
                (np.max([np.sqrt(nd) * np.linalg.norm(x - x_old, 'fro'),
                         np.sqrt(nw) * np.linalg.norm(u - u_old, 'fro')]) <= eps2):
            print('y')
            return x, u, num_it

    return x, u, num_it


def soft(x, T):
    if np.sum(np.abs(T)) == 0:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0)
        y = np.sign(x) * y
    return y
