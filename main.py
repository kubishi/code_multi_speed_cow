from itertools import product
import pathlib
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# enable latex rendering
plt.rc('text', usetex=True)

# ALG_COLORS = {
#     "Fast": "#D81B60",
#     "Hybrid": "#1E88E5",
#     "Slow": "#FFC107",
# }

# https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
ALG_COLORS = {
    "Fast": "#332288",
    "Hybrid": "#DDCC77",
    "Slow": "#117733",
}

def expr1(a, b, v, p):
    return ((1 + a) * ((-1 + a) * (a * (-1 + p)**2 - (-2 + p) * p) + 
            (a**2 + 2 * (-1 + a) * a**3 * b + 2 * p - 2 * a * (a + b + a**2 * (-3 + 2 * a) * b) * p +
            (-1 + a) * (1 + a + 2 * (-1 + a) * a**2 * b) * p**2) * v)) / ((-1 + a) * a * v)

def expr2(a, b, v):
    return (-1 + b + a * (-b + v + a * (1 + v + (-1 + a) * b * (1 + 2 * (1 + a) * v)))) / ((-1 + a) * (1 + (-1 + a**2) * b) * v)

def objective(params, v, p):
    a, b = params
    return max(expr1(a, b, v, p), expr2(a, b, v))

def optimize_a_b(v, p):
    initial_guess = [2, 0.5]
    bounds = [(1.001, 10), (0, 1)]  
    result = opt.minimize(objective, initial_guess, args=(v, p), bounds=bounds, method='Nelder-Mead')
    
    return result.x if result.success else (2, 0.5)  

@np.vectorize
def cr_fast(p):
    return 8 / p + p / (2 - p)

@np.vectorize
def cr_slow(v):
    return 3 + 2 * np.sqrt(2 + 2 / v) + 2 / v

def cr_hybrid(V, P):
    A = np.zeros_like(V)
    B = np.zeros_like(V)
    CRHybrid = np.zeros_like(P)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = V[i, j]
            p = P[i, j]
            a_opt, b_opt = optimize_a_b(v, p)
            A[i, j] = a_opt
            B[i, j] = b_opt
            CRHybrid[i, j] = max(expr1(a_opt, b_opt, v, p), expr2(a_opt, b_opt, v))
    
    return A, B, CRHybrid

def plot_regions(V, P, CRFast, CRSlow, CRHybrid):
    """
    Fixes:
    - Ensures all three regions are visible distinctly.
    - Uses numerical encoding for region identification.
    - Plots with a discrete colormap.
    """    
    # 0 = Fast is best (default)
    # 1 = Hybrid is best
    # 2 = Slow is best
    Zslow = np.zeros_like(V)
    Zhybrid = np.zeros_like(V)

    Zhybrid[CRHybrid < CRFast] = 1  # Hybrid is better than Fast
    Zslow[CRSlow < CRFast] = 2  # Slow is better than Fast

    ax_slow: plt.Axes
    ax_hybrid: plt.Axes
    fig, (ax_slow, ax_hybrid) = plt.subplots(1, 2, figsize=(12, 6))
    ctr_slow = ax_slow.contourf(V, P, Zslow, levels=[-0.5, 1.0, 2.5], colors=[ALG_COLORS[k] for k in ["Fast", "Slow"]])
    ctr_hybrid = ax_hybrid.contourf(V, P, Zhybrid, levels=[-0.5, 0.5, 1.5], colors=[ALG_COLORS[k] for k in ["Fast", "Hybrid"]])

    for ctr in [ctr_slow, ctr_hybrid]:
        for a in ctr.collections:
            a.set_edgecolor("face")

    for ax in [ax_slow, ax_hybrid]:
        ax.set_xlabel(r'$v$', fontsize=16)
        ax.set_ylabel(r'$p$', fontsize=16)
        ax.grid(True)
        ax.set_aspect('equal')
        # set tick font size
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

        if ax == ax_slow:
            labels = [r"$CR_{Fast} \leq CR_{Slow}$", r"$CR_{Slow} < CR_{Fast}$"]
            handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in [ALG_COLORS["Fast"], ALG_COLORS["Slow"]]]
            ax.legend(handles, labels, loc="upper left", fontsize=16)
        else:
            labels = [r"$CR_{Fast} \leq CR_{Hybrid}$", r"$CR_{Hybrid} < CR_{Fast}$"]
            handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in [ALG_COLORS["Fast"], ALG_COLORS["Hybrid"]]]
            ax.legend(handles, labels, loc="upper left", fontsize=16)


    fig.tight_layout()

    fig.savefig('regions.pdf')
    plt.close(fig)

def plot_heatmap(V: np.ndarray,
                 P: np.ndarray,
                 CR: np.ndarray,
                 savepath: pathlib.Path):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(V, P, CR, cmap='viridis')
    ax.set_xlabel(r'$v$', fontsize=16)
    ax.set_ylabel(r'$p$', fontsize=16)
    fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def main():
    n_points = 100
    p_vals = np.linspace(0.001, 1, n_points)
    v_vals = np.linspace(0.001, 1, n_points)

    P, V = np.meshgrid(p_vals, v_vals)

    print("Calculating CRFast, CRSlow, and CRHybrid...")
    CRFast = cr_fast(P)
    CRSlow = cr_slow(V)
    A, B, CRHybrid = cr_hybrid(V, P)

    print("Generating plots...")
    plot_regions(V, P, CRFast, CRSlow, CRHybrid)

    # Clip for CR Plotting
    idx_p = np.where(p_vals > 0.1)
    idx_v = np.where(v_vals > 0.1)
    V = V[idx_v[0]][:, idx_p[0]]
    P = P[idx_v[0]][:, idx_p[0]]
    CRFast = CRFast[idx_v[0]][:, idx_p[0]]
    CRSlow = CRSlow[idx_v[0]][:, idx_p[0]]
    CRHybrid = CRHybrid[idx_v[0]][:, idx_p[0]]

    plot_heatmap(V, P, CRFast, pathlib.Path('CRFast.pdf'))
    plot_heatmap(V, P, CRSlow, pathlib.Path('CRSlow.pdf'))
    plot_heatmap(V, P, CRHybrid, pathlib.Path('CRHybrid.pdf'))

    CRMin = np.minimum(np.minimum(CRFast, CRSlow), CRHybrid)
    plot_heatmap(V, P, CRMin, pathlib.Path('CRMin.pdf'))

    CRMinNoHybrid = np.minimum(CRFast, CRSlow)
    plot_heatmap(V, P, CRMinNoHybrid, pathlib.Path('CRMinNoHybrid.pdf'))

    CRDiff = CRMinNoHybrid - CRMin
    plot_heatmap(V, P, CRDiff, pathlib.Path('CRDiff.pdf'))

if __name__ == "__main__":
    main()
