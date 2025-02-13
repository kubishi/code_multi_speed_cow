from functools import lru_cache, wraps
from typing import Tuple
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def expr1(a: float, b: float, v: float, p: float) -> float:
    """Case 1 competitive ratio (agent finds target during scout-ahead)
    
    Args:
        a (float): expansion ratio
        b (float): scout-ahead percentage
        v (float): slow speed of agent
        p (float): probability of finding a target while moving at fast speed

    Returns:
        float: competitive ratio
    """
    return ((1 + a) * ((-1 + a) * (a * (-1 + p)**2 - (-2 + p) * p) + (a**2 +
              2 * (-1 + a) * a**3 * b + 2 * p - 2 * a * (a + b + a**2 * (-3 + 2 * a) * b) * p +
              (-1 + a) * (1 + a + 2 * (-1 + a) * a**2 * b) * p**2) * v)) / ((-1 + a) * a * v)

def expr2(a, b, v):
    """Case 2 competitive ratio (agent does not find target during scout-ahead)
    
    Args:
        a (float): expansion ratio
        b (float): scout-ahead percentage
        v (float): slow speed of agent

    Returns:
        float: competitive ratio
    """
    return (-1 + b + a * (-b + v + a * (1 + v + (-1 + a) * b * (1 + 2 * (1 + a) * v)))) / ((-1 + a) * (1 + (-1 + a**2) * b) * v)

def objective(params, v, p):
    a, b = params
    return max(expr1(a, b, v, p), expr2(a, b, v))

def optimize_a_b(v, p):
    initial_guess = [2, 0.5]  # Start with a = 2 and b = 0.5
    bounds = [(1, None), (0, 1)]  # Ensure a > 1 and 0 <= b <= 1 
    result = opt.minimize(objective, initial_guess, args=(v, p), bounds=bounds, method='Nelder-Mead')
    return result.x if result.success else (None, None)

def cr_fast(p):
    return 8 / p + p / (2 - p)

def cr_slow(v):
    return 3 + 2 * np.sqrt(2 + 2 / v) + 2 / v

def cr_hybrid(V: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            if a_opt is not None and b_opt is not None:
                CRHybrid[i, j] = max(expr1(a_opt, b_opt, v, p), expr2(a_opt, b_opt, v))
            else:
                CRHybrid[i, j] = np.nan  # Set invalid points to NaN

    return A, B, CRHybrid

def plot_results(V: np.ndarray,
                 P: np.ndarray,
                 CRFast: np.ndarray,
                 CRSlow: np.ndarray,
                 CRHybrid: np.ndarray,
                 alpha: float = 0.6):
    ax: Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(V, P, CRHybrid, color='blue', alpha=alpha, label='CRHybrid')
    ax.plot_surface(V, P, CRFast, color='red', alpha=alpha, label='CRFast')    
    ax.plot_surface(V, P, CRSlow, color='green', alpha=alpha, label='CRSlow')
    
    ax.set_xlabel('v')
    ax.set_ylabel('p')
    ax.set_zlabel('Max Expression Value')
    ax.set_title('Comparison of CRHybrid, CRFast, and CRSlow')

    # legend
    plt.legend()

    # plt.show()
    for elev, azim in [(30, 30), (30, 150), (30, 270), (30, 390)]:
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f'CR_{elev}_{azim}.png')
    
    plt.close()

def plot_comparison_curves(V: np.ndarray,
                           P: np.ndarray,
                           CRFast: np.ndarray,
                           CRSlow: np.ndarray,
                           CRHybrid: np.ndarray):
    # Find where CRHybrid == CRFast
    diff_hf = np.abs(CRHybrid - CRFast)
    curve_hf = np.where(diff_hf < 0.1)  # Tolerance for equality

    # Find where CRHybrid == CRSlow
    diff_hs = np.abs(CRHybrid - CRSlow)
    curve_hs = np.where(diff_hs < 0.1)

    # Find where CRSlow == CRFast
    diff_sf = np.abs(CRSlow - CRFast)
    curve_sf = np.where(diff_sf < 0.1)

    # Plot the curves
    plt.figure(figsize=(8, 6))
    plt.contour(P, V, CRHybrid, levels=20, cmap='coolwarm', alpha=0.3)
    plt.scatter(P[curve_hf], V[curve_hf], color='red', label='CRHybrid == CRFast', s=5)
    plt.scatter(P[curve_hs], V[curve_hs], color='blue', label='CRHybrid == CRSlow', s=5)
    plt.scatter(P[curve_sf], V[curve_sf], color='green', label='CRSlow == CRFast', s=5)

    plt.xlabel('p')
    plt.ylabel('v')
    plt.title('Comparison Curves: CRHybrid, CRFast, CRSlow')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('comparison_curves.png')
    plt.close()

def plot_advantage_regions(V: np.ndarray,
                           P: np.ndarray,
                           CRFast: np.ndarray,
                           CRSlow: np.ndarray,
                           CRHybrid: np.ndarray):
    """
    Plots regions where CRHybrid is better than CRFast and where CRSlow is better than CRFast.

    Args:
        V (np.ndarray): Meshgrid of v values.
        P (np.ndarray): Meshgrid of p values.
        CRFast (np.ndarray): Competitive ratio of fast search.
        CRSlow (np.ndarray): Competitive ratio of slow search.
        CRHybrid (np.ndarray): Competitive ratio of hybrid search.
    """
    # Compute regions
    hybrid_better_than_fast = (CRHybrid < CRFast)  # Where CRHybrid is better than CRFast
    slow_better_than_fast = (CRSlow < CRFast)  # Where CRSlow is better than CRFast

    plt.figure(figsize=(8, 6))

    # Plot the regions
    plt.contourf(P, V, hybrid_better_than_fast, levels=1, colors=['blue'], alpha=0.5, label="CRHybrid < CRFast")
    plt.contourf(P, V, slow_better_than_fast, levels=1, colors=['green'], alpha=0.5, label="CRSlow < CRFast")

    # Contours for visualization
    plt.contour(P, V, CRHybrid, levels=20, cmap='coolwarm', alpha=0.3)

    # Labels and legend
    plt.xlabel('p')
    plt.ylabel('v')
    plt.title('Regions where CRHybrid and CRSlow are Better than CRFast')
    plt.legend(["CRHybrid < CRFast (Blue)", "CRSlow < CRFast (Green)"])
    plt.grid(True)

    # Save and show
    plt.savefig('advantage_regions.png')
    # plt.show()
    plt.close()


def main():
    # Define p_vals and v_vals
    p_vals = np.linspace(0.0, 1, 100)
    v_vals = np.linspace(0.0, 1, 100)

    # Create meshgrid
    P, V = np.meshgrid(p_vals, v_vals)

    # Compute CRFast and CRSlow
    CRFast = cr_fast(P)
    CRSlow = cr_slow(V)
    A, B, CRHybrid = cr_hybrid(V, P)

    # Find index ranges where values are greater than 0.25
    p_index = np.where(p_vals > 0.25)[0]
    v_index = np.where(v_vals > 0.25)[0]
    idx = np.ix_(v_index, p_index)

    plot_results(
        V=V[idx],
        P=P[idx],
        CRFast=CRFast[idx],
        CRSlow=CRSlow[idx],
        CRHybrid=CRHybrid[idx]
    )
    plot_comparison_curves(
        V=V, P=P, CRFast=CRFast, CRSlow=CRSlow, CRHybrid=CRHybrid
    )
    plot_advantage_regions(
        V=V, P=P, CRFast=CRFast, CRSlow=CRSlow, CRHybrid=CRHybrid
    )
    


if __name__ == "__main__":
    main()
