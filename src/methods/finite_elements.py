import numpy as np
from scipy.linalg import solve


def fem_price(S0, K, r, sigma, T, **kwargs):
    """
    Solves the Black–Scholes PDE using the Finite Element Method with
    piecewise linear basis functions and Crank–Nicolson time stepping.
    
    PDE: V_t + 0.5*sigma^2 S^2 V_SS + rS V_S - r V = 0,  t in [0, T],
         with terminal condition: V(S, T) = max(S-K,0),
         and Dirichlet BCs: V(0,t)=0, V(S_max,t)=S_max-K*exp(-r*(T-t)).
    
    We discretize the spatial domain [0, S_max] into M elements (M+1 nodes) and
    the time interval [0, T] into N steps. The PDE is solved backward in time.
    
    Parameters (via kwargs):
      - S_max: maximum asset price (default 300)
      - M: number of elements (default 200)
      - N: number of time steps (default 200)
    
    Returns:
      - Option price at S = S0 at t=0.
    """
    # Set parameters
    S_max = kwargs.get("S_max", 300)
    M = kwargs.get("M", 200)  # number of elements; there will be M+1 nodes
    N = kwargs.get("N", 200)  # number of time steps
    h = S_max / M
    dt = T / N

    # Spatial grid: nodes S_0, S_1, ..., S_M
    S_nodes = np.linspace(0, S_max, M+1)
    
    # Initialize global matrices for mass M_mat and operator A_mat (both (M+1)x(M+1))
    M_mat = np.zeros((M+1, M+1))
    A_mat = np.zeros((M+1, M+1))
    
    # Two-point Gauss-Legendre quadrature on reference interval [-1, 1]
    quad_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    quad_wts = np.array([1.0, 1.0])
    
    # Loop over elements (each element e spans nodes i and i+1)
    for i in range(M):
        a = S_nodes[i]
        b = S_nodes[i+1]
        he = b - a  # element length (should be equal to h)
        # Map quadrature points from [-1,1] to [a,b]: S = (he/2)*xi + (a+b)/2
        S_q = (he/2)*quad_pts + (a+b)/2
        # Jacobian for the transformation
        J = he / 2

        # Local basis functions and derivatives on element e:
        # phi0(S) = (b - S) / he, phi1(S) = (S - a) / he.
        # Their derivatives are constant: dphi0/dS = -1/he, dphi1/dS = 1/he.
        phi0 = (b - S_q) / he
        phi1 = (S_q - a) / he
        dphi0 = -np.ones_like(S_q) / he
        dphi1 = np.ones_like(S_q) / he
        
        # Initialize local matrices (2x2)
        M_local = np.zeros((2,2))
        A_local = np.zeros((2,2))
        
        # Loop over quadrature points to compute local integrals
        for q in range(len(quad_pts)):
            wq = quad_wts[q]
            S_val = S_q[q]
            # Mass term: phi_i * phi_j
            M_local[0,0] += phi0[q] * phi0[q] * wq * J
            M_local[0,1] += phi0[q] * phi1[q] * wq * J
            M_local[1,0] += phi1[q] * phi0[q] * wq * J
            M_local[1,1] += phi1[q] * phi1[q] * wq * J
            
            # Diffusion term: 0.5*sigma^2 * S^2 * phi'_i * phi'_j
            diff = 0.5 * sigma**2 * S_val**2
            A_local[0,0] += diff * dphi0[q] * dphi0[q] * wq * J
            A_local[0,1] += diff * dphi0[q] * dphi1[q] * wq * J
            A_local[1,0] += diff * dphi1[q] * dphi0[q] * wq * J
            A_local[1,1] += diff * dphi1[q] * dphi1[q] * wq * J
            
            # Convection term: r * S * (phi'_j) * (phi_i)
            # Note: we sum for each pair (i,j); here we use the convention: test function index i,
            # trial function index j.
            A_local[0,0] += r * S_val * dphi0[q] * phi0[q] * wq * J
            A_local[0,1] += r * S_val * dphi1[q] * phi0[q] * wq * J
            A_local[1,0] += r * S_val * dphi0[q] * phi1[q] * wq * J
            A_local[1,1] += r * S_val * dphi1[q] * phi1[q] * wq * J
            
            # Reaction term: -r * phi_i * phi_j (added to operator)
            A_local[0,0] += -r * phi0[q] * phi0[q] * wq * J
            A_local[0,1] += -r * phi0[q] * phi1[q] * wq * J
            A_local[1,0] += -r * phi1[q] * phi0[q] * wq * J
            A_local[1,1] += -r * phi1[q] * phi1[q] * wq * J
        
        # Assemble into global matrices (add contributions to nodes i and i+1)
        indices = [i, i+1]
        for ii in range(2):
            for jj in range(2):
                M_mat[indices[ii], indices[jj]] += M_local[ii, jj]
                A_mat[indices[ii], indices[jj]] += A_local[ii, jj]
    
    # --- Time stepping using Crank–Nicolson (backward in time) ---
    # We solve: (M + dt/2*A) V^n = (M - dt/2*A) V^{n+1}, for n = N-1,...,0,
    # with Dirichlet boundary conditions:
    #   V(0, t) = 0,  V(S_max, t) = S_max - K*exp(-r*(T-t)).
    A1 = M_mat + (dt/2)*A_mat
    B1 = M_mat - (dt/2)*A_mat
    
    # Terminal condition at t = T: V(S,T) = max(S-K,0)
    V = np.maximum(S_nodes - K, 0)
    
    # Time stepping: n = N-1 down to 0, where t_n = n*dt.
    for n in range(N-1, -1, -1):
        t = n * dt
        # Right-hand side for current time step:
        RHS = B1.dot(V)
        # Impose Dirichlet BC in the solution V^n:
        # At S=0: V=0.
        RHS[0] = 0
        # At S=S_max: V = S_max - K*exp(-r*t)
        RHS[-1] = S_max - K * np.exp(-r * t)
        
        # Modify the system matrix A1 for boundary nodes.
        A_mod = A1.copy()
        # For node 0:
        A_mod[0, :] = 0
        A_mod[0, 0] = 1
        # For node M:
        A_mod[-1, :] = 0
        A_mod[-1, -1] = 1
        
        # Solve for V at current time step.
        V = solve(A_mod, RHS)
    
    # Interpolate to get the price at S = S0.
    price = np.interp(S0, S_nodes, V)
    return price