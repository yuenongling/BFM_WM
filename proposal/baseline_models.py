"""
Baseline wall model predictors for comparison
"""
import numpy as np
from scipy.optimize import fsolve
from typing import Optional, Tuple, Dict, Any

# Constants for wall models
KAPPA = 0.41
B = 5.2
YSTAR = 11.0  # Transition y+ value between viscous sublayer and log region
A = 0.5  # Coefficient for quadratic term in near-wall model

def log_law(utau, y, nu, u):
    """
    Log law equation for wall-bounded turbulent flows
    
    Args:
        utau: Friction velocity
        y: Wall-normal distance
        nu: Kinematic viscosity
        u: Streamwise velocity
    
    Returns:
        Residual of the log law equation
    """
    eq = u / utau - B - 1 / KAPPA * np.log(y * utau / nu)
    return eq

def near_wall(utau, y, nu, u):
    """
    Near-wall equation for viscous sublayer
    
    Args:
        utau: Friction velocity
        y: Wall-normal distance
        nu: Kinematic viscosity
        u: Streamwise velocity
    
    Returns:
        Residual of the near-wall equation
    """
    eq = u / utau - y * utau / nu - A * (y * utau / nu)**2
    return eq

def log_law_solve(y, nu, u, initial=0.001, tol=1e-5):
    """
    Solve for friction velocity using log law or near-wall model
    
    Args:
        y: Wall-normal distance
        nu: Kinematic viscosity
        u: Streamwise velocity
        initial: Initial guess for friction velocity
        tol: Tolerance for convergence
    
    Returns:
        Solved friction velocity
    """
    sol = fsolve(log_law, initial, args=(y, nu, u))
    
    # Check the y in plus units
    yplus = y * sol / nu
    eq = log_law(sol, y, nu, u)
    
    if yplus < YSTAR:
        # If yplus is less than YSTAR, use the near-wall model
        sol = fsolve(near_wall, initial, args=(y, nu, u))
        eq = near_wall(sol, y, nu, u)

    if abs(eq) > tol:
        print(f"Warning: log_law_solve did not converge for y={y}, nu={nu}, u={u}")

    return sol

def log_law_Re(utau, y, Re, u):
    """
    Log law equation using Reynolds number instead of viscosity
    
    Args:
        utau: Friction velocity
        y: Normalized wall-normal distance
        Re: Reynolds number
        u: Normalized streamwise velocity
    
    Returns:
        Residual of the log law equation
    """
    eq = abs(u) / utau - B - 1 / KAPPA * np.log(abs(y * utau * Re))
    return eq

def near_wall_Re(utau, y, Re, u):
    """
    Near-wall equation using Reynolds number instead of viscosity
    
    Args:
        utau: Friction velocity
        y: Normalized wall-normal distance
        Re: Reynolds number
        u: Normalized streamwise velocity
    
    Returns:
        Residual of the near-wall equation
    """
    eq = abs(u) / utau - y * utau * Re - A * (y * utau * Re)**2
    return eq

def log_law_solve_Re(y, Re, u, initial=0.001, tol=1e-5):
    """
    Solve for friction velocity using Reynolds number formulation
    
    Args:
        y: Normalized wall-normal distance
        Re: Reynolds number
        u: Normalized streamwise velocity
        initial: Initial guess for friction velocity
        tol: Tolerance for convergence
    
    Returns:
        Solved friction velocity
    """
    sol = fsolve(log_law_Re, initial, args=(y, Re, u))
    
    # Check the y in plus units
    yplus = y * sol * Re
    eq = log_law_Re(sol, y, Re, u)
    
    if yplus < YSTAR:
        # If yplus is less than YSTAR, use the near-wall model
        sol = fsolve(near_wall_Re, initial, args=(y, Re, u))
        eq = near_wall_Re(sol, y, Re, u)

    if abs(eq) > tol:
        print(f"Warning: log_law_solve_Re did not converge for y={y}, Re={Re}, u={u}")

    return sol

class LogLawPredictor:
    """
    Log law predictor as a baseline model for wall shear stress
    """
    
    def __init__(self, kappa: float = KAPPA, B: float = B, ystar: float = YSTAR):
        """
        Initialize log law predictor
        
        Args:
            kappa: von Karman constant
            B: Log law intercept constant
            ystar: Transition y+ value
        """
        self.kappa = kappa
        self.B = B
        self.ystar = ystar
    
    def predict(self, 
               unnormalized_inputs: np.ndarray,
               flow_type: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict wall shear stress using the log law
        
        Args:
            unnormalized_inputs: Unnormalized inputs with physical quantities
                               Expected columns: [y, U, nu, ...] 
            flow_type: Optional flow type information
            
        Returns:
            Predicted wall shear stress values
        """
        # Extract needed values
        y = unnormalized_inputs[:, 0]  # Wall-normal distance
        U = unnormalized_inputs[:, 1]  # Velocity
        nu = unnormalized_inputs[:, 2]  # Kinematic viscosity
        
        # Initialize predictions
        u_tau = np.zeros_like(y)
        
        # Solve for u_tau at each point
        for i in range(len(y)):
            u_tau[i] = log_law_solve(y[i], nu[i], U[i])
        
        # Convert u_tau to wall shear stress (τ_w = ρ * u_tau²)
        tau_w = u_tau**2
        
        return tau_w

class WallFunctionPredictor:
    """
    Enhanced wall function predictor for wall shear stress
    """
    
    def __init__(self, kappa: float = KAPPA, B: float = B, 
                y_visc: float = YSTAR, a: float = 8.3, b: float = 1/7):
        """
        Initialize enhanced wall function predictor
        
        Args:
            kappa: von Karman constant
            B: Log law intercept constant
            y_visc: Upper limit of viscous sublayer
            a: Power law coefficient
            b: Power law exponent
        """
        self.kappa = kappa
        self.B = B
        self.y_visc = y_visc
        self.a = a
        self.b = b
    
    def predict(self, 
               unnormalized_inputs: np.ndarray,
               flow_type: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict wall shear stress using enhanced wall functions
        
        Args:
            unnormalized_inputs: Unnormalized inputs with physical quantities
                               Expected columns: [y, U, nu, ...] 
            flow_type: Optional flow type information
            
        Returns:
            Predicted wall shear stress values
        """
        # Extract needed values
        y = unnormalized_inputs[:, 0]  # Wall-normal distance
        U = unnormalized_inputs[:, 1]  # Velocity
        nu = unnormalized_inputs[:, 2]  # Kinematic viscosity
        
        # Initialize predictions
        u_tau_guess = np.zeros_like(y)
        
        # Iterative solution for each point
        for i in range(len(y)):
            # Initial guess based on power law
            u_tau = 0.05 * U[i]
            
            # Iterative refinement (typically converges in <10 iterations)
            for _ in range(20):
                y_plus = y[i] * u_tau / nu[i]
                
                # Different treatment based on y+ value
                if y_plus < self.y_visc:  # Viscous sublayer
                    u_plus = y_plus
                elif y_plus < 30:  # Buffer layer - use blended formulation
                    # Blend viscous and log law
                    weight = (y_plus - self.y_visc) / (30 - self.y_visc)
                    u_plus_visc = y_plus
                    u_plus_log = (1/self.kappa) * np.log(y_plus) + self.B
                    u_plus = (1-weight) * u_plus_visc + weight * u_plus_log
                elif y_plus < 300:  # Log law region
                    u_plus = (1/self.kappa) * np.log(y_plus) + self.B
                else:  # Wake region - use power law
                    u_plus = self.a * y_plus**self.b
                
                # Update u_tau
                u_tau_new = U[i] / u_plus
                
                # Check convergence
                if abs(u_tau_new - u_tau) / u_tau < 1e-6:
                    break
                
                # Apply relaxation for stability
                u_tau = 0.8 * u_tau + 0.2 * u_tau_new
            
            u_tau_guess[i] = u_tau
        
        # Convert u_tau to wall shear stress
        tau_w = u_tau_guess**2
        
        return tau_w

class EqWallModelPredictor:
    """
    Equilibrium wall model predictor
    """
    
    def __init__(self, kappa: float = KAPPA, B: float = B, 
                ystar: float = YSTAR, max_iter: int = 50, tol: float = 1e-6):
        """
        Initialize equilibrium wall model predictor
        
        Args:
            kappa: von Karman constant
            B: Log law intercept constant
            ystar: Transition y+ value
            max_iter: Maximum iterations for solver
            tol: Convergence tolerance
        """
        self.kappa = kappa
        self.B = B
        self.ystar = ystar
        self.max_iter = max_iter
        self.tol = tol
    
    def predict(self, 
               unnormalized_inputs: np.ndarray,
               flow_type: Optional[np.ndarray] = None,
               pressure_gradient: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict wall shear stress using equilibrium wall model approach
        
        Args:
            unnormalized_inputs: Unnormalized inputs with physical quantities
                               Expected columns: [y, U, nu, ...] 
            flow_type: Optional flow type information
            pressure_gradient: Optional pressure gradient term
            
        Returns:
            Predicted wall shear stress values
        """
        # Extract needed values
        y = unnormalized_inputs[:, 0]  # Wall-normal distance
        U = unnormalized_inputs[:, 1]  # Velocity
        nu = unnormalized_inputs[:, 2]  # Kinematic viscosity
        
        # Use pressure gradient if provided, otherwise assume zero
        dp_dx = np.zeros_like(y) if pressure_gradient is None else pressure_gradient
        
        # Initialize predictions
        tau_w = np.zeros_like(y)
        
        # Process each point using the functionality from log_law_solve but with pressure gradient
        for i in range(len(y)):
            def eq_with_pressure(utau, y_val, nu_val, u_val, dp_dx_val):
                yplus = y_val * utau / nu_val
                if yplus < self.ystar:
                    # Near-wall model with pressure gradient
                    eq = u_val / utau - y_val * utau / nu_val - A * (y_val * utau / nu_val)**2
                else:
                    # Log law with pressure gradient
                    eq = u_val / utau - self.B - 1 / self.kappa * np.log(y_val * utau / nu_val)
                
                # Add pressure gradient term if significant
                if abs(dp_dx_val) > 1e-10:
                    eq -= y_val * dp_dx_val / utau**2
                    
                return eq
            
            # Initial guess
            utau_guess = 0.05 * U[i]
            
            # Solve for u_tau with pressure gradient
            sol = fsolve(eq_with_pressure, utau_guess, args=(y[i], nu[i], U[i], dp_dx[i]))
            
            # Store result
            tau_w[i] = sol**2
            
        return tau_w
