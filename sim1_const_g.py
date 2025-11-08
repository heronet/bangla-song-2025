import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical constants
G = 9.81  # m/s^2 - gravitational acceleration
RHO_SEA = 1.225  # kg/m^3 at sea level - air density
MASS = 5  # kg - projectile mass
DIAMETER = 0.11  # m - projectile diameter
RADIUS = DIAMETER / 2
AREA = np.pi * RADIUS**2
CD = 0.47  # drag coefficient for sphere

# Atmospheric constants
SCALE_HEIGHT = 8500  # meters - atmospheric scale height (exponential air density decay)
# This is derived from atmospheric physics: H = RT/Mg where:
# R = gas constant, T = temperature, M = molar mass of air, g = gravity
# For Earth's troposphere, this is approximately 8500m

def air_density(altitude):
    """
    Calculate air density at given altitude using barometric formula
    ρ(h) = ρ₀ * exp(-h/H) where H is the scale height
    """
    return RHO_SEA * np.exp(-altitude / SCALE_HEIGHT)

def derivatives(state, t, rho):
    """
    Calculate derivatives for projectile motion with air resistance
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    
    # Calculate velocity magnitude
    v = np.sqrt(vx**2 + vy**2)
    
    # Calculate drag force magnitude
    if v > 0:
        drag = 0.5 * rho * CD * AREA * v**2 / MASS
        
        # Acceleration components
        ax = -drag * vx / v
        ay = -G - drag * vy / v
    else:
        ax = 0
        ay = -G
    
    return [vx, vy, ax, ay]

def simulate_trajectory(v0, angle_deg, altitude, dt=0.01, max_time=60):
    """
    Simulate projectile trajectory with air resistance
    Returns trajectory points and final position
    """
    angle_rad = np.radians(angle_deg)
    rho = air_density(altitude)
    
    # Initial conditions [x, y, vx, vy]
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    state0 = [0, 0, vx0, vy0]
    
    # Time array
    t = np.arange(0, max_time, dt)
    
    # Solve ODE
    solution = odeint(derivatives, state0, t, args=(rho,))
    
    # Extract x, y coordinates
    x = solution[:, 0]
    y = solution[:, 1]
    
    # Find where projectile hits ground (y <= 0)
    ground_idx = np.where(y <= 0)[0]
    if len(ground_idx) > 1:
        idx = ground_idx[1]  # First point after launch where y <= 0
        x = x[:idx+1]
        y = y[:idx+1]
        t = t[:idx+1]
    
    return x, y, t

def simulate_trajectory_euler(v0, angle_deg, altitude, dt=0.01, max_time=60):
    """
    Simulate trajectory using Euler method (alternative to odeint)
    More explicit control over the integration
    """
    angle_rad = np.radians(angle_deg)
    rho = air_density(altitude)
    
    # Initial conditions
    x, y = 0, 0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    # Storage arrays
    x_arr, y_arr, t_arr = [x], [y], [0]
    t = 0
    
    while y >= 0 and t < max_time and x < 5000:
        # Calculate velocity magnitude
        v = np.sqrt(vx**2 + vy**2)
        
        # Calculate drag
        if v > 0:
            drag = 0.5 * rho * CD * AREA * v**2 / MASS
            ax = -drag * vx / v
            ay = -G - drag * vy / v
        else:
            ax = 0
            ay = -G
        
        # Update velocities
        vx += ax * dt
        vy += ay * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        t += dt
        
        x_arr.append(x)
        y_arr.append(y)
        t_arr.append(t)
    
    return np.array(x_arr), np.array(y_arr), np.array(t_arr)

def find_solution(target_distance, altitude, max_velocity=450, method='euler'):
    """
    Find velocity and angle combination to hit target
    
    Args:
        target_distance: horizontal distance to target (m)
        altitude: altitude above sea level (m)
        max_velocity: maximum available muzzle velocity (m/s)
        method: 'euler' or 'odeint' for integration method
    """
    print(f"\nSearching for solution...")
    print(f"Target: {target_distance}m at altitude {altitude}m")
    print(f"Max velocity available: {max_velocity} m/s\n")
    
    best_error = float('inf')
    best_solution = None
    solutions = []
    
    # Define search range
    MIN_VELOCITY = 50  # minimum reasonable velocity (m/s)
    VELOCITY_STEP_COARSE = 10  # m/s - coarse search step
    MIN_ANGLE = 10  # degrees - minimum firing angle
    MAX_ANGLE = 80  # degrees - maximum firing angle
    ANGLE_STEP_COARSE = 2  # degrees - coarse search step

    # Coarse search
    for v0 in range(MIN_VELOCITY, max_velocity + 1, VELOCITY_STEP_COARSE):
        for angle in range(MIN_ANGLE, MAX_ANGLE + 1, ANGLE_STEP_COARSE):
            if method == 'euler':
                x, y, t = simulate_trajectory_euler(v0, angle, altitude)
            else:
                x, y, t = simulate_trajectory(v0, angle, altitude)
            
            final_x = x[-1]
            error = abs(final_x - target_distance)
            
            COARSE_ERROR_THRESHOLD = 20  # meters - if within this, consider for refinement
            if error < COARSE_ERROR_THRESHOLD:
                solutions.append({
                    'velocity': v0,
                    'angle': angle,
                    'error': error,
                    'range': final_x
                })
                
                if error < best_error:
                    best_error = error
                    best_solution = {
                        'velocity': v0,
                        'angle': angle,
                        'x': x,
                        'y': y,
                        't': t,
                        'error': error
                    }
    
    if best_solution is None:
        print("No solution found - target may be out of range!")
        return None
    
    # Fine-tune the best solution
    print("Refining solution...")
    v0_init = best_solution['velocity']
    angle_init = best_solution['angle']
    
    VELOCITY_RANGE_FINE = 15  # m/s - search range around best coarse solution
    VELOCITY_STEP_FINE = 1  # m/s - fine search step
    ANGLE_RANGE_FINE = 5  # degrees - search range around best coarse solution
    ANGLE_STEP_FINE = 0.5  # degrees - fine search step
    TIME_STEP_FINE = 0.005  # seconds - smaller time step for accuracy
    

    for v0 in np.arange(v0_init - VELOCITY_RANGE_FINE, v0_init + VELOCITY_RANGE_FINE, VELOCITY_STEP_FINE):
        for angle in np.arange(angle_init - ANGLE_RANGE_FINE, angle_init + ANGLE_RANGE_FINE, ANGLE_STEP_FINE):
            if method == 'euler':
                x, y, t = simulate_trajectory_euler(v0, angle, altitude, dt=TIME_STEP_FINE)
            else:
                x, y, t = simulate_trajectory(v0, angle, altitude, dt=TIME_STEP_FINE)
            
            final_x = x[-1]
            error = abs(final_x - target_distance)
            
            if error < best_error:
                best_error = error
                best_solution = {
                    'velocity': v0,
                    'angle': angle,
                    'x': x,
                    'y': y,
                    't': t,
                    'error': error
                }
    
    return best_solution

def plot_trajectory(solution, target_distance):
    """Plot the trajectory"""
    plt.figure(figsize=(12, 6))
    
    x = solution['x']
    y = solution['y']
    
    plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    plt.plot(target_distance, 0, 'r*', markersize=15, label='Target')
    plt.plot(0, 0, 'go', markersize=10, label='Cannon')
    
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title(f'Artillery Trajectory - v₀={solution["velocity"]:.1f} m/s, θ={solution["angle"]:.1f}°', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def print_solution(solution, target_distance):
    """Print solution details"""
    max_height = np.max(solution['y'])
    flight_time = solution['t'][-1]
    final_range = solution['x'][-1]
    
    print("\n" + "="*50)
    print("SOLUTION FOUND")
    print("="*50)
    print(f"Muzzle Velocity:  {solution['velocity']:.2f} m/s")
    print(f"Firing Angle:     {solution['angle']:.2f}°")
    print(f"Flight Time:      {flight_time:.2f} s")
    print(f"Maximum Height:   {max_height:.1f} m")
    print(f"Final Range:      {final_range:.1f} m")
    print(f"Target Distance:  {target_distance} m")
    print(f"Error:            {solution['error']:.2f} m")
    print("="*50 + "\n")

# Main execution
if __name__ == "__main__":
    # Problem parameters
    TARGET_DISTANCE = 1200  # meters
    ALTITUDE = 500  # meters
    MAX_VELOCITY = 450  # m/s
    
    print("\n" + "="*50)
    print("ARTILLERY TRAJECTORY SIMULATOR")
    print("="*50)
    print(f"Projectile mass: {MASS} kg")
    print(f"Projectile diameter: {DIAMETER*100} cm")
    print(f"Drag coefficient: {CD}")
    print(f"Air density at {ALTITUDE}m: {air_density(ALTITUDE):.3f} kg/m³")
    
    # Find solution
    solution = find_solution(TARGET_DISTANCE, ALTITUDE, MAX_VELOCITY, method='euler')
    
    if solution:
        print_solution(solution, TARGET_DISTANCE)
        plot_trajectory(solution, TARGET_DISTANCE)
        
        # Compare with no-drag case
        angle_no_drag = 45  # optimal angle without drag
        v0_no_drag = np.sqrt(TARGET_DISTANCE * G / np.sin(2 * np.radians(angle_no_drag)))
        print(f"Without air resistance:")
        print(f"  Required velocity at 45°: {v0_no_drag:.1f} m/s")
        print(f"  Difference: +{solution['velocity'] - v0_no_drag:.1f} m/s due to drag")