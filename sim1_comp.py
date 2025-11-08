import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical constants
G_SEA_LEVEL = 9.81  # m/s^2 - gravitational acceleration at sea level
RHO_SEA = 1.225  # kg/m^3 at sea level - air density
MASS = 5  # kg - projectile mass
DIAMETER = 0.11  # m - projectile diameter
RADIUS = DIAMETER / 2
AREA = np.pi * RADIUS**2
CD = 0.47  # drag coefficient for sphere

# Earth and atmospheric constants
EARTH_RADIUS = 6371000  # meters - Earth's mean radius
GAS_CONSTANT = 8.314  # J/(mol·K) - universal gas constant
MOLAR_MASS_AIR = 0.029  # kg/mol - molar mass of air
TEMPERATURE = 288  # K - standard temperature (15°C)
SCALE_HEIGHT_SEA = 8500  # meters - atmospheric scale height at sea level (reference)

def gravity(altitude):
    """
    Calculate gravitational acceleration at given altitude
    g(h) = g₀ × (R / (R + h))²
    
    Args:
        altitude: height above sea level (m)
    Returns:
        gravitational acceleration (m/s²)
    """
    return G_SEA_LEVEL * (EARTH_RADIUS / (EARTH_RADIUS + altitude))**2

def scale_height(altitude):
    """
    Calculate atmospheric scale height at given altitude
    H = RT / (Mg)
    
    Since g varies with altitude, H also varies.
    
    Args:
        altitude: height above sea level (m)
    Returns:
        scale height (m)
    """
    g = gravity(altitude)
    return (GAS_CONSTANT * TEMPERATURE) / (MOLAR_MASS_AIR * g)

def air_density(altitude, use_variable_H=True):
    """
    Calculate air density at given altitude
    
    Args:
        altitude: height above sea level (m)
        use_variable_H: if True, use variable scale height; if False, use constant
    Returns:
        air density (kg/m³)
    """
    if altitude <= 0:
        return RHO_SEA
    
    if use_variable_H:
        # Calculate average scale height between sea level and altitude
        H_sea = scale_height(0)
        H_alt = scale_height(altitude)
        H_avg = (H_sea + H_alt) / 2
        return RHO_SEA * np.exp(-altitude / H_avg)
    else:
        # Use constant scale height
        return RHO_SEA * np.exp(-altitude / SCALE_HEIGHT_SEA)

def air_density_accurate(altitude, num_steps=100):
    """
    More accurate calculation of air density using numerical integration
    Divides the altitude into small steps and integrates dρ/dh = -ρ/H(h)
    
    This is more accurate but slower - use for verification
    
    Args:
        altitude: height above sea level (m)
        num_steps: number of integration steps
    Returns:
        air density (kg/m³)
    """
    if altitude <= 0:
        return RHO_SEA
    
    dh = altitude / num_steps
    rho = RHO_SEA
    
    for i in range(num_steps):
        h = i * dh
        H = scale_height(h)
        rho = rho * np.exp(-dh / H)
    
    return rho

def derivatives(state, t, base_altitude, use_variable_g=True, use_variable_H=True):
    """
    Calculate derivatives for projectile motion with air resistance
    state = [x, y, vx, vy]
    
    Args:
        state: [x, y, vx, vy]
        t: time
        base_altitude: altitude of the launch point above sea level
        use_variable_g: if True, use g(h); if False, use constant g
        use_variable_H: if True, use variable scale height for air density
    """
    x, y, vx, vy = state
    
    # Current altitude above sea level
    current_altitude = base_altitude + y
    
    # Get gravitational acceleration
    if use_variable_g:
        g = gravity(current_altitude)
    else:
        g = G_SEA_LEVEL
    
    # Get air density
    rho = air_density(current_altitude, use_variable_H=use_variable_H)
    
    # Calculate velocity magnitude
    v = np.sqrt(vx**2 + vy**2)
    
    # Calculate drag force magnitude
    if v > 0:
        drag = 0.5 * rho * CD * AREA * v**2 / MASS
        
        # Acceleration components
        ax = -drag * vx / v
        ay = -g - drag * vy / v
    else:
        ax = 0
        ay = -g
    
    return [vx, vy, ax, ay]

def simulate_trajectory(v0, angle_deg, altitude, dt=0.01, max_time=60):
    """
    Simulate projectile trajectory with air resistance using scipy.odeint
    Now with variable g and air density
    
    Returns trajectory points and final position
    """
    angle_rad = np.radians(angle_deg)
    
    # Initial conditions [x, y, vx, vy]
    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    state0 = [0, 0, vx0, vy0]
    
    # Time array
    t = np.arange(0, max_time, dt)
    
    # Solve ODE with altitude parameter
    solution = odeint(derivatives, state0, t, args=(altitude,))
    
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

def simulate_trajectory_euler(v0, angle_deg, altitude, dt=0.01, max_time=60, use_variable_g=True, use_variable_H=True):
    """
    Simulate trajectory using Euler method
    
    Args:
        v0: initial velocity (m/s)
        angle_deg: launch angle (degrees)
        altitude: launch altitude above sea level (m)
        dt: time step (s)
        max_time: maximum simulation time (s)
        use_variable_g: if True, use g(h); if False, use constant g
        use_variable_H: if True, use variable H for air density
    """
    angle_rad = np.radians(angle_deg)
    
    # Initial conditions
    x, y = 0, 0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    # Storage arrays
    x_arr, y_arr, t_arr = [x], [y], [0]
    t = 0
    
    MAX_RANGE = 5000  # meters - stop if projectile goes too far
    
    while y >= 0 and t < max_time and x < MAX_RANGE:
        # Current altitude above sea level
        current_altitude = altitude + y
        
        # Get parameters at current altitude
        if use_variable_g:
            g = gravity(current_altitude)
        else:
            g = G_SEA_LEVEL
            
        rho = air_density(current_altitude, use_variable_H=use_variable_H)
        
        # Calculate velocity magnitude
        v = np.sqrt(vx**2 + vy**2)
        
        # Calculate drag
        if v > 0:
            drag = 0.5 * rho * CD * AREA * v**2 / MASS
            ax = -drag * vx / v
            ay = -g - drag * vy / v
        else:
            ax = 0
            ay = -g
        
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

def find_solution(target_distance, altitude, max_velocity=450, method='euler', use_variable_g=True, use_variable_H=True):
    """
    Find velocity and angle combination to hit target
    
    Args:
        target_distance: horizontal distance to target (m)
        altitude: altitude above sea level (m)
        max_velocity: maximum available muzzle velocity (m/s)
        method: 'euler' or 'odeint' for integration method
        use_variable_g: if True, use g(h); if False, use constant g
        use_variable_H: if True, use variable H for air density
    """
    mode_str = f"{'Variable' if use_variable_g else 'Constant'} g, {'Variable' if use_variable_H else 'Constant'} H"
    
    print(f"\nSearching for solution ({mode_str})...")
    print(f"Target: {target_distance}m at altitude {altitude}m")
    print(f"Max velocity available: {max_velocity} m/s")
    if use_variable_g:
        print(f"Gravity at {altitude}m: {gravity(altitude):.4f} m/s²")
    else:
        print(f"Gravity (constant): {G_SEA_LEVEL} m/s²")
    print(f"Air density at {altitude}m: {air_density(altitude, use_variable_H):.4f} kg/m³")
    if use_variable_g:
        print(f"Scale height at {altitude}m: {scale_height(altitude):.1f} m\n")
    else:
        print(f"Scale height (constant): {SCALE_HEIGHT_SEA} m\n")
    
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
                x, y, t = simulate_trajectory_euler(v0, angle, altitude, 
                                                    use_variable_g=use_variable_g, 
                                                    use_variable_H=use_variable_H)
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
                x, y, t = simulate_trajectory_euler(v0, angle, altitude, dt=TIME_STEP_FINE,
                                                    use_variable_g=use_variable_g,
                                                    use_variable_H=use_variable_H)
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

def plot_trajectory(solution, target_distance, altitude):
    """Plot the trajectory with altitude information"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = solution['x']
    y = solution['y']
    
    # Plot 1: Trajectory
    ax1.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    ax1.plot(target_distance, 0, 'r*', markersize=15, label='Target')
    ax1.plot(0, 0, 'go', markersize=10, label='Cannon')
    ax1.axhline(y=0, color='brown', linestyle='--', alpha=0.5, label='Ground level')
    
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Height above launch point (m)', fontsize=12)
    ax1.set_title(f'Artillery Trajectory - v₀={solution["velocity"]:.1f} m/s, θ={solution["angle"]:.1f}°', 
              fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Variable parameters along trajectory
    altitudes = altitude + y
    g_values = [gravity(alt) for alt in altitudes]
    rho_values = [air_density(alt) for alt in altitudes]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(x, g_values, 'r-', linewidth=2, label='Gravity (g)')
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('Gravity (m/s²)', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    line2 = ax2_twin.plot(x, rho_values, 'b-', linewidth=2, label='Air density (ρ)')
    ax2_twin.set_ylabel('Air Density (kg/m³)', fontsize=12, color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title('Variable Parameters Along Trajectory', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def print_solution(solution, target_distance, altitude):
    """Print solution details with variable parameter information"""
    max_height = np.max(solution['y'])
    flight_time = solution['t'][-1]
    final_range = solution['x'][-1]
    
    # Calculate parameters at key points
    max_height_idx = np.argmax(solution['y'])
    max_altitude = altitude + max_height
    
    print("\n" + "="*60)
    print("SOLUTION FOUND")
    print("="*60)
    print(f"Muzzle Velocity:      {solution['velocity']:.2f} m/s")
    print(f"Firing Angle:         {solution['angle']:.2f}°")
    print(f"Flight Time:          {flight_time:.2f} s")
    print(f"Maximum Height:       {max_height:.1f} m above launch point")
    print(f"Maximum Altitude:     {max_altitude:.1f} m above sea level")
    print(f"Final Range:          {final_range:.1f} m")
    print(f"Target Distance:      {target_distance} m")
    print(f"Error:                {solution['error']:.2f} m")
    print("="*60)
    print("\nVariable Parameters:")
    print(f"At launch (altitude={altitude}m):")
    print(f"  g = {gravity(altitude):.5f} m/s²")
    print(f"  ρ = {air_density(altitude):.5f} kg/m³")
    print(f"  H = {scale_height(altitude):.1f} m")
    print(f"\nAt max height (altitude={max_altitude:.1f}m):")
    print(f"  g = {gravity(max_altitude):.5f} m/s²")
    print(f"  ρ = {air_density(max_altitude):.5f} kg/m³")
    print(f"  H = {scale_height(max_altitude):.1f} m")
    print(f"\nChange in g: {gravity(altitude) - gravity(max_altitude):.6f} m/s² ({100*(gravity(altitude) - gravity(max_altitude))/gravity(altitude):.4f}%)")
    print(f"Change in ρ: {air_density(altitude) - air_density(max_altitude):.6f} kg/m³ ({100*(air_density(altitude) - air_density(max_altitude))/air_density(altitude):.2f}%)")
    print("="*60 + "\n")

# Main execution
if __name__ == "__main__":
    # Problem parameters
    TARGET_DISTANCE = 1200  # meters
    ALTITUDE = 500  # meters above sea level
    MAX_VELOCITY = 450  # m/s - maximum muzzle velocity
    
    print("\n" + "="*60)
    print("ARTILLERY TRAJECTORY SIMULATOR - COMPARISON")
    print("="*60)
    print(f"Projectile mass:      {MASS} kg")
    print(f"Projectile diameter:  {DIAMETER*100} cm")
    print(f"Drag coefficient:     {CD}")
    print(f"Launch altitude:      {ALTITUDE} m above sea level")
    print(f"Earth radius:         {EARTH_RADIUS/1000:.0f} km")
    
    # Find solution with CONSTANT g and H
    print("\n" + "="*60)
    print("METHOD 1: CONSTANT g AND CONSTANT H")
    print("="*60)
    solution_const = find_solution(TARGET_DISTANCE, ALTITUDE, MAX_VELOCITY, 
                                   method='euler', use_variable_g=False, use_variable_H=False)
    
    if solution_const:
        print("\n" + "="*60)
        print("SOLUTION FOUND (Constant g, Constant H)")
        print("="*60)
        print(f"Muzzle Velocity:      {solution_const['velocity']:.2f} m/s")
        print(f"Firing Angle:         {solution_const['angle']:.2f}°")
        print(f"Flight Time:          {solution_const['t'][-1]:.2f} s")
        print(f"Maximum Height:       {np.max(solution_const['y']):.1f} m")
        print(f"Error:                {solution_const['error']:.2f} m")
        print("="*60)
    
    # Find solution with VARIABLE g and H
    print("\n" + "="*60)
    print("METHOD 2: VARIABLE g AND VARIABLE H")
    print("="*60)
    solution_var = find_solution(TARGET_DISTANCE, ALTITUDE, MAX_VELOCITY, 
                                 method='euler', use_variable_g=True, use_variable_H=True)
    
    if solution_var:
        print_solution(solution_var, TARGET_DISTANCE, ALTITUDE)
        
        # COMPARISON
        if solution_const:
            print("\n" + "="*60)
            print("COMPARISON: Constant vs Variable")
            print("="*60)
            print(f"Velocity:  {solution_const['velocity']:.2f} m/s (const) vs {solution_var['velocity']:.2f} m/s (var)")
            print(f"           Difference: {solution_var['velocity'] - solution_const['velocity']:.2f} m/s")
            print(f"Angle:     {solution_const['angle']:.2f}° (const) vs {solution_var['angle']:.2f}° (var)")
            print(f"           Difference: {solution_var['angle'] - solution_const['angle']:.2f}°")
            print(f"Max Height: {np.max(solution_const['y']):.1f}m (const) vs {np.max(solution_var['y']):.1f}m (var)")
            print("="*60)
        
        # DEBUG: Air density comparison
        print("\n" + "="*60)
        print("DEBUG - Air Density Comparison")
        print("="*60)
        rho_constant = RHO_SEA * np.exp(-ALTITUDE / SCALE_HEIGHT_SEA)
        rho_variable = air_density(ALTITUDE, use_variable_H=True)
        print(f"Method 1 (constant H={SCALE_HEIGHT_SEA}m):")
        print(f"  ρ at {ALTITUDE}m = {rho_constant:.6f} kg/m³")
        print(f"\nMethod 2 (variable H):")
        print(f"  H at 0m     = {scale_height(0):.2f} m")
        print(f"  H at {ALTITUDE}m   = {scale_height(ALTITUDE):.2f} m")
        print(f"  ρ at {ALTITUDE}m   = {rho_variable:.6f} kg/m³")
        print(f"\nDifference: {rho_variable - rho_constant:.6f} kg/m³ ({100*(rho_variable - rho_constant)/rho_constant:.3f}%)")
        print("="*60)
        
        # Plot both trajectories
        if solution_const:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(solution_const['x'], solution_const['y'], 'r-', linewidth=2, 
                   label=f'Constant g,H: v={solution_const["velocity"]:.1f}m/s, θ={solution_const["angle"]:.1f}°')
            ax.plot(solution_var['x'], solution_var['y'], 'b-', linewidth=2,
                   label=f'Variable g,H: v={solution_var["velocity"]:.1f}m/s, θ={solution_var["angle"]:.1f}°')
            ax.plot(TARGET_DISTANCE, 0, 'g*', markersize=15, label='Target')
            ax.plot(0, 0, 'ko', markersize=10, label='Cannon')
            ax.axhline(y=0, color='brown', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Distance (m)', fontsize=12)
            ax.set_ylabel('Height above launch point (m)', fontsize=12)
            ax.set_title('Trajectory Comparison: Constant vs Variable Parameters', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            plot_trajectory(solution_var, TARGET_DISTANCE, ALTITUDE)