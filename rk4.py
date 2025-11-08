import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G_SEA_LEVEL = 9.81  # m/s^2
RHO_SEA = 1.225  # kg/m^3
MASS = 5  # kg
DIAMETER = 0.11  # m
RADIUS = DIAMETER / 2
AREA = np.pi * RADIUS**2
CD = 0.47

# Earth and atmospheric constants
EARTH_RADIUS = 6371000  # meters
GAS_CONSTANT = 8.314  # J/(mol·K)
MOLAR_MASS_AIR = 0.029  # kg/mol
TEMPERATURE = 288  # K
SCALE_HEIGHT_SEA = 8500  # meters

def gravity(altitude):
    """Calculate gravitational acceleration at given altitude"""
    return G_SEA_LEVEL * (EARTH_RADIUS / (EARTH_RADIUS + altitude))**2

def scale_height(altitude):
    """Calculate atmospheric scale height at given altitude"""
    g = gravity(altitude)
    return (GAS_CONSTANT * TEMPERATURE) / (MOLAR_MASS_AIR * g)

def air_density(altitude, use_variable_H=True):
    """Calculate air density at given altitude"""
    if altitude <= 0:
        return RHO_SEA
    
    if use_variable_H:
        H_sea = scale_height(0)
        H_alt = scale_height(altitude)
        H_avg = (H_sea + H_alt) / 2
        return RHO_SEA * np.exp(-altitude / H_avg)
    else:
        return RHO_SEA * np.exp(-altitude / SCALE_HEIGHT_SEA)

def derivatives(state, base_altitude, use_variable_g=True, use_variable_H=True):
    """
    Calculate derivatives for projectile motion
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    
    current_altitude = base_altitude + y
    
    if use_variable_g:
        g = gravity(current_altitude)
    else:
        g = G_SEA_LEVEL
    
    rho = air_density(current_altitude, use_variable_H=use_variable_H)
    
    v = np.sqrt(vx**2 + vy**2)
    
    if v > 0:
        drag = 0.5 * rho * CD * AREA * v**2 / MASS
        ax = -drag * vx / v
        ay = -g - drag * vy / v
    else:
        ax = 0
        ay = -g
    
    return np.array([vx, vy, ax, ay])

def simulate_trajectory_rk4(v0, angle_deg, altitude, dt=0.01, max_time=60, 
                            use_variable_g=True, use_variable_H=True):
    """
    Simulate trajectory using RK4 (Runge-Kutta 4th order) method
    
    RK4 is a classic method from ~1900, perfect for historical artillery!
    Much more accurate than Euler with same time step.
    
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
    
    # Initial state [x, y, vx, vy]
    state = np.array([0.0, 0.0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)])
    
    # Storage arrays
    x_arr, y_arr, t_arr = [state[0]], [state[1]], [0]
    t = 0
    
    MAX_RANGE = 5000  # meters
    
    while state[1] >= 0 and t < max_time and state[0] < MAX_RANGE:
        # RK4 method - calculates 4 derivative estimates per step
        k1 = derivatives(state, altitude, use_variable_g, use_variable_H)
        k2 = derivatives(state + 0.5 * dt * k1, altitude, use_variable_g, use_variable_H)
        k3 = derivatives(state + 0.5 * dt * k2, altitude, use_variable_g, use_variable_H)
        k4 = derivatives(state + dt * k3, altitude, use_variable_g, use_variable_H)
        
        # Weighted average of derivatives
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        
        x_arr.append(state[0])
        y_arr.append(state[1])
        t_arr.append(t)
    
    return np.array(x_arr), np.array(y_arr), np.array(t_arr)

def simulate_trajectory_euler(v0, angle_deg, altitude, dt=0.01, max_time=60, 
                              use_variable_g=True, use_variable_H=True):
    """Simulate trajectory using Euler method (for comparison)"""
    angle_rad = np.radians(angle_deg)
    
    x, y = 0, 0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    x_arr, y_arr, t_arr = [x], [y], [0]
    t = 0
    
    MAX_RANGE = 5000
    
    while y >= 0 and t < max_time and x < MAX_RANGE:
        current_altitude = altitude + y
        
        if use_variable_g:
            g = gravity(current_altitude)
        else:
            g = G_SEA_LEVEL
            
        rho = air_density(current_altitude, use_variable_H=use_variable_H)
        
        v = np.sqrt(vx**2 + vy**2)
        
        if v > 0:
            drag = 0.5 * rho * CD * AREA * v**2 / MASS
            ax = -drag * vx / v
            ay = -g - drag * vy / v
        else:
            ax = 0
            ay = -g
        
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        
        x_arr.append(x)
        y_arr.append(y)
        t_arr.append(t)
    
    return np.array(x_arr), np.array(y_arr), np.array(t_arr)

def compare_methods(v0, angle_deg, altitude):
    """Compare Euler vs RK4 accuracy"""
    print(f"\nComparing methods: v0={v0} m/s, angle={angle_deg}°, altitude={altitude}m")
    print("="*70)
    
    # Use different time steps
    dt_coarse = 0.05  # 50ms
    dt_fine = 0.001   # 1ms (reference)
    
    # Get reference solution with very fine time step using RK4
    x_ref, y_ref, t_ref = simulate_trajectory_rk4(v0, angle_deg, altitude, dt=dt_fine)
    ref_range = x_ref[-1]
    ref_time = t_ref[-1]
    ref_max_height = np.max(y_ref)
    
    print(f"Reference (RK4, dt={dt_fine}s):")
    print(f"  Range: {ref_range:.3f} m, Time: {ref_time:.3f} s, Max height: {ref_max_height:.3f} m")
    
    # Test Euler with coarse time step
    x_euler, y_euler, t_euler = simulate_trajectory_euler(v0, angle_deg, altitude, dt=dt_coarse)
    euler_range = x_euler[-1]
    euler_time = t_euler[-1]
    euler_max_height = np.max(y_euler)
    
    print(f"\nEuler (dt={dt_coarse}s):")
    print(f"  Range: {euler_range:.3f} m, Time: {euler_time:.3f} s, Max height: {euler_max_height:.3f} m")
    print(f"  Error in range: {abs(euler_range - ref_range):.3f} m ({100*abs(euler_range-ref_range)/ref_range:.3f}%)")
    print(f"  Error in time: {abs(euler_time - ref_time):.3f} s")
    
    # Test RK4 with coarse time step
    x_rk4, y_rk4, t_rk4 = simulate_trajectory_rk4(v0, angle_deg, altitude, dt=dt_coarse)
    rk4_range = x_rk4[-1]
    rk4_time = t_rk4[-1]
    rk4_max_height = np.max(y_rk4)
    
    print(f"\nRK4 (dt={dt_coarse}s):")
    print(f"  Range: {rk4_range:.3f} m, Time: {rk4_time:.3f} s, Max height: {rk4_max_height:.3f} m")
    print(f"  Error in range: {abs(rk4_range - ref_range):.3f} m ({100*abs(rk4_range-ref_range)/ref_range:.3f}%)")
    print(f"  Error in time: {abs(rk4_time - ref_time):.3f} s")
    
    print(f"\nRK4 is {abs(euler_range - ref_range) / abs(rk4_range - ref_range):.1f}x more accurate than Euler!")
    print("="*70)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Full trajectories
    ax1.plot(x_ref, y_ref, 'k-', linewidth=2, label=f'Reference (RK4, dt={dt_fine}s)', alpha=0.7)
    ax1.plot(x_euler, y_euler, 'r--', linewidth=2, label=f'Euler (dt={dt_coarse}s)')
    ax1.plot(x_rk4, y_rk4, 'b:', linewidth=2, label=f'RK4 (dt={dt_coarse}s)')
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Trajectory Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Error over distance
    # Interpolate to compare at same x values
    from scipy.interpolate import interp1d
    
    x_common = np.linspace(0, min(x_ref[-1], x_euler[-1], x_rk4[-1]), 500)
    
    f_ref = interp1d(x_ref, y_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    f_euler = interp1d(x_euler, y_euler, kind='cubic', bounds_error=False, fill_value='extrapolate')
    f_rk4 = interp1d(x_rk4, y_rk4, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    y_ref_interp = f_ref(x_common)
    y_euler_interp = f_euler(x_common)
    y_rk4_interp = f_rk4(x_common)
    
    euler_error = np.abs(y_euler_interp - y_ref_interp)
    rk4_error = np.abs(y_rk4_interp - y_ref_interp)
    
    ax2.semilogy(x_common, euler_error, 'r-', linewidth=2, label='Euler error')
    ax2.semilogy(x_common, rk4_error, 'b-', linewidth=2, label='RK4 error')
    ax2.set_xlabel('Distance (m)', fontsize=12)
    ax2.set_ylabel('Absolute Error in Height (m)', fontsize=12)
    ax2.set_title('Numerical Error Accumulation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Demo
if __name__ == "__main__":
    print("="*70)
    print("RK4 vs EULER METHOD COMPARISON")
    print("="*70)
    
    # Test case: 1200m target at 500m altitude
    TARGET_DISTANCE = 1200
    ALTITUDE = 500
    
    # Use a typical solution
    v0 = 200
    angle = 25
    
    compare_methods(v0, angle, ALTITUDE)
    
    print("\n" + "="*70)
    print("COMPUTATIONAL EFFICIENCY")
    print("="*70)
    print("RK4 requires 4x more derivative evaluations per step than Euler,")
    print("BUT you can use a much larger time step (5-10x) for same accuracy.")
    print("Net result: RK4 is often 2-3x FASTER for same accuracy!")
    print("="*70)