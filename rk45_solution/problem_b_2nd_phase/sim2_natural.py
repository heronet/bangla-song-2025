import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Constants
G_SEA_LEVEL = 9.81
RHO_SEA = 1.225
MASS = 5
DIAMETER = 0.11
RADIUS = DIAMETER / 2
AREA = np.pi * RADIUS**2
CD = 0.47

EARTH_RADIUS = 6371000
GAS_CONSTANT = 8.314
MOLAR_MASS_AIR = 0.029
TEMPERATURE = 288
SCALE_HEIGHT_SEA = 8500

def gravity(altitude):
    # g decreases with altitude
    return G_SEA_LEVEL * (EARTH_RADIUS / (EARTH_RADIUS + altitude))**2

def air_density(altitude):
    # constant scale height
    if altitude <= 0:
        return RHO_SEA
    return RHO_SEA * np.exp(-altitude / SCALE_HEIGHT_SEA)

def derivatives_with_wind(t, state, base_altitude, wind_x, wind_y):
    # state = [x, y, vx, vy]
    x, y, vx, vy = state

    current_altitude = base_altitude + y

    g = G_SEA_LEVEL  # constant g

    rho = air_density(current_altitude)

    # relative velocity
    vx_rel = vx - wind_x
    vy_rel = vy - wind_y
    v_rel = np.sqrt(vx_rel**2 + vy_rel**2)

    if v_rel > 0:
        drag = 0.5 * rho * CD * AREA * v_rel**2 / MASS

        ax = -drag * vx_rel / v_rel
        ay = -g - drag * vy_rel / v_rel
    else:
        ax = 0
        ay = -g

    return [vx, vy, ax, ay]

def ground_event(t, state, base_altitude, wind_x, wind_y):
    return state[1]

ground_event.terminal = True
ground_event.direction = -1

def simulate_trajectory_rk45(v0, angle_deg, altitude, wind_speed=0, wind_angle=0, max_time=60):
    # simulate trajectory with optional wind
    angle_rad = np.radians(angle_deg)
    wind_angle_rad = np.radians(wind_angle)

    # wind components
    wind_x = wind_speed * np.cos(wind_angle_rad)
    wind_y = 0  # horizontal wind only

    vx0 = v0 * np.cos(angle_rad)
    vy0 = v0 * np.sin(angle_rad)
    state0 = [0, 0, vx0, vy0]

    solution = solve_ivp(
        derivatives_with_wind,
        t_span=(0, max_time),
        y0=state0,
        method='RK45',
        events=ground_event,
        args=(altitude, wind_x, wind_y),
        dense_output=False,
        max_step=0.1
    )

    if len(solution.t) > 0:
        final_x = solution.y[0, -1]
        final_t = solution.t[-1]
    else:
        final_x = 0
        final_t = 0

    return final_x, final_t

def generate_base_range_table(altitudes, velocities, angles):
    # generate base range table
    print("Generating base range table using RK45...")
    print(f"Total simulations: {len(altitudes) * len(velocities) * len(angles)}")

    results = []
    total = len(altitudes) * len(velocities) * len(angles)
    count = 0

    for alt in altitudes:
        for v0 in velocities:
            for angle in angles:
                range_m, time_s = simulate_trajectory_rk45(v0, angle, alt, wind_speed=0, wind_angle=0)
                results.append({
                    'altitude': alt,
                    'velocity': v0,
                    'angle': angle,
                    'range': range_m,
                    'time': time_s
                })
                count += 1
                if count % 100 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

    df = pd.DataFrame(results)
    print(f"Generated {len(df)} range table entries")
    return df

def calculate_wind_corrections(base_altitude, base_velocity, base_angle, wind_speeds, wind_angles):
    # calculate wind correction factors
    print("\nCalculating wind correction factors using RK45...")

    # no wind baseline
    base_range, _ = simulate_trajectory_rk45(base_velocity, base_angle, base_altitude,
                                            wind_speed=0, wind_angle=0)

    print(f"Base case: v={base_velocity} m/s, θ={base_angle}°, alt={base_altitude}m → {base_range:.1f}m")

    corrections = []
    total = len(wind_speeds) * len(wind_angles)
    count = 0

    for wind_speed in wind_speeds:
        for wind_angle in wind_angles:
            range_with_wind, _ = simulate_trajectory_rk45(
                base_velocity, base_angle, base_altitude, wind_speed, wind_angle
            )

            correction = range_with_wind - base_range
            corrections.append({
                'wind_speed': wind_speed,
                'wind_angle': wind_angle,
                'range_correction': correction,
                'percent_change': 100 * correction / base_range if base_range > 0 else 0
            })
            count += 1
            if count % 5 == 0:
                print(f"  Progress: {count}/{total}")

    print(f"Generated {len(corrections)} wind correction entries")
    return pd.DataFrame(corrections), base_range

def create_firing_table_for_distance(target_distance, altitude_range=(0, 1000, 100),
                                     velocity_range=(100, 450, 10)):
    # for a given target, find required angle and velocity for various altitudes
    print(f"\nCreating firing table for {target_distance}m target using RK45...")

    altitudes = range(*altitude_range)
    velocities = range(*velocity_range)

    firing_solutions = []

    for alt in altitudes:
        best_solution = None
        min_velocity = float('inf')

        # find solution with minimum velocity
        for v0 in velocities:
            for angle in range(10, 80, 1):
                range_m, time_s = simulate_trajectory_rk45(v0, angle, alt)
                error = abs(range_m - target_distance)

                if error < 10 and v0 < min_velocity:
                    min_velocity = v0
                    best_solution = {
                        'altitude': alt,
                        'velocity': v0,
                        'angle': angle,
                        'time': time_s,
                        'error': error
                    }

        if best_solution:
            firing_solutions.append(best_solution)
            print(f"  Altitude {alt}m: v={best_solution['velocity']} m/s, θ={best_solution['angle']}°")

    print(f"Generated firing table with {len(firing_solutions)} entries")
    return pd.DataFrame(firing_solutions)

def create_interpolation_guide(df_base):
    print("\nCreating interpolation guide...")

    # use 500m altitude
    df_500 = df_base[df_base['altitude'] == 500].copy()

    if len(df_500) == 0:
        print("  Warning: No data for 500m altitude")
        return pd.DataFrame()

    guide = []

    velocities = sorted(df_500['velocity'].unique())
    for v0 in velocities[::5]:
        df_v = df_500[df_500['velocity'] == v0]

        range_30 = df_v[df_v['angle'] == 30]['range'].values
        range_45 = df_v[df_v['angle'] == 45]['range'].values

        if len(range_30) > 0 and len(range_45) > 0:
            # rate of change: meters per degree
            rate = (range_45[0] - range_30[0]) / 15

            guide.append({
                'velocity': v0,
                'range_at_30deg': range_30[0],
                'range_at_45deg': range_45[0],
                'meters_per_degree': rate
            })

    print(f"Generated interpolation guide with {len(guide)} entries")
    return pd.DataFrame(guide)

def generate_nomograph_data(df_base, fixed_altitude=500):
    # data for creating a nomograph
    print(f"\nGenerating nomograph data for altitude={fixed_altitude}m...")

    df_alt = df_base[df_base['altitude'] == fixed_altitude]

    if len(df_alt) == 0:
        print(f"  Warning: No data for altitude {fixed_altitude}m")
        return None, None, None

    # range as function of velocity and angle
    velocities = sorted(df_alt['velocity'].unique())
    angles = sorted(df_alt['angle'].unique())

    V, A = np.meshgrid(velocities, angles)
    R = np.zeros_like(V, dtype=float)

    for i, angle in enumerate(angles):
        for j, velocity in enumerate(velocities):
            range_val = df_alt[(df_alt['velocity'] == velocity) &
                              (df_alt['angle'] == angle)]['range'].values
            if len(range_val) > 0:
                R[i, j] = range_val[0]

    print(f"Generated nomograph data: {len(velocities)} velocities × {len(angles)} angles")
    return V, A, R

def create_simplified_formula(df_base, altitude=500):
    # derive empirical formula using polynomial regression
    print(f"\nDeriving simplified formula for altitude={altitude}m...")

    df_alt = df_base[df_base['altitude'] == altitude]

    if len(df_alt) == 0:
        print(f"  Warning: No data for altitude {altitude}m")
        return None, None, None, 0

    # Range = f(velocity, angle)
    # Range ≈ a*v + b*v² + c*sin(2θ) + d*v*sin(2θ) + e

    X = df_alt[['velocity', 'angle']].values
    y = df_alt['range'].values

    v = X[:, 0]
    angle_rad = np.radians(X[:, 1])
    sin_2a = np.sin(2 * angle_rad)

    X_features = np.column_stack([
        v,
        v**2,
        sin_2a,
        v * sin_2a,
        np.ones_like(v)
    ])

    model = LinearRegression()
    model.fit(X_features, y)

    coeffs = model.coef_
    intercept = model.intercept_

    score = model.score(X_features, y)

    print(f"  Formula R² score: {score:.4f}")
    print(f"\n  Simplified Formula (altitude={altitude}m):")
    print(f"  Range ≈ {coeffs[0]:.4f}*v + {coeffs[1]:.6f}*v² + {coeffs[2]:.2f}*sin(2θ)")
    print(f"          + {coeffs[3]:.4f}*v*sin(2θ) + {intercept:.2f}")
    print(f"  where v is velocity (m/s) and θ is angle (degrees)")

    return model, coeffs, intercept, score

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("ARTILLERY RANGE TABLE GENERATOR - RK45 METHOD")
    print("="*70)
    print("Integration method: Runge-Kutta 45 (RK45)")
    print("Physics: Constant g and H")
    print("="*70)

    # parameter ranges
    ALTITUDES = [0, 250, 500, 750, 1000]
    VELOCITIES = list(range(100, 451, 25))
    ANGLES = list(range(10, 81, 5))

    print(f"\nParameter ranges:")
    print(f"  Altitudes: {ALTITUDES}")
    print(f"  Velocities: {VELOCITIES[0]} to {VELOCITIES[-1]} m/s (step: 25)")
    print(f"  Angles: {ANGLES[0]}° to {ANGLES[-1]}° (step: 5°)")

    print("\n" + "="*70)
    print("BASE RANGE TABLE")
    print("="*70)
    df_base = generate_base_range_table(ALTITUDES, VELOCITIES, ANGLES)

    df_base.to_csv('artillery_range_table_rk45.csv', index=False)
    print(f"\nBase range table saved to 'artillery_range_table_rk45.csv'")

    print("\n" + "="*70)
    print("FIRING TABLES FOR SPECIFIC DISTANCES")
    print("="*70)
    TARGET_DISTANCES = [1000, 1100, 1200, 1300, 1400, 1500]

    firing_tables = {}
    for distance in TARGET_DISTANCES:
        df_firing = create_firing_table_for_distance(distance)
        firing_tables[distance] = df_firing
        if len(df_firing) > 0:
            df_firing.to_csv(f'firing_table_{distance}m_rk45.csv', index=False)
            print(f"Firing table for {distance}m saved to 'firing_table_{distance}m_rk45.csv'")

    # wind corrections
    print("\n" + "="*70)
    print("WIND CORRECTION FACTORS")
    print("="*70)
    WIND_SPEEDS = [0, 5, 10, 15, 20]
    WIND_ANGLES = [0, 45, 90, 135, 180]  # 0=tailwind, 180=headwind

    df_wind, base_range = calculate_wind_corrections(
        base_altitude=500,
        base_velocity=200,
        base_angle=45,
        wind_speeds=WIND_SPEEDS,
        wind_angles=WIND_ANGLES
    )

    df_wind.to_csv('wind_correction_table_rk45.csv', index=False)
    print(f"Wind correction table saved to 'wind_correction_table_rk45.csv'")

    # interpolation guide
    print("\n" + "="*70)
    print("INTERPOLATION GUIDE")
    print("="*70)
    df_interp = create_interpolation_guide(df_base)
    if len(df_interp) > 0:
        df_interp.to_csv('interpolation_guide_rk45.csv', index=False)
        print(f"Interpolation guide saved to 'interpolation_guide_rk45.csv'")

    # simplified formula
    print("\n" + "="*70)
    print("SIMPLIFIED EMPIRICAL FORMULA")
    print("="*70)
    model, coeffs, intercept, score = create_simplified_formula(df_base, altitude=500)

    # visualizations
    print("\n" + "="*70)
    print("VISUALIZATIONS")
    print("="*70)
    V, A, R = generate_nomograph_data(df_base, fixed_altitude=500)

    print("\nGenerating plots...")

    if V is not None and A is not None and R is not None:
        # nomograph plot
        fig, ax = plt.subplots(figsize=(12, 8))
        contour = ax.contour(V, A, R, levels=15, colors='black', linewidths=0.5)
        contourf = ax.contourf(V, A, R, levels=15, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%d m')

        cbar = plt.colorbar(contourf, ax=ax, label='Range (meters)')
        ax.set_xlabel('Muzzle Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Firing Angle (degrees)', fontsize=12)
        ax.set_title('Artillery Nomograph - Range at 500m Altitude (RK45)\n(Read intersection of velocity and angle to find range)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('nomograph_range_rk45.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Nomograph saved to 'nomograph_range_rk45.png'")

    # firing solutions plot
    if len(firing_tables) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))

        for distance, df_firing in firing_tables.items():
            if len(df_firing) > 0:
                ax.plot(df_firing['altitude'], df_firing['angle'],
                       marker='o', label=f'{distance}m target', linewidth=2)

        ax.set_xlabel('Launch Altitude (m)', fontsize=12)
        ax.set_ylabel('Required Firing Angle (degrees)', fontsize=12)
        ax.set_title('Firing Angle vs Altitude for Different Target Distances (RK45)\n(Using minimum velocity for each case)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('firing_angles_vs_altitude_rk45.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Firing angle chart saved to 'firing_angles_vs_altitude_rk45.png'")

    # wind correction plot
    if len(df_wind) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        wind_angle_labels = {
            0: 'Tailwind (0°)',
            45: 'Quartering Tailwind (45°)',
            90: 'Crosswind (90°)',
            135: 'Quartering Headwind (135°)',
            180: 'Headwind (180°)'
        }

        for wind_angle in WIND_ANGLES:
            df_wind_angle = df_wind[df_wind['wind_angle'] == wind_angle]
            label = wind_angle_labels.get(wind_angle, f'{wind_angle}° wind')
            ax.plot(df_wind_angle['wind_speed'], df_wind_angle['range_correction'],
                   marker='o', label=label, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
        ax.set_ylabel('Range Correction (meters)', fontsize=12)
        ax.set_title(f'Wind Correction Factors (RK45)\n(Base: v=200 m/s, θ=45°, altitude=500m, range={base_range:.0f}m)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('wind_corrections_rk45.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Wind correction chart saved to 'wind_corrections_rk45.png'")

    print("\n" + "="*70)
    print("ALL TABLES AND CHARTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles created (all using RK45 method):")
    print("  • artillery_range_table_rk45.csv - Complete range data")
    print("  • firing_table_XXXXm_rk45.csv - Firing solutions for specific distances")
    print("  • wind_correction_table_rk45.csv - Wind adjustment factors")
    print("  • interpolation_guide_rk45.csv - Simple interpolation rules")
    print("  • nomograph_range_rk45.png - Graphical calculator")
    print("  • firing_angles_vs_altitude_rk45.png - Altitude corrections")
    print("  • wind_corrections_rk45.png - Wind effect visualization")
    print("="*70)