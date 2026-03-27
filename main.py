"""
Assignment 2: Bayesian Place Cell Decoder
Implementation of Bayesian and Population Vector decoding for place cells
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import poisson
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("="*70)
print("BAYESIAN PLACE CELL DECODER")
print("="*70)

#%% PART 1: SIMULATE PLACE CELLS

class PlaceCell:
    """Single place cell with Gaussian tuning curve"""
    
    def __init__(self, place_field_center, place_field_width, 
                 peak_firing_rate, track_length=100):
        self.center = place_field_center
        self.width = place_field_width
        self.peak_rate = peak_firing_rate
        self.track_length = track_length
        
    def firing_rate(self, position):
        """
        Compute firing rate at given position(s)
        Gaussian tuning curve
        """
        return self.peak_rate * np.exp(
            -(position - self.center)**2 / (2 * self.width**2)
        )
    
    def generate_spikes(self, position, dt=0.1):
        """
        Generate spike count using Poisson process
        
        Parameters:
        -----------
        position : float or array
            Current position(s)
        dt : float
            Time bin size in seconds
            
        Returns:
        --------
        spike_count : int or array
            Number of spikes in time bin
        """
        rate = self.firing_rate(position)
        expected_spikes = rate * dt
        return np.random.poisson(expected_spikes)


def create_place_cell_population(n_cells=30, track_length=100,
                                place_field_width_range=(5, 15),
                                peak_rate_range=(5, 20)):
    """
    Create population of place cells tiling the track
    
    Parameters:
    -----------
    n_cells : int
        Number of place cells
    track_length : float
        Length of linear track (cm)
    place_field_width_range : tuple
        Range of place field widths (cm)
    peak_rate_range : tuple
        Range of peak firing rates (Hz)
        
    Returns:
    --------
    place_cells : list
        List of PlaceCell objects
    """
    
    print(f"\nCreating {n_cells} place cells...")
    
    place_cells = []
    
    # Tile the track uniformly
    centers = np.linspace(0, track_length, n_cells + 2)[1:-1]
    
    for i, center in enumerate(centers):
        # Random width and peak rate
        width = np.random.uniform(*place_field_width_range)
        peak_rate = np.random.uniform(*peak_rate_range)
        
        cell = PlaceCell(center, width, peak_rate, track_length)
        place_cells.append(cell)
    
    print(f"✓ Created {n_cells} place cells")
    print(f"  Place field centers: {min(centers):.1f} - {max(centers):.1f} cm")
    print(f"  Place field widths: {place_field_width_range}")
    print(f"  Peak firing rates: {peak_rate_range} Hz")
    
    return place_cells


def simulate_trajectory(duration=60, track_length=100, dt=0.1, 
                       velocity_range=(5, 15)):
    """
    Simulate animal's trajectory on linear track
    
    Parameters:
    -----------
    duration : float
        Duration in seconds
    track_length : float
        Track length (cm)
    dt : float
        Time step (seconds)
    velocity_range : tuple
        Range of velocities (cm/s)
        
    Returns:
    --------
    time : array
        Time points
    position : array
        Position at each time point
    """
    
    n_steps = int(duration / dt)
    time = np.arange(n_steps) * dt
    position = np.zeros(n_steps)
    
    # Start at random position
    position[0] = np.random.uniform(0, track_length)
    
    # Random walk with variable velocity
    direction = 1  # 1 = right, -1 = left
    
    for i in range(1, n_steps):
        # Variable velocity
        velocity = np.random.uniform(*velocity_range)
        
        # Update position
        position[i] = position[i-1] + direction * velocity * dt
        
        # Bounce at boundaries
        if position[i] >= track_length:
            position[i] = track_length - (position[i] - track_length)
            direction = -1
        elif position[i] <= 0:
            position[i] = -position[i]
            direction = 1
        
        # Occasionally change direction
        if np.random.rand() < 0.02:  # 2% chance per timestep
            direction *= -1
    
    return time, position


def generate_population_activity(place_cells, position, dt=0.1):
    """
    Generate spike counts for all place cells
    
    Parameters:
    -----------
    place_cells : list
        List of PlaceCell objects
    position : array
        Position trajectory
    dt : float
        Time bin size
        
    Returns:
    --------
    spike_counts : array (n_cells, n_timepoints)
        Spike counts for each cell at each time
    firing_rates : array (n_cells, n_timepoints)
        Underlying firing rates
    """
    
    n_cells = len(place_cells)
    n_timepoints = len(position)
    
    spike_counts = np.zeros((n_cells, n_timepoints))
    firing_rates = np.zeros((n_cells, n_timepoints))
    
    for i, cell in enumerate(place_cells):
        firing_rates[i] = cell.firing_rate(position)
        spike_counts[i] = cell.generate_spikes(position, dt)
    
    return spike_counts, firing_rates


# Generate data
print("\n" + "="*70)
print("PART 1: Simulate Place Cell Population")
print("="*70)

n_cells = 30
track_length = 100
duration = 60
dt = 0.1

place_cells = create_place_cell_population(n_cells, track_length)

print(f"\nSimulating {duration}s trajectory...")
time, true_position = simulate_trajectory(duration, track_length, dt)

print(f"\nGenerating population activity...")
spike_counts, firing_rates = generate_population_activity(place_cells, true_position, dt)

print(f"✓ Generated data")
print(f"  Time points: {len(time)}")
print(f"  Total spikes: {spike_counts.sum():.0f}")
print(f"  Mean firing rate: {spike_counts.sum() / (n_cells * duration):.2f} Hz")


#%% PART 2: BAYESIAN DECODER

class BayesianDecoder:
    """
    Bayesian decoder for place cells
    P(position | spikes) ∝ P(spikes | position) * P(position)
    """
    
    def __init__(self, place_cells, track_length=100, position_bins=100):
        self.place_cells = place_cells
        self.track_length = track_length
        self.n_cells = len(place_cells)
        
        # Discretize position space
        self.position_bins = position_bins
        self.positions = np.linspace(0, track_length, position_bins)
        self.bin_width = self.positions[1] - self.positions[0]
        
        # Compute tuning curves for all positions
        self.tuning_curves = self._compute_tuning_curves()
        
    def _compute_tuning_curves(self):
        """Precompute firing rates at all positions"""
        tuning = np.zeros((self.n_cells, self.position_bins))
        
        for i, cell in enumerate(self.place_cells):
            tuning[i] = cell.firing_rate(self.positions)
        
        return tuning
    
    def decode(self, spike_counts, dt=0.1, prior=None):
        """
        Decode position from spike counts
        
        Parameters:
        -----------
        spike_counts : array (n_cells,) or (n_cells, n_timepoints)
            Observed spike counts
        dt : float
            Time bin size
        prior : array or None
            Prior distribution over positions (uniform if None)
            
        Returns:
        --------
        decoded_position : float or array
            Most likely position(s)
        posterior : array
            Posterior distribution(s)
        """
        
        # Handle single timepoint
        if spike_counts.ndim == 1:
            spike_counts = spike_counts.reshape(-1, 1)
        
        n_timepoints = spike_counts.shape[1]
        decoded_positions = np.zeros(n_timepoints)
        posteriors = np.zeros((self.position_bins, n_timepoints))
        
        # Uniform prior if not provided
        if prior is None:
            prior = np.ones(self.position_bins) / self.position_bins
        
        for t in range(n_timepoints):
            # Compute likelihood for each position
            # P(spikes | position) using Poisson likelihood
            likelihood = np.ones(self.position_bins)
            
            for pos_idx in range(self.position_bins):
                for cell_idx in range(self.n_cells):
                    # Expected rate at this position
                    expected_rate = self.tuning_curves[cell_idx, pos_idx]
                    expected_count = expected_rate * dt
                    
                    # Poisson probability
                    observed_count = spike_counts[cell_idx, t]
                    likelihood[pos_idx] *= poisson.pmf(
                        observed_count, expected_count
                    )
            
            # Posterior = likelihood * prior
            posterior = likelihood * prior
            
            # Normalize
            if posterior.sum() > 0:
                posterior /= posterior.sum()
            else:
                posterior = prior
            
            posteriors[:, t] = posterior
            
            # Maximum a posteriori estimate
            decoded_positions[t] = self.positions[np.argmax(posterior)]
        
        if n_timepoints == 1:
            return decoded_positions[0], posteriors[:, 0]
        
        return decoded_positions, posteriors


print("\n" + "="*70)
print("PART 2: Bayesian Decoder")
print("="*70)

bayesian_decoder = BayesianDecoder(place_cells, track_length)
print(f"\n✓ Initialized Bayesian decoder")
print(f"  Position bins: {bayesian_decoder.position_bins}")

print(f"\nDecoding trajectory...")
decoded_position_bayes, posteriors = bayesian_decoder.decode(spike_counts, dt)

# Compute decoding error
decoding_error_bayes = np.abs(decoded_position_bayes - true_position)
print(f"✓ Bayesian decoding complete")
print(f"  Mean absolute error: {np.mean(decoding_error_bayes):.2f} cm")
print(f"  Median absolute error: {np.median(decoding_error_bayes):.2f} cm")


#%% PART 3: POPULATION VECTOR DECODER

class PopulationVectorDecoder:
    """
    Population vector decoder
    Weighted average of preferred positions by firing rate
    """
    
    def __init__(self, place_cells, track_length=100):
        self.place_cells = place_cells
        self.track_length = track_length
        self.n_cells = len(place_cells)
        
        # Preferred positions
        self.preferred_positions = np.array([cell.center for cell in place_cells])
    
    def decode(self, spike_counts, dt=0.1):
        """
        Decode position using population vector
        
        Parameters:
        -----------
        spike_counts : array (n_cells,) or (n_cells, n_timepoints)
            Observed spike counts
        dt : float
            Time bin size
            
        Returns:
        --------
        decoded_position : float or array
            Decoded position(s)
        """
        
        if spike_counts.ndim == 1:
            spike_counts = spike_counts.reshape(-1, 1)
        
        n_timepoints = spike_counts.shape[1]
        decoded_positions = np.zeros(n_timepoints)
        
        for t in range(n_timepoints):
            # Weighted average
            weights = spike_counts[:, t]
            
            if weights.sum() > 0:
                decoded_positions[t] = np.average(
                    self.preferred_positions, 
                    weights=weights
                )
            else:
                # Default to center if no spikes
                decoded_positions[t] = self.track_length / 2
        
        if n_timepoints == 1:
            return decoded_positions[0]
        
        return decoded_positions


print("\n" + "="*70)
print("PART 3: Population Vector Decoder")
print("="*70)

pv_decoder = PopulationVectorDecoder(place_cells, track_length)
print(f"\n✓ Initialized Population Vector decoder")

print(f"\nDecoding trajectory...")
decoded_position_pv = pv_decoder.decode(spike_counts, dt)

decoding_error_pv = np.abs(decoded_position_pv - true_position)
print(f"✓ Population Vector decoding complete")
print(f"  Mean absolute error: {np.mean(decoding_error_pv):.2f} cm")
print(f"  Median absolute error: {np.median(decoding_error_pv):.2f} cm")


#%% PART 4: VISUALIZATION 1 - TRUE VS DECODED POSITION

print("\n" + "="*70)
print("PART 4: Visualizations")
print("="*70)

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(4, 1, figure=fig, hspace=0.3)

# Plot 1: True vs decoded position (Bayesian)
ax1 = fig.add_subplot(gs[0])
time_window = slice(0, 200)  # First 20 seconds

ax1.plot(time[time_window], true_position[time_window], 
        'k-', linewidth=2, label='True position', alpha=0.7)
ax1.plot(time[time_window], decoded_position_bayes[time_window], 
        'r-', linewidth=1.5, label='Bayesian decoder', alpha=0.8)

ax1.set_ylabel('Position (cm)', fontsize=11)
ax1.set_title('True vs Decoded Position (Bayesian)', 
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(time[time_window][0], time[time_window][-1])

# Plot 2: True vs decoded position (Population Vector)
ax2 = fig.add_subplot(gs[1])

ax2.plot(time[time_window], true_position[time_window], 
        'k-', linewidth=2, label='True position', alpha=0.7)
ax2.plot(time[time_window], decoded_position_pv[time_window], 
        'b-', linewidth=1.5, label='Population Vector', alpha=0.8)

ax2.set_ylabel('Position (cm)', fontsize=11)
ax2.set_title('True vs Decoded Position (Population Vector)', 
             fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(time[time_window][0], time[time_window][-1])

# Plot 3: Comparison
ax3 = fig.add_subplot(gs[2])

ax3.plot(time[time_window], true_position[time_window], 
        'k-', linewidth=2.5, label='True position', alpha=0.7)
ax3.plot(time[time_window], decoded_position_bayes[time_window], 
        'r-', linewidth=1.5, label='Bayesian', alpha=0.8)
ax3.plot(time[time_window], decoded_position_pv[time_window], 
        'b-', linewidth=1.5, label='Population Vector', alpha=0.8)

ax3.set_ylabel('Position (cm)', fontsize=11)
ax3.set_title('Decoder Comparison', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(time[time_window][0], time[time_window][-1])

# Plot 4: Decoding errors
ax4 = fig.add_subplot(gs[3])

ax4.plot(time[time_window], decoding_error_bayes[time_window], 
        'r-', linewidth=1.5, label='Bayesian error', alpha=0.7)
ax4.plot(time[time_window], decoding_error_pv[time_window], 
        'b-', linewidth=1.5, label='Pop. Vector error', alpha=0.7)

ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Decoding Error (cm)', fontsize=11)
ax4.set_title('Decoding Errors Over Time', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(time[time_window][0], time[time_window][-1])

plt.suptitle('Place Cell Decoding: True vs Decoded Position (60s session)',
            fontsize=14, fontweight='bold')
plt.savefig('decoder_position_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 5: DECODING ERROR VS NUMBER OF CELLS

print("\nAnalyzing effect of population size...")

def analyze_population_size(place_cells, true_position, spike_counts, dt,
                           max_cells=30, n_repeats=10):
    """
    Test decoding accuracy vs number of cells included
    """
    
    cell_counts = np.arange(5, max_cells + 1, 5)
    
    results = {
        'n_cells': cell_counts,
        'bayes_error_mean': [],
        'bayes_error_std': [],
        'pv_error_mean': [],
        'pv_error_std': []
    }
    
    for n in tqdm(cell_counts, desc="Population size"):
        bayes_errors = []
        pv_errors = []
        
        for _ in range(n_repeats):
            # Randomly select n cells
            cell_indices = np.random.choice(len(place_cells), n, replace=False)
            selected_cells = [place_cells[i] for i in cell_indices]
            selected_spikes = spike_counts[cell_indices]
            
            # Bayesian decoder
            decoder_bayes = BayesianDecoder(selected_cells, track_length)
            decoded_bayes, _ = decoder_bayes.decode(selected_spikes, dt)
            error_bayes = np.mean(np.abs(decoded_bayes - true_position))
            bayes_errors.append(error_bayes)
            
            # Population vector decoder
            decoder_pv = PopulationVectorDecoder(selected_cells, track_length)
            decoded_pv = decoder_pv.decode(selected_spikes, dt)
            error_pv = np.mean(np.abs(decoded_pv - true_position))
            pv_errors.append(error_pv)
        
        results['bayes_error_mean'].append(np.mean(bayes_errors))
        results['bayes_error_std'].append(np.std(bayes_errors))
        results['pv_error_mean'].append(np.mean(pv_errors))
        results['pv_error_std'].append(np.std(pv_errors))
    
    return results

population_size_results = analyze_population_size(
    place_cells, true_position, spike_counts, dt,
    max_cells=30, n_repeats=10
)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(population_size_results['n_cells'],
           population_size_results['bayes_error_mean'],
           yerr=population_size_results['bayes_error_std'],
           marker='o', capsize=5, linewidth=2, markersize=8,
           color='red', label='Bayesian decoder')

ax.errorbar(population_size_results['n_cells'],
           population_size_results['pv_error_mean'],
           yerr=population_size_results['pv_error_std'],
           marker='s', capsize=5, linewidth=2, markersize=8,
           color='blue', label='Population Vector')

ax.set_xlabel('Number of Cells', fontsize=12)
ax.set_ylabel('Mean Absolute Error (cm)', fontsize=12)
ax.set_title('Decoding Error vs Population Size',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decoder_population_size.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 6: DECODER COMPARISON ACROSS NOISE LEVELS

print("\nAnalyzing decoder robustness to noise...")

def compare_decoders_noise(place_cells, true_position, dt,
                          noise_levels=np.linspace(0, 2, 11)):
    """
    Compare decoder performance across noise levels
    Noise added as multiplicative factor on spike counts
    """
    
    results = {
        'noise_levels': noise_levels,
        'bayes_error': [],
        'pv_error': [],
        'bayes_std': [],
        'pv_std': []
    }
    
    for noise_factor in tqdm(noise_levels, desc="Noise level"):
        # Generate clean spikes
        _, clean_rates = generate_population_activity(place_cells, true_position, dt)
        
        # Add noise
        noisy_spikes = np.zeros_like(clean_rates)
        for i in range(clean_rates.shape[0]):
            for t in range(clean_rates.shape[1]):
                # Add noise to expected rate
                noisy_rate = clean_rates[i, t] * (1 + noise_factor * np.random.randn())
                noisy_rate = max(0, noisy_rate)  # Keep positive
                noisy_spikes[i, t] = np.random.poisson(noisy_rate * dt)
        
        # Decode with Bayesian
        decoder_bayes = BayesianDecoder(place_cells, track_length)
        decoded_bayes, _ = decoder_bayes.decode(noisy_spikes, dt)
        error_bayes = np.abs(decoded_bayes - true_position)
        
        # Decode with Population Vector
        decoder_pv = PopulationVectorDecoder(place_cells, track_length)
        decoded_pv = decoder_pv.decode(noisy_spikes, dt)
        error_pv = np.abs(decoded_pv - true_position)
        
        results['bayes_error'].append(np.mean(error_bayes))
        results['bayes_std'].append(np.std(error_bayes))
        results['pv_error'].append(np.mean(error_pv))
        results['pv_std'].append(np.std(error_pv))
    
    return results

noise_results = compare_decoders_noise(place_cells, true_position, dt)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Mean error
ax1.plot(noise_results['noise_levels'],
        noise_results['bayes_error'],
        marker='o', linewidth=2, markersize=8,
        color='red', label='Bayesian decoder')

ax1.plot(noise_results['noise_levels'],
        noise_results['pv_error'],
        marker='s', linewidth=2, markersize=8,
        color='blue', label='Population Vector')

ax1.set_xlabel('Noise Level (σ)', fontsize=12)
ax1.set_ylabel('Mean Absolute Error (cm)', fontsize=12)
ax1.set_title('Decoder Performance vs Noise Level',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Error ratio
ratio = np.array(noise_results['pv_error']) / np.array(noise_results['bayes_error'])
ax2.plot(noise_results['noise_levels'], ratio,
        marker='o', linewidth=2, markersize=8, color='purple')
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Noise Level (σ)', fontsize=12)
ax2.set_ylabel('Error Ratio (PV / Bayesian)', fontsize=12)
ax2.set_title('Relative Performance\n(>1 = Bayesian better)',
             fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decoder_noise_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 7: ADDITIONAL VISUALIZATIONS

# Tuning curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

positions_plot = np.linspace(0, track_length, 200)

for idx in range(6):
    ax = axes[idx]
    cell = place_cells[idx]
    
    rates = cell.firing_rate(positions_plot)
    ax.plot(positions_plot, rates, linewidth=2, color='blue')
    ax.axvline(x=cell.center, color='red', linestyle='--', 
              linewidth=1.5, label=f'Center: {cell.center:.1f} cm')
    
    ax.set_xlabel('Position (cm)', fontsize=10)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
    ax.set_title(f'Place Cell {idx} Tuning Curve\nPeak: {cell.peak_rate:.1f} Hz',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Example Place Cell Tuning Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tuning_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# Population activity raster
fig, axes = plt.subplots(3, 1, figsize=(15, 9))

time_window = slice(0, 300)  # 30 seconds

# Raster plot
ax = axes[0]
for cell_idx in range(n_cells):
    spike_times = time[time_window][spike_counts[cell_idx, time_window] > 0]
    ax.scatter(spike_times, [cell_idx] * len(spike_times),
              s=2, c='black', alpha=0.5)

ax.set_ylabel('Cell ID', fontsize=11)
ax.set_title('Population Activity Raster', fontsize=12, fontweight='bold')
ax.set_xlim(time[time_window][0], time[time_window][-1])

# Firing rates heatmap
ax = axes[1]
im = ax.imshow(firing_rates[:, time_window], aspect='auto',
              cmap='hot', interpolation='nearest',
              extent=[time[time_window][0], time[time_window][-1], 
                     n_cells, 0])
ax.set_ylabel('Cell ID', fontsize=11)
ax.set_title('Firing Rate Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')

# Position
ax = axes[2]
ax.plot(time[time_window], true_position[time_window], 
       'k-', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Position (cm)', fontsize=11)
ax.set_title('Animal Position', fontsize=12, fontweight='bold')
ax.set_xlim(time[time_window][0], time[time_window][-1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('population_activity.png', dpi=150, bbox_inches='tight')
plt.show()


# Posterior probability heatmap
fig, ax = plt.subplots(figsize=(15, 6))

time_window_short = slice(0, 100)  # 10 seconds
im = ax.imshow(posteriors[:, time_window_short], aspect='auto',
              cmap='viridis', interpolation='nearest',
              extent=[time[time_window_short][0], time[time_window_short][-1],
                     0, track_length],
              origin='lower')

ax.plot(time[time_window_short], true_position[time_window_short],
       'r-', linewidth=3, label='True position', alpha=0.8)
ax.plot(time[time_window_short], decoded_position_bayes[time_window_short],
       'w--', linewidth=2, label='Decoded position', alpha=0.8)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Position (cm)', fontsize=12)
ax.set_title('Bayesian Posterior Probability Over Time',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
plt.colorbar(im, ax=ax, label='Posterior Probability')

plt.tight_layout()
plt.savefig('posterior_probability.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 8: SUMMARY STATISTICS

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nSimulation Parameters:")
print(f"  Number of place cells: {n_cells}")
print(f"  Track length: {track_length} cm")
print(f"  Duration: {duration} s")
print(f"  Time bins: {len(time)}")
print(f"  Bin size: {dt} s")

print(f"\nPlace Cell Properties:")
centers = [cell.center for cell in place_cells]
widths = [cell.width for cell in place_cells]
peak_rates = [cell.peak_rate for cell in place_cells]
print(f"  Field centers: {min(centers):.1f} - {max(centers):.1f} cm")
print(f"  Mean field width: {np.mean(widths):.1f} ± {np.std(widths):.1f} cm")
print(f"  Mean peak rate: {np.mean(peak_rates):.1f} ± {np.std(peak_rates):.1f} Hz")

print(f"\nDecoding Performance:")
print(f"\n  Bayesian Decoder:")
print(f"    Mean error: {np.mean(decoding_error_bayes):.2f} cm")
print(f"    Median error: {np.median(decoding_error_bayes):.2f} cm")
print(f"    95th percentile: {np.percentile(decoding_error_bayes, 95):.2f} cm")

print(f"\n  Population Vector:")
print(f"    Mean error: {np.mean(decoding_error_pv):.2f} cm")
print(f"    Median error: {np.median(decoding_error_pv):.2f} cm")
print(f"    95th percentile: {np.percentile(decoding_error_pv, 95):.2f} cm")

improvement = (np.mean(decoding_error_pv) - np.mean(decoding_error_bayes)) / np.mean(decoding_error_pv) * 100
print(f"\n  Bayesian improvement: {improvement:.1f}%")

print(f"\nPopulation Size Analysis:")
print(f"  Tested: {population_size_results['n_cells'][0]} - {population_size_results['n_cells'][-1]} cells")
print(f"  Error reduction (5→30 cells):")
print(f"    Bayesian: {population_size_results['bayes_error_mean'][0]:.2f} → {population_size_results['bayes_error_mean'][-1]:.2f} cm")
print(f"    Pop. Vector: {population_size_results['pv_error_mean'][0]:.2f} → {population_size_results['pv_error_mean'][-1]:.2f} cm")

print(f"\nNoise Robustness:")
print(f"  Noise levels tested: {noise_results['noise_levels'][0]:.1f} - {noise_results['noise_levels'][-1]:.1f} σ")
print(f"  Bayesian error increase: {noise_results['bayes_error'][0]:.2f} → {noise_results['bayes_error'][-1]:.2f} cm")
print(f"  Pop. Vector error increase: {noise_results['pv_error'][0]:.2f} → {noise_results['pv_error'][-1]:.2f} cm")

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - decoder_position_comparison.png")
print("  - decoder_population_size.png")
print("  - decoder_noise_comparison.png")
print("  - tuning_curves.png")
print("  - population_activity.png")
print("  - posterior_probability.png")
