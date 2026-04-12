# Bayesian Place Cell Decoder

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.5%2B-8CAAE6.svg)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3%2B-11557c.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implementation and comparison of Bayesian and Population Vector decoding algorithms for hippocampal place cell activity

## Table of Contents
- [Overview](#overview)
- [What This Project Does](#what-this-project-does)
- [Key Features](#key-features)
- [Scientific Background](#scientific-background)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Pipeline](#analysis-pipeline)
- [Output & Visualizations](#output--visualizations)
- [Results](#results)
- [Project Structure](#project-structure)
- [Mathematical Models](#mathematical-models)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements two neural decoding algorithms used in computational neuroscience to infer an animal's spatial location from hippocampal place cell activity: **Bayesian decoding** and **Population Vector decoding**. The implementation includes:

- Realistic place cell simulation with Gaussian tuning curves
- Trajectory generation on a linear track
- Poisson spike generation with biologically plausible parameters
- Comprehensive comparison of decoding algorithms
- Analysis of population size effects and noise robustness

This work demonstrates fundamental principles of neural coding, statistical inference in neuroscience, and how the brain represents spatial information.

---

## What This Project Does

### Core Functionality:

1. **Place Cell Simulation**
   - Generates population of hippocampal place cells with Gaussian receptive fields
   - Simulates realistic firing patterns using Poisson statistics
   - Models animal movement on a linear track

2. **Bayesian Decoding**
   - Implements probabilistic inference: P(position | spikes)
   - Uses Poisson likelihood model with uniform prior
   - Computes full posterior distribution over positions
   - Provides maximum a posteriori (MAP) estimates

3. **Population Vector Decoding**
   - Implements center-of-mass decoding approach
   - Weighted average of place field centers by firing rate
   - Computationally efficient alternative to Bayesian method

4. **Comparative Analysis**
   - Performance vs. population size (5-30 cells)
   - Robustness to neural noise
   - Decoding accuracy metrics
   - Statistical comparison of methods

### Real-World Applications:

This type of analysis is used in:
- **Brain-machine interfaces** for decoding movement intentions
- **Neuroscience research** studying spatial memory and navigation
- **Prosthetic development** for neural control systems
- **Computational psychiatry** understanding memory disorders

---

## Key Features

- **Realistic Neural Simulation**: Biologically plausible place cells with Gaussian tuning curves
- **Dual Decoding Methods**: Complete implementation of Bayesian and Population Vector approaches
- **Comprehensive Testing**: Population size scaling and noise robustness analysis
- **Probabilistic Framework**: Full posterior distributions, not just point estimates
- **Visualization Suite**: 6 multi-panel publication-quality figures
- **Performance Metrics**: Mean, median, and percentile error statistics
- **Modular Design**: Reusable classes for place cells and decoders
- **Statistical Rigor**: Error bars, confidence intervals, and comparative statistics

---

## Scientific Background

### What are Place Cells?

**Place cells** are neurons in the hippocampus that fire when an animal occupies specific locations in space. Discovered by John O'Keefe (Nobel Prize 2014), they form the brain's "cognitive map" of the environment.

Key properties:
- Each place cell has a **place field** - a preferred location where it fires maximally
- Firing rate follows a **Gaussian tuning curve** around the place field center
- Population of place cells collectively represents the entire environment
- Spike timing follows **Poisson statistics**

### Neural Decoding Problem

**Neural decoding** aims to infer external variables (like position) from neural activity:
- **Forward model**: Position → Neural activity (encoding)
- **Inverse problem**: Neural activity → Position (decoding)

Two main approaches:

1. **Bayesian Decoding**
   - Principled probabilistic framework
   - Optimal under certain assumptions
   - Computes full posterior distribution
   - Accounts for uncertainty

2. **Population Vector Decoding**
   - Simple weighted average
   - Computationally efficient
   - Intuitive geometric interpretation
   - No explicit uncertainty quantification

### Why This Matters

Understanding neural decoding helps:
- Design better brain-machine interfaces
- Understand how the brain represents information
- Develop neuroprosthetics and assistive devices
- Study memory and navigation disorders
- Validate computational theories of neural coding

---

## Technologies Used

### Core Libraries:
- **Python 3.7+**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **SciPy**: Statistical distributions (Poisson) and signal processing
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Enhanced statistical visualizations
- **tqdm**: Progress bars for long computations

### Key Algorithms:
- Bayesian inference with Poisson likelihood
- Population vector decoding
- Gaussian tuning curve modeling
- Monte Carlo simulation
- Statistical hypothesis testing

---

## Prerequisites

### Required Knowledge:
- Basic Python programming
- Understanding of probability and statistics
- Familiarity with Bayes' theorem (helpful)
- Basic neuroscience concepts (optional but beneficial)

### System Requirements:
- **Python**: Version 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: ~100MB for code and outputs
- **OS**: Windows, macOS, or Linux

### Python Packages:
```bash
numpy >= 1.19.0
scipy >= 1.5.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
tqdm >= 4.50.0
```

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/hadbierox196/bayesian-place-cell-decoder.git
cd bayesian-place-cell-decoder
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# OR using conda
conda create -n place-cell python=3.8
conda activate place-cell
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```text
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
```

### Step 4: Verify Installation
```bash
python -c "import numpy, scipy, matplotlib; print('All packages installed successfully!')"
```

---

## Usage

### Quick Start:
```bash
python bayesian_place_cell_decoder.py
```

### What Happens:
1. Generates 30 place cells tiling a 100 cm linear track
2. Simulates 60 seconds of animal movement and neural activity
3. Decodes position using both Bayesian and Population Vector methods
4. Analyzes performance vs. population size (5-30 cells)
5. Tests robustness to neural noise
6. Generates 6 comprehensive visualization figures

### Expected Runtime:
- **Place cell generation**: < 1 second
- **Trajectory simulation**: < 1 second
- **Bayesian decoding**: approximately 3-5 seconds
- **Population Vector decoding**: < 1 second
- **Population size analysis**: approximately 30-60 seconds
- **Noise analysis**: approximately 60-90 seconds
- **Visualization**: approximately 5 seconds
- **Total**: approximately 2-3 minutes

### Customization Options:

#### Modify Simulation Parameters:
```python
# In the main script
n_cells = 30              # Number of place cells
track_length = 100        # Track length in cm
duration = 60             # Recording duration in seconds
dt = 0.1                  # Time bin size in seconds
```

#### Adjust Place Cell Properties:
```python
place_cells = create_place_cell_population(
    n_cells=30,
    track_length=100,
    place_field_width_range=(5, 15),    # Width in cm
    peak_rate_range=(5, 20)             # Peak firing rate in Hz
)
```

#### Change Decoding Resolution:
```python
bayesian_decoder = BayesianDecoder(
    place_cells,
    track_length=100,
    position_bins=100       # Number of position bins (higher = finer resolution)
)
```

---

## Analysis Pipeline

### Part 1: Place Cell Simulation
- **PlaceCell class** with Gaussian tuning curves
- **Spike generation** using Poisson process
- **Population creation** with uniform tiling of track
- **Trajectory simulation** with realistic movement dynamics

### Part 2: Bayesian Decoder Implementation
- **Likelihood computation** using Poisson model: P(spikes | position)
- **Prior specification** (uniform over positions)
- **Posterior calculation**: P(position | spikes) ∝ P(spikes | position) × P(position)
- **MAP estimation** for position decoding

### Part 3: Population Vector Decoder
- **Weighted averaging** of place field centers
- Weights determined by **spike counts**
- Handles zero-spike cases with default to track center

### Part 4: Position Decoding Visualization
- True vs. decoded position over time
- Separate plots for each decoder
- Comparison overlay
- Decoding error time series

### Part 5: Population Size Analysis
- Test 5, 10, 15, 20, 25, 30 cells
- 10 repetitions with random cell selection
- Error bars showing variability
- Comparison of scaling behavior

### Part 6: Noise Robustness Analysis
- Noise levels from 0 to 2 sigma
- Multiplicative noise on firing rates
- Performance degradation curves
- Relative performance ratio

### Part 7: Additional Visualizations
- Individual place cell tuning curves
- Population activity raster plots
- Firing rate heatmaps
- Posterior probability evolution

### Part 8: Summary Statistics
- Comprehensive performance metrics
- Statistical comparisons
- Population size effects
- Noise robustness summary

---

## Output & Visualizations

### Generated Files:

#### 1. **decoder_position_comparison.png**
Four-panel figure showing:
- True vs. Bayesian decoded position
- True vs. Population Vector decoded position
- Combined comparison of both methods
- Decoding errors over time
- **Interpretation**: Shows real-time decoding accuracy and temporal dynamics

#### 2. **decoder_population_size.png**
Analysis of performance scaling:
- Mean absolute error vs. number of cells
- Error bars showing variability across repetitions
- Both decoders plotted for comparison
- **Interpretation**: Reveals how many neurons are needed for accurate decoding

#### 3. **decoder_noise_comparison.png**
Two-panel figure:
- Mean error vs. noise level for both decoders
- Error ratio (PV/Bayesian) showing relative performance
- **Interpretation**: Tests robustness to neural noise and variability

#### 4. **tuning_curves.png**
Six example place cells:
- Firing rate vs. position
- Place field centers marked
- Peak rates annotated
- **Interpretation**: Shows diversity of place field properties

#### 5. **population_activity.png**
Three-panel figure:
- Spike raster across population
- Firing rate heatmap over time
- Animal position trajectory
- **Interpretation**: Visualizes population code for position

#### 6. **posterior_probability.png**
Bayesian posterior heatmap:
- Probability distribution over positions and time
- True position overlay
- Decoded position overlay
- **Interpretation**: Shows uncertainty and confidence in decoding

### Console Output Example:
```
======================================================================
BAYESIAN PLACE CELL DECODER
======================================================================

Creating 30 place cells...
✓ Created 30 place cells
  Place field centers: 3.2 - 96.8 cm
  Place field widths: (5, 15)
  Peak firing rates: (5, 20) Hz

Simulating 60s trajectory...

Generating population activity...
✓ Generated data
  Time points: 600
  Total spikes: 1847
  Mean firing rate: 1.03 Hz

======================================================================
PART 2: Bayesian Decoder
======================================================================

✓ Initialized Bayesian decoder
  Position bins: 100

Decoding trajectory...
✓ Bayesian decoding complete
  Mean absolute error: 4.23 cm
  Median absolute error: 2.87 cm

======================================================================
PART 3: Population Vector Decoder
======================================================================

✓ Initialized Population Vector decoder

Decoding trajectory...
✓ Population Vector decoding complete
  Mean absolute error: 5.67 cm
  Median absolute error: 4.12 cm

Analyzing effect of population size...
Population size: 100%|████████████████| 6/6 [00:45<00:00,  7.58s/it]

Analyzing decoder robustness to noise...
Noise level: 100%|██████████████████| 11/11 [01:23<00:00,  7.59s/it]

======================================================================
ANALYSIS SUMMARY
======================================================================

Simulation Parameters:
  Number of place cells: 30
  Track length: 100 cm
  Duration: 60 s
  Time bins: 600
  Bin size: 0.1 s

Place Cell Properties:
  Field centers: 3.2 - 96.8 cm
  Mean field width: 9.8 ± 2.9 cm
  Mean peak rate: 12.3 ± 4.2 Hz

Decoding Performance:

  Bayesian Decoder:
    Mean error: 4.23 cm
    Median error: 2.87 cm
    95th percentile: 12.45 cm

  Population Vector:
    Mean error: 5.67 cm
    Median error: 4.12 cm
    95th percentile: 15.23 cm

  Bayesian improvement: 25.4%

Population Size Analysis:
  Tested: 5 - 30 cells
  Error reduction (5→30 cells):
    Bayesian: 8.92 → 4.23 cm
    Pop. Vector: 11.34 → 5.67 cm

Noise Robustness:
  Noise levels tested: 0.0 - 2.0 σ
  Bayesian error increase: 4.23 → 8.76 cm
  Pop. Vector error increase: 5.67 → 12.34 cm

======================================================================
ALL ANALYSES COMPLETE
======================================================================
```

---

## Results

### Typical Performance Metrics:

1. **Decoding Accuracy**
   - Bayesian decoder: 4-5 cm mean error
   - Population Vector: 5-7 cm mean error
   - Bayesian improvement: 20-30%
   - Result: Bayesian method is more accurate

2. **Population Size Effects**
   - 5 cells: ~9 cm error (Bayesian)
   - 30 cells: ~4 cm error (Bayesian)
   - Result: Logarithmic improvement with more cells

3. **Noise Robustness**
   - Both decoders degrade with noise
   - Bayesian maintains advantage at all noise levels
   - Error ratio increases slightly with noise
   - Result: Bayesian is more robust to noise

4. **Computational Efficiency**
   - Population Vector: < 1 second
   - Bayesian: 3-5 seconds
   - Result: 3-5x speed difference for this dataset

5. **Uncertainty Quantification**
   - Bayesian provides full posterior distribution
   - Can compute confidence intervals
   - Population Vector gives point estimate only
   - Result: Bayesian offers richer information

---

## Project Structure

```
bayesian-place-cell-decoder/
│
├── bayesian_place_cell_decoder.py    # Main analysis script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── LICENSE                            # MIT License
│
├── outputs/                           # Generated figures (created on run)
│   ├── decoder_position_comparison.png
│   ├── decoder_population_size.png
│   ├── decoder_noise_comparison.png
│   ├── tuning_curves.png
│   ├── population_activity.png
│   └── posterior_probability.png
│
└── docs/                              # Additional documentation
    ├── THEORY.md                      # Mathematical theory
    ├── METHODS.md                     # Detailed methods
    └── APPLICATIONS.md                # Real-world applications
```

---

## Mathematical Models

### 1. Place Cell Tuning Curve (Gaussian):
```
f(x) = f_max × exp(-(x - μ)² / (2σ²))
```
Where:
- `f(x)` = firing rate at position x
- `f_max` = peak firing rate
- `μ` = place field center
- `σ` = place field width

### 2. Poisson Spike Generation:
```
P(n spikes | λ) = (λ^n × e^(-λ)) / n!
```
Where:
- `n` = observed spike count
- `λ = f(x) × Δt` = expected spike count
- `Δt` = time bin size

### 3. Bayesian Decoding:
```
P(x | spikes) ∝ P(spikes | x) × P(x)
```

Likelihood for all cells:
```
P(spikes | x) = ∏ᵢ P(nᵢ | λᵢ(x))
```

Where:
- `x` = position
- `nᵢ` = spike count of cell i
- `λᵢ(x)` = expected spike count of cell i at position x
- `P(x)` = prior (uniform)

### 4. Population Vector Decoding:
```
x̂ = Σᵢ (nᵢ × μᵢ) / Σᵢ nᵢ
```
Where:
- `x̂` = decoded position
- `nᵢ` = spike count of cell i
- `μᵢ` = place field center of cell i

### 5. Mean Absolute Error:
```
MAE = (1/T) × Σₜ |x̂ₜ - xₜ|
```
Where:
- `T` = number of time bins
- `x̂ₜ` = decoded position at time t
- `xₜ` = true position at time t

---

## Future Enhancements

### Planned Features:
- **2D environments**: Extend to open field navigation
- **Ring attractor model**: Compare with continuous attractor dynamics
- **Temporal integration**: Bayesian filtering with motion model
- **Spike timing**: Incorporate temporal precision
- **Grid cells**: Add grid cell population for multi-scale coding
- **Real data support**: Import actual neural recordings
- **Online decoding**: Real-time position estimation
- **Adaptive binning**: Optimize temporal resolution

### Research Extensions:
- Compare with actual hippocampal data
- Test different prior distributions
- Implement particle filtering
- Study replay event decoding
- Analyze theta phase precession
- Investigate place cell remapping
- Model spatial memory consolidation

---

## Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs:
1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Minimal code to reproduce
   - Expected vs. actual behavior
   - Python version and OS

### Suggesting Enhancements:
1. Open an issue describing the feature
2. Explain the scientific motivation
3. Provide references if applicable

### Pull Requests:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/ImprovedDecoder`)
3. Commit changes (`git commit -m 'Add Kalman filter decoder'`)
4. Push to branch (`git push origin feature/ImprovedDecoder`)
5. Open a Pull Request

### Code Style:
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Update README for new features
- Add unit tests for new functionality

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Hassan Farooq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

[Full license text...]
```

---



## Acknowledgments

- Inspired by work from the O'Keefe and Moser labs on place cells and grid cells
- Decoding methods based on Zhang et al. (1998) and Brown et al. (1998)
- Bayesian framework follows Wilson & McNaughton (1993)
- Statistical methods adapted from computational neuroscience literature
- Special thanks to the open-source scientific Python community

---

## References

### Key Papers:

1. **O'Keefe, J., & Dostrovsky, J. (1971).** "The hippocampus as a spatial map." *Brain Research*
   - Original discovery of place cells

2. **Zhang, K., et al. (1998).** "Interpreting neuronal population activity by reconstruction: unified framework with application to hippocampal place cells." *Journal of Neurophysiology*
   - Bayesian decoding framework

3. **Brown, E. N., et al. (1998).** "A statistical paradigm for neural spike train decoding applied to position prediction from ensemble firing patterns of rat hippocampal place cells." *Journal of Neuroscience*
   - Point process methods

4. **Wilson, M. A., & McNaughton, B. L. (1993).** "Dynamics of the hippocampal ensemble code for space." *Science*
   - Population coding principles

5. **Georgopoulos, A. P., et al. (1986).** "Neuronal population coding of movement direction." *Science*
   - Population vector method

### Documentation:
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Stats Module](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Theoretical Neuroscience (Dayan & Abbott)](http://www.gatsby.ucl.ac.uk/~lmate/biblio/dayanabbott.pdf)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bayesian_place_cell_decoder,
  author = {Hassan Farooq},
  title = {Bayesian Place Cell Decoder: Implementation and Analysis},
  year = {2024},
  url = {https://github.com/hadbierox196/bayesian-place-cell-decoder}
}
```

---

## Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/hadbierox196/bayesian-place-cell-decoder)
![GitHub code size](https://img.shields.io/github/languages/code-size/hadbierox196/bayesian-place-cell-decoder)
![Lines of code](https://img.shields.io/tokei/lines/github/hadbierox196/bayesian-place-cell-decoder)
![GitHub last commit](https://img.shields.io/github/last-commit/hadbierox196/bayesian-place-cell-decoder)

---

<div align="center">

**Decoding the neural code of spatial memory through computational neuroscience**

[Back to Top](#bayesian-place-cell-decoder)

</div>
