"""
Spectral generator for the compositional generalization tutorial.
Generates synthetic spectra with configurable spectral lines.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import yaml
from dataclasses import dataclass, field


@dataclass
class SpectralLine:
    """Configuration for a single spectral line."""
    center: float  # Peak wavelength location
    width: float   # Gaussian width (sigma)
    label: str     # Component label (e.g., "comp1", "comp2")
    strength: float = 1.0  # Fixed strength parameter [-1, 1]


@dataclass
class SamplingConfig:
    """Configuration for sampling distributions."""
    range: Tuple[float, float]
    distribution: str = "uniform"  # "uniform" or "log_uniform"


@dataclass
class SpectralConfig:
    """Configuration for spectral generation."""
    n_bins: int = 512
    wavelength_min: float = 400.0
    wavelength_max: float = 700.0
    temperature_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig((3000, 7000), "uniform"))
    noise_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig((0.01, 0.15), "log_uniform"))
    abundance_sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig((0.0, 1.0), "uniform"))
    lines: List[SpectralLine] = None
    
    def __post_init__(self):
        if self.lines is None:
            self.lines = []


class SpectralGenerator:
    """Generates synthetic spectra with configurable spectral lines."""
    
    def __init__(self, config: SpectralConfig):
        self.config = config
        self.wavelengths = torch.linspace(
            config.wavelength_min, 
            config.wavelength_max, 
            config.n_bins
        )
        self.component_map = {line.label: i for i, line in enumerate(config.lines)}
        
    def _sample_from_config(self, sampling_config: SamplingConfig) -> float:
        """Sample a value based on the sampling configuration."""
        min_val, max_val = sampling_config.range
        if sampling_config.distribution == "uniform":
            return np.random.uniform(min_val, max_val)
        elif sampling_config.distribution == "log_uniform":
            return np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
        else:
            raise ValueError(f"Unknown distribution: {sampling_config.distribution}")
        
    def generate_spectrum(self, abundances: torch.Tensor, 
                         temperature: Optional[float] = None,
                         noise_level: Optional[float] = None) -> torch.Tensor:
        """
        Generate a synthetic spectrum from abundance vector.
        
        Args:
            abundances: Tensor of shape [n_components] with values in [-1, 1]
            temperature: Optional temperature scaling factor
            noise_level: Optional noise level override
            
        Returns:
            spectrum: Tensor of shape [n_bins] containing the generated spectrum
        """
        if temperature is None:
            temperature = self._sample_from_config(self.config.temperature_sampling)
        if noise_level is None:
            noise_level = self._sample_from_config(self.config.noise_sampling)
            
        # Start with blackbody continuum
        spectrum = self._generate_blackbody_spectrum(temperature)
        
        # Modulate spectrum with spectral lines
        for i, line in enumerate(self.config.lines):
            if i < len(abundances):
                abundance = abundances[i]  # [0, 1] abundance
                
                # Generate Gaussian line profile
                line_profile = torch.exp(
                    -0.5 * ((self.wavelengths - line.center) / line.width) ** 2
                )
                
                # Modulate blackbody: spectrum *= (1 + strength * abundance * line_profile)
                modulation = 1.0 + line.strength * abundance * line_profile
                spectrum *= modulation
        
        # Apply multiplicative noise: spectrum *= (1 + noise_level * gaussian_noise)
        gaussian_noise = torch.randn_like(spectrum)
        noise_modulation = 1.0 + noise_level * gaussian_noise
        spectrum *= noise_modulation
        
        return spectrum
    
    def _generate_blackbody_spectrum(self, temperature: float) -> torch.Tensor:
        """Generate blackbody spectrum using Planck's law."""
        # Convert nm to meters
        wavelengths_m = self.wavelengths * 1e-9
        
        # Physical constants
        h = 6.626e-34  # Planck constant (J⋅s)
        c = 3e8        # Speed of light (m/s)
        k_b = 1.381e-23  # Boltzmann constant (J/K)
        
        # Temperature is now directly in Kelvin
        temp_k = temperature
        
        # Planck's law
        numerator = 2 * h * c**2
        exponent = h * c / (wavelengths_m * k_b * temp_k)
        denominator = wavelengths_m**5 * (torch.exp(exponent) - 1)
        
        # Calculate intensity and normalize
        intensity = numerator / denominator
        intensity = intensity / torch.max(intensity)  # Normalize to [0, 1]
        
        return intensity
    
    def generate_batch(self, abundance_batch: torch.Tensor,
                      temperature_batch: Optional[torch.Tensor] = None,
                      noise_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a batch of spectra.
        
        Args:
            abundance_batch: Tensor of shape [batch_size, n_components]
            temperature_batch: Optional tensor of shape [batch_size]
            noise_levels: Optional tensor of shape [batch_size]
            
        Returns:
            spectra: Tensor of shape [batch_size, n_bins]
        """
        batch_size = abundance_batch.shape[0]
        spectra = torch.zeros(batch_size, self.config.n_bins)
        
        for i in range(batch_size):
            temp = temperature_batch[i] if temperature_batch is not None else None
            noise = noise_levels[i] if noise_levels is not None else None
            spectra[i] = self.generate_spectrum(abundance_batch[i], temp, noise)
            
        return spectra
    
    def sample_abundances(self, n_samples: int, 
                         allowed_combinations: List[List[str]]) -> torch.Tensor:
        """
        Sample abundance vectors according to allowed combinations.
        
        Args:
            n_samples: Number of samples to generate
            allowed_combinations: List of allowed component combinations
            
        Returns:
            abundances: Tensor of shape [n_samples, n_components]
        """
        n_components = len(self.config.lines)
        abundances = torch.zeros(n_samples, n_components)
        
        for i in range(n_samples):
            # Randomly select an allowed combination
            combo = np.random.choice(len(allowed_combinations))
            active_components = allowed_combinations[combo]
            
            for comp_name in active_components:
                if comp_name in self.component_map:
                    comp_idx = self.component_map[comp_name]
                    # Sample abundance from configured distribution
                    abundances[i, comp_idx] = self._sample_from_config(self.config.abundance_sampling)
        
        return abundances
    
    def plot_spectrum(self, spectrum: torch.Tensor, title: str = "Spectrum"):
        """Plot a single spectrum."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.wavelengths.numpy(), spectrum.numpy())
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_components(self, abundances: torch.Tensor, 
                       title: str = "Component Abundances"):
        """Plot component abundances as a bar chart."""
        plt.figure(figsize=(12, 6))
        component_names = [line.label for line in self.config.lines]
        plt.bar(component_names, abundances.numpy())
        plt.xlabel('Component')
        plt.ylabel('Abundance')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def load_spectral_config(config_dict: dict) -> SpectralConfig:
    """Load spectral configuration from config dictionary."""
    
    # Parse spectral lines
    lines = []
    for line_config in config_dict.get('lines', []):
        lines.append(SpectralLine(
            center=line_config['center'],
            width=line_config['width'],
            label=line_config['label'],
            strength=line_config.get('strength', 1.0)
        ))
    
    # Parse sampling configurations
    sampling = config_dict.get('sampling', {})
    
    temp_config = sampling.get('temperature', {})
    temperature_sampling = SamplingConfig(
        range=tuple(temp_config.get('range', [3000, 7000])),
        distribution=temp_config.get('distribution', 'uniform')
    )
    
    noise_config = sampling.get('noise', {})
    noise_sampling = SamplingConfig(
        range=tuple(noise_config.get('range', [0.01, 0.15])),
        distribution=noise_config.get('distribution', 'log_uniform')
    )
    
    abundance_config = sampling.get('abundance', {})
    abundance_sampling = SamplingConfig(
        range=tuple(abundance_config.get('range', [0.0, 1.0])),
        distribution=abundance_config.get('distribution', 'uniform')
    )
    
    return SpectralConfig(
        n_bins=config_dict.get('n_bins', 512),
        wavelength_min=config_dict.get('wavelength_min', 400.0),
        wavelength_max=config_dict.get('wavelength_max', 700.0),
        temperature_sampling=temperature_sampling,
        noise_sampling=noise_sampling,
        abundance_sampling=abundance_sampling,
        lines=lines
    )


def generate_train_test_data(config: SpectralConfig, 
                           train_combinations: List[List[str]],
                           test_combinations: List[List[str]],
                           n_train: int = 1000,
                           n_test: int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training and test datasets.
    
    Args:
        config: Spectral configuration
        train_combinations: Allowed combinations for training
        test_combinations: Allowed combinations for testing
        n_train: Number of training samples
        n_test: Number of test samples
        
    Returns:
        train_spectra, train_abundances, test_spectra, test_abundances
    """
    generator = SpectralGenerator(config)
    
    # Generate training data
    train_abundances = generator.sample_abundances(n_train, train_combinations)
    train_spectra = generator.generate_batch(train_abundances)
    
    # Generate test data
    test_abundances = generator.sample_abundances(n_test, test_combinations)
    test_spectra = generator.generate_batch(test_abundances)
    
    return train_spectra, train_abundances, test_spectra, test_abundances


def generate_data_from_distribution(config: SpectralConfig, 
                                  distribution_config: Dict[str, Any],
                                  n_samples: int,
                                  seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data based on distribution configuration.
    
    Args:
        config: Spectral configuration
        distribution_config: Distribution specification (type, combinations/code)
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        spectra, abundances
    """
    generator = SpectralGenerator(config)
    
    if seed is not None:
        np.random.seed(seed)
    
    if distribution_config['type'] == 'component_combinations':
        # Sample from weighted component combinations
        combinations_config = distribution_config['combinations']
        
        # Extract weights and normalize
        weights = [c['weight'] for c in combinations_config]
        weights = np.array(weights) / np.sum(weights)
        
        # Sample abundances
        abundances = torch.zeros(n_samples, len(config.lines))
        
        for i in range(n_samples):
            # Choose combination based on weights
            combo_idx = np.random.choice(len(combinations_config), p=weights)
            active_components = combinations_config[combo_idx]['components']
            
            # Set abundances for active components
            for comp_name in active_components:
                if comp_name in generator.component_map:
                    comp_idx = generator.component_map[comp_name]
                    abundances[i, comp_idx] = generator._sample_from_config(config.abundance_sampling)
        
        # Generate spectra
        spectra = generator.generate_batch(abundances)
        
    elif distribution_config['type'] == 'sampling_code':
        # Execute custom sampling code
        code = distribution_config['code']
        
        # Create a namespace for the code execution
        namespace = {}
        exec(code, namespace)
        
        # Get the sampling function
        if 'sample_abundances' not in namespace:
            raise ValueError("sampling_code must define a 'sample_abundances' function")
        
        sample_func = namespace['sample_abundances']
        
        # Create random generator for reproducibility
        rng = np.random.default_rng(seed)
        
        # Sample abundances using custom function
        abundances = torch.zeros(n_samples, len(config.lines))
        
        for i in range(n_samples):
            abundance_dict = sample_func(rng)
            for comp_name, value in abundance_dict.items():
                if comp_name in generator.component_map:
                    comp_idx = generator.component_map[comp_name]
                    abundances[i, comp_idx] = value
        
        # Generate spectra
        spectra = generator.generate_batch(abundances)
        
    else:
        raise ValueError(f"Unknown distribution type: {distribution_config['type']}")
    
    return spectra, abundances


def generate_split_data(config: SpectralConfig,
                       splits_config: Dict[str, Dict[str, Any]],
                       seed: int = 42) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate data for all splits defined in configuration.
    
    Args:
        config: Spectral configuration
        splits_config: Dictionary of split configurations
        seed: Base random seed
        
    Returns:
        Dictionary mapping split names to data dictionaries
    """
    all_data = {}
    
    for split_idx, (split_name, split_config) in enumerate(splits_config.items()):
        # Use different seed for each split
        split_seed = seed + split_idx
        
        n_samples = split_config['n_samples']
        distribution = split_config['distribution']
        
        spectra, abundances = generate_data_from_distribution(
            config, distribution, n_samples, split_seed
        )
        
        all_data[split_name] = {
            'spectra': spectra,
            'abundances': abundances
        }
    
    return all_data