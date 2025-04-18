import os
import pathlib
import shutil

data_dir = pathlib.Path(__file__).parent / "data"
eplus_dir = pathlib.Path("C:/EnergyPlus-24.2.0") # energyplus installation directory
results_dir = data_dir / "Results"

def create_directories():
    for key, value in CONFIG['paths'].items():
        pathlib.Path(value).mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Paths configuration
    "paths":{
        "data_dir": data_dir,
        "results_dir": results_dir, # Results Output Directory
        "prototypes_dir": data_dir / "Prototypes", # Prototype Building IDF Files Directory
        "tmy_dir": data_dir / "TMYs", # Typical Meteorological Year Files Directory
        "ftmy_dir": data_dir / "FTMYs", # Future Typical Meteorological Year Files Directory
        "epsim_dir": results_dir / "EPSim", # EnergyPlus Benchmark Output Directory
        "sensitivity_dir": results_dir / "Sensitivity", # Sensitivity Analysis and Optimization Results Output Directory
        "future_load_dir": results_dir / "FutureLoad", # Future Loads Simulation Output Directory
        "ensemble_dir": results_dir / "Ensemble", # Ensemble Simulation Output Directory
        "eplus_executable": eplus_dir / "energyplus.exe", # EnergyPlus executable file path
        "log_dir": results_dir / "Logs", # Log files output directory
        },
    # Constants configuration
    "constants":{
        'ng_conversion_factor': 3.2, # conversion factor for natural gas to energy
        'cpu_count_override': os.cpu_count() - 1, # number of cores to use for simulation
        'debug': False, # debug mode
    },

    # Different Building Types Ratio of the city
    "building_ratios":{
        "office_large": 0.000663, # Large office building
        "office_medium": 0.001725, # Medium office building
        "apartment_high_rise": 0.000384, # High-rise apartment building
        "sf": 0.608149, # Single-family residential
        "mf": 0.389079 # Multi-family residential
    },

    # Energy conservation measures parameter ranges
    "ecm_ranges":{
        "shgc": [0.2, 0.4, 0.6, 0.8], # Solar Heat Gain Coefficient
        "win_u": [0.5, 1, 1.5, 2, 2.5, 3], # Window U-value (W/m2K)
        "nv_area": [0.4, 0.8, 1.2, 1.6, 2, 2.4], # Natural Ventilation Area (m2)
        "insu": [1, 2, 3, 4], # Wall Insulation R-value (m2K/W)
        "infl": [0.25, 0.5, 0.75, 1, 1.25, 1.5], # Air Infiltration Rate (ACH)
        "cool_cop": [3.5, 4, 4.5, 5, 5.5, 6], # Cooling Coefficient of Performance
        "cool_air_temp": [10, 12, 14, 16], # Cooling Air Supply Temperature (C)
        "lighting": [1, 2, 3], # Lighting Power Density (W/m2)
        "vt": [0.6] # Visible Light Transmittance
    },

    # Lighting reduction rate
    "lighting_reduction_map":{
        "office_large": {1: 0.2, 2: 0.47, 3: 0.53},
        "office_medium": {1: 0.2, 2: 0.47, 3: 0.53},
        "apartment_high_rise": {1: 0.35, 2: 0.45, 3: 0.55},
        "sf": {1: 0.45, 2: 0.5, 3: 0.64},
        "mf": {1: 0.35, 2: 0.45, 3: 0.55},
    },

    # Simulation settings
    "simulation":{
        "start_year": 2040,
        "end_year": 2040,
        "default_output_suffix": "C", # suffix for the output file
        "cleanup_files": ['.eso', '.mtr', '.rdd', '.mdd', '.err', '.svg', '.dxf', '.audit', '.bnd', '.eio', '.shd', '.edd', '.end', '.mtd', '.rvaudit', '.sql'], # files to be cleaned up
    },

    # sensitivity analysis settings
    "analysis":{
        "sensitivity_samples_n": 2, # Number of samples for Saltelli's sampling
        "n_estimators": 100, # Number of trees for Random Forest
        "random_state": 10, # Random state for Random Forest
        "optimization_model": 'rf', # Optimization model example: ['ols', 'rf', etc]
        "ga_population_size": 100, # Population size for genetic algorithm
        "ga_generations": 100, # Number of generations for genetic algorithm
    },

    # PV system settings
    'pv_analysis': {
        'enabled': True, # Enable PV analysis process
        'pv_efficiency': 0.18, # PV module efficiency
        'pv_coverage': 0.8, # PV module coverage on surfaces (consider gaps)
        'shadow_calculation_surface_types': ['ROOF', 'WALL'], # Surface types to calculate shadows/radiation (uppercase)
        'radiation_threshold_high': 1000.0, # High radiation threshold (kWh/m2)
        'radiation_threshold_low': 600.0,   # Low radiation threshold (kWh/m2)
        'radiation_score_threshold': 70, # Minimum radiation score (0-100)
        'max_score': 100.0, # Maximum radiation score
        'min_score': 0.0,   # Minimum radiation score
        'pv_output_prefix': 'pv', # PV simulation output prefix
        'shadow_output_prefix': 'shadow', # Shadow analysis simulation output prefix
    }
}

create_directories()