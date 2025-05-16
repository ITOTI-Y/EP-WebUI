import os
import pathlib
import dotenv
import torch

dotenv.load_dotenv()
data_dir = pathlib.Path(__file__).parent / "data"
eplus_dir = pathlib.Path("/usr/local/bin/") # energyplus installation directory
results_dir = data_dir / "Results"


def create_directories():
    for key, value in CONFIG['paths'].items():
        if key == "eplus_executable":
            pass
        else:
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)


CONFIG = {
    # Paths configuration
    "paths": {
        "data_dir": data_dir,
        "results_dir": results_dir, # Results Output Directory
        "prototypes_dir": data_dir / "Prototypes", # Prototype Building IDF Files Directory
        "tmy_dir": data_dir / "TMYs", # Typical Meteorological Year Files Directory
        "ftmy_dir": data_dir / "FTMYs", # Future Typical Meteorological Year Files Directory
        "epsim_dir": results_dir / "EPSim", # EnergyPlus Benchmark Output Directory
        "sensitivity_dir": results_dir / "Sensitivity", # Sensitivity Analysis and Optimization Results Output Directory
        "future_load_dir": results_dir / "FutureLoad", # Future Loads Simulation Output Directory
        "ensemble_dir": results_dir / "Ensemble", # Ensemble Simulation Output Directory
        "eplus_executable": eplus_dir / "energyplus", # EnergyPlus executable file path
        "log_dir": results_dir / "Logs", # Log files output directory
        },
    # Constants configuration
    "constants": {
        'ng_conversion_factor': 3.2,  # conversion factor for natural gas to energy
        # number of cores to use for simulation
        'cpu_count_override': os.cpu_count() - 1,
        'debug': False,  # debug mode
    },

    # Different Building Types Ratio of the city
    "building_ratios": {
        "office_large": 0.000663,  # Large office building
        "office_medium": 0.001725,  # Medium office building
        "apartment_high_rise": 0.000384,  # High-rise apartment building
        "sf": 0.608149,  # Single-family residential
        "mf": 0.389079  # Multi-family residential
    },

    # Energy conservation measures parameter ranges
    "ecm_ranges": {
        "shgc": [0.2, 0.4, 0.6, 0.8],  # Solar Heat Gain Coefficient
        "win_u": [0.5, 1, 1.5, 2, 2.5, 3],  # Window U-value (W/m2K)
        # Natural Ventilation Area (m2)
        "nv_area": [0.4, 0.8, 1.2, 1.6, 2, 2.4],
        "insu": [1, 2, 3, 4],  # Wall Insulation R-value (m2K/W)
        "infl": [0.25, 0.5, 0.75, 1, 1.25, 1.5],  # Air Infiltration Rate (ACH)
        # Cooling Coefficient of Performance
        "cool_cop": [3.5, 4, 4.5, 5, 5.5, 6],
        # Cooling Air Supply Temperature (C)
        "cool_air_temp": [10, 12, 14, 16],
        "lighting": [1, 2, 3],  # Lighting Power Density (W/m2)
        "vt": [0.4, 0.6, 0.7]  # Visible Light Transmittance
    },

    # Lighting reduction rate
    "lighting_reduction_map": {
        "officelarge": {1: 0.2, 2: 0.47, 3: 0.53},
        "officemedium": {1: 0.2, 2: 0.47, 3: 0.53},
        "apartmenthighrise": {1: 0.35, 2: 0.45, 3: 0.55},
        "singlefamilyresidential": {1: 0.45, 2: 0.5, 3: 0.64},
        "multifamilyresidential": {1: 0.35, 2: 0.45, 3: 0.55},
    },

    # Simulation settings
    "simulation": {
        "start_year": 2040,
        "end_year": 2040,
        "default_output_suffix": "C",  # suffix for the output file
        # Files to be cleaned up (except .sql)
        "cleanup_files": ['.eso', '.mtr', '.rdd', '.mdd', '.err', '.svg', '.dxf', '.audit', '.bnd', '.eio', '.shd', '.edd', '.end', '.mtd', '.rvaudit'],
    },

    # sensitivity analysis settings
    "analysis": {
        "output_intermediary_files": True,  # Output intermediary files
        "sensitivity_samples_n": 32,  # Number of samples for Saltelli's sampling
        "n_estimators": 100,  # Number of trees for Random Forest
        "random_state": 10,  # Random state for Random Forest
        "surrogate_model": 'xgb', # ['ols', 'rf', 'xgb']

        "xgboost_params": {
            "load_saved_model": True,  # Attempt to load a previously saved XGBoost model
            # Filename for the saved model (relative to work_dir)
            "model_save_filename": "xgboost_surrogate_model.json",
            # Filename for best hyperparameters
            "best_params_filename": "xgboost_best_params.json",

            "use_optuna_tuning": True,  # Enable Optuna for hyperparameter tuning
            "optuna_n_trials": 50,      # Number of Optuna trials
            # Optuna timeout in seconds (e.g., 1 hour)
            "optuna_timeout": 3600,
            "optuna_cv_folds": 3,       # Number of K-Folds for Optuna's internal CV
            # Number of parallel jobs for Optuna trials (-1 to use all processors)
            "optuna_n_jobs": 1,

            # Ratio of data for early stopping validation set in final model training
            "validation_split_ratio": 0.2,
            "early_stopping_rounds": 10,  # Early stopping rounds for XGBoost training

            # Default fixed parameters for XGBoost (used if not tuning or as base for tuning)
            "fixed_params": {
                "objective": "reg:squarederror",
                "booster": "gbtree",
                # "random_state": 10, # Will be taken from general analysis random_state
                # Add other fixed params like tree_method if needed
            },
            # Parameter space for Optuna (if use_optuna_tuning is True)
            "optuna_param_space": {
                "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 50},
                "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
                "gamma": {"type": "float", "low": 0, "high": 0.5, "step": 0.05},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                # "min_child_weight": {"type": "int", "low": 1, "high": 10}
            }
        },

        # Optimizer settings
        "optimizer_type": "ga",  # 'lbfgsb' or 'ga'
        "lbfgsb_params": {       # Parameters for L-BFGS-B
            "maxiter": 150,
            "disp": False # scipy.optimize.minimize display option
        },
        "ga_params": {           # Parameters for Genetic Algorithm (DEAP)
            "population_size": 100,
            "num_generations": 50,
            "cxpb": 0.7,          # Crossover probability
            "mutpb": 0.3,         # Mutation probability for the individual
            "indpb": 0.1,        # Independent probability for each attribute to be mutated
            "tournament_size": 3,
            "hall_of_fame_size": 1
        },
    },

    # PV system settings
    'pv_analysis': {
        'enabled': True,  # Enable PV analysis process
        'pv_model_type': 'Sandia',  # Choose PV model type: 'Simple', 'Sandia', 'PVWatts'

        # --- Simple PV Model (If pv_model_type == 'Simple') ---
        'simple_pv_efficiency': 0.18,       # PV Module Efficiency
        # PV Module Coverage on the surface (considering gaps)
        'simple_pv_coverage': 0.8,

        # --- Sandia PV Model (If pv_model_type == 'Sandia') ---
        # 注意: 以下 Sandia 参数是示例值，你需要为你选择的组件填充真实数据
        # 这些参数通常来自 NREL SAM 组件库或制造商数据表
        'sandia_module_params': {
            # 单个组件有效面积 (m2) - Sanyo HIP-200BA3
            'active_area': 1.179,
            # 单个组件串联电池片数量 - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'num_cells_series': 36,
            # 单个组件并联电池片数量 - SAND2004-3535 (B17)
            'num_cells_parallel': 1,
            # 短路电流 (Amps) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'short_circuit_current': 4.80,
            # 开路电压 (Volts) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'open_circuit_voltage': 21.40,
            # 最大功率点电流 (Amps) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'current_at_mpp': 4.10,
            # 最大功率点电压 (Volts) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'voltage_at_mpp': 17.10,
            # 短路电流的温度系数 (1/degC) - SAND2004-3535 (B17)
            'aIsc': 0.0006,
            # 最大功率点电流的温度系数 (1/degC) - SAND2004-3535 (B17)
            'aImp': 0.0001,
            'c0': 0.9604,  # - SAND2004-3535
            'c1': 0.0396,  # - SAND2004-3535
            # 开路电压的温度系数 (Volts/degC) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'BVoc0': -0.080,
            'mBVoc': 0.0,  # - SAND2004-3535 (B19)
            # 最大功率点电压的温度系数 (Volts/degC) - SAND2004-3535 (B19) 示例组件 (ASE-70-ALF)
            'BVmp0': -0.083,
            'mBVmp': 0.0,  # - SAND2004-3535 (B19)
            'diode_factor': 1.217,             # 二极管因子 - SAND2004-3535 (B19)
            'c2': -0.40718,  # - SAND2004-3535 (B19)
            'c3': -14.0746,  # - SAND2004-3535 (B19)
            # IAM 参数 - De Soto (2006)多晶硅推荐值
            'a0': 0.918093, 'a1': 0.086255, 'a2': -0.020356, 'a3': 0.002004, 'a4': -0.000072,
            # IAM 参数 - PVsyst 默认值 / SAND2004-3535 (B17)
            'b0': 1.0, 'b1': -0.002438, 'b2': 0.003103, 'b3': -0.0001246, 'b4': 1.211e-7, 'b5': -1.36e-9,
            # NOCT 相关温度差 (deg C) - SAND2004-3535 安装方式：Open rack
            'delta_tc': 3.0,
            # 漫反射IAM因子 - SAND2004-3535 (B5, B17, S32, S38)
            'fd': 1.0,
            'a': -3.56, 'b': -0.075,          # 温度模型系数 - SAND2004-3535 安装方式：Open rack
            'c4': 0.9789, 'c5': 0.0211,  # - SAND2004-3535 (B19)
            'Ix0': 4.70, 'Ixx0': 4.30,  # - 典型c-Si I-V曲线形状
            'c6': 1.1468, 'c7': -0.1468,  # - SAND2004-3535 (B19)
            # --- 以下为电气连接参数，将由代码根据表面积计算模块数量来确定 ---
            # 'number_of_series_strings_in_parallel': 1, # 这个在 Generator:Photovoltaic 中
            # 'number_of_modules_in_series': 1,         # 这个在 Generator:Photovoltaic 中
            # --- 热传递模式也将在代码中设置 ---
            # 'heat_transfer_integration_mode': "Decoupled" # 或 "IntegratedSurfaceOutsideFace"
        },
        'sandia_pv_coverage': 0.9,  # 每个组件自身的有效面积与总面积比

        # --- PVWatts Model (如果 pv_model_type == 'PVWatts') ---
        # 每平方米屋顶的直流容量 (W/m2) - 用于单一系统估算
        'pvwatts_dc_system_capacity_per_sqm': 144,
        'pvwatts_module_type': 'Standard',
        'pvwatts_array_type': 'FixedRoofMounted',
        'pvwatts_system_losses': 0.14,
        'pvwatts_dc_ac_ratio': 1.1,
        'pvwatts_inverter_efficiency': 0.96,
        'pvwatts_ground_coverage_ratio': 0.4,


        # --- 通用参数 ---
        'pv_inverter_efficiency': 0.96,      # 通用逆变器效率 (用于 Simple 和 Sandia 模型)
        'heat_transfer_integration_mode': "Decoupled",  # PV 传热集成模式

        # --- 阴影/辐射分析参数 (保持不变) ---
        'shadow_calculation_surface_types': ['ROOF', 'WALL'],
        'radiation_threshold_high': 1000.0,
        'radiation_threshold_low': 600.0,
        'radiation_score_threshold': 70,
        'max_score': 100.0,
        'min_score': 0.0,
        'pv_output_prefix': 'pv',
        'shadow_output_prefix': 'shadow',
    },

    # EUI prediction settings
    'eui_prediction': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feature_columns': ["city", "btype", "ssp_code", "shgc", "win_u", "nv_area", "insu", "infl", "cool_cop", "cool_air_temp", "lighting", "vt"],
        'target_column': 'eui',
        'group_by_columns': ["city", "btype", "ssp_code"],
        # train, val, test split ratio
        'train_val_test_split': [0.8, 0.1, 0.1],
        'random_state': 10,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'num_epochs': 500,
        'scale_features': True,
    },

    # Supabase settings
    'supabase': {
        'url': os.getenv('SUPABASE_URL'),  # Supabase URL
        'key': os.getenv('SUPABASE_KEY'),  # Supabase key
        'table': os.getenv('SUPABASE_TABLE'),  # Supabase table name
    }
}

create_directories()
