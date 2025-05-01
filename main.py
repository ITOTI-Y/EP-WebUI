import logging
import sys
import argparse
import asyncio
from tqdm import tqdm
from backend.services.optimization_service import OptimizationPipeline
from backend.services.azure_service import AzureDockerService
from backend.services.eui_data_pipeline import EUIDataPipeline
from backend.services.eui_prediction_service import EUIPredictionService
from config import CONFIG
from backend.services.supabase_services import SensitivityDataUploader

logging.basicConfig(level=logging.WARNING,
                    filename=CONFIG['paths']['log_dir'] / "optimization.log",
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

def run_optimization(target_cities, target_ssps, target_btypes):
    """Run the standard optimization pipeline."""
    logging.info("Starting optimization process...")
    logging.info(f"Configuration info: CPU core count = {CONFIG['constants']['cpu_count_override']}")
    logging.info(f"EnergyPlus executable path = '{CONFIG['paths']['eplus_executable']}'")

    for city in tqdm(target_cities, desc="Cities"):
        for ssp in tqdm(target_ssps, desc="SSPs"):
            weather_file = False
            if str(ssp).upper() == "TMY":
                if any(city.lower() in str(f).lower() for f in CONFIG['paths']['tmy_dir'].glob("*.epw")):
                    weather_file = True
            else:
                if any(str(ssp).lower() in str(f).lower() for f in CONFIG['paths']['ftmy_dir'].glob("*.epw")): # Need to add a city check
                    weather_file = True
            if not weather_file:
                logging.warning(f"Warning: Weather file for city '{city}' SSP '{ssp}' not found; skipping this scenario.")
                continue
            for btype in tqdm(target_btypes, desc="Building Types"):
                proto_path_check = False
                if any(btype.lower() in str(f).lower() for f in CONFIG['paths']['prototypes_dir'].glob("*.idf")):
                    proto_path_check = True
                if not proto_path_check:
                    logging.warning(f"Warning: Prototype file for city '{city}' SSP '{ssp}' and btype '{btype}' not found; skipping this scenario.")
                    continue
                try:
                    pipeline = OptimizationPipeline(city, ssp, btype, CONFIG)
                    pipeline.run_full_pipeline(
                        run_sens=True,
                        build_model=True,
                        run_opt=True,
                        validate=True,
                        save=True
                    )
                except FileNotFoundError as e:
                    logging.error(f"Error: Initialization failed for {city}/{ssp}/{btype} - {e}")
                    sys.exit(1)
                except Exception as e:
                    logging.error(f"Error occurred for {city}/{ssp}/{btype} - {e}")

    logging.info("Optimization process completed successfully.")

def collect_eui_data(target_cities, target_ssps, target_btypes):
    """Collect EUI data for training the prediction model."""
    if not CONFIG['eui_prediction']['enabled']:
        logging.error("EUI prediction is disabled in config.")
        return
        
    logging.info("Starting EUI data collection...")
    
    data_pipeline = EUIDataPipeline(CONFIG)
    
    try:
        training_data = data_pipeline.prepare_training_data(target_cities, target_ssps, target_btypes)
        logging.info(f"Collected {len(training_data)} training samples.")
    except Exception as e:
        logging.error(f"Error occurred during EUI data collection - {e}")
        
    logging.info("EUI data collection completed successfully.")

def train_eui_model():
    """Train the EUI prediction model."""
    if not CONFIG['eui_prediction']['enabled']:
        logging.error("EUI prediction is disabled in config.")
        return
        
    logging.info("Starting EUI model training...")
    
    prediction_service = EUIPredictionService(CONFIG)
    data_pipeline = EUIDataPipeline(CONFIG)
    
    try:
        training_data = data_pipeline.load_training_data()
        prediction_service.train_model(
            training_data,
            epochs=CONFIG['eui_prediction']['epochs'],
            batch_size=CONFIG['eui_prediction']['batch_size']
        )
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred during EUI model training - {e}")
        
    logging.info("EUI model training completed successfully.")

def predict_eui(city, ssp, btype):
    """Predict EUI for a specific building type."""
    if not CONFIG['eui_prediction']['enabled']:
        logging.error("EUI prediction is disabled in config.")
        return
        
    logging.info(f"Predicting EUI for {city}/{ssp}/{btype}...")
    
    prediction_service = EUIPredictionService(CONFIG)
    
    try:
        building_data = {
            'building_type': btype,
            'zones': [],
            'surfaces': [],
            'equipment': [],
            'zone_connections': [],
            'surface_zone_map': [],
            'equipment_zone_map': []
        }
        
        eui = prediction_service.predict_eui(building_data)
        logging.info(f"Predicted EUI for {city}/{ssp}/{btype}: {eui}")
        return eui
    except Exception as e:
        logging.error(f"Error occurred during EUI prediction - {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='EP-WebUI: Energy Performance Web User Interface')
    parser.add_argument('--mode', type=str, default='collect',
                        choices=['optimize', 'collect', 'train', 'predict'],
                        help='Operation mode: optimize, azure, collect, train, or predict')
    parser.add_argument('--cities', type=str, nargs='+', default=['Chicago'],
                        help='Target cities')
    parser.add_argument('--ssps', type=str, nargs='+', default=['126', '245', '370', '434', '585'],
                        help='Target SSP scenarios')
    parser.add_argument('--btypes', type=str, nargs='+', default=['OfficeMedium', 'OfficeLarge','ApartmentHighRise','SingleFamilyResidential','MultiFamilyResidential'],
                        help='Target building types')
    parser.add_argument('--city', type=str, help='City for prediction')
    parser.add_argument('--ssp', type=str, help='SSP scenario for prediction')
    parser.add_argument('--btype', type=str, help='Building type for prediction')
    
    args = parser.parse_args()
    
    target_ssps = []
    for ssp in args.ssps:
        try:
            target_ssps.append(int(ssp))
        except ValueError:
            target_ssps.append(ssp)
    
    if args.mode == 'optimize':
        run_optimization(args.cities, target_ssps, args.btypes)
    elif args.mode == 'collect':
        collect_eui_data(args.cities, target_ssps, args.btypes)
    elif args.mode == 'train':
        train_eui_model()
    elif args.mode == 'predict':
        if not all([args.city, args.ssp, args.btype]):
            logging.error("City, SSP, and building type must be specified for prediction mode.")
            sys.exit(1)
        predict_eui(args.city, args.ssp, args.btype)

if __name__ == "__main__":
    main()