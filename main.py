import logging
import sys
from backend.services.optimization_service import OptimizationPipeline
from config import CONFIG

logging.basicConfig(level=logging.INFO)

def main():
    target_cities = ['Chicago']
    target_ssps = [126]
    target_btypes = ['Office_Small']

    logging.info("Starting optimization process...")
    logging.info(f"Configuration info: CPU core count = {CONFIG['constants']['cpu_count_override']}")
    logging.info(f"EnergyPlus executable path = '{CONFIG['paths']['eplus_executable']}'")

    for city in target_cities:
        for ssp in target_ssps:
            weather_file = False
            if str(ssp).upper() == "TMY":
                if any(city.lower() in str(f).lower() for f in CONFIG['paths']['tmy_dir'].glob("*.epw")):
                    weather_file = True
            else:
                if any(str(ssp).lower() in str(f).lower() for f in CONFIG['paths']['ftmy_dir'].glob("*.epw")):
                    weather_file = True
            if not weather_file:
                logging.warning(f"Warning: Weather file for city '{city}' SSP '{ssp}' not found; skipping this scenario.")
                continue
            for btype in target_btypes:
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

if __name__ == "__main__":
    main()
