import logging
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
            for btype in target_btypes:
                try:
                    pipeline = OptimizationPipeline(city, ssp, btype, CONFIG)
                    pipeline.run_baseline_simulation()
                except Exception as e:
                    logging.error(f"Error occurred for {city}, SSP {ssp}, Btype {btype}: {e}")

    logging.info("Optimization process completed successfully.")

if __name__ == "__main__":
    main()
