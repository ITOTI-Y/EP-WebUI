import pandas as pd
import json
from pathlib import Path
from config import CONFIG

def data_analysis():
    data_dir = CONFIG['paths']['epsim_dir']
    eui_data_dir = CONFIG['paths']['eui_data_dir']
    json_list = data_dir.rglob("pipeline_results.json")

    method_list = ['OLS', 'RandomForest']

    columns = ['city','btype', 'ssp','building_floor_area_m2', 'baseline_eui',
                            'optimal_eui_simulated', 'optimal_eui_predicted', 'optimization_improvement_percent',
                            'optimization_bias_percent', 'gross_eui_with_pv',
                            'net_eui_with_pv', 'optimization_improvement_with_pv', 
                            ]

    params = ['shgc', 'win_u', 'nv_area', 'insu', 'infl', 'cool_cop', 'cool_air_temp', 'lighting', 'vt']

    df = pd.DataFrame(columns=columns)

    data_list = []

    for json_file in json_list:
        with open(json_file, "r") as f:
            data = json.load(f)
            columns_data = [data[i] for i in columns]
            params_data = [data['optimal_params'][i] for i in params]
            data_list.append([json_file.parents[1].name] + columns_data + params_data)

    all_columns =['method'] + columns + params
    df = pd.DataFrame(data_list, columns=all_columns)

    df.to_csv(eui_data_dir / "data_analysis.csv", index=False)