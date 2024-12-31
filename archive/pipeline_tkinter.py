import os, shutil, time, collections, random, warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import EPPreprocessor, ECMs
from SALib.sample import saltelli
from SALib.analyze import sobol
from pyDOE import lhs
from sklearn.ensemble import RandomForestRegressor
from statsmodels.formula.api import ols
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
from sklearn.linear_model import ElasticNet, BayesianRidge
from deap import base, creator, tools, algorithms
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import json


parent_folder = '\\'.join(os.path.abspath(os.getcwd()).split('\\')[:-1])+"\\"
epsim_folder = parent_folder + 'EPSim\\'
sensitivity_folder = parent_folder + 'Sensitivity\\'
futureLoad_folder = parent_folder + 'FutureLoad\\'
commercials_folder = parent_folder + 'Prototypes\\Commercials\\'
residentials_folder = parent_folder + 'Prototypes\\Residentials\\'
TMY_folder = parent_folder + 'Climate\\TMYs\\'
FTMY_folder = parent_folder + 'Climate\\FTMYs\\'
Ensemble_folder = parent_folder + 'Climate\\Ensemble\\'
city_list = parent_folder + '''Climate\\world's largest metros.csv'''
cpu_count = os.cpu_count()
print ('Total number of threads of this computer:', cpu_count)

def getBuildingInfo(city, ssp, btype, ECM_dict=None):
    if not ECM_dict:
        folder = epsim_folder + city + '\\%s\\' % (ssp)
        f = [folder + f for f in os.listdir(folder) if f.endswith('Table.csv') if btype in f][0]
    else:
        idf_name = IDFname(ECM_dict)
        IDF_folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
        f = [IDF_folder + f for f in os.listdir(IDF_folder) if f.endswith('Table.csv')][0]
    with open(f, 'r') as doc:
        line = doc.readline()
        while line:
            # print (line)
            if "exterior fenestration" in line.lower():
                for i in range(3):
                    line = doc.readline()
                win_U = float(line.split(',')[8])
                shgc = float(line.split(',')[9])
                v_trans = float(line.split(',')[10])
                return win_U, shgc, v_trans
            line = doc.readline()

def copyBase(ep_base_file, new_ep_file):
    new_ep = open(new_ep_file, 'w')
    ep_base = open(ep_base_file, 'r')
        
    line = ep_base.readline()
    while line:
        new_ep.writelines(line)
        line = ep_base.readline()
    ep_base.close()
    new_ep.close()

def copyIDF(city, ssp, btype, if_copy = True):        
    target_folder = epsim_folder+"%s\\%s\\" % (city, ssp) 
    if not os.path.exists(target_folder): 
        try:
            os.makedirs(target_folder)
        except FileExistsError:
            pass
    source_folder = parent_folder + 'Prototypes\\%s\\' % (city)
    source_file = [f for f in os.listdir(source_folder) if btype in f and f.endswith('.idf')][0]
    if if_copy:
        shutil.copy(source_folder+source_file, target_folder)
    return source_folder + source_file, target_folder + source_file

def toECMDict(item):
    "turn the combination into ECM dictionaries"
    dct = collections.OrderedDict()
    dct['shgc'] = item[0]
    dct['win_U'] = item[1]
    dct['nv_area'] = item[2]
    dct['insu'] = item[3]
    dct['infl'] = item[4]
    dct['cool_COP'] = item[5]
    dct['cool_air_temp'] = item[6]
    dct['lighting'] = item[7]
    return dct

def IDFname(ECM_dict):
    para_name = [v for v in ECM_dict.values()]
    name = "_".join(map(str, para_name))
    name = name.replace(', ', '_')
    string = name + ".idf"
    return string

def createIDF(city, ssp, btype, ECM_dict, year=None):
    if not year:
        idf_name = IDFname(ECM_dict)
        folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'            
    else:
        idf_name = "%s_%s_%s_%s.idf" % (city, btype, ssp, year)
        folder = futureLoad_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
    new_IDF = folder+idf_name
    base_IDF = folder + "base.idf"
    copyBase(base_IDF, new_IDF)

    lighting_dict = {'OfficeLarge': {1: 0.2, 2: 0.47, 3: 0.53},
    'OfficeMedium': {1: 0.2, 2: 0.47, 3: 0.53},
    'MF': {1: 0.35, 2: 0.45, 3: 0.55},
    'ApartmentHighRise': {1: 0.35, 2: 0.45, 3: 0.55},
    'SF': {1: 0.45, 2: 0.5, 3: 0.64}}

    win_U, shgc, v_trans = getBuildingInfo(city, ssp, btype)
    if ECM_dict['shgc'] != 0 and ECM_dict['win_U'] != 0:
        ECMs.writeWindowInput(ECM_dict['win_U'], ECM_dict['shgc'], v_trans, new_IDF)
    else:
        if ECM_dict['shgc'] != 0:
            ECMs.writeWindowInput(win_U, ECM_dict['shgc'], v_trans, new_IDF)
        if ECM_dict['win_U'] != 0:
            ECMs.writeWindowInput(ECM_dict['win_U'], shgc, v_trans, new_IDF)
    if ECM_dict['nv_area'] != 0:
        ECMs.writeNVInput(ECM_dict['nv_area'], new_IDF)
    if ECM_dict['insu'] != 0:
        wall_IM_list = [ECM_dict['insu'], 0.7, 0.75, 0.75]
        ECMs.writeWallInsuInput(wall_IM_list, new_IDF)
    if ECM_dict['infl'] != 0:
        ECMs.writeAIInput(ECM_dict['infl'], new_IDF)
    if ECM_dict['cool_COP'] != 0:
        ECMs.writeCoolingCOP(ECM_dict['cool_COP'], new_IDF)
    if ECM_dict['cool_air_temp'] != 0:
        ECMs.writeCoolingAirSupply(ECM_dict['cool_air_temp'], new_IDF)
    if ECM_dict['lighting'] != 0:
        reduct_rate = lighting_dict[btype][ECM_dict['lighting']]
        # print(reduct_rate)
        ECMs.writeLightings(reduct_rate, new_IDF)
    os.remove(folder+"base.idf")
    return new_IDF

def getResult(city, ssp, btype, ECM_dict=None, optimization_rate=None, year=None):
    if not ECM_dict:
        folder = epsim_folder + city + '\\%s\\' % (ssp)
        f = [folder + f for f in os.listdir(folder) if f.endswith('Meter.csv') and btype in f][0]
    else:
        if not year:
            idf_name = IDFname(ECM_dict)
            folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
        else:
            idf_name = "%s_%s_%s_%s.idf" % (city, btype, ssp, year)
            folder = futureLoad_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
        f = [folder + f for f in os.listdir(folder) if f.endswith('Meter.csv')][0]
    df = pd.read_csv(f)
    if not year:
        df['Date/Time'] = '2022/' + df['Date/Time'].astype(str).str.replace(' ', '').replace('/', '')
    else:
        df['Date/Time'] = '%s/'%(year) + df['Date/Time'].astype(str).str.replace(' ', '').replace('/', '')
    df['Date/Time'] = df['Date/Time'].apply(custom_to_datetime)  
    df.set_index('Date/Time', inplace=True)
    df.rename(columns = {'InteriorLights:Electricity [J](Hourly)': 'Lighting (kWh)', 'Cooling:Electricity [J](Hourly)': 'Cooling Electricity (kWh)', 'Heating:Electricity [J](Hourly)': 'Heating Electricity (kWh)', \
        'Electricity:Facility [J](Hourly)': 'Total Electricity (kWh)', 'Heating:EnergyTransfer [J](Hourly)': 'Heating Load (kWh)','Cooling:EnergyTransfer [J](Hourly)': 'Cooling Load (kWh)', \
        'NaturalGas:Facility [J](Hourly)': 'Heating Gas (kWh)', }, inplace = True)
    if btype == 'SF' or btype == 'MF':
        df = df[['Cooling Load (kWh)', 'Cooling Electricity (kWh)',  'Heating Load (kWh)', 'Heating Electricity (kWh)', 'Total Electricity (kWh)']].iloc[:-1]
    else:
        df = df[['Cooling Load (kWh)', 'Cooling Electricity (kWh)',  'Heating Load (kWh)', 'Heating Electricity (kWh)', 'Total Electricity (kWh)', 'Heating Gas (kWh)']].iloc[:-1] 

    if ECM_dict:
        # Remove columns with "Load" in the name
        df = df[[col for col in df.columns if 'Load' not in col]] / 3600000
        df.to_csv(sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + 'optimized_%s_%s_%s_by_%s.csv' % (btype, city, ssp, optimization_rate))
    else:
        if not year:
            df = (df.resample('M').sum() / 3600000)
        else:
            df = df[[col for col in df.columns if 'Load' not in col]] / 3600000
    return df
        
def custom_to_datetime(date):
    # If the time is 24, set it to 0 and increment day by 1
    if date[13:15] == '24':
        return pd.to_datetime(date[:8], format='%Y%m%d%H') + pd.Timedelta(days=1)
    else:
        date = pd.to_datetime(date[:10], infer_datetime_format=True)
        return date
    
def get_sourceEUI(city, ssp, btype, ECM_dict=None, year=None):
    if not ECM_dict:
        folder = epsim_folder + city + '\\%s\\' % (ssp)
        f = [folder + f for f in os.listdir(folder) if f.endswith('Table.csv') if btype in f][0]
    else:
        if not year:
            idf_name = IDFname(ECM_dict)
            IDF_folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
            f = [IDF_folder + f for f in os.listdir(IDF_folder) if f.endswith('Table.csv')][0]
        else:
            idf_name = "%s_%s_%s_%s.idf" % (city, btype, ssp, year)
            IDF_folder = futureLoad_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'
            f = [IDF_folder + f for f in os.listdir(IDF_folder) if f.endswith('Table.csv')][0]
    with open(f, 'r') as doc:
        line = doc.readline()
        while line:
            # print (line)
            if "utility use per total floor area" in line.lower():
                for i in range(6):
                    line = doc.readline()
                EUI = float(line.split(',')[2]) + float(line.split(',')[3])/NG_source_energy_conversion
                return round(EUI, 2)          
            line = doc.readline()

def cal_baseline_city_EUI(city):
    EUI_dict = {}
    for ssp in ['TMY', 126, 245, 370, 585]:
        mean_EUI = 0            
        for btype in btype_list:   
            mean_EUI += get_sourceEUI(city, ssp, btype) * btype_ratio_dict[btype]
        EUI_dict[ssp] = mean_EUI 
    #cnvert the dict to dataframe
    EUI_df = pd.DataFrame.from_dict(EUI_dict, orient = 'index', columns = ['EUI'])
    EUI_df.index.rename('SSP', inplace = True)
    return EUI_df

def runEP(city, ssp, btype, ECM_dict=None, year = None):
    if not ECM_dict:
        idf = copyIDF(city, ssp, btype)[1]
        folder = epsim_folder + city + '\\%s\\' % (ssp) 
        if not os.path.exists(folder): 
            os.makedirs(folder)
        if len([f for f in os.listdir(folder) if f.endswith('Meter.csv') and btype in f]) != 0:
            pass
        else:
            if ssp == 'TMY':
                wea_file = [TMY_folder+f for f in os.listdir(TMY_folder) if f.endswith('.epw') and city in f][0]
            else:
                wea_file = [FTMY_folder+city+'\\'+f for f in os.listdir(FTMY_folder+city+'\\') if f.endswith('.epw') and str(ssp) in f][0]
            preprocess = EPPreprocessor.Preprocessor(idf, wea_file)
            preprocess.AYIDF()

            print("Waiting for EnergyPlus simulation to complete for %s under SSP: %s in the city of %s..." % (btype, ssp, city))
            start_time = time.time() 
            os.chdir(folder)
            os.system("energyplus -w %s -p %s -s C -r %s" % (wea_file, idf.rstrip('.idf'), idf))
            while len([f for f in os.listdir(folder) if f.endswith('Meter.csv') and btype in f]) == 0:
                time.sleep(3)
            time.sleep(3)
            end_time = time.time()        
            print("Total Simulation time:",  str(end_time - start_time)) 
            # Remove files with extensions other .idf and those including "Meter" and "Table" in the file name
            for f in os.listdir(folder):
                if f.endswith('.idf') or ('Meter' in f) or ('Table' in f):
                    pass
                else:
                    os.remove(f)
            os.chdir(parent_folder+'Codes\\')
        return get_sourceEUI(city, ssp, btype)
    else:
        if not year:
            idf_name = IDFname(ECM_dict)
            IDF_folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'        
            if not os.path.exists(IDF_folder):
                os.makedirs(IDF_folder)
            if len([f for f in os.listdir(IDF_folder) if f.endswith('Meter.csv')]) != 0:
                pass
            else:
                if ssp == 'TMY':
                    wea_file = [TMY_folder+f for f in os.listdir(TMY_folder) if f.endswith('.epw') and city in f][0]
                else:
                    wea_file = [FTMY_folder+city+'\\'+f for f in os.listdir(FTMY_folder+city+'\\') if f.endswith('.epw') and str(ssp) in f][0]
                base_IDF = IDF_folder + "base.idf"
                copyBase(copyIDF(city, ssp, btype, if_copy=False)[0], base_IDF)
                preprocess = EPPreprocessor.Preprocessor(base_IDF, wea_file)
                preprocess.AYIDF()    

                print("Waiting for EnergyPlus simulation to complete for %s under SSP: %s in the city of %s..." % (btype, ssp, city))
                start_time = time.time() 
                os.chdir(IDF_folder)
                ECM_idf = createIDF(city, ssp, btype, ECM_dict)
                os.system("energyplus -w %s -p %s -s C -r %s" % (wea_file, ECM_idf.rstrip('.idf'), ECM_idf))
                while len([f for f in os.listdir(IDF_folder) if f.endswith('Meter.csv')]) == 0:
                    time.sleep(3)
                time.sleep(3)
                end_time = time.time()        
                print("Total Simulation time:",  str(end_time - start_time))
                # Remove files with extensions other .idf and those including "Meter" and "Table" in the file name
                for f in os.listdir(IDF_folder):
                    if f.endswith('.idf') or ('Meter' in f) or ('Table' in f):
                        pass
                    else:
                        os.remove(f)
                os.chdir(parent_folder+'Codes\\')
            return get_sourceEUI(city, ssp, btype, ECM_dict) 
        else:
            idf_name = "%s_%s_%s_%s.idf" % (city, btype, ssp, year)
            IDF_folder = futureLoad_folder + city + '\\%s\\' % (ssp) + btype + '\\' + idf_name.rstrip('.idf') + '\\'        
            if not os.path.exists(IDF_folder):
                os.makedirs(IDF_folder)
            if len([f for f in os.listdir(IDF_folder) if f.endswith('Meter.csv')]) != 0:
                pass
            else:
                base_IDF = IDF_folder + "base.idf"
                copyBase(copyIDF(city, ssp, btype, if_copy=False)[0], base_IDF)
                wea_file = Ensemble_folder + city + '\\SSP%s\\%s.epw' % (ssp, year) 
                preprocess = EPPreprocessor.Preprocessor(base_IDF, wea_file)
                preprocess.AYIDF()    

                print("Waiting for EnergyPlus simulation to complete for %s under SSP: %s in the city of %s..." % (btype, ssp, city))
                start_time = time.time() 
                os.chdir(IDF_folder)
                ECM_idf = createIDF(city, ssp, btype, ECM_dict, year)
                os.system("energyplus -w %s -p %s -s C -r %s" % (wea_file, ECM_idf.rstrip('.idf'), ECM_idf))
                while len([f for f in os.listdir(IDF_folder) if f.endswith('Meter.csv')]) == 0:
                    time.sleep(3)
                time.sleep(3)
                end_time = time.time()        
                print("Total Simulation time:",  str(end_time - start_time))
                # Remove files with extensions other .idf and those including "Meter" and "Table" in the file name
                for f in os.listdir(IDF_folder):
                    if f.endswith('.idf') or ('Meter' in f) or ('Table' in f):
                        pass
                    else:
                        os.remove(f)
                os.chdir(parent_folder+'Codes\\')
            return get_sourceEUI(city, ssp, btype, ECM_dict, year) 

def evalObj_discrete(city, ssp, btype, param_df):
    problem = pd.Series(data = [0, 0, 0, 0, 0, 0, 0, 0],
        index = ['shgc', 'win_U', 'nv_area', 'insu', 'infl', 'cool_COP', 'cool_air_temp', 'lighting'])
    problem.update(param_df)
    problem = problem.to_dict()
    EUI = runEP(city, ssp, btype, problem)
    param_df['EUI'] = EUI
    return param_df

def continuousToDiscrete(x, lst):
    space = np.linspace(0, 1, len(lst)+1)
    for i in range(len(space)):
        if x >= space[i] and x < space[i+1]:
            idx = i
    val = lst[idx]
    return 0 if val == 0 else val

def continuousSpace2Discrete(param_values):
    para_dict = Para_dict()
    #convert continuous numpy array param_values to discrete space using continuousToDiscrete function
    for i in range(param_values.shape[0]):
        for j in range(param_values.shape[1]):
            param_values[i][j] = continuousToDiscrete(param_values[i][j], para_dict[list(para_dict.keys())[j]])            
    #remove duplicate rows
    param_values = np.unique(param_values, axis=0)
    return param_values

def refillContinuousSpace(discrete_df):
    para_dict = Para_dict()
    ecm = list(para_dict.keys())
    problem = pd.DataFrame(columns = ecm, data = [[0]*len(ecm)])
    problem = {'num_vars': len(ecm), 'names': problem.columns, 'bounds': [[0,0.999]]* len(ecm)}
    param_values = saltelli.sample(problem, 32)  
    Y = np.zeros([param_values.shape[0]])
    
    for i in range(param_values.shape[0]):
        paras = []
        for j in range(param_values.shape[1]):
            paras.append(continuousToDiscrete(param_values[i][j], para_dict[list(para_dict.keys())[j]]))
        #iterately fillout 'Y' by looking up the discrete_df using paras
        Y[i] = discrete_df.loc[(discrete_df[list(discrete_df.columns)[:-1]] == paras).all(axis=1), 'EUI'].values[0]
    return Y      

def sensitivity(city, ssp, btype):
    work_folder = sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\'
    if not os.path.exists(work_folder):
        os.makedirs(work_folder)

    # # Test    
    # param_df = pd.Series(index = ['shgc', 'win_U', 'nv_area', 'insu', 'infl', 'cool_COP', 'cool_air_temp', 'lighting'],
    #     data = [0.4,2.5,2.4,4.0,1.25,3.5,10.0,3.0])
    # print(evalObj_discrete(city, ssp, btype, param_df)['EUI'])

    ecm = list(Para_dict().keys())
    problem = pd.DataFrame(columns = ecm, data = [[0]*len(ecm)])
    problem = {'num_vars': len(ecm), 'names': problem.columns, 'bounds': [[0,0.999]]* len(ecm)}
    param_values = saltelli.sample(problem, 32)   
    param_values = continuousSpace2Discrete(param_values)

    if not os.path.exists(work_folder + '%s.csv' % ('samples')):
        lst = Parallel(n_jobs = cpu_count-4)(delayed(evalObj_discrete)(city, ssp, btype, pd.Series(data = X, index = ecm, name = i)) for i, X in enumerate(param_values))
        pd.concat(lst, axis = 1).transpose().to_csv(work_folder + '%s.csv' % ('samples'), index=False)
    else:
        print ("The samples have been generated for %s in the city of %s under %s carbon emission scenario!" % (btype, city, ssp))
    df = pd.read_csv(work_folder + '%s.csv' % ('samples'))
    
    Y =  refillContinuousSpace(df)
    Si = sobol.analyze(problem, Y)
    if not os.path.exists(work_folder + '%s.csv' % ('samples_continuous')):
        #create a dataframe concatenating param_values and Y('EUI')
        continuous = saltelli.sample(problem, 32) 
        continuous = np.concatenate((continuous, Y.reshape(-1,1)), axis = 1)
        pd.DataFrame(columns = ecm + ['EUI'], data = continuous).to_csv(work_folder + '%s.csv' % ('samples_continuous'), index=False)
    first_order = pd.Series(data = Si['S1'], index = problem['names'], name = '1st-order sensitivity')
    first_order.to_csv(work_folder + 'sensitivity_1st-order.csv')
    second_order = pd.DataFrame(data = Si['S2'], index = problem['names'], columns = problem['names']).fillna(0).transpose()
    second_order.to_csv(work_folder + 'sensitivity_2nd-order.csv')
    # df_plot([first_order], parent_folder + 'Projects\\'+building+'\\sensitivity\\sensitivity_1st-order', style = 'bar')
    # df_plot([second_order], parent_folder + 'Projects\\'+building+'\\sensitivity\\sensitivity_2nd-order', style = 'heatmap')

def responseSurface(city, ssp, btype, model_type):
    df = pd.read_csv(sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\%s.csv' % ('samples'))
    lhs_df = df.drop('EUI', axis = 1)
    lhs_df['Y'] = df['EUI'].values

    if model_type == 'ols':
        formula = 'Y ~ shgc + win_U + nv_area + insu + infl + cool_COP + cool_air_temp + lighting'
        model = ols(formula, data=lhs_df).fit()        
        with open(sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\model_summary.txt', 'w') as fh:
            fh.write(model.summary().as_text())
            
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=0)
        model.fit(lhs_df.drop('Y', axis=1), lhs_df['Y'])        
        feature_importances = pd.DataFrame(model.feature_importances_,
                                           index = lhs_df.drop('Y', axis=1).columns,
                                           columns=['importance']).sort_values('importance', ascending=False)        
        feature_importances.to_csv(sensitivity_folder + city + '\\%s\\' % (ssp) + btype + '\\rf_feature_importance.txt')

    elif model_type == 'elasticnet':
        model = ElasticNet()
        model.fit(lhs_df.drop('Y', axis=1), lhs_df['Y'])
    elif model_type == 'bayesianridge':
        model = BayesianRidge()
        model.fit(lhs_df.drop('Y', axis=1), lhs_df['Y'])
    else:
        model = BayesianRidge()
        model.fit(lhs_df.drop('Y', axis=1), lhs_df['Y'])

    return model

def optimize(city, ssp, btype, model_type):
    # Find the optimized parameters and Y value based on the response surface model
    DOE_dict = Para_dict()
    model = responseSurface(city, ssp, btype, model_type)
    
    # Define the bounds for each parameter
    bounds = [(min(DOE_dict[key]), max(DOE_dict[key])) for key in DOE_dict.keys()]
    min_bounds, max_bounds = zip(*bounds)  # Extract the min and max bounds separately

    # Initial guess for the parameters
    x0 = np.array([np.mean(DOE_dict[key]) for key in DOE_dict.keys()])

    # Define the function to optimize
    def fun(x):
        # Calculate the response surface model   
        params = pd.DataFrame(dict(zip(DOE_dict.keys(), x)), index = [0])      
        y = model.predict(params)
        return y[0]
    
    if model_type == 'rf':
        # Define the individual and fitness
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create the toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1) # normalize parameters
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(DOE_dict.keys()))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Define the evaluation function to maximize
        def evalModel(individual):
            # Rescale parameters
            params = [min_bounds[i] + (max_bounds[i] - min_bounds[i]) * individual[i] for i in range(len(individual))]
            return fun(params),

        # Register the evaluation and optimization functions
        toolbox.register("evaluate", evalModel)
        toolbox.register("mate", tools.cxTwoPoint)
        # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=1, low=0, up=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create the population and run the GA
        def create_population(n):
            lhd = lhs(len(DOE_dict.keys()), samples=n, criterion='center')
            # Rescale to the parameter bounds
            for i, key in enumerate(DOE_dict.keys()):
                lhd[:, i] = DOE_dict[key][0] + lhd[:, i] * (DOE_dict[key][1] - DOE_dict[key][0])                
            # Convert to individuals and create population
            population = [creator.Individual(l) for l in lhd]
            
            return population

        pop = create_population(100)  
        min_ind = [0]*len(DOE_dict.keys())
        max_ind = [1]*len(DOE_dict.keys())
        pop.extend([creator.Individual(min_ind), creator.Individual(max_ind)])

        # Define statistics to be calculated
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)
        stats.register("min", np.min)
        
        # Evolve population
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=50, 
                                            stats=stats, verbose=True)
        
        # Find the best individual after the GA has run
        best_individual = tools.selBest(pop, k=1)[0]
        best_params = dict(zip(DOE_dict.keys(), [min_bounds[i] + (max_bounds[i] - min_bounds[i]) * best_individual[i] for i in range(len(best_individual))]))
        for key in best_params.keys():
            best_params[key] = round(best_params[key], 2)
        best_y = fun([min_bounds[i] + (max_bounds[i] - min_bounds[i]) * best_individual[i] for i in range(len(best_individual))])
    else:
        res = minimize(fun, x0, bounds=bounds, method='L-BFGS-B')
        best_params = dict(zip(DOE_dict.keys(), res.x))
        for key in best_params.keys():
            best_params[key] = round(best_params[key], 2)
        best_y = res.fun

    # Print the optimized parameters and Y
    print('Optimized parameters:', best_params)
    print('Optimized Y value:', best_y)
    
    # Validate the optimized EUI compared to EUI calculated by EnergyPlus
    simulated_best = runEP(city, ssp, btype, best_params)
    print('Simulated EUI of %s in the city of %s under %s scenario: ' % (btype, city, ssp), simulated_best)
    err = round((abs(best_y - simulated_best)/simulated_best) * 100, 2)
    print('Bias level for the optimized EUI of %s in the city of %s under %s scenario: ' % (btype, city, ssp), err, '%')
    baseline_EUI = runEP(city, ssp, btype)
    print('The EUI of %s in the city of %s under %s scenario has been improved by: ' % (btype, city, ssp), round(((baseline_EUI - simulated_best)/baseline_EUI) * 100, 2), '%')
    print('\n')
    getResult(city, ssp, btype, best_params, round(((baseline_EUI - simulated_best)/baseline_EUI) * 100, 2))
    return best_params

def optimizeAllBuildings(city):
    # Running through all building types
    source_EUI_dict = collections.OrderedDict()
    source_city_EUI_dict = collections.OrderedDict()
    best_params_dict = collections.OrderedDict()
    for ssp in ['TMY', 126, 245, 370, 585]:
        source_EUI_dict[ssp] = collections.OrderedDict()
        best_params_dict[ssp] = collections.OrderedDict()
        optimized_mean_EUI = 0
        for btype in btype_list:
            sensitivity(city, ssp, btype)
            best_params = optimize(city, ssp, btype, model_type = 'ols')
            best_params_dict[ssp][btype] = best_params
            source_EUI_dict[ssp][btype] = runEP(city, ssp, btype, best_params)
            optimized_mean_EUI += source_EUI_dict[ssp][btype] * btype_ratio_dict[btype]
        source_city_EUI_dict[ssp] = optimized_mean_EUI
    copyOptimizedResults(city)
    #convert the best_param_dict dictionary to multi-index dataframe
    best_params_df = pd.DataFrame.from_dict({(i,j): best_params_dict[i][j] 
                           for i in best_params_dict.keys() 
                           for j in best_params_dict[i].keys()},
                       orient='index')
    best_params_df.index.rename(['SSP', 'btype'], inplace = True)
    best_params_df.to_csv(os.path.join(sensitivity_folder, city, 'results', '%s_building_stock_optimized_parameters.csv' % (city)))
    #save the best_params_dict dictionary to a jason file
    with open(os.path.join(sensitivity_folder, city, 'results', '%s_building_stock_optimized_parameters.json' % (city)), 'w') as fp:
        json.dump(best_params_dict, fp)
    optimized_mean_EUI_df = pd.DataFrame.from_dict(source_city_EUI_dict, orient = 'index', columns = ['EUI'])
    optimized_mean_EUI_df.index.rename('SSP', inplace = True)
    baseline_mean_EUI_df = cal_baseline_city_EUI(city)
    #concatenate the two dataframes
    mean_EUI_df = pd.concat([baseline_mean_EUI_df, optimized_mean_EUI_df], axis = 1)
    mean_EUI_df.columns = ['baseline_EUI', 'optimized_EUI']
    mean_EUI_df.to_csv(os.path.join(sensitivity_folder, city, 'results', '%s_building_stock_mean_source_EUI.csv' % (city)))
    
    source_EUI_df = pd.DataFrame.from_dict(source_EUI_dict)
    source_EUI_df.to_csv(os.path.join(sensitivity_folder, city, 'results', '%s_building_source_EUI.csv' % (city)))

def copyOptimizedResults(city):
    #create a "results" folder under the city folder of sensitivity folder and create a "ssp" folder under the "results" folder
    if not os.path.exists(os.path.join(sensitivity_folder, city, 'results')):
        os.makedirs(os.path.join(sensitivity_folder, city, 'results'))
    for ssp in ['TMY', 126, 245, 370, 585]:
        if not os.path.exists(os.path.join(sensitivity_folder, city, 'results', str(ssp))):
            os.makedirs(os.path.join(sensitivity_folder, city, 'results', str(ssp)))

        #copy all the files under each ssp folder of each btype to the created results folder
        for btype in btype_list:
            for file in os.listdir(os.path.join(sensitivity_folder, city, str(ssp), btype)):
                if file.endswith('.csv') or file.endswith('.txt'):
                    #copy and rename the file by putting 'btype_city_ssp_' in front of the original file name except for the file containing 'optimized' in its file name
                    if 'optimized' not in file:
                        shutil.copy(os.path.join(sensitivity_folder, city, str(ssp), btype, file), os.path.join(sensitivity_folder, city, 'results', str(ssp), btype + '_' + city + '_' + str(ssp) + '_' + file))
                    else:
                        shutil.copy(os.path.join(sensitivity_folder, city, str(ssp), btype, file), os.path.join(sensitivity_folder, city, 'results', file))

def runBest(city):
    # read the best_params dictionary from the jason file
    with open(os.path.join(sensitivity_folder, city, 'results', '%s_building_stock_optimized_parameters.json' % (city)), 'r') as fp:
        best_params_dict = json.load(fp)
    Parallel(n_jobs = cpu_count - 4, verbose = 10)(delayed(runEP)(city, ssp, btype, best_params_dict[str(ssp)][btype], year) for ssp in [126, 245, 370, 585] for btype in btype_list for year in range(2040, 2070))
    
    if not os.path.exists(os.path.join(futureLoad_folder, city, 'results')):
        os.makedirs(os.path.join(futureLoad_folder, city, 'results'))
    EUI_dict = collections.OrderedDict()
    for ssp in [126, 245, 370, 585]:
        EUI_dict[ssp] = collections.OrderedDict()
        for btype in btype_list:
            results_dict = collections.OrderedDict()
            EUI_dict[ssp][btype] = collections.OrderedDict()
            EUI_list = []
            for year in range(2040, 2070):
                results_dict[year] = getResult(city, ssp, btype, best_params_dict[str(ssp)][btype], None, year)
                EUI_list.append(get_sourceEUI(city, ssp, btype, best_params_dict[str(ssp)][btype], year))
            # turn the EUI_list into one dataframe with the year as the index and the EUI as the column name
            EUI_df = pd.DataFrame(data = EUI_list, index = range(2040, 2070))
            EUI_df.columns = ['EUI']
            EUI_dict[ssp][btype] = EUI_df

            # concatenate all the dfs in the results_dict dictionary into one dataframe
            results_df = pd.concat(results_dict, axis = 0)
            results_df.to_csv(os.path.join(futureLoad_folder, city, 'results', '%s_%s_%s_EnergyDemand2040to2070.csv' % (city, btype, ssp)))
    
    for btype in btype_list:
        # plot EUIs in all the year of one building type under different ssps in one figure
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette(sns.color_palette("Set2", 4))
        fig, ax = plt.subplots(figsize=(8, 6))
        for ssp in [126, 245, 370, 585]:
            ax.plot(EUI_dict[ssp][btype].index, EUI_dict[ssp][btype]['EUI'], label = 'SSP%s' % (ssp))
        ax.set_xlabel('Year')
        ax.set_ylabel('kWh/m2')
        ax.set_title('Future annual EUI of retrofitted %s in the city of %s' % (btype, city))
        # save the figure
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(futureLoad_folder, city, 'results', '%s_%s_EUI.png' % (city, btype)), dpi=300, bbox_inches='tight')

            

def analysis(city):
    results_folder = os.path.join(sensitivity_folder, city, 'results')
    optimization_dict = collections.OrderedDict()
    optimized_files = [f for f in os.listdir(os.path.join(results_folder)) if f.endswith('.csv') and 'optimized' in f]
    for ssp in ['TMY', 126, 245, 370, 585]:
        optimization_dict[ssp] = collections.OrderedDict()
        for btype in btype_list:
            for f in optimized_files:
                if "optimized_%s_%s_%s" % (btype, city, ssp) in f and 'none' not in f.lower():
                    optimization_dict[ssp][btype] = float(f.rstrip('.csv').split('_')[-1])
    optimization_df = pd.DataFrame.from_dict(optimization_dict)
    optimization_df.to_csv(os.path.join(results_folder, 'ECM_optimized_EUI_by_percentage.csv'))

    #create a dictionary containing all the coefficients of the variables in the model for each ssp and btype
    coef_dict = collections.OrderedDict()
    #read the model_summary.txt file in each ssp folder of each btype under the results folder
    for ssp in ['TMY', 126, 245, 370, 585]:
        coef_dict[ssp] = collections.OrderedDict()
        for btype in btype_list:
            coef_dict[ssp][btype] = collections.OrderedDict()
            with open(os.path.join(results_folder, str(ssp), btype + '_' + city + '_' + str(ssp) + '_model_summary.txt'), 'r') as f:
                #get the coefficient of all the variables in the model and their variable names as the key, store them in the coef_dict[ssp][btype] dictionary
                for line in f:
                    for var in ['Intercept', 'shgc', 'win_U', 'nv_area', 'insu', 'infl', 'cool_COP', 'cool_air_temp', 'lighting']:
                        if var in line:
                            coef_dict[ssp][btype][var] = float(line.split()[1])

    ssps = ['TMY', 126, 245, 370, 585]
    features = ['Intercept', 'shgc', 'win_U', 'nv_area', 'insu', 'infl', 'cool_COP', 'cool_air_temp', 'lighting']
    multi_index = pd.MultiIndex.from_product([btype_list, features, ssps], names=['Building Type', 'Features', 'SSP'])
    coef_df = pd.DataFrame(index=multi_index, columns=[])
    for btype in btype_list:        
        for feature in features:
            for ssp in ssps:
                coef_df.loc[(btype, feature, ssp), 'Coefficient'] = coef_dict[ssp][btype][feature]
    coef_df.to_csv(os.path.join(results_folder,'ecm_coef.csv'))
    coef_df = coef_df.reset_index()

    # Create a pivot table from the DataFrame
    coef_df = coef_df.pivot_table(index=['Building Type', 'SSP'], columns='Features', values='Coefficient')

    # Create a sns heatmap with sorted building types and ssps on the y-axis
    coef_df = coef_df.reindex(index=btype_list, level=0)
    coef_df = coef_df.reindex(index=['TMY', 126, 245, 370, 585], level=1)
    coef_df = coef_df.reindex(columns=['Intercept', 'shgc', 'win_U', 'nv_area', 'insu', 'infl', 'cool_COP', 'cool_air_temp', 'lighting'])
    sns.heatmap(coef_df, cmap='RdBu_r', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f', annot_kws={"size": 7})
    # Adjust x and y axis label size
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    # Adjust color bar legend size
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=7)
    # Adjust x and y title size
    plt.xlabel('Measures', fontsize=8)
    plt.ylabel('Building Type and Climate Scenario', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'ecm_coefficients_heatmap.png'), dpi=300, bbox_inches='tight')    

def Para_dict():
    # ECM parameters
    DOE_dict = collections.OrderedDict()
    DOE_dict['shgc'] = [0.2, 0.4, 0.6, 0.8]
    DOE_dict['win_U'] = [0.5, 1, 1.5, 2, 2.5, 3] #W/m2K
    DOE_dict['nv_area'] = [0, 1, 2, 3, 4] #m2
    DOE_dict['insu'] = [1, 2, 3, 4] #R-value in SI: m2K/W
    DOE_dict['infl'] = [0.25, 0.5, 0.75, 1, 1.25, 1.5] #ACH: h-1
    DOE_dict['cool_COP'] = [3.5, 4, 4.5, 5, 5.5, 6] 
    DOE_dict['cool_air_temp'] = [10, 12, 14, 16] #Celcius
    DOE_dict['lighting'] = [1, 2, 3] #Proportion, ratio
    return DOE_dict

def main():
    warnings.filterwarnings("ignore")    
    global NG_source_energy_conversion, btype_ratio_dict, btype_list

    # default values
    NG_source_energy_conversion = 3.2
    btype_ratio_dict = {'OfficeLarge': 0.000663, 'OfficeMedium': 0.001725, 'ApartmentHighRise': 0.000384, 'SF': 0.608149, 'MF': 0.389079}
    btype_list = []
    city = 'Chicago'

    # create main window
    root = tk.Tk()
    root.title("Building Energy Retrofit Analysis for Megacities")

    # create entry for city
    city_label = tk.Label(root, text="City")
    city_label.pack()
    city_entry = tk.Entry(root, justify='center')
    city_entry.insert(0, city)  # set default value
    city_entry.pack()

    # create entry for NG_source_energy_conversion
    ng_label = tk.Label(root, text="NG_source_energy_conversion")
    ng_label.pack()
    ng_entry = tk.Entry(root, justify='center')
    ng_entry.insert(0, NG_source_energy_conversion)  # set default value
    ng_entry.pack()

    # create entries for btype_ratio_dict
    btype_entries = {}
    for btype, ratio in btype_ratio_dict.items():
        btype_label = tk.Label(root, text=btype)
        btype_label.pack()
        btype_entry = tk.Entry(root, justify='center')
        btype_entry.insert(0, ratio)  # set default value
        btype_entry.pack()
        btype_entries[btype] = btype_entry

    # create console output
    console_output = tk.Text(root, state='disabled', height=10, width=50)
    console_output.pack()

    def submit():
        # retrieve the input values
        NG_source_energy_conversion = ng_entry.get()
        city = city_entry.get()
        btype_ratio_dict = {btype: btype_entry.get() for btype, btype_entry in btype_entries.items()}
        for btype, ratio in btype_ratio_dict.items():
            if ratio != 0:
                btype_list.append(btype)

        # output to console
        console_output.configure(state='normal')
        console_output.insert('end', 'Submitted\n')
        console_output.insert('end', f'NG_source_energy_conversion: {NG_source_energy_conversion}\n')
        console_output.insert('end', f'City: {city}\n')
        for btype, ratio in btype_ratio_dict.items():
            console_output.insert('end', f'{btype}: {ratio}\n')
        console_output.configure(state='disabled')

    def run():
        submit()  # submit the values first

        cpu_count = os.cpu_count()  # assuming you have imported os
        city = city_entry.get()  # retrieve the input value for city

        # output to console
        console_output.configure(state='normal')
        console_output.insert('end', 'Running runEP...\n')
        console_output.configure(state='disabled')

        Parallel(n_jobs=cpu_count - 4, verbose=10)(delayed(runEP)(city, ssp, btype) for ssp in ['TMY', 126, 245, 370, 585] for btype in btype_list)

        # output to console
        console_output.configure(state='normal')
        console_output.insert('end', 'Running optimizeAllBuildings...\n')
        console_output.configure(state='disabled')

        optimizeAllBuildings(city)

        # output to console
        console_output.configure(state='normal')
        console_output.insert('end', 'Running analysis...\n')
        console_output.configure(state='disabled')

        analysis(city)
        runBest(city)

    # create run button
    run_button = tk.Button(root, text="Run", command=run)
    run_button.pack()

    # run main loop
    root.mainloop()

# def main():
#     warnings.filterwarnings("ignore")    
#     global NG_source_energy_conversion, btype_ratio_dict, btype_list
#     btype_list = []
#     NG_source_energy_conversion = 3.2
#     btype_ratio_dict = {'OfficeLarge': 0.000663, 'OfficeMedium': 0.001725, 'ApartmentHighRise': 0.000384, 'SF': 0.608149, 'MF': 0.389079}
#     for btype, ratio in btype_ratio_dict.items():
#         if ratio != 0:
#             btype_list.append(btype)

#     # #For testing
#     # city = 'Chicago'
#     # sensitivity(city, '126', 'OfficeLarge')

#     #For running
#     city = 'Chicago'
#     Parallel(n_jobs = cpu_count - 4, verbose = 10)(delayed(runEP)(city, ssp, btype) for ssp in ['TMY', 126, 245, 370, 585] for btype in btype_list)
#     optimizeAllBuildings(city)   
#     analysis(city)
#     runBest(city)
          

if __name__ == '__main__':
    main()

