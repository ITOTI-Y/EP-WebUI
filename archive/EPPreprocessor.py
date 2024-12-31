import itertools, os, fnmatch, csv, pickle, collections
from tempfile import mkstemp
import numpy as np
from shutil import move

class Preprocessor():
    def __init__(self, base_IDF, wea_file):
        self.base_IDF = base_IDF
        self.wea_file = wea_file
        
    def outputStr(self):
        string = """

OutputControl:Table:Style,
    Comma,             !- Column Separator
    JtoKWH;                   !- Unit Conversion
    
Output:Table:SummaryReports,
    AllSummaryAndMonthly;    !- Report 1 Name

Output:Meter:MeterFileOnly,
    InteriorLights:Electricity,  !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    InteriorEquipment:Electricity,  !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Refrigeration:Electricity,  !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Fans:Electricity,        !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Cooling:Electricity,     !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Pumps:Electricity,       !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Heating:Electricity,     !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Heating:NaturalGas,      !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Water Heater:WaterSystems:NaturalGas,  !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Water:Facility,          !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Electricity:Facility,    !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    NaturalGas:Facility,     !- Name
    hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Heating:EnergyTransfer,  !- Name
    Hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Cooling:EnergyTransfer,  !- Name
    Hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    EnergyTransfer:Building, !- Name
    Hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    EnergyTransfer:HVAC,     !- Name
    Hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Cooling:Electricity,     !- Name
    Hourly;                  !- Reporting Frequency

Output:Meter:MeterFileOnly,
    Heating:NaturalGas,      !- Name
    Hourly;                  !- Reporting Frequency\n\n"""            

        return string

    def runPeriodStr(self, name, str_list):
        string = 'RunPeriod,\n'+'    '+name+",                  !- Name\n"+"    "+\
                 str(str_list[0])+",                       !- Begin Month\n"+"    "+\
                 str(str_list[1])+",                       !- Begin Day of Month\n"+"    "+\
                 str(str_list[2])+",                      !- End Month\n"+"    "+\
                 str(str_list[3])+",                      !- End Day of Month\n" + "    " + '''UseWeatherFile,          !- Day of Week for Start Day
    Yes,                     !- Use Weather File Holidays and Special Days
    Yes,                     !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes,                     !- Use Weather File Snow Indicators
    1;                       !- Number of Times Runperiod to be Repeated\n\n'''
        return string

    def AYIDF(self):
        fh, abs_path = mkstemp()
        new_ep = open(abs_path, 'w')
        ep_base = open(self.base_IDF, 'r')

        line = ep_base.readline()
        while line:
            # print (line)
            if line.lower() == 'simulationcontrol,\n':
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines("No," + line.split(',')[1])
                line = ep_base.readline()
                new_ep.writelines("Yes;" + line.split(';')[1])
                line = ep_base.readline()
            
            while "groundheattransfer:" in line.lower():
                line = ep_base.readline()
                while '!' in line or line == '\n':
                    line = ep_base.readline()

            while 'output:' in line.lower() and "!" not in line:
                line = ep_base.readline()
                while line != '\n':
                    line = ep_base.readline()
            while 'outputcontrol:' in line.lower() and "!" not in line:
                line = ep_base.readline()
                while line != '\n':
                    line = ep_base.readline()        
            new_ep.writelines(line)
            line = ep_base.readline()

        new_ep.writelines(self.outputStr()) 
        ep_base.close()
        new_ep.close()
        os.close(fh)
        os.remove(self.base_IDF)
        move(abs_path, self.base_IDF)

    def plotAnnualTempHist(self):
        wea = Helper.getWeather(self.wea_file)
        temp = wea[0]
        Helper.plotHist(temp)

    def plotMonthlyTempHist(self):
        temp = Helper.getWeather(self.wea_file)[0]
        for i in range(12):
            Helper.plotHist(temp[i*730: (i+1)*730])

    def findColdestDay(self):
        'find special days for simulaiton'
        wea = Helper.getWeather(self.wea_file)
        month = wea[-3]
        day = wea[-2]
        hour = wea[-1]
        temp = wea[0]
        
        temp_ind = np.argmin(temp)
        for i in range(24):
            if hour[temp_ind-i] == 1:
                start_idx = temp_ind - i
                break
        end_idx = start_idx + 47
        start_month = month[start_idx]
        end_month = month[end_idx]
        start_day = day[start_idx]
        end_day = day[end_idx]
        return start_month,start_day, end_month, end_day

    def findHottestDay(self):
        'find special days for simulaiton'
        wea = Helper.getWeather(self.wea_file)
        month = wea[-3]
        day = wea[-2]
        hour = wea[-1]
        temp = wea[0]
        
        temp_ind = np.argmax(temp)
        for i in range(24):
            if hour[temp_ind-i] == 1:
                start_idx = temp_ind - i
                break
        end_idx = start_idx + 47
        start_month = month[start_idx]
        end_month = month[end_idx]
        start_day = day[start_idx]
        end_day = day[end_idx]
        return start_month,start_day, end_month, end_day

    def findSwingDay(self):
        'find special days for simulaiton'
        wea = Helper.getWeather(self.wea_file)
        month = wea[-3]
        day = wea[-2]
        hour = wea[-1]
        temp = wea[0]

        for i in range(len(temp)):
            if temp[i] >= np.median(temp):
                temp_ind = i
                break
        for i in range(24):
            if hour[temp_ind-i] == 1:
                start_idx = temp_ind - i
                break
        end_idx = start_idx + 47
        start_month = month[start_idx]
        end_month = month[end_idx]
        start_day = day[start_idx]
        end_day = day[end_idx]
        return start_month,start_day, end_month, end_day
    

                
                
