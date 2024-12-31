from shutil import move
from os import close, remove
from tempfile import mkstemp

def getalllines(new_ep_path):   
    with open(new_ep_path, 'r') as reader:  
            all = reader.readlines()
    return all
#-------------------------------------------------------------------------------Wall Insulation
def wallInsuMaterial(IM_list):
    string = "Material:NoMass, SINGLEPANEOPAQUE, VerySmooth, "+ str(IM_list).strip("[]")+ "; \n\n"
    return string

def wallMovableInsu(counter, wall_list):
    string = "SurfaceControl:MovableInsulation, Outside, " + wall_list[counter] + ", SINGLEPANEOPAQUE, wall_insulation;"
    return string

def wallInsuSched():
    string = """Schedule:Compact,
    wall_insulation,                      !- Name
    FRACTION,                !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: Alldays,            !- Field 2
    Until: 24:00,1.00;       !- Field 3\n\n"""
    return string

def writeWallInsuInput(IM_list, new_ep_path):
    wall_list = []
    text = getalllines(new_ep_path)
    for i in range(0, len(text)):
        if "buildingsurface:detailed," in text[i].lower():
            if 'outdoors,' in text[i+6].lower(): 
                if 'wall,' in text[i+2].lower() or 'roof,' in text[i+2].lower():
                    wall_list.append(text[i+1].split(",")[0].strip())
    # print (wall_list)
    
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')
    new_ep.writelines(text)

    insu_material = wallInsuMaterial(IM_list)
    sched = wallInsuSched()
    new_ep.writelines('\n'+sched)
    new_ep.writelines('\n'+insu_material)
    for i in range(0, len(wall_list)):
        wall_insu = wallMovableInsu(i, wall_list)
        new_ep.writelines(wall_insu)
        new_ep.writelines("\n")
        
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)

#-------------------------------------------------------------------------------Air Infiltration
def AISchedStr():
    string = """Schedule:Compact,
    Always_On_AI,     !- Name
    On/Off,                  !- Schedule Type Limits Name
    Through: 12/31,          !- Field 1
    For: AllDays,            !- Field 2
    Until: 24:00,            !- Field 3
    1;                       !- Field 4 \n\n"""
    return string
    

def airInflStr(counter, zone, infl_rate):
    string = "ZoneInfiltration:DesignFlowRate, \n" + "    AirInfiltration_" + str(counter) + ",            !- Name \n" + \
             "    " + zone + "        !- Zone Name \n" + "    " + \
    """Always_On_AI,     !- Schedule Name
    AirChanges/Hour,        !- Design Flow Rate Calculation Method
    ,                        !- Design Flow Rate {m3/s}
    ,                        !- Flow per Zone Floor Area {m3/s-m2}
    ,                        !- Flow per Exterior Surface Area {m3/s-m2} \n""" \
    + "    " +str(infl_rate)+";                     !- Air Changes per Hour {1/hr}\n"""
    return string

def writeAIInput(infl_rate, new_ep_path):
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')
    
    zone_list = []
    with open(new_ep_path, 'r') as reader:
        line = reader.readline()
        while line:
            if "zoneinfiltration:" in line.lower() and '=' not in line:
                line = reader.readline()
                line = reader.readline()
                zoneline_string = line.strip('!-').split(',')[0].strip()
                zone_list.append(zoneline_string+',')
            line = reader.readline()
    # print (zone_list)

    ep_base = open(new_ep_path, 'r')
    line = ep_base.readline()
    while line:
        while "zoneinfiltration:" in line.lower():
            line = ep_base.readline()
            while "    " in line or line == '\n':
                line = ep_base.readline()
        while "zoneventilation:" in line.lower():
            line = ep_base.readline()
            while "    " in line or line == '\n':
                line = ep_base.readline()
        while "airflownetwork:" in line.lower():
            line = ep_base.readline()
            while "    " in line or line == '\n':
                line = ep_base.readline()
        while "schedule:compact," in line.lower():
            line = ep_base.readline()
            if "always_on_ai," in line.lower():
                while "    " in line or ";" in line:
                    line = ep_base.readline()
            else:
                new_ep.writelines("Schedule:Compact,\n")
                new_ep.writelines(line)
                line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()                 
        
        
    AISched = AISchedStr()
    new_ep.writelines(AISched)
    for i in range(0, len(zone_list)):
        airInfl = airInflStr(i + 1, zone_list[i], infl_rate)
        new_ep.writelines(airInfl)
        new_ep.writelines("\n\n")
    new_ep.writelines(line)
    line = ep_base.readline()
        
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)
    
#-------------------------------------------------------------------------------Window
def windowConstruction(construction, window_name):
    WC_string = "Construction, \n  ", window_name, ", ", construction, "; \n\n"
    return WC_string

def writeWindowInput(U, SHGC, v_trans, new_ep_path):
    construction_w_shade = None
    with open(new_ep_path, 'r') as reader:  
        all = reader.readlines()
        for i in range(0, len(all)):
            if "windowshadingcontrol," in all[i].lower():
                construction_w_shade = all[i+5].split(',')[0]
                break
        # print ('found!', construction_w_shade)

    window_name = "new_window"
    glazing_name = "new_glazing"
    WM_string = "WindowMaterial:SimpleGlazingSystem, " + glazing_name + ", " + str(U) + ", " + str(SHGC) + ", %s;" % (v_trans) 
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')
    line = ep_base.readline()
    while line:
        line_words_list = line.split()
        for words in line_words_list:
            if construction_w_shade:
                if 'construction' in words.lower():
                    new_ep.writelines(line)
                    line = ep_base.readline()
                    if construction_w_shade in line:
                        new_ep.writelines(line)
                        new_ep.writelines("    " + glazing_name + ",  !- Outside Layer \n")   
                        line = ep_base.readline()
                        line = ep_base.readline()
                    else:
                        new_ep.writelines(line)
                        line = ep_base.readline()
            if "FenestrationSurface:" in words:
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines(line)
                line = ep_base.readline()
                if "window," in line.lower():
                    new_ep.writelines(line)
                    new_ep.writelines("    " + window_name + ",  !- Construction Name \n")
                    line = ep_base.readline()
                    line = ep_base.readline()
                else:
                    new_ep.writelines(line)
                    line = ep_base.readline()
            if line.strip(' ') == "Window,\n":
                new_ep.writelines(line)
                line = ep_base.readline()
                new_ep.writelines(line)
                new_ep.writelines("    " + window_name + ",  !- Construction Name \n")
                line = ep_base.readline()
                line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()

    new_ep.writelines('\n'+ WM_string+'\n\n')
    new_ep.writelines(windowConstruction(glazing_name, window_name))
        
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)
    
    
#-------------------------------------------------------------------------------Cooling Efficiency
def writeCoolingCOP(COP, new_ep_path):
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')

##    if ifchiller == 0: 
    line = ep_base.readline()           
    while line:        
        line_words_list = line.split(',')
        if line_words_list:
            #print line_words_list
            if "Cooling COP" in line_words_list[-1]:
                line_words_list[0] = '  '+str(COP)
                line = ', '.join(line_words_list)
##                print line
                new_ep.writelines(line)
                line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)
##    else:
##        line = ep_base.readline() 
##        while line:        
##            line_words_list = line.split(',')
##            if line_words_list:      
##                if "Cooling COP" in line_words_list[-1]:
##                    line_words_list[0] = '  '+str(COP)
##                    line = ', '.join(line_words_list)
####                    print line
##                    new_ep.writelines(line)
##                    line = ep_base.readline()  
##            new_ep.writelines(line)
##            line = ep_base.readline()        
##        ep_base.close()
##        new_ep.close()
##        close(fh)
##        remove(new_ep_path)
##        move(abs_path, new_ep_path)
    
#-------------------------------------------------------------------------------Cooling Supply Air Temperature
def writeCoolingAirSupply(Temp, new_ep_path):
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')
    
    line = ep_base.readline()           
    while line:        
        line_words_list = line.split(',')
        if line_words_list:
            #print line_words_list
            if 'Zone Cooling Design Supply Air Temperature {C}' in line_words_list[-1]:
                line_words_list[0] = '  '+str(Temp)
                line = ', '.join(line_words_list)
##                print line
                new_ep.writelines(line)
                line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)

#-------------------------------------------------------------------------------Efficient lighting
def writeLightings(percent, new_ep_path):
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')

    line = ep_base.readline()
    while line:
        if line.lower().strip(' ') == "lights,\n":
            for i in range(5):
                new_ep.writelines(line)
                line = ep_base.readline()
            if line.split(',')[0] != "    ":
                print (line)
                string = "    "+str(float(line.split(',')[0])*percent)+",                        !- Lighting Level W\n" 
                new_ep.writelines(string)
                line = ep_base.readline()
            else:                
                new_ep.writelines(line)
                line = ep_base.readline()            
                if line.split(',')[0]!= "    ":
                    string = "    "+str(float(line.split(',')[0])*percent)+",                  !- Watts per Zone Floor Area {W/m2}\n" 
                    new_ep.writelines(string)
                    line = ep_base.readline()
                else:
                    new_ep.writelines(line)
                    line = ep_base.readline()
                    if line.split(',')[0]!= "    ":
                        string = "    "+str(float(line.split(',')[0])*percent)+",                        !- Watts per Person {W/person}\n" 
                        new_ep.writelines(string)
                        line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()

    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)


#-----------------------------------------------------------------------------------Natural Ventilation
def NVSchedStr():
    string = """Schedule:Compact,
    Natural_Ventilation,     !- Name
    Fraction,                !- Schedule Type Limits Name
    Through: 12/31,           !- Field 1
    For: AllDays,            !- Field 2
    Until: 24:00,            !- Field 3
    1;                       !- Field 4\n\n"""
    return string
    

def zoneVentStr(counter, zone, open_area):
    string = "  ZoneVentilation:WindandStackOpenArea, \n" + "    NaturalVent_" + str(counter) + ",            !- Name \n" + \
             "    " + zone + "        !- Zone Name \n" + "    " + str(open_area) + ",                     !- Opening Area {m2} \n" + \
             """    Natural_Ventilation,     !- Opening Area Fraction Schedule Name
    autocalculate,           !- Opening Effectiveness {dimensionless}
    ,                        !- Effective Angle {deg}
    ,                        !- Height Difference {m}
    autocalculate,           !- Discharge Coefficient for Opening
    22,                      !- Minimum Indoor Temperature {C}
    ,                        !- Minimum Indoor Temperature Schedule Name
    100,                     !- Maximum Indoor Temperature {C}
    ,                        !- Maximum Indoor Temperature Schedule Name
    1,                       !- Delta Temperature {deltaC}
    ,                        !- Delta Temperature Schedule Name
    18,                    !- Minimum Outdoor Temperature {C}
    ,                        !- Minimum Outdoor Temperature Schedule Name
    28,                      !- Maximum Outdoor Temperature {C}
    ,                        !- Maximum Outdoor Temperature Schedule Name
    15;                      !- Maximum Wind Speed {m/s}"""
    return string

def writeNVInput(open_area, new_ep_path):
    fh, abs_path = mkstemp()
    new_ep = open(abs_path, 'w')
    ep_base = open(new_ep_path, 'r')
    counter = 0
    
    zone_list = []
    line = ep_base.readline()
    while line:
        if line == "ZoneVentilation:DesignFlowRate,\n":
##            print "Default venttilation detected!"
            for i in range(27):
                line = ep_base.readline()
        if line.strip() == 'Zone,':
            new_ep.writelines(line)
            line = ep_base.readline()
            new_ep.writelines(line)
            zoneline_string = line.split()
            zone_list.append(zoneline_string[0])
            line = ep_base.readline()
        new_ep.writelines(line)
        line = ep_base.readline()
    # print (zone_list)
        
    NVSched = NVSchedStr()
    new_ep.writelines(NVSched)
    for i in range(0, len(zone_list)):
        zoneVent = zoneVentStr(i + 1, zone_list[i], open_area)
        new_ep.writelines(zoneVent)
        new_ep.writelines("\n \n")
    new_ep.writelines(line)
    line = ep_base.readline()
        
    ep_base.close()
    new_ep.close()
    close(fh)
    remove(new_ep_path)
    move(abs_path, new_ep_path)


