#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C reated on Thu Sep  9 15:22:33 2021

@author: johanvanhaperen
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pvlib
import numpy as np
import sklearn
import math
import matplotlib.dates as mdates


#Importing Data - PV/Irrad
irradiance = pd.read_csv('Irradiance_2015_UPOT.csv', sep = ";" , index_col = 'timestamp', parse_dates=(True))
irradiance = irradiance.resample(rule = '5Min').mean()
irradiance.index = irradiance.index.tz_localize('UTC')
irradiance.index = irradiance.index.tz_convert('CET')
module_par = pd.read_excel('Module parameters.xlsx', index_col='Parameters')
KNMI_weather = pd.read_csv('Hupsel_weer.csv', sep = ",", parse_dates=(True))
KNMI_weather.date = KNMI_weather.date.astype(str)
KNMI_weather.hour = KNMI_weather.hour.apply(lambda x: str(x).zfill(2))
KNMI_weather.hour =KNMI_weather.hour.replace('24', '00')
KNMI_weather ['datetime'] = KNMI_weather.date + KNMI_weather.hour + '00'
KNMI_weather ['datetime'] = pd.to_datetime(KNMI_weather.datetime, format = '%Y%m%d%H%M')
KNMI_weather.index = KNMI_weather.datetime 
KNMI_weather = KNMI_weather[['windspeed', 'temperature', 'ghi']]      
KNMI_weather.ghi = KNMI_weather.ghi * 2.77778
KNMI_weather.temperature = KNMI_weather.temperature / 10
KNMI_weather.windspeed = KNMI_weather.windspeed / 10
KNMI_weather.index = KNMI_weather.index.tz_localize('UTC')
KNMI_weather.index = KNMI_weather.index.tz_convert('CET')
KNMI_weather.index = KNMI_weather.index + pd.Timedelta(minutes=-30)                    

#importing Variables
capacity_hit = module_par.iloc[41,0]
capacity_cdte = module_par.iloc[41,1]
capacity_monosi = module_par.iloc[41,2]
area_hit = module_par.iloc[1,0]
area_cdte = module_par.iloc[1,1]
area_monosi = module_par.iloc[1,2]

#Calculating Zenith
longitude_UU = 5.167502412464747
latitude_UU = 52.08777166948196
pressure = 101325
solarposition = pvlib.solarposition.ephemeris(irradiance.index, latitude_UU, longitude_UU, pressure, irradiance.temp_air)
solarposition = solarposition.loc[solarposition['elevation'] > 0]
irradiance = irradiance[irradiance.index.isin(solarposition.index)]

'Calculating Different DNI Values'
DNI_UU = irradiance.DNI
DNI_UU = DNI_UU.dropna()
DNI_disc = pvlib.irradiance.disc(irradiance.GHI, solarposition.zenith, irradiance.index)
DNI_disc = DNI_disc.dropna()
DNI_dirint = pvlib.irradiance.dirint(irradiance.GHI, solarposition.zenith, irradiance.index, pressure=101325.0, use_delta_kt_prime=True, temp_dew= irradiance.temp_air, min_cos_zenith=0.065, max_zenith=87)
DNI_dirint = DNI_dirint.dropna()
DNI_UU_dirint = DNI_UU[DNI_UU.index.isin(DNI_dirint.index)]
DNI_erbs = pvlib.irradiance.erbs(irradiance.GHI, solarposition.zenith, irradiance.index)
DNI_erbs = DNI_erbs.dropna()

#DNI_dirindex additional calculated values
linke_turbidity_uu = pvlib.clearsky.lookup_linke_turbidity(irradiance.index, latitude_UU, longitude_UU)
absolute_airmas = pvlib.atmosphere.get_absolute_airmass(pvlib.atmosphere.get_relative_airmass(solarposition.apparent_zenith, model='kastenyoung1989'))
uuclearsky = pvlib.clearsky.ineichen(solarposition.apparent_zenith, absolute_airmas, linke_turbidity_uu, perez_enhancement=True)
DNI_dirindex = pvlib.irradiance.dirindex(irradiance.GHI, uuclearsky.ghi, uuclearsky.dni, solarposition.zenith, irradiance.index)
DNI_dirindex = DNI_dirindex.replace(np.inf, np.nan)
DNI_dirindex = DNI_dirindex.dropna()
DNI_UU_dirindex = DNI_UU[DNI_UU.index.isin(DNI_dirindex.index)]
DNI_UU_dirindex = DNI_UU_dirindex.dropna()    
              
'Calculating Square Root Mean Errors'
rmse_disc = sklearn.metrics.mean_squared_error(DNI_UU, DNI_disc.dni)
rmse_disc = np.sqrt(rmse_disc)
rmse_dirint = sklearn.metrics.mean_squared_error(DNI_UU_dirint, DNI_dirint)
rmse_dirint = np.sqrt(rmse_dirint)
rmse_erbs = sklearn.metrics.mean_squared_error(DNI_UU, DNI_erbs.dni)
rmse_erbs = np.sqrt(rmse_erbs)
rmse_dirindex = sklearn.metrics.mean_squared_error(DNI_UU_dirindex, DNI_dirindex) 
rmse_dirindex = np.sqrt(rmse_dirindex)


'Calculating The MBE'
mbe_disc = (DNI_disc.dni-DNI_UU).mean()
mbe_dirint = (DNI_dirint-DNI_UU_dirint).mean()
mbe_erbs = (DNI_erbs.dni - DNI_UU).mean()
mbe_dirindex = (DNI_dirindex - DNI_UU_dirindex).mean()

'Calculating the MAE'
mae_disc = sklearn.metrics.mean_absolute_error(DNI_UU, DNI_disc.dni)
mae_dirint = sklearn.metrics.mean_absolute_error(DNI_UU_dirint, DNI_dirint)
mae_erbs = sklearn.metrics.mean_absolute_error(DNI_UU, DNI_erbs.dni)
mae_dirindex = sklearn.metrics.mean_absolute_error(DNI_UU_dirindex, DNI_dirindex)

'Calculating R2'
r2_disc = sklearn.metrics.r2_score(DNI_UU, DNI_disc.dni)
r2_dirint = sklearn.metrics.r2_score(DNI_UU_dirint, DNI_dirint)
r2_erbs = sklearn.metrics.r2_score(DNI_UU, DNI_erbs.dni)
r2_dirindex = sklearn.metrics.r2_score(DNI_UU_dirindex, DNI_dirindex)

'Creating Dataframe'
modeled_DNI = pd.DataFrame({
    'disc': [mae_disc, mbe_disc, r2_disc, rmse_disc],
    'dirint': [mae_dirint, mbe_dirint, r2_dirint, rmse_dirint],
    'erbs': [mae_erbs, mbe_erbs, r2_erbs, rmse_erbs],
    'dirindex':[mae_dirindex, mbe_dirindex, r2_dirindex, rmse_dirindex]
    })
modeled_DNI.index=['mae', 'mbe', 'r2', 'rmse']

fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(321)
ax1.set_xlabel('Measured DNI [W/m²]', fontsize=14)
ax1.set_ylabel('Modelled DNI [W/m²]', fontsize=14)
ax2 = fig.add_subplot(322)
ax2.set_xlabel('Measured DNI [W/m²]', fontsize=14)
ax2.set_ylabel('Modelled DNI [W/m²]', fontsize=14)
ax3 = fig.add_subplot(323)
plt.xlim([0,1000])
plt.ylim([0,1000])
ax3.set_xlabel('Measured DNI [W/m²]', fontsize=14)
ax3.set_ylabel('Modelled DNI [W/m²]', fontsize=14)
ax4 = fig.add_subplot(324)
ax4.set_xlabel('Measured DNI [W/m²]', fontsize=14)
ax4.set_ylabel('Modelled DNI [W/m²]', fontsize=14)
disc_plt = ax1.scatter(DNI_UU, DNI_disc.dni, s=0.05, c="green", marker = ".")
ax1.title.set_text('DISC model')
dirint_plt = ax2.scatter(DNI_UU, DNI_dirint, s=0.05, c="red", marker=".")
ax2.title.set_text('DIRINT model')
erbs_plt = ax3.scatter(DNI_UU, DNI_erbs.dni, s=0.05, c="blue", marker=".")
ax3.title.set_text('Erbs model')
dirindex_plt = ax4.scatter(DNI_UU_dirindex, DNI_dirindex, s=0.05, c="black", marker=".")
ax4.title.set_text('DIRINDEX model')

'Question 2'
building_para = pd.DataFrame({
    'A': [100, 50, 60, 0],
    'B': [30, 50, 30,0 ],
    'CD': [6, 0, 50, 40]})
building_para.index = ['height', 'width', 'length','roof incline']
facade_se_surface_a = building_para.A.length * building_para.A.height
facade_sw_surface_a = building_para.A.width * building_para.A.height
roof_surface_a = building_para.A.length * building_para.A.width
facade_s_surface_b = building_para.B.width * building_para.B.height
facade_sw_surface_b = building_para.B.length * building_para.B.height
facade_se_surface_b = building_para.B.length * building_para.B.height
roof_surface_b = building_para.A.length * building_para.A.width
roof_surface_c = 3/np.sin(np.deg2rad(40)) * building_para.CD.length
roof_surface_d = 3/np.sin(np.deg2rad(40)) * building_para.CD.length

def surface (name, facade, height, width, tilt, surface_azimuth):
    surface = {}
    surface['name']=name
    surface['surface']=facade
    surface['height']=height
    surface['width']=width
    surface['tilt']=tilt
    surface['orientation']=surface_azimuth
    return surface


surfaces = {}
surfaces['building_a_fsw'] = surface('BA', 'F SW',100,50,90,225)
surfaces['building_a_fse'] = surface('BA', 'F SE',100,60,90,135)
surfaces['building_a_roof']= surface('BA', 'Roof', 50, 60, 35, 225)
#Building B, Office building, oriented North South
surfaces['building_b_fe'] = surface('BB', 'F E',30,30,90,90)
surfaces['building_b_fs'] = surface('BB', 'F S',30,50,90,180)
surfaces['building_b_fw'] = surface('BB', 'F W',30,30,90,270)
surfaces['building_b_roof'] = surface('BB', 'Roof', 50,30,35,180)
#House C, located on a East-West line
surfaces['house_c_rfn'] = surface('HC','Roof N',4.67,50,40,0)
surfaces['house_c_rfs'] = surface('HC','Roof S',4.67,50,40,180)
#House D, located on a North-South line
surfaces['house_d_rfe'] = surface('HD','Roof E',4.67,50,40,90)
surfaces['house_d_rfw'] = surface('HD','Roof W',4.67,50,40,270)

'Question 2.3'
longitude_Hupsel = 6.616657
latitude_Hupsel = 52.0819672
solarposition_hupsel = pvlib.solarposition.ephemeris(KNMI_weather.index, latitude_Hupsel, longitude_Hupsel, pressure, KNMI_weather.temperature)
solarposition_hupsel = solarposition_hupsel.loc[solarposition_hupsel['elevation'] > 0]
KNMI_weather = KNMI_weather[KNMI_weather.index.isin(solarposition_hupsel.index)]
DNI_hupsel = pvlib.irradiance.dirint(KNMI_weather.ghi, solarposition_hupsel.zenith, KNMI_weather.index, pressure=101325.0, use_delta_kt_prime=True, temp_dew= KNMI_weather.temperature, min_cos_zenith=0.065, max_zenith=87)
DHI_hupsel = KNMI_weather.ghi - (np.cos(np.radians(solarposition_hupsel.zenith)) * DNI_hupsel)

        

def total_irradiance(tilt, surface_azimuth, zenith, solar_azimuth,
                     dni, ghi, dhi, surface):
    
    total_irradiance = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt, surface_azimuth=surface_azimuth,solar_zenith = zenith, 
                                           solar_azimuth= solar_azimuth, 
                                           dni=dni, 
                                           ghi=ghi, 
                                           dhi=dhi)
  
    surface['poa_global'] = total_irradiance.poa_global 
    surface['poa_direct']=total_irradiance.poa_direct
    surface['poa_diffuse']=total_irradiance.poa_diffuse
    surface['poa_sky_diffuse']=total_irradiance.poa_sky_diffuse
    surface['poa_ground_diffuse']=total_irradiance.poa_ground_diffuse

for i in surfaces.keys():
     total_irradiance(surfaces[i]['tilt'], surfaces[i]['orientation'], solarposition_hupsel.zenith, solarposition_hupsel.azimuth, DNI_hupsel, KNMI_weather.ghi, DHI_hupsel, surfaces[i])  


surfaces_irradiance = {}
for i in surfaces.keys():
    surfaces_irradiance[i] = {}
    surfaces_irradiance[i]['poa_global']= surfaces[i]['poa_global']
    del surfaces[i]['poa_global']
    surfaces_irradiance[i]['poa_direct']= surfaces[i]['poa_direct']
    del surfaces[i]['poa_direct']
    surfaces_irradiance[i]['poa_diffuse']= surfaces[i]['poa_diffuse']
    del surfaces[i]['poa_diffuse']
    surfaces_irradiance[i]['poa_sky_diffuse']= surfaces[i]['poa_sky_diffuse']
    del surfaces[i]['poa_sky_diffuse']
    surfaces_irradiance[i]['poa_ground_diffuse']= surfaces[i]['poa_ground_diffuse']
    del surfaces[i]['poa_ground_diffuse']


roofs_AB = {}
for i in range(10,46,5):
    j = 'building_b_rf{}'.format(i)
    k = 'building_a_rfse{}'.format(i)
    l = 'building_a_rfsw{}'.format(i)
    roofs_AB [j] = surface('Building B', 'Rooftop S {}°'.format(i),50,30,i,180)
    roofs_AB [k] = surface('Building A', 'Rooftop SE {}°'.format(i),50,60,i,135)
    roofs_AB [l] = surface('Building A', 'Rooftop SW {}°'.format(i),60,50,i,225)

for i in roofs_AB :
    total_irradiance(roofs_AB[i]['tilt'], roofs_AB[i]['orientation'], solarposition_hupsel.zenith, solarposition_hupsel.azimuth, DNI_hupsel, KNMI_weather.ghi, DHI_hupsel, roofs_AB[i])

roofs_AB_irradiance = {}
for i in roofs_AB.keys():
    roofs_AB_irradiance[i] = {}
    
    roofs_AB_irradiance[i]['poa_global']= roofs_AB[i]['poa_global']
    del roofs_AB[i]['poa_global']
    roofs_AB_irradiance[i]['poa_direct']= roofs_AB[i]['poa_direct']
    del roofs_AB[i]['poa_direct']
    roofs_AB_irradiance[i]['poa_diffuse']= roofs_AB[i]['poa_diffuse']
    del roofs_AB[i]['poa_diffuse']
    roofs_AB_irradiance[i]['poa_sky_diffuse']= roofs_AB[i]['poa_sky_diffuse']
    del roofs_AB[i]['poa_sky_diffuse']
    roofs_AB_irradiance[i]['poa_ground_diffuse']= roofs_AB[i]['poa_ground_diffuse']
    del roofs_AB[i]['poa_ground_diffuse']
    
def consolidation (buildings_dictionary, parameter, time):
    consolidate = {}
    for i in buildings_dictionary.keys():
        consolidate[i]= buildings_dictionary[i][parameter]
    consolidation = pd.DataFrame(consolidate, time)
    return consolidation

poa_global_rf = consolidation(roofs_AB_irradiance,'poa_global', KNMI_weather.index)
poa_global_surfaces = consolidation(surfaces_irradiance, 'poa_global', KNMI_weather.index)
poa_direct_surfaces = consolidation(surfaces_irradiance, 'poa_direct', KNMI_weather.index)
poa_diffuse_surfaces = consolidation(surfaces_irradiance, 'poa_diffuse', KNMI_weather.index)

surfaces_poa_total_annual= {}
surfaces_poa_direct_annual= {}
surfaces_poa_diffuse_annual= {}


for i in poa_global_surfaces.keys():
    surfaces_poa_total_annual[i] = poa_global_surfaces[i].sum()/1000

for i in poa_direct_surfaces.keys():
    surfaces_poa_direct_annual[i] = poa_direct_surfaces[i].sum()/1000    

for i in poa_diffuse_surfaces.keys():
    surfaces_poa_diffuse_annual[i] = poa_diffuse_surfaces[i].sum()/1000 
    
new_names_roofs = []
for i in poa_global_rf.keys():
    new_names_roofs.append(str(roofs_AB[i]['name']+ ' ' +roofs_AB[i]['surface']))
poa_global_rf.columns=new_names_roofs
 
  
poa_global_rf_annual = {}
for i in poa_global_rf.keys():
    poa_global_rf_annual[i]=poa_global_rf[i].sum()/1000    


poa_global_surfaces = consolidation(surfaces_irradiance, 'poa_global', KNMI_weather.index)

poa_global_surfaces_annual = {}
for i in poa_global_surfaces.keys():
    poa_global_surfaces_annual[i] = poa_global_surfaces[i].sum()/1000

buildinga_annual = {}
buildingb_annual = {}
for i in poa_global_rf_annual.keys():
    if 'Building A' in i:
        j = i.replace("Building A Rooftop", '' )
        buildinga_annual[j]=poa_global_rf_annual[i]
    if 'Building B' in i:
        j = i.replace("Building B Rooftop", 'South' )
        buildingb_annual[j]=poa_global_rf_annual[i]

buildinga_annual_graph = pd.DataFrame(data=buildinga_annual, index=[0])
buildinga_annual_graph = buildinga_annual_graph.sort_values(by=[0], axis=1)
buildingb_annual_graph = pd.DataFrame(data=buildingb_annual, index=[0])
buildingb_annual_graph = buildingb_annual_graph.sort_values(by=[0], axis=1)
        
def bar_graph (dataframe, color, xaxis, yaxis, title, filename):
    fig, ax = plt.subplots(figsize = (15,10))
    bars = ax.bar(dataframe.columns, dataframe.iloc[0], 
            color = color)
    ax.set_xlabel(xaxis, fontsize = 14, fontweight =400)
    ax.tick_params(axis='x', rotation = 15, labelsize=13)
    ax.set_ylabel(yaxis, fontsize = 14, fontweight =400)
    ax.set_title(title)
    for i in bars:
        height = i.get_height()
        ax.text(i.get_x() + i.get_width()/2., 1.01*height,
                '%.1f' % float(height),
                ha='center', va='bottom', color='black', fontweight=510)
    plt.show()
    fig.savefig(filename)
    plt.close()
    
bar_graph (buildinga_annual_graph, "midnightblue",
           "Combinations Rooftop Building A","POA Global Annual (kWh/m²)", 
           '','Part A Q2 Figure 4 POA Global A')

bar_graph (buildingb_annual_graph, "darkred",
           "Combinations Rooftop Building B","POA Global Annual (kWh/m²)", 
           '','Part A Q2 Figure 4 POA Global B')


'Question 2.5'
total_irradiance_roof_b = pvlib.irradiance.get_total_irradiance(35, 180, solarposition_hupsel.zenith, solarposition_hupsel.azimuth, DNI_hupsel, KNMI_weather.ghi, DHI_hupsel)
total_irradiance_roof_a = pvlib.irradiance.get_total_irradiance(35, 225, solarposition_hupsel.zenith, solarposition_hupsel.azimuth, DNI_hupsel, KNMI_weather.ghi, DHI_hupsel)

#2.6
surfaces_annual_graph = pd.DataFrame(data=surfaces_poa_total_annual, index=[0])
surfaces_annual_graph = surfaces_annual_graph.sort_values(by=[0], axis=1)


    
bar_graph (surfaces_annual_graph, "orange",
           "Annual Irradiance Building Surfaces","POA Global Annual (kWh/m²)", 
           '','Part A Q2 Figure 4 POA Global A')

'Question 3.1'


for i in surfaces.keys():
    #To determine the office rooftops
    if surfaces[i]['surface'] == 'Roof':
        av_area = (surfaces[i]['height']) * (surfaces[i]['width']) * 0.5
    #To determine the house rooftops
    elif surfaces[i]['name'] == 'HC':
        av_area = (surfaces[i]['height']) * (surfaces[i]['width']) * 0.6
    elif surfaces[i]['name'] == 'HD':  
        av_area = (surfaces[i]['height']) * (surfaces[i]['width']) * 0.6
    #To determine the facades (= the rest)
    else:
        av_area = (surfaces[i]['height']) * (surfaces[i]['width']) * 0.3
    surfaces[i]['available area'] = av_area 

'Calculating Number of Panels'    
for i in surfaces.keys():
    number_panels_hit = math.floor((surfaces[i]['available area']/area_hit))
    surfaces[i]['amount of hit panels']= number_panels_hit
    
for i in surfaces.keys():
    number_panels_monosi = math.floor((surfaces[i]['available area']/area_monosi))
    surfaces[i]['amount of monosi panels']= number_panels_monosi

for i in surfaces.keys():
    number_panels_cdte = math.floor((surfaces[i]['available area']/area_cdte))
    surfaces[i]['amount of cdte panels']= number_panels_cdte

'Calculating installed capacity'

for i in surfaces.keys():
    installed_capacity_hit = surfaces[i]['amount of hit panels'] * capacity_hit 
    surfaces[i]['installed Capacity hit'] = installed_capacity_hit #in W
    
for i in surfaces.keys():
    installed_capacity_monosi = surfaces[i]['amount of monosi panels'] * capacity_monosi
    surfaces[i]['installed Capacity monosi'] = installed_capacity_monosi #in W

for i in surfaces.keys():
    installed_capacity_cdte = surfaces[i]['amount of cdte panels'] * capacity_cdte 
    surfaces[i]['installed Capacity cdte'] = installed_capacity_cdte #in W
    
"Question 3.2"

for i in surfaces.keys():
    if 'building_a_roof' or 'building_b_roof' in surfaces.keys():
        cell_temp_hit = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -3.7489, -0.1287, 2.03) 
        surfaces[i]['cell temperature hit'] = cell_temp_hit
    else:
        cell_temp_hit = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -2.98, -0.0471, 1) 
        surfaces[i]['cell temperature'] = cell_temp_hit

for i in surfaces.keys():
    if 'building_a_roof' or 'building_b_roof' in surfaces.keys():
        cell_temp_monosi = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -3.7566, -0.156, 2.58) 
        surfaces[i]['cell temperature monosi'] = cell_temp_monosi
    else:
        cell_temp_monosi = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -2.98, -0.0471, 1) 
        surfaces[i]['cell temperature monosi'] = cell_temp_monosi
        
for i in surfaces.keys():
    if 'building_a_roof' or 'building_b_roof' in surfaces.keys():
        cell_temp_cdte = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -3.47, -0.059, 3) 
        surfaces[i]['cell temperature cdte'] = cell_temp_cdte
    else:
        cell_temp_cdte = pvlib.pvsystem.temperature.sapm_cell(surfaces_irradiance[i]['poa_global'], KNMI_weather['temperature'], KNMI_weather['windspeed'], -2.98, -0.0471, 1) 
        surfaces[i]['cell temperature cdte'] = cell_temp_cdte


for i in surfaces.keys():
    aoi = pvlib.irradiance.aoi(surfaces[i]['tilt'], surfaces[i]['orientation'], solarposition_hupsel.zenith, solarposition_hupsel.azimuth)
    surfaces[i]['aoi'] = aoi

linke_turbidity_hupsel = pvlib.clearsky.lookup_linke_turbidity(KNMI_weather.index, latitude_Hupsel, longitude_Hupsel)
absolute_airmas_hupsel = pvlib.atmosphere.get_absolute_airmass(pvlib.atmosphere.get_relative_airmass(solarposition_hupsel.apparent_zenith, model='kastenyoung1989'))


# for e in irradiance_building.keys():
#     #assumption that mounted close to a roof for the values of a,b and delta T
#     cell_temp = pvlib.pvsystem.temperature.sapm_cell(irradiance_building[e]['poa_global'], KNMI_weatherHeino['temperature'],
#                                             KNMI_weatherHeino['windspeed'], -2.98, -0.0471, 1) 
#     buildings[e]['cell temperature'] = cell_temp

'Calculating Effective Irradiance'

for i in surfaces.keys():
    effective_irradiance_hit = pvlib.pvsystem.sapm_effective_irradiance(surfaces_irradiance[i]['poa_global'], surfaces_irradiance[i]['poa_diffuse'], absolute_airmas_hupsel, surfaces[i]['aoi'], module_par['HIT'])
    surfaces[i]['effective irradiance hit'] = effective_irradiance_hit
    
for i in surfaces.keys():
    effective_irradiance_monosi = pvlib.pvsystem.sapm_effective_irradiance(surfaces_irradiance[i]['poa_global'], surfaces_irradiance[i]['poa_diffuse'], absolute_airmas_hupsel, surfaces[i]['aoi'], module_par['mono-Si'])
    surfaces[i]['effective irradiance monosi'] = effective_irradiance_monosi
    
for i in surfaces.keys():
    effective_irradiance_cdte = pvlib.pvsystem.sapm_effective_irradiance(surfaces_irradiance[i]['poa_global'], surfaces_irradiance[i]['poa_diffuse'], absolute_airmas_hupsel, surfaces[i]['aoi'], module_par['CdTe'])
    surfaces[i]['effective irradiance cdte'] = effective_irradiance_cdte  
    

for i in surfaces.keys():
    dc_hit = pvlib.pvsystem.sapm(surfaces[i]['effective irradiance hit'], surfaces[i]['cell temperature hit'], module_par['HIT'])
    surfaces[i]['dc_hit']=dc_hit

for i in surfaces.keys():
    dc_monosi = pvlib.pvsystem.sapm(surfaces[i]['effective irradiance monosi'], surfaces[i]['cell temperature monosi'], module_par['mono-Si'])
    surfaces[i]['dc_monosi']=dc_monosi

for i in surfaces.keys():
    dc_cdte = pvlib.pvsystem.sapm(surfaces[i]['effective irradiance cdte'], surfaces[i]['cell temperature cdte'], module_par['CdTe'])
    surfaces[i]['dc_cdte']=dc_cdte

for i in surfaces.keys():
    surfaces[i]['power_hit'] = surfaces[i]['dc_hit']['p_mp']
    surfaces[i]['power_hit'] = surfaces[i]['power_hit'].fillna(0)
    surfaces[i]['power_monosi'] = surfaces[i]['dc_monosi']['p_mp']
    surfaces[i]['power_monosi'] = surfaces[i]['power_monosi'].fillna(0)
    surfaces[i]['power_cdte'] = surfaces[i]['dc_cdte']['p_mp']
    surfaces[i]['power_cdte'] = surfaces[i]['power_cdte'].fillna(0)

for i in surfaces.keys():
    total_power_hit = surfaces[i]['power_hit']*surfaces[i]['amount of hit panels']
    surfaces[i]['total power hit']=total_power_hit
    
for i in surfaces.keys():
    total_power_monosi = surfaces[i]['power_monosi']*surfaces[i]['amount of monosi panels']
    surfaces[i]['total power monosi']=total_power_monosi
    
for i in surfaces.keys():
    total_power_cdte = surfaces[i]['power_cdte']*surfaces[i]['amount of cdte panels']
    surfaces[i]['total power cdte']=total_power_cdte
    
#annual yields (division by 1000000 to go from Wh/yr to MWh/yr)

for i in surfaces.keys():
    annual_yield_hit = sum(surfaces[i]['total power hit'])/1000000
    surfaces[i]['annual yield hit'] = annual_yield_hit

for i in surfaces.keys():
    annual_yield_monosi = sum(surfaces[i]['total power monosi'])/1000000
    surfaces[i]['annual yield monosi'] = annual_yield_monosi
    
for i in surfaces.keys():
    annual_yield_cdte = sum(surfaces[i]['total power cdte'])/1000000
    surfaces[i]['annual yield cdte'] = annual_yield_cdte

#new dataframe to make graphs about the annual yield for all module types

annual_yield_all = pd.DataFrame(data = [surfaces['building_a_fse'], surfaces['building_a_fsw'], surfaces['building_a_roof'],
                                   surfaces['building_b_fe'], surfaces['building_b_fs'], surfaces['building_b_fw'], surfaces['building_b_roof'],
                                   surfaces['house_c_rfn'], surfaces['house_c_rfs'],
                                   surfaces['house_d_rfe'], surfaces['house_d_rfw']],
                           columns = ['name', 'surface', 'annual yield hit', 'annual yield monosi', 'annual yield cdte']) 

#graphs for Q3.3
annual_yield_all.index = annual_yield_all.name + ' ' + annual_yield_all.surface

annual_yield_all.plot.bar(rot = 30, color=['red', 'orange', 'yellow'])
plt.title('Annual yield building surfaces per module type', fontsize = 20)
plt.legend(fontsize = 14)
plt.ylabel('Annual yield (MWh)', fontsize = 16)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 14)


#Q3.4 annual yield per area in kWh/m2

for i in surfaces.keys():
    annual_yield_hit_per_area = (surfaces[i]['annual yield hit']) / ((surfaces[i]['height']) * (surfaces[i]['width'])) * 1000
    surfaces[i]['annual yield hit per area'] = annual_yield_hit_per_area

for i in surfaces.keys():
    annual_yield_monosi_per_area = (surfaces[i]['annual yield monosi']) / ((surfaces[i]['height']) * (surfaces[i]['width'])) * 1000
    surfaces[i]['annual yield monosi per area'] = annual_yield_monosi_per_area
    
for i in surfaces.keys():
    annual_yield_cdte_per_area = (surfaces[i]['annual yield cdte']) / ((surfaces[i]['height']) * (surfaces[i]['width'])) * 1000
    surfaces[i]['annual yield cdte per area'] = annual_yield_cdte_per_area    
    
#Q3.4 graph same as before    
    
    
annual_yield_all_per_area = pd.DataFrame(data = [surfaces['building_a_fse'], surfaces['building_a_fsw'], surfaces['building_a_roof'],
                                   surfaces['building_b_fe'], surfaces['building_b_fs'], surfaces['building_b_fw'], surfaces['building_b_roof'],
                                   surfaces['house_c_rfn'], surfaces['house_c_rfs'],
                                   surfaces['house_d_rfe'], surfaces['house_d_rfw']],
                           columns = ['name', 'surface', 'annual yield hit per area', 'annual yield monosi per area', 'annual yield cdte per area'])     
    
annual_yield_all_per_area.index = annual_yield_all_per_area.name + ' ' + annual_yield_all_per_area.surface

annual_yield_all_per_area.plot.bar(rot = 30, color=['red','orange','yellow'])
plt.title('Annual yield per area for different modules ', fontsize = 16)
plt.legend(loc = 1, fontsize = 10)
plt.ylabel('Annual yield per unit area (kWh/m2)', fontsize = 16)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 14)    
    
'Question 4'
#Paco = rated power of the inverter
#𝜂 nom = nominal efficiency of the interver 

nnom = 0.96
pac0 = module_par['HIT']['Wp']
pdc0 = pac0/nnom

def inverter_module (pdc, nnmom, pac0):
    pac  = np.zeros(len(pdc))
    for i in range(len(pdc)):
        if pdc[i] >0 :
            𝜁 = pdc[i]/pdc0
            𝜂 = -0.0162* 𝜁 - (0.0059/𝜁) + 0.9858 
            if pdc[i]<pdc0 and 𝜂>0:
                pac[i] = 𝜂*pdc[i]
            elif pdc[i] >= pdc0:
                pac[i] = pac0
        else:
            pac[i] = 0
    return pac


for i in surfaces.keys(): #per panel
    ac_power = inverter_module(surfaces[i]['power_hit'], nnom, pac0)*surfaces[i]['amount of hit panels']
    surfaces[i]['ac_power'] = ac_power
    surfaces[i]['ac_power'] = pd.Series(surfaces[i]['ac_power'], index = KNMI_weather.index)


for i in surfaces.keys():
    annual_ac_power = sum(surfaces[i]['ac_power'])/1000000 #mwh
    surfaces[i]['annual ac power'] = annual_ac_power

#Q4.2 graphs

annual_ac_surfaces = pd.DataFrame(data = [surfaces['building_a_fse'], surfaces['building_a_fsw'], surfaces['building_a_roof'],
                                   surfaces['building_b_fe'], surfaces['building_b_fs'], surfaces['building_b_fw'], surfaces['building_b_roof'],
                                   surfaces['house_c_rfn'], surfaces['house_c_rfs'],
                                   surfaces['house_d_rfe'], surfaces['house_d_rfw']],
                           columns = ['name', 'surface', 'annual ac power'])  


annual_ac_surfaces.index = annual_ac_surfaces.name + ' ' + annual_ac_surfaces.surface
annual_ac_surfaces.plot.bar(rot = 30, color=['cyan'])
plt.title('Annual AC output per surface', fontsize = 16)
plt.ylabel('Annual AC output (MWh)', fontsize = 16)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 14)   

#adding all surfaces to their corresponding buildings for second graph of 4.2
annual_ac_output_building_a = surfaces['building_a_fse']['annual ac power'] + surfaces['building_a_fsw']['annual ac power'] + surfaces['building_a_roof']['annual ac power']
annual_ac_output_building_b = surfaces['building_b_fe']['annual ac power'] + surfaces['building_b_fs']['annual ac power'] + surfaces['building_b_fw']['annual ac power']+ surfaces['building_b_roof']['annual ac power']
annual_ac_output_building_c = surfaces['house_c_rfn']['annual ac power'] + surfaces['house_c_rfs']['annual ac power']
annual_ac_output_building_d = surfaces['house_d_rfe']['annual ac power'] + surfaces['house_d_rfw']['annual ac power']

#second graph of 4.2
annual_ac_output_buildings = pd.DataFrame(data = [annual_ac_output_building_a, annual_ac_output_building_b, annual_ac_output_building_c, annual_ac_output_building_d])
buildings_a_to_d = ['Building A', 'Building B', 'House C', 'House D']
annual_ac_output_buildings.index = buildings_a_to_d
annual_ac_output_buildings.plot.bar(rot=30, color=['pink'])
plt.title('Annual AC output per building', fontsize = 16)
plt.ylabel('Annual AC output (MWh)', fontsize = 16)
legend4_2=['AC Output']
plt.legend(legend4_2, loc=1)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 14)   




#4.3
#Choose one day for summer (June - Aug), spring (Mar - May), autumn (Sep - Nov)
#Summer: July 4, from 04:00 to 19:00, highest GHI noted at 852
#Spring: April 5, from 06:00 to 18:00, highest GHI noted at 330
#Autumn: Nov 7, start from 07:00 to 15:00, highest GHI noted at 177

for i in surfaces.keys():
    KNMI_weather_summer = (surfaces[i]['ac_power'].loc['2019-07-04']) / 1000 #kW
    surfaces[i]['ac power summer day'] = KNMI_weather_summer

#graph for building A summer    
x = KNMI_weather_summer.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_a_fse']['ac power summer day']
y2 = surfaces['building_a_fsw']['ac power summer day']
y3 = surfaces['building_a_roof']['ac power summer day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_a_fse'], color='darkred')
ax.plot(x,y2, label = surfaces['building_a_fsw'], color='orangered')
ax.plot(x,y3, label = surfaces['building_a_roof'], color ='peru')
legend4_3_1=['Facade SE', 'Facade SW', 'Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building A per surface in summer', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')
#graph for building B summer
x = KNMI_weather_summer.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_b_fe']['ac power summer day']
y2 = surfaces['building_b_fs']['ac power summer day']
y3 = surfaces['building_b_fw']['ac power summer day']
y4 = surfaces['building_b_roof']['ac power summer day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_b_fe'], color='darkgreen')
ax.plot(x,y2, label = surfaces['building_b_fs'], color='lime')
ax.plot(x,y3, label = surfaces['building_b_fw'], color='cyan')
ax.plot(x,y4, label = surfaces['building_b_roof'], color='aquamarine')
legend4_3_1=['Facade E', 'Facade S', 'Facade W','Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building B per surface in summer', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house C summer
x = KNMI_weather_summer.index
plt.xticks(fontsize = 10)
y1 = surfaces['house_c_rfn']['ac power summer day']
y2 = surfaces['house_c_rfs']['ac power summer day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_c_rfn'], color='purple')
ax.plot(x,y2, label = surfaces['house_c_rfs'],color='magenta')
legend4_3_2=['Roof north', 'Roof south']
plt.legend(legend4_3_2, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house C per surface in summer', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house D summer
x = KNMI_weather_summer.index
plt.xticks(fontsize = 10)
y1 = surfaces['house_d_rfe']['ac power summer day']
y2 = surfaces['house_d_rfw']['ac power summer day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_d_rfe'], color='blue')
ax.plot(x,y2, label = surfaces['house_d_rfw'], color='dodgerblue')
legend4_3_3=['Roof east', 'Roof west']
plt.legend(legend4_3_3, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house D per surface in summer', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#now we move to spring

for i in surfaces.keys():
    KNMI_weather_spring = (surfaces[i]['ac_power'].loc['2019-04-05']) / 1000 #kW
    surfaces[i]['ac power spring day'] = KNMI_weather_spring

#graph for building A spring
x = KNMI_weather_spring.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_a_fse']['ac power spring day']
y2 = surfaces['building_a_fsw']['ac power spring day']
y3 = surfaces['building_a_roof']['ac power spring day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_a_fse'], color='darkred')
ax.plot(x,y2, label = surfaces['building_a_fsw'], color='orangered')
ax.plot(x,y3, label = surfaces['building_a_roof'], color ='peru')
legend4_3_1=['Facade SE', 'Facade SW', 'Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building A per surfaces in spring', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for building B spring
x = KNMI_weather_spring.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_b_fe']['ac power spring day']
y2 = surfaces['building_b_fs']['ac power spring day']
y3 = surfaces['building_b_fw']['ac power spring day']
y4 = surfaces['building_b_roof']['ac power spring day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_b_fe'], color='darkgreen')
ax.plot(x,y2, label = surfaces['building_b_fs'], color='lime')
ax.plot(x,y3, label = surfaces['building_b_fw'], color='cyan')
ax.plot(x,y4, label = surfaces['building_b_roof'], color='aquamarine')
legend4_3_1=['Facade E', 'Facade S', 'Facade W','Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building B per surface in spring', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house C spring
x = KNMI_weather_spring.index
plt.xticks(fontsize = 10)
y1 = surfaces['house_c_rfn']['ac power spring day']
y2 = surfaces['house_c_rfs']['ac power spring day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_c_rfn'], color='purple')
ax.plot(x,y2, label = surfaces['house_c_rfs'],color='magenta')
legend4_3_2=['Roof north', 'Roof south']
plt.legend(legend4_3_2, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house C per surfaces in spring', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house D spring
x = KNMI_weather_spring.index
plt.xticks(fontsize = 10)
y1 = surfaces['house_d_rfe']['ac power spring day']
y2 = surfaces['house_d_rfw']['ac power spring day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_d_rfe'], color='blue')
ax.plot(x,y2, label = surfaces['house_d_rfw'], color='dodgerblue')
legend4_3_3=['Roof east', 'Roof west']
plt.legend(legend4_3_3, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house D per surfaces in spring', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#now.... same for autumn

for i in surfaces.keys():
    KNMI_weather_autumn = (surfaces[i]['ac_power'].loc['2019-11-07']) / 1000 #kW
    surfaces[i]['ac power autumn day'] = KNMI_weather_autumn


#graph for building A autumn
x = KNMI_weather_autumn.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_a_fse']['ac power autumn day']
y2 = surfaces['building_a_fsw']['ac power autumn day']
y3 = surfaces['building_a_roof']['ac power autumn day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_a_fse'], color='darkred')
ax.plot(x,y2, label = surfaces['building_a_fsw'], color='orangered')
ax.plot(x,y3, label = surfaces['building_a_roof'], color ='peru')
legend4_3_1=['Facade SE', 'Facade SW', 'Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building A per surfaces in autumn', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')


#graph for building B autumn
x = KNMI_weather_autumn.index
plt.xticks(fontsize = 10)
y1 = surfaces['building_b_fe']['ac power autumn day']
y2 = surfaces['building_b_fs']['ac power autumn day']
y3 = surfaces['building_b_fw']['ac power autumn day']
y4 = surfaces['building_b_roof']['ac power autumn day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['building_b_fe'], color='darkgreen')
ax.plot(x,y2, label = surfaces['building_b_fs'], color='lime')
ax.plot(x,y3, label = surfaces['building_b_fw'], color='cyan')
ax.plot(x,y4, label = surfaces['building_b_roof'], color='aquamarine')
legend4_3_1=['Facade E', 'Facade S', 'Facade W','Roof']
plt.legend(legend4_3_1, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of building B per surface in autumn', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house C autumn
x = KNMI_weather_autumn.index
plt.xticks(fontsize = 10)
y1 = surfaces['house_c_rfn']['ac power autumn day']
y2 = surfaces['house_c_rfs']['ac power autumn day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_c_rfn'], color='purple')
ax.plot(x,y2, label = surfaces['house_c_rfs'],color='magenta')
legend4_3_2=['Roof north', 'Roof south']
plt.legend(legend4_3_2, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house C per surface in autumn', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

#graph for house D autumn
x = KNMI_weather_autumn.index
plt.xticks(fontsize = 10)
dpi=400
y1 = surfaces['house_d_rfe']['ac power autumn day']
y2 = surfaces['house_d_rfw']['ac power autumn day']
fig, ax = plt.subplots()
ax.plot(x,y1, label = surfaces['house_d_rfe'], color='blue')
ax.plot(x,y2, label = surfaces['house_d_rfw'], color='dodgerblue')
legend4_3_3=['Roof east', 'Roof west']
plt.legend(legend4_3_3, loc=1)
plt.xticks(rotation=30)
plt.title('Hourly AC output of house D per surface in autumn', fontsize=14)
plt.ylabel('AC Output (kW)')
plt.xlabel('Month, Day, Hour')

