# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:59:47 2020

@author: Weather Radar Team
"""

from datetime import datetime
from mpl_toolkits.basemap import Basemap 
import matplotlib.pyplot as plt
import wradlib as wrl
import numpy as np
import warnings, math
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/1013SOR-20200620-122001-PPIVol.nc'
f = wrl.util.get_wradlib_data_file(radarFile)
raw = wrl.io.read_generic_netcdf(f)

radarLon = float(raw['variables']['longitude']['data'])
radarLat = float(raw['variables']['latitude']['data'])
radarAlt = float(raw['variables']['altitude']['data'])
sitecoords=(radarLon,radarLat,radarAlt)

res=250. # resolusi data yang diinginkan dalam meter
resCoords=res/111229. # resolusi data dalam derajat
rmax=250000./111229. # range maksimum
lonMax,lonMin=radarLon+(rmax),radarLon-(rmax) 
latMax,latMin=radarLat+(rmax),radarLat-(rmax)
nGrid=int(np.floor((lonMax-lonMin)/resCoords))+1 # jumlah grid
lonGrid=np.linspace(lonMin,lonMax,nGrid) # grid longitude
latGrid=np.linspace(latMin,latMax,nGrid) # grid latitude           
dataContainer = np.zeros((len(lonGrid),len(latGrid))) # penampung data
  
# ekstrak waktu radar (end of observation)
try:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%SZ")
except:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%S.%fZ") 

# define flag option untuk melihat apakah gates vary atau tidak
sweep_start_idx = raw['variables']['sweep_start_ray_index']['data']
sweep_end_idx = raw['variables']['sweep_end_ray_index']['data']
try:
    if raw['gates_vary']=='true':
        ray_n_gates=raw['variables']['ray_n_gates']['data']
        ray_start_index=raw['variables']['ray_start_index']['data']
        flag='true'
    elif raw['gates_vary']=='false':
        flag='false'
except :
    if raw['n_gates_vary']=='true':
        ray_n_gates=raw['variables']['ray_n_gates']['data']
        ray_start_index=raw['variables']['ray_start_index']['data']
        flag='true'
    elif raw['n_gates_vary']=='false':
        flag='false'

nElevation=np.size(raw['variables']['fixed_angle']['data'])
for i in range(nElevation):
    elevation=float('{0:.1f}'.format(raw['variables']['fixed_angle']['data'][i]))
    print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
    
    # ekstrak azimuth data
    azi = raw['variables']['azimuth']['data'][sweep_start_idx[i]:sweep_end_idx[i]]   
    
    # ekstrak range data dan radar (dBZ) data berdasarkan nilai flag
    r_all = raw['variables']['range']['data'] 
    if flag == 'false':
        data = raw['variables']['DBZH']['data'][sweep_start_idx[i]:sweep_end_idx[i], :]
        r = r_all    
    else:              
        data = np.array([])
        n_azi = sweep_end_idx[i]-sweep_start_idx[i]        
        try:
            for ll in range(sweep_start_idx[i],sweep_end_idx[i]):
                data = np.append(data,raw['variables']['DBZH']['data'][ray_start_index[ll]:ray_start_index[ll+1]])
            data = data.reshape((n_azi,ray_n_gates[sweep_start_idx[i]]))
        except:
            pass
        r = r_all[0:ray_n_gates[sweep_start_idx[i]]]
    
    # transformasi dari koordinat bola ke koordinat kartesian
    rangeMesh, azimuthMesh =np.meshgrid(r,azi) # meshgrid azimuth dan range
    lonlatalt = wrl.georef.polar.spherical_to_proj(
        rangeMesh, azimuthMesh, elevation, sitecoords
    ) 
    x, y = lonlatalt[:, :, 0], lonlatalt[:, :, 1]
    
    # proses regriding ke data container yang sudah dibuat sebelumnya
    lonMesh, latMesh=np.meshgrid(lonGrid,latGrid)
    gridLatLon = np.vstack((lonMesh.ravel(), latMesh.ravel())).transpose()
    xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
    radius=r[np.size(r)-1]
    center=[x.mean(),y.mean()]
    gridded = wrl.comp.togrid(
        xy, gridLatLon,
        radius, center, data.ravel(),
        wrl.ipol.Linear
    )
    griddedData = np.ma.masked_invalid(gridded).reshape((len(lonGrid), len(latGrid)))
    dataContainer=np.dstack((dataContainer,griddedData))

# plotting CMAX data menggunakan Basemap
cmaxData=np.nanmax(dataContainer[:,:,:],axis=2)
cmaxData[cmaxData<0]=np.nan;cmaxData[cmaxData>100]=np.nan

m=Basemap(llcrnrlat=latMin,urcrnrlat=latMax,\
            llcrnrlon=lonMin,urcrnrlon=lonMax,\
            resolution='i')
x0,y0=m(radarLon,radarLat)
x1,y1=m(lonMesh,latMesh)
clevsZ = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
colors=['#07FEF6','#0096FF','#0002FE','#01FE03','#00C703','#009902','#FFFE00','#FFC801','#FF7707','#FB0103','#C90002','#980001','#FF00FF','#9800FE']
plt.figure(figsize=(10,10))
m.plot(x0, y0, 'ko', markersize=3)
m.contourf(x1, y1, cmaxData,clevsZ,colors=colors)
m.colorbar(ticks=clevsZ,location='right',size='4%',label='Reflectivity (dBZ)')
m.drawparallels(np.arange(math.floor(latMin),math.ceil(latMax),1.5),labels=[1,0,0,0],linewidth=0.2)
m.drawmeridians(np.arange(math.floor(lonMin),math.ceil(lonMax),1.5),labels=[0,0,0,1],linewidth=0.2)
m.drawcoastlines() 
title='CMAX '+timeEnd.strftime("%Y%m%d-%H%M")+' UTC'
plt.title(title,weight='bold',fontsize=15)
plt.savefig('EECTest.png',bbox_inches='tight',dpi=200,pad_inches=0.1)
plt.close()

    
