# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:45:09 2020

@author: Weather Radar Team
"""

from datetime import datetime
from mpl_toolkits.basemap import Basemap 
import matplotlib.pyplot as plt
import wradlib as wrl
import numpy as np
import warnings, os, math
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/YOG201711271250.mvol'
f = wrl.util.get_wradlib_data_file(radarFile)
data, metadata = wrl.io.read_gamic_hdf5(f)

radarLon=float(metadata['VOL']['Longitude'])
radarLat=float(metadata['VOL']['Latitude'])
radarAlt=float(metadata['VOL']['Height'])
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

nElevation=len(data)
for i in range(nElevation):
    sweep='SCAN'+str(i)
    timeEnd=datetime.strptime(str(metadata[sweep]['Time']),"b'%Y-%m-%dT%H:%M:%S.%fZ'")
    elevation=float('{0:.1f}'.format(metadata[sweep]['elevation'])) # ekstrak data elevasi
    print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
    
    azi=metadata[sweep]['az'] # mengekstrak data azimuth disetiap elevasi
    r=metadata[sweep]['r'] # mengekstrak data range disetiap elevasi
    sweep_data=data[sweep]['Z']['data'] # mengekstrak data radar
    
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
        radius, center, sweep_data.ravel(),
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
plt.savefig('BARONTest.png',bbox_inches='tight',dpi=200,pad_inches=0.1)
plt.close()
