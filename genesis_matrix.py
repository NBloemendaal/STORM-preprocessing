# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This part is to generate the genesis matrix used in the STORM model

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0

!!This script needs to be run in Python 2.7!!
"""

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import maskoceans
from scipy.interpolate import griddata
import preprocessing
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def Dataframe_Genesis_Basin(idx,month):
    lat0,lat1,lon0,lon1=preprocessing.BOUNDARIES_BASINS(idx)    
    
    genesis=np.load(os.path.join(__location__,'GEN_LOC.npy'),allow_pickle=True).item() 
    
    lat_init_list=[genesis[idx][month][i][0] for i in range(len(genesis[idx][month]))]    
    lon_init_list=[genesis[idx][month][i][1] for i in range(len(genesis[idx][month]))]
    
    lat_init_filter=[]
    lon_init_filter=[]
    #filter out all longitude-latitude genesis points inside the basin domain
    for i in range(len(lat_init_list)):
        if lat_init_list[i]<=lat1 and lat_init_list[i]>=lat0 and lon_init_list[i]<=lon1 and lon_init_list[i]>=lon0:
            lat_init_filter.append(lat_init_list[i])
            lon_init_filter.append(lon_init_list[i])
            
    df_GB=pd.DataFrame({'Latitude':lat_init_filter, 'Longitude':lon_init_filter})
        
    return df_GB    

def create_mask(lon0,lon1,lat0,lat1,stepsize):
    x=int(abs(lon1-lon0)*stepsize)
    y=int(abs(lat1-lat0)*stepsize)
    if lon0<180: #south pacific
        xmask,ymask=np.mgrid[lon0:lon1:complex(0,x),lat0:lat1:complex(0,y)]
    else:
        xmask,ymask=np.mgrid[lon0-360:lon1-360:complex(0,x),lat0:lat1:complex(0,y)]
    data=np.ones((x,y))
    mdata=maskoceans(xmask,ymask,data,resolution='h',grid=1.25,inlands=False)
        
    return mdata,x,y

def Makegrid(idx,month):
#==============================================================================
# Split data in bins by using Groupby
#==============================================================================
    df=Dataframe_Genesis_Basin(idx,month) #generate dataframe for genesis counts (step 2)
    step=5. #define grid size of initial data binning
    
    lat0,lat1,lon0,lon1=preprocessing.BOUNDARIES_BASINS(idx) 
    latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)
    lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
    
    to_bin=lambda x:np.floor(x/step)*step
    df["latbin"]=df.Latitude.map(to_bin)
    df["lonbin"]=df.Longitude.map(to_bin)
    groups=df.groupby(("latbin","lonbin"))
    count_df=pd.DataFrame({'count':groups.size()}).reset_index()
    counts=count_df["count"]       
    latbin=groups.count().index.get_level_values('latbin')
    lonbin=groups.count().index.get_level_values('lonbin')
    count_matrix=np.zeros((len(latspace),len(lonspace)))
    
    for lat,lon,count in zip(latbin,lonbin,counts):
        i=latspace.tolist().index(lat)
        j=lonspace.tolist().index(lon)
        count_matrix[i,j]=count
        
#==============================================================================
# Create land-sea mask, to check whether a grid box has over 50% of land
#==============================================================================
    stepsize=10 #0.1 deg
    if lon0<180 and lon1>180: #south pacific
        xall=(lon1-lon0)*stepsize    
        #split the mask up in two parts:
        #Part 1: from lon0 till 180
        mdata,x0,y=create_mask(lon0,179.9,lat0,lat1,stepsize)
        ocean_mask=np.zeros((y,xall))
        for i in range(0,x0):
            for j in range(0,y):
                if mdata[i,j]>-0.1:
                    ocean_mask[j,i]=mdata[i,j]
        ocean_mask[:,x0]=0 
        ocean_mask[:,x0+1]=0                   
        #Part 2: from 180 till lon1            
        mdata,x,y=create_mask(180.1,lon1,lat0,lat1,stepsize)
        for i in range(0,x):
            for j in range(0,y):
                if mdata[i,j]>-0.1:
                    ocean_mask[j,i+x0+1]=mdata[i,j]

    else:
        mdata,x,y=create_mask(lon0,lon1,lat0,lat1,stepsize)
        ocean_mask=np.zeros((y,x))
        for i in range(0,x): #rows
            for j in range(0,y): #columns
                if mdata[i,j]>-0.1:
                    ocean_mask[j,i]=mdata[i,j]
                    
#==============================================================================
# Interpolate genesis locations matrix to a 1x1 degree grid
#==============================================================================
    xg=int(abs(lon1-lon0))
    yg=int(abs(lat1-lat0))
    xgrid,ygrid=np.mgrid[lon0:lon1:complex(0,xg),lat0:lat1:complex(0,yg)]
    points=[]
    for i in range(len(lonspace)):
        for j in range(len(latspace)):
            points.append((lonspace[i],latspace[j]))
     
    values=np.reshape(count_matrix.T,int(len(lonspace))*int(len(latspace)))
    grid=griddata(points,values,(xgrid,ygrid),method='cubic')
    grid=grid.transpose()
    grid=np.flipud(grid)
    stepsize=1 #1 degree
    if lon0<180 and lon1>180:
        #split the mask up in 2 parts:
        #Part 1: from lon0 till 180
        mdata,x0,y=create_mask(lon0,179,lat0,lat1,stepsize)
        mdata=mdata.transpose()
        mdata=np.flipud(mdata)
        for i in range(0,y):
            for j in range(0,x0):
                if mdata[i,j]>-10:
                    grid[i,j]='nan'
                    
        #Part 2: from 180 till lon1
        mdata,x,y=create_mask(181,lon1,lat0,lat1,stepsize)
        mdata=mdata.transpose()
        mdata=np.flipud(mdata)
        for i in range(0,y):
            for j in range(0,x):
                if mdata[i,j]>-10:
                    grid[i,j+x0+2]='nan'
        
    else:
        mdata,x,y=create_mask(lon0,lon1,lat0,lat1,stepsize)
        mdata=mdata.transpose()
        mdata=np.flipud(mdata)
        for i in range(0,y):
            for j in range(0,x):
                if mdata[i,j]>-10:
                    grid[i,j]='nan'
                     
    np.savetxt(os.path.join(__location__,'COUNTS_GENESIS_MATRIX_'+str(idx)+'_'+str(month)+'.txt'),count_matrix)
    np.savetxt(os.path.join(__location__,'GRID_GENESIS_MATRIX_'+str(idx)+'_'+str(month)+'.txt'),grid)
    return grid
   