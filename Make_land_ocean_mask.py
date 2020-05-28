# -*- coding: utf-8 -*-
"""
@author: Nadia Bloemendaal, nadia.bloemendaal@vu.nl

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This part is needed for the creation of the land-ocean mask. This needs to be run in Python 2.7!!!

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""


from mpl_toolkits.basemap import maskoceans
import numpy as np
import preprocessing
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def create_mask(lon0,lon1,lat0,lat1,stepsize):
    """
    Create land-sea mask

    Parameters
    ----------
    lon0 : lower left corner longitude.
    lon1 : upper right corner longitude
    lat0 : lower left corner latitude.
    lat1 : upper right corner latitude.
    stepsize : 1/stepsize = degrees.

    Returns
    -------
    mdata : mask.
    x : longitudinal length of mask.
    y : latitudinal length of mask.

    """
    x=int(abs(lon1-(lon0-0.1))*stepsize)
    y=int(abs(lat1-(lat0-0.1))*stepsize)
    if lon0<180: #south pacific
        xmask,ymask=np.mgrid[lon0:lon1:complex(0,x),lat0:lat1:complex(0,y)]
    else:
        xmask,ymask=np.mgrid[lon0-360:lon1-360:complex(0,x),lat0:lat1:complex(0,y)]
    data=np.ones((x,y))
    mdata=maskoceans(xmask,ymask,data,resolution='h',grid=1.25,inlands=False)
        
    return mdata,x,y

def Generate_landmask(idx):
    """
    Generate the land-ocean mask    

    Parameters
    ----------
    idx : basin index (0=EP 1=NA 2=NI 3=SI 4=SP 5=WP).

    Returns
    -------
    saves land-ocean mask

    """
    name,lat0,lat1,lon0,lon1=preprocessing.BOUNDARIES_BASINS(idx)
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
                    
        #Part 2: from 180 till lon1            
        mdata,x,y=create_mask(180.1,lon1,lat0,lat1,stepsize)
        for i in range(0,x):
            for j in range(0,y):
                if mdata[i,j]>-0.1:
                    ocean_mask[j,i+x0]=mdata[i,j]

    else:
        mdata,x,y=create_mask(lon0,lon1,lat0,lat1,stepsize)
        ocean_mask=np.zeros((y,x))
        for i in range(0,x): #rows
            for j in range(0,y): #columns
                if mdata[i,j]>-0.1:
                    ocean_mask[j,i]=mdata[i,j]
                    
    ocean_mask=np.flipud(ocean_mask)
    np.savetxt(os.path.join(__location__,'Land_ocean_mask_'+str(name)+'.txt'),ocean_mask)

for idx in range(0,6):
    Generate_landmask(idx)
                    