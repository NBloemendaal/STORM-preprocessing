# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing and calculate the environmental
conditions + wind-pressure relationship.

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
import math
import preprocessing
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def monthly_mean_pressure(data):
    """
    Create the monthly mean MSLP fields. This function outputs a txt-file of a global field of monthly mean MSLP for every month.
    
    Parameters
    ----------
    data : dataset with monthly mean MSLP values for 38 years of data (ERA-5)

    """
    mslp=data.msl.values
    lon=data.longitude.values
    lat=data.latitude.values
    
    for month in range(0,12):
        mean_matrix=np.zeros((len(lat),len(lon)))
        
        for t in range(0,38):    
            #loop over 38 years
            mean_matrix=mean_matrix+mslp[month+t*12,:,:]/100.
            
        mean_matrix=mean_matrix/38.                         
        np.savetxt(os.path.join(__location__,'Monthly_mean_MSLP_'+str(month+1)+'.txt'),mean_matrix)

def monthly_mean_sst(data):
    """
    Create the monthly mean SST fields. This function outputs a txt-file of a global field of monthly mean SSTs for every month.
    
    Parameters
    ----------
    data : dataset with monthly mean SST values for 38 years of data (ERA-5)

    """
    sst=data.sst.values
    lon=data.longitude.values
    lat=data.latitude.values
    
    for month in range(0,12):
        mean_matrix=np.zeros((len(lat),len(lon)))
        
        for t in range(0,38):
            mean_matrix=mean_matrix+sst[month+t*12,:,:]
    
        mean_matrix=mean_matrix/38.
        np.savetxt(os.path.join(__location__,'Monthly_mean_SST_'+str(month+1)+'.txt'),mean_matrix)

def check_season(idx,month):
    """
    Check if TC occurred in TC season. 
    
    Parameters
    ----------
    idx : Basin index (EP=0,NA=1,NI=2,SI=3,SP=4,WP=5)
    month : month in which TC occurred

    Returns
    -------
    check : 0 if TC did not occur in TC season, 1 if TC did occur in TC season. 
    """
    check=0
    if idx==0 or idx==1:
        if month>5 and month<12:
            check=1
    elif idx==2:
        if month>3 and month<7:
            check=1
        elif month>8 and month<12:
            check=1
    elif idx==3 or idx==4:
        if month<5 or month>10:
            check=1
    elif idx==5:
        if month>4 and month<12:
            check=1
    return check

def Vmax_function(DP,A,B):
    """  
    This is the wind-pressure relationship. Here, we calculate the values of the coefficients
    A en B for the wind and pressure found in the dataset. 
    Parameters
    ----------
    DP : Difference between environmental pressure and central pressure (hPa)
    A,B : Coefficients for wind-pressure relationship.
    """
    return A*(DP)**B

def wind_pressure_relationship():
    """
    This function calculates the coefficients for the wind-pressure relationship.
    The wind-pressure relationship is based on the empirical wind-pressure relationship (for overview, see Harper 2002:
        Tropical Cyclone Parameter Estimation in the Australian Region: Wind-Pressure Relationships and 
        Related Issues for Engineering Planning and Design - A Discussion Paper)
        
    Adapted by e.g. Atkinson and Holliday (1977), Love and Murphy (1985) and Crane (1985)

    This script saves the coefficients list for the wind-pressure relationship, per month as an npy-file.
    """
    latlist=np.load(os.path.join(__location__,'LATLIST_INTERP.npy'),allow_pickle=True).item()
    lonlist=np.load(os.path.join(__location__,'LONLIST_INTERP.npy'),allow_pickle=True).item()
    windlist=np.load(os.path.join(__location__,'WINDLIST_INTERP.npy'),allow_pickle=True).item()
    preslist=np.load(os.path.join(__location__,'PRESLIST_INTERP.npy'),allow_pickle=True).item()
    monthlist=np.load(os.path.join(__location__,'MONTHLIST_INTERP.npy'),allow_pickle=True).item()
    basinlist=np.load(os.path.join(__location__,'BASINLIST_INTERP.npy'),allow_pickle=True).item()

    data=xr.open_dataset(os.path.join(__location__,'Monthly_mean_SST.nc'))
    
    lon=data.longitude.values
    lat=data.latitude.values
    data.close()

    pres_basin={i:[] for i in range(0,6)}
    wind_basin={i:[] for i in range(0,6)}
    month_basin={i:[] for i in range(0,6)}
    
    for i in range(len(latlist)):
        if len(latlist[i])>0:            
            idx=basinlist[i][0]
            month=monthlist[i][0]            
            check=check_season(idx,month) 
            print(idx,month,check)
            if check==1:
                MSLP=np.loadtxt(os.path.join(__location__,'Monthly_mean_MSLP_'+str(month)+'.txt'))                
                for j in range(0,len(latlist[i])):
                    #Wind needs to be greater than 15 kt.                         
                        latn=np.abs(lat-latlist[i][j]).argmin()
                        lonn=np.abs(lon-lonlist[i][j]).argmin()                           
                        if preslist[i][j]>0 and MSLP[latn][lonn]-preslist[i][j]>0 and windlist[i][j]>15.*0.5144444:  
                            pres_basin[idx].append(MSLP[latn][lonn]-preslist[i][j])
                            wind_basin[idx].append(windlist[i][j])
                            month_basin[idx].append(month)
      
    coeff_list={i:[] for i in range(0,6)}    
    months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    
    for idx in range(0,6):        
        coeff_list[idx]={i:[] for i in months[idx]}    
        df=pd.DataFrame({"Wind":wind_basin[idx],"Pressure":pres_basin[idx],"Month":month_basin[idx]})        
        for i in range(len(months[idx])):
            m=months[idx][i]
            df1=df[(df["Month"]==m)]    
            step=2. #Group in 2 m/s bins to eliminate the effect of more weaker storms (that might skew the fit)
            to_bin=lambda x:np.floor(x/step)*step            
            df1["windbin"]=df1.Wind.map(to_bin)                
            minpres=df1.groupby(["windbin"]).agg({"Pressure":"mean"})["Pressure"]   
            maxwind=np.unique(df1["windbin"])  
            
            try:     
                opt,l=curve_fit(Vmax_function,minpres,maxwind,maxfev=5000)
                [a,b]=opt
                coeff_list[idx][m]=[a,b]
        
            except RuntimeError:
                print('Optimal parameters not found')
            except TypeError:
                print('Too few items')
            
    np.save(os.path.join(__location__,'COEFFICIENTS_WPR_PER_MONTH.npy'),coeff_list) 

def MPI_function(T,A,B,C):
    """
    Fit the MPI function to the data. This function returns the optimal coefficients.
    Parameters
    ----------
    T : Sea-surface temperature in Celcius.
    A,B,C : coefficients 

    """
    return A+B*np.exp(C*(T-30.))

def Calculate_P(V,Penv,a,b):
    """
    Convert Vmax to Pressure following the empirical wind-pressure relationship (Harper 2002, Atkinson and Holliday 1977)
    
    Input: 
        Vmax: 10-min mean maximum wind speed in m/s
        Penv: environmental pressure (hPa)
        a,b: coefficients. See Atkinson_Holliday_wind_pressure_relationship.py
    
    Returns:
        Pc: central pressure in the eye
    
    """
    
    Pc=Penv-(V/a)**(1./b)  
    return Pc

def calculate_MPI_fields():  
    """
    Calculate the MPI fields from the pressure drop and environmental conditions.
    """
    # =============================================================================
    # Calculate the MPI and SST - NOTE: THIS PART TAKES VERY LOOONG
    # =============================================================================
    data=xr.open_dataset(os.path.join(__location__,'Monthly_mean_SST.nc'))
     
    lon=data.longitude.values
    lat=data.latitude.values
    data.close()
    latlist=np.load(os.path.join(__location__,'LATLIST_INTERP.npy')).item()
    lonlist=np.load(os.path.join(__location__,'LONLIST_INTERP.npy')).item()
    monthlist=np.load(os.path.join(__location__,'MONTHLIST_INTERP.npy')).item()
    basinlist=np.load(os.path.join(__location__,'BASINLIST_INTERP.npy')).item()
    preslist=np.load(os.path.join(__location__,'PRESLIST_INTERP.npy')).item()

    sst_list={i:[] for i in range(0,6)}
    month_list={i:[] for i in range(0,6)}
    intensity_list={i:[] for i in range(0,6)}
    pressure_drop_list={i:[] for i in range(0,6)}
    
    MSLP_field_all={i:[] for i in range(1,13)}
    SST_field_all={i:[] for i in range(1,13)}
    
    for month in range(1,13):
        MSLP_field_all[month]=np.loadtxt(os.path.join(__location__,'Monthly_mean_MSLP_'+str(month)+'.txt'))
        SST_field_all[month]=np.loadtxt(os.path.join(__location__,'Monthly_mean_SST_'+str(month)+'.txt'))
    
    for i in range(len(latlist)):
        if len(preslist[i])>0:
            idx=basinlist[i][0]
            month=monthlist[i][0]
            
            SST_field=SST_field_all[month]
            MSLP_field=MSLP_field_all[month]
            
            for j in range(len(preslist[i])):
                lat_index=np.abs(lat-latlist[i][j]).argmin()
                lon_index=np.abs(lon-lonlist[i][j]).argmin()
    
                if SST_field[lat_index,lon_index]>288.15 and preslist[i][j]>0: #only use SST>15C for the fit.
                    sst_list[idx].append(SST_field[lat_index,lon_index]-273.15)
                    intensity_list[idx].append(preslist[i][j])
                    pressure_drop_list[idx].append(MSLP_field[lat_index,lon_index]-preslist[i][j])
                    month_list[idx].append(month)
    #=============================================================================
    #Calculate the MPI coefficients (see DeMaria & Kaplan 1994)
    #=============================================================================
    basins=['EP','NA','NI','SI','SP','WP']
    coeflist={i:[] for i in range(0,6)}
    #Only consider those in the hurricane season
    months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    months_for_coef=[[6,7,8,9,10,10],[6,7,8,9,10,11],[6,6,6,10,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    for idx in range(0,6):
        
        coeflist[idx]={i:[] for i in months[idx]}
        
        df=pd.DataFrame({'Drop':pressure_drop_list[idx],'SST':sst_list[idx],'Month':month_list[idx]})
        
        df=df[df["Drop"]>-99999.]
        
        for i in range(len(months[idx])):
            m=months_for_coef[idx][i]
            mc=months[idx][i]
            print(idx,mc)
            if idx==2 and m<7.:
                df1=df[(df["Month"]==4) | (df["Month"]==5) | (df["Month"]==6)]
            
            elif idx==2 and m>7.:
                df1=df[(df["Month"]==9) | (df["Month"]==10) | (df["Month"]==11)]
            
            elif m>10 and idx==3 or idx==4:
                df1=df[(df['Month']==11) | (df['Month']==12)]
            
            elif m<5 and idx==3 or idx==4:
                df1=df[(df['Month']==1) |(df['Month']==2) | (df['Month']==3)| (df['Month']==4) ]             
            
            else:            
                df1=df[(df["Month"]==m)]
    
            df1=df1[(df1["SST"]<30.)]
    
            step=1.0
            to_bin=lambda x:np.floor(x/step)*step
            df1["sstbin"]=df1.SST.map(to_bin)
            
              
            droplist=df1.groupby(["sstbin"]).agg({"Drop":"max"})["Drop"]   
            sstlist=df1.groupby(["sstbin"]).agg({"SST":"mean"})["SST"]       
            
            try:         
                opt,l=curve_fit(MPI_function,sstlist,droplist,maxfev=5000)
                [a,b,c]=opt
                coeflist[idx][mc]=[a,b,c] 
            except RuntimeError:
                print('Optimal parameters not found for '+str(basins[idx]))
            except TypeError:
                print('Too few items')
                
    np.save(os.path.join(__location__,'COEFFICIENTS_MPI_PRESSURE_DROP_MONTH.npy'),coeflist)
    # =============================================================================
    #  Calculate the new MPI in hPa         
    # =============================================================================
    months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    #these are the lowest mpi values per basin and serve as the lower bound, derived from Bister & Emanuel 2002
    mpi_bounds=[[860,880,900,900,880,860],[920,900,900,900,880,880],[840,860,880,900,880,860],[840,880,860,860,840,860],[840,840,860,860,840,840],[860,860,860,870,870,860,860]]
    
    for idx in range(0,6):
        for m,midx in zip(months[idx],range(len(months[idx]))):
            [A,B,C]=coeflist[idx][m]    
        
            SST=SST_field_all[m]
            MSLP=MSLP_field_all[m]
            
            lat0,lat1,lon0,lon1=preprocessing.BOUNDARIES_BASINS(idx)
    
            lat_0=np.abs(lat-lat1).argmin()
            lat_1=np.abs(lat-lat0).argmin()
            lon_0=np.abs(lon-lon0).argmin()
            lon_1=np.abs(lon-lon1).argmin()
            
            SST_field=SST[lat_0:lat_1,lon_0:lon_1]
            MSLP_field=MSLP[lat_0:lat_1,lon_0:lon_1]
            PC_MATRIX=np.zeros((SST_field.shape))
            PC_MATRIX[:]=np.nan
    
            PRESDROP=MPI_function(SST_field-273.15,A,B,C) #Vmax is given in m/s
            PC_MATRIX=MSLP_field-PRESDROP
            boundary=mpi_bounds[idx][midx]
            
            PC_MATRIX[PC_MATRIX<boundary]=boundary
    
            np.savetxt(os.path.join(__location__,'MPI_FIELDS_'+str(idx)+str(m)+'.txt'),PC_MATRIX)

def PRESFUNCTION(X,a,b,c,d):
    """
    Fit the data to the pressure function. 
    Parameters
    ----------
    X : array of change in pressure and difference between pressure and mpi ([dp0,p-mpi])
    a,b,c,d : Coefficients

    """
    dp,presmpi=X
    return a+b*dp+c*np.exp(-d*presmpi)

def PRESEXPECTED(dp,presmpi,a,b,c,d):
    """
    Calculate the forward change in pressure (dp1, p[i+1]-p[i])    

    Parameters
    ----------
    dp : backward change in pressure (dp0, p[i]-p[i-1])
    presmpi : difference between pressure and mpi (p-mpi).
    a,b,c,d : coefficients

    Returns
    -------
    dp1_list : array of forward change in pressure (dp1, p[i+1]-p[i])

    """
    dp1_list=[]
    for k in range(len(dp)):
        dp1_list.append(a+b*dp[k]+c*np.exp(-d*presmpi[k]))
    return dp1_list

def pressure_coefficients():
    """
    Calculate the pressure coefficients
    """
    data=xr.open_dataset(os.path.join(__location__,'Monthly_mean_SST.nc'))
    
    lon=data.longitude.values
    lat=data.latitude.values
    data.close()
    step=5
    pres_variables=np.load(os.path.join(__location__,'TC_PRESSURE_VARIABLES.npy'),allow_pickle=True).item()

    coeflist={i:[] for i in range(0,6)}
    
    months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,10,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    
    months_for_coef=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
        
    for idx in range(0,6):
        coeflist[idx]={i:[] for i in months_for_coef[idx]}
        
        lat0,lat1,lon0,lon11=preprocessing.BOUNDARIES_BASINS(idx) 
        
        lat_0=np.abs(lat-lat1).argmin()
        lon_0=np.abs(lon-lon0).argmin()
        
        for i in range(len(months[idx])):
            m=months[idx][i]
            print(idx,m)
            
            m_coef=months_for_coef[idx][i]
    
            MPI_MATRIX=np.loadtxt(os.path.join(__location__,'MPI_FIELDS_'+str(idx)+str(m)+'.txt'))
        
            lat_df,lon_df,mpi_df=[],[],[]
            
            for i in range(len(MPI_MATRIX[:,0])):
                for j in range(len(MPI_MATRIX[0,:])):
                    lat_df.append(lat[i+lat_0])
                    lon_df.append(lon[j+lon_0])
                    mpi_df.append(MPI_MATRIX[i,j])
                               
            df=pd.DataFrame({'Latitude':lat_df,'Longitude':lon_df,'MPI':mpi_df})
            to_bin=lambda x:np.floor(x/step)*step
            df["latbin"]=df.Latitude.map(to_bin)
            df["lonbin"]=df.Longitude.map(to_bin)
            MPI=df.groupby(["latbin","lonbin"])['MPI'].apply(list)  
            
            latbins1=np.linspace(lat0,lat1-5,(lat1-5-lat0)//step+1)
            lonbins1=np.linspace(lon0,lon11-5,(lon11-5-lon0)//step+1)
            
            matrix_mpi=-100*np.ones((int((lat1-lat0)/5),int((lon11-lon0)/5)))
            for latidx in latbins1:
                for lonidx in lonbins1:
                    i_ind=int((latidx-lat0)/5.)
                    j_ind=int((lonidx-lon0)/5.)
                    matrix_mpi[i_ind,j_ind]=np.nanmin(MPI[latidx][lonidx])
                    
            if idx==1:
                matrix_mpi=np.c_[matrix_mpi,matrix_mpi[:,-1]]        
                    
            df_data=pd.DataFrame({'Latitude':pres_variables[3][idx],'Longitude':pres_variables[4][idx],'Pressure':pres_variables[2][idx],'DP0':pres_variables[0][idx],'DP1':pres_variables[1][idx],'Month':pres_variables[5][idx]})
            df_data=df_data[(df_data['Pressure']>0.) & (df_data['DP0']>-10000.) & (df_data['DP1']>-10000.) & (df_data['Longitude']>=lon0) &(df_data['Longitude']<lon11) & (df_data["Latitude"]>=lat0) & (df_data["Latitude"]<lat1)]
            df_data1=df_data[df_data["Month"]==m]
            
            df_data1["latbin"]=df_data1.Latitude.map(to_bin)
            df_data1["lonbin"]=df_data1.Longitude.map(to_bin)    
        
            latbins=np.unique(df_data1["latbin"])
            lonbins=df_data1.groupby("latbin")["lonbin"].apply(list)
            Pressure=df_data1.groupby(["latbin","lonbin"])["Pressure"].apply(list)
            DP1=df_data1.groupby(["latbin","lonbin"])['DP1'].apply(list)
            DP0=df_data1.groupby(["latbin","lonbin"])['DP0'].apply(list) 
            
            if idx==1:
                lon1=lon11+5
            else:
                lon1=lon11
        
            matrix_mean=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_std=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c0=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c1=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c2=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c3=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
        
            count=0
            lijst=[]
            for latidx in latbins:
                lonlist=np.unique(lonbins[latidx])
                for lonidx in lonlist:
                    lijst.append((latidx,lonidx))
                    
            for latidx in latbins:
                lonlist=np.unique(lonbins[latidx])
                for lonidx in lonlist:            
                    i_ind=int((latidx-lat0)/5.)
                    j_ind=int((lonidx-lon0)/5.)
                    preslist=[]
                    dp0list=[]
                    dp1list=[]
                    mpi=[]
                    #include all bins from lat-5 to lat+5 and lon-5 to lon+5
                    for lat_sur in [-5,0,5]:
                        for lon_sur in [-5,0,5]:
                            if (int(latidx+lat_sur),int(lonidx+lon_sur)) in lijst:
                                if np.nanmin(MPI[latidx+lat_sur][lonidx+lon_sur])>0.:
                                    for pr,d0,d1 in zip(Pressure[latidx+lat_sur][lonidx+lon_sur],DP0[latidx+lat_sur][lonidx+lon_sur],DP1[latidx+lat_sur][lonidx+lon_sur]):
                                        preslist.append(pr)
                                        dp0list.append(d0)
                                        dp1list.append(d1)
                                        mpi.append(np.nanmin(MPI[latidx+lat_sur][lonidx+lon_sur]))
                                    
                    if len(preslist)>9.:
                        presmpi_list=[]
                        for y in range(len(preslist)):
                            if preslist[y]<mpi[y]:
                                presmpi_list.append(0)
                            else:
                                presmpi_list.append(preslist[y]-mpi[y])
                                
                        X=[dp0list,presmpi_list]
                        try:
                            opt,l=curve_fit(PRESFUNCTION,X,dp1list,p0=[0,0,0,0],maxfev=5000)
                            [c0,c1,c2,c3]=opt
                            expected=PRESEXPECTED(dp1list,presmpi_list,c0,c1,c2,c3)
                            Epres=[]
                            for ind in range(len(expected)):
                                Epres.append(expected[ind]-dp0list[ind])
                                
                            mu,std=norm.fit(Epres)
                            if abs(mu)<1 and c2>0: #otherwise: the fit didn't go as planned: large deviation from expected values..
                                matrix_mean[i_ind,j_ind]=mu
                                matrix_std[i_ind,j_ind]=std
                                matrix_c0[i_ind,j_ind]=c0
                                matrix_c1[i_ind,j_ind]=c1
                                matrix_c2[i_ind,j_ind]=c2
                                matrix_c3[i_ind,j_ind]=c3
                        except RuntimeError:
                            count=count+1
            print (str(count)+' fields out of '+str(len(latbins1)*len(lonbins1))+' bins do not have a fit')
            
            
            (X,Y)=matrix_mean.shape
            neighbors=lambda x, y : [(x2, y2) for (x2,y2) in [(x,y-1),(x,y+1),(x+1,y),(x-1,y),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]
                                       if (-1 < x < X and
                                           -1 < y < Y and
                                           (x != x2 or y != y2) and
                                           (0 <= x2 < X) and
                                           (0 <= y2 < Y))]
            var=100
            while var!=0:
                shadowmatrix=np.zeros((X,Y))
                zeroeslist=[[i1,j1] for i1,x in enumerate(matrix_mean) for j1,y in enumerate(x) if y==-100]
                var=len(zeroeslist)
                for [i,j] in zeroeslist:       
                        lijst=neighbors(i,j)
                        for item in lijst:
                            (i0,j0)=item
                            if matrix_mean[i0,j0]!=-100 and shadowmatrix[i0,j0]==0:
                                matrix_mean[i,j]=matrix_mean[i0,j0]
                                matrix_std[i,j]=matrix_std[i0,j0]
                                matrix_c0[i,j]=matrix_c0[i0,j0]
                                matrix_c1[i,j]=matrix_c1[i0,j0]
                                matrix_c2[i,j]=matrix_c2[i0,j0]
                                matrix_c3[i,j]=matrix_c3[i0,j0]
                                shadowmatrix[i,j]=1                     
                                break
            
            print('Filling succeeded')                 
            var=100
            (X,Y)=matrix_mpi.shape
            while var!=0:
                shadowmatrix=np.zeros((X,Y))
                zeroeslist=[[i1,j1] for i1,x in enumerate(matrix_mpi) for j1,y in enumerate(x) if math.isnan(y)]
                var=len(zeroeslist)
                for [i,j] in zeroeslist:       
                        lijst=neighbors(i,j)
                        for item in lijst:
                            (i0,j0)=item
                            if math.isnan(matrix_mpi[i0,j0])==False and shadowmatrix[i0,j0]==0:
                                matrix_mpi[i,j]=matrix_mpi[i0,j0]
                                shadowmatrix[i,j]=1                     
                                break
        
            print(np.mean(matrix_c0),np.mean(matrix_c1),np.mean(matrix_c2),np.mean(matrix_c3))
                     
            for i in range(0,X):
                for j in range(0,Y):
                    coeflist[idx][m_coef].append([matrix_c0[i,j],matrix_c1[i,j],matrix_c2[i,j],matrix_c3[i,j],matrix_mean[i,j],matrix_std[i,j],matrix_mpi[i,j]])
            
        np.save(os.path.join(__location__,'COEFFICIENTS_JM_PRESSURE.npy'),coeflist)      
