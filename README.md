# STORM-preprocessing
These are the scripts to conduct the data-preprocessing for the STORM model. Input data is taken from IBTrACS, see https://www.ncdc.noaa.gov/ibtracs/. 
Please reach out if you have any questions!

master_preprocessing.py is the python script from which the other modules are run. Please run the scripts as follows:

1. preprocessing.py
2. coefficients.py
3. environmental.py
4. Make_land_ocean_mask.py <-- in Python 2.7 (uses Basemap), this file stores a .txt file that can be loaded in Python 3.x
5. genesis_matrix.py
6. master_preprocessing.py
