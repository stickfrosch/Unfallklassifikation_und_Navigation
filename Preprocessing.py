import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import osmnx as ox
import networkx as nx
import folium
import webbrowser
import datetime

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

############### Lese CSV Dateien ein #######################
dataset2016 = pd.read_csv("Unfallorte2016_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'])
dataset2016 = dataset2016.drop(['LINREFX','LINREFY'], axis=1)
#print(dataset2016.head())
dataset2017 = pd.read_csv("Unfallorte2017_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','UIDENTSTLA','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','LICHT','IstRad','IstPKW','IstFuss','IstKrad','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'], low_memory=False)
dataset2017 = dataset2017.drop(['UIDENTSTLA','LINREFX','LINREFY'], axis=1)
dataset2017 = dataset2017.rename(columns={"LICHT": "ULICHTVERH"})
#print(dataset2017.head())
dataset2018 = pd.read_csv("Unfallorte2018_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID_1','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'])
dataset2018 = dataset2018.drop(['LINREFX','LINREFY'], axis=1)
dataset2018 = dataset2018.rename(columns={"OBJECTID_1": "OBJECTID"})
#print(dataset2018.head())
dataset2019 = pd.read_csv("Unfallorte2019_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84','STRZUSTAND'])
dataset2019 = dataset2019.drop(['LINREFX','LINREFY'], axis=1)
#print(dataset2019.head())
dataset2020 = pd.read_csv("Unfallorte2020_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','UIDENTSTLAE','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84','STRZUSTAND'])
dataset2020 = dataset2020.drop(['UIDENTSTLAE','LINREFX','LINREFY'], axis=1)
#print(dataset2020.head())

weather = pd.read_csv("weather_data.csv", delimiter=',')
print(weather.dtypes)
weather["date"] = pd.to_datetime(weather["date"])
weather["weekday"] = weather["date"].dt.dayofweek
weather["month"] = weather["date"].dt.month
weather["year"] = weather["date"].dt.year
weather["weekday"] = weather["weekday"] + 2
weather.loc[weather["weekday"] == 8, "weekday"] = 1
weather = weather.groupby(["year", "month", "weekday"]).mean().reset_index()
print(weather)

dataset_merged = pd.concat([dataset2016, dataset2017, dataset2018, dataset2019, dataset2020], sort=False)
dataset_merged = dataset_merged.reset_index(drop=True)

dataset_merged = dataset_merged.merge(weather, left_on=["UJAHR", "UMONAT", "UWOCHENTAG"], right_on=["year", "month", "weekday"], how="left")

for col in ['XGCSWGS84', 'YGCSWGS84']:
    dataset_merged[col] = pd.to_numeric(dataset_merged[col].apply(lambda x: re.sub(',', '.', str(x))))

dataset_merged['FullAGS'] = dataset_merged['ULAND'].astype(str).str.zfill(2) + dataset_merged['UREGBEZ'].astype(str).str.zfill(1) + dataset_merged['UKREIS'].astype(str).str.zfill(2) + dataset_merged['UGEMEINDE'].astype(str).str.zfill(3)

augsburg = dataset_merged[dataset_merged.FullAGS.str.startswith('09761')]
augsburg = augsburg.drop_duplicates(subset=['OBJECTID'])
print(augsburg)
augsburg.to_pickle("bayern.pkl")


