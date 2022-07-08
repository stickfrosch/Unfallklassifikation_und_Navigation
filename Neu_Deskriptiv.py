import pandas as pd
import re
import osmnx as ox
import networkx as nx
import folium
import webbrowser

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

############### Lese CSV Dateien ein #######################
dataset2016 = pd.read_csv("Unfallorte2016_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'])
dataset2016 = dataset2016.drop(['LINREFX','LINREFY'], axis=1)
dataset2017 = pd.read_csv("Unfallorte2017_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','UIDENTSTLA','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','LICHT','IstRad','IstPKW','IstFuss','IstKrad','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'])
dataset2017 = dataset2017.drop(['UIDENTSTLA','LINREFX','LINREFY'], axis=1)
dataset2017 = dataset2017.rename(columns={"LICHT": "ULICHTVERH"})
dataset2018 = pd.read_csv("Unfallorte2018_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID_1','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','STRZUSTAND','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84'])
dataset2018 = dataset2018.drop(['LINREFX','LINREFY'], axis=1)
dataset2018 = dataset2018.rename(columns={"OBJECTID_1": "OBJECTID"})
dataset2019 = pd.read_csv("Unfallorte2019_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84','STRZUSTAND'])
dataset2019 = dataset2019.drop(['LINREFX','LINREFY'], axis=1)
dataset2020 = pd.read_csv("Unfallorte2020_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','UIDENTSTLAE','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84','STRZUSTAND'])
dataset2020 = dataset2020.drop(['UIDENTSTLAE','LINREFX','LINREFY'], axis=1)

########################## Preprocessing Gemeindeschlüssel CSV (Gemeindenamen, Einwohneranzahl, Männlich Weiblich, PLZ, Flächengröße) ########################
ags_namen = pd.read_csv('AuszugGemeindeschlüssel.csv', delimiter=';', header=[0], skiprows=1)
ags_namen = ags_namen.drop([0])
ags_namen = ags_namen.rename(columns={"Unnamed: 5": "Gemeindename", "Unnamed: 6": "Flaechekm2", "insgesamt": "Einwohneranzahl", "männlich": "maennlich", "Unnamed: 11": "PLZ"})
ags_namen.Einwohneranzahl = ags_namen.groupby("Gemeindename").Einwohneranzahl.apply(lambda x: x.ffill().bfill())
ags_namen.Flaechekm2 = ags_namen.groupby("Gemeindename").Flaechekm2.apply(lambda x: x.ffill().bfill())
ags_namen.maennlich = ags_namen.groupby("Gemeindename").maennlich.apply(lambda x: x.ffill().bfill())
ags_namen.weiblich = ags_namen.groupby("Gemeindename").weiblich.apply(lambda x: x.ffill().bfill())
ags_namen.PLZ = ags_namen.groupby("Gemeindename").PLZ.apply(lambda x: x.ffill().bfill())
ags_namen = ags_namen.fillna(0)
ags_namen = ags_namen.astype({'RB': 'int64', 'Kreis': 'int64', 'VB': 'int64', 'Gem': 'int64', 'PLZ': 'int64'})
ags_namen = ags_namen.astype({'Land': 'string', 'RB': 'string', 'Kreis': 'string', 'VB': 'string', 'Gem': 'string'})
ags_namen['Land'] = ags_namen['Land'].str.zfill(2)
ags_namen['RB'] = ags_namen['RB'].str.zfill(1)
ags_namen['Kreis'] = ags_namen['Kreis'].str.zfill(2)
ags_namen['VB'] = ags_namen['VB'].str.zfill(4)
ags_namen['Gem'] = ags_namen['Gem'].str.zfill(3)
ags_namen['FullAGS'] = ags_namen['Land'] + ags_namen['RB'] + ags_namen['Kreis'] + ags_namen['Gem']
ags_namen["Gemeindename"] = ags_namen["Gemeindename"].str.replace(", Stadt", "")
ags_namen["Gemeindename"] = ags_namen["Gemeindename"].str.replace(", Landeshauptstadt", "")

############### Merge Datensätze zusammen #######################
dataset_merged = pd.concat([dataset2016, dataset2017, dataset2018, dataset2019, dataset2020], sort=False)
dataset_merged = dataset_merged.reset_index(drop=True)

############### Wandele , zu . um für Koordinaten #######################
for col in ['XGCSWGS84', 'YGCSWGS84']:
    dataset_merged[col] = pd.to_numeric(dataset_merged[col].apply(lambda x: re.sub(',', '.', str(x))))

################ Fülle Nullen für Gemeindeschlüssel auf ########################
dataset_merged['FullAGS'] = dataset_merged['ULAND'].astype(str).str.zfill(2) + dataset_merged['UREGBEZ'].astype(str).str.zfill(1) + dataset_merged['UKREIS'].astype(str).str.zfill(2) + dataset_merged['UGEMEINDE'].astype(str).str.zfill(3)

####################### Merge Unfallatlas Dataframe mit Gemeindeschlüssel Dataframe anhand beider FullAGS Columns ##########################
df = ags_namen.merge(dataset_merged, left_on='FullAGS', right_on='FullAGS')

################ Erstelle Augsburg Dataframe ########################
augsburg = df.loc[df.loc[:,"Land"] == "09", :]
augsburg = augsburg.loc[df.loc[:,"RB"] == "7", :]
augsburg = augsburg.loc[df.loc[:,"Kreis"] == "61", :]
augsburg.to_csv(index=False)
augsburg.to_pickle("augsburg.pkl")

df.dropna(inplace=True)

df['Land'] = df['Land'].astype(int)
df = df.loc[df['Land'] == 9]

