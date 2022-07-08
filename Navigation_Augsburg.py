import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import osmnx as ox
import networkx as nx
import webbrowser
import geopandas
#import geopy
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

########## Lese Augsburg Unfälle ein #############
augsburg = pd.read_pickle("augsburg.pkl")

########## Nutze Nominatim Geocoding Serivce um Adresse in Lat und Lon umzurechnen #############
locator = Nominatim(user_agent="myGeocoder")
location = locator.geocode("Rathaus, Augsburg, Germany")
endlocation = locator.geocode("Bürgeramt, Augsburg, Germany")

while True:
    var_mobility_input = input("Geben Sie bitte 1 für Auto, 2 für zu Fuß oder 3 für Fahrrad als ihr Fortbewegungsmittel ein")

    if var_mobility_input == "1":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='drive')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_drive"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_drive'].isin(route2)]

    elif var_mobility_input == "2":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='walk')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_walk"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_walk'].isin(route2)]

    elif var_mobility_input == "3":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='bike')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_bike"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_bike'].isin(route2)]

    else:
        print("Invalid Input. Bitte nochmal eingeben")
        continue
    break


Lon = unfaelle_auf_route['XGCSWGS84'].values[:]
Lat = unfaelle_auf_route['YGCSWGS84'].values[:]
ids = np.array(unfaelle_auf_route.loc[:,'UART'])
#print(unfaelle_auf_route)
#unfaelle_auf_route.to_pickle("unfaelle_auf_route.pkl")


Version4 = ['UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG',
            'ULICHTVERH', 'IstRad', 'IstPKW', 'IstFuss', 'IstKrad',
            'IstGkfz', 'IstSonstige', 'STRZUSTAND']

Version5 = ['UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG',
            'ULICHTVERH', 'IstRad', 'IstPKW', 'IstFuss', 'IstKrad',
            'IstGkfz', 'IstSonstige', 'STRZUSTAND', 'tavg',
            'tmin', 'tmax', 'prcp']


# erhält die Anzahl der Datenpunkte für die Unfaelle auf der Route und für die Unfälle in Augsburg
rownumber_unfaelle_auf_route_df = unfaelle_auf_route_df.shape[0]
rownumber_unfallatlas_df = augsburg.shape[0]

# Berechnet, um wie viel die Unfaelle auf der Route vermehrfacht werden müssen, um 10% auszumachen
Multiplikator = (0.1 * rownumber_unfallatlas_df) / rownumber_unfaelle_auf_route_df
Multiplikator = round(Multiplikator)

# Vermehrfacht die Unfaelle auf der Route
unfaelle_auf_route_df_final = pd.DataFrame(np.repeat(unfaelle_auf_route_df.values, Multiplikator, axis=0))
unfaelle_auf_route_df_final.columns = unfaelle_auf_route_df.columns
#print(unfaelle_auf_route_df_final)

# Hänge vermehrte Anzahl der Unfälle auf der Route an Ursprungsdataframe
augsburg = pd.concat([augsburg, unfaelle_auf_route_df_final])


##########  Trainiere Random Forest auf den Augsburg Daten mit (Version 4) und ohne (Version 5) Wetterdaten ########
X = augsburg[Version4]
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)

y = augsburg['UTYP1']
y_art = augsburg['UART']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
RFC = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=21)
modelRFC = RFC.fit(X_train, y_train)
modelRFC.predict(X_test)
print("Unfalltyp: Accuracy without weather data:", round(RFC.score(X_test,y_test), 4))

X_train, X_test, y_train, y_test = train_test_split(X, y_art, test_size=0.3, random_state=1) # 70% training and 30% test
RFC = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=21)
modelRFC = RFC.fit(X_train, y_train)
modelRFC.predict(X_test)
print("Unfallart: Accuracy without weather data:", round(RFC.score(X_test,y_test), 4))




X_weather = augsburg[Version5]
imp = SimpleImputer(strategy="mean")
X_weather = imp.fit_transform(X_weather)

y_weather = augsburg['UTYP1']
y_weather_art = augsburg['UART']

X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather, test_size=0.3, random_state=1) # 70% training and 30% test
RFC_weather = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=26)
weathermodelRFC = RFC_weather.fit(X_train, y_train)
weathermodelRFC.predict(X_test)
print("Unfalltyp: Accuracy with weather data:", round(RFC_weather.score(X_test,y_test), 4))

X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather_art, test_size=0.3, random_state=1) # 70% training and 30% test
RFC_weather = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=26)
weathermodelRFC = RFC_weather.fit(X_train, y_train)
weathermodelRFC.predict(X_test)
print("Unfallart: Accuracy with weather data:", round(RFC_weather.score(X_test,y_test), 4))



############ Plotte Route mit Folium auf eine karte und speichere als html ##########
route_map = ox.plot_route_folium(G, route2, opacity=0.5, color="#cc0000")

for Lon, Lat, idd in zip(Lon, Lat, ids):
    folium.Circle(
            radius=10,
            location=[Lat, Lon],
            popup=str(idd),
            color="blue",
            fill=False,
            ).add_to(route_map)


folium.Marker(
    location=[location.latitude, location.longitude],
    popup='Bitte passen Sie besonders auf ..... Es ist sehr wahrscheinlich dass Sie....',
    icon=folium.Icon(color='red', icon='ok-sign'),
).add_to(route_map)
route_map.save('route.html')
