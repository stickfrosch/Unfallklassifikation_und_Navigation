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
start = input("Geben Sie bitte ihre Start Adresse ein:")
end = input("Geben Sie bitte ihre Ziel Adresse ein:")
#location = locator.geocode("Rathaus, Augsburg, Germany")
#Universität Augsburg, Augsburg, Germany
#Zoo Augsburg, Augsburg, Germany
location = locator.geocode(start)
#endlocation = locator.geocode("Bürgeramt, Augsburg, Germany")
endlocation = locator.geocode(end)

while True:
    var_mobility_input = input("Geben Sie bitte 1 für Auto, 2 für zu Fuß oder 3 für Fahrrad als ihr Fortbewegungsmittel ein")

    if var_mobility_input == "1":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='drive')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_drive"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_drive'].isin(route2)]
        var_auto = 1
        var_fuß = 0
        var_fahrrad = 0

    elif var_mobility_input == "2":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='walk')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_walk"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_walk'].isin(route2)]
        var_auto = 0
        var_fuß = 1
        var_fahrrad = 0

    elif var_mobility_input == "3":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='bike')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_bike"].isin(route2)]
        unfaelle_auf_route_df = augsburg.loc[augsburg['unfallnodes_bike'].isin(route2)]
        var_auto = 0
        var_fuß = 0
        var_fahrrad = 1

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
modelRFC_art = RFC.fit(X_train, y_train)
modelRFC_art.predict(X_test)
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

############ Abfrage der Reiseparameter ##########
while True:
    var_light = input("Wie sind die Lichterverhältnisse? Geben Sie bitte eine 0 für Tageslicht, eine 1 für Dämmerung oder eine 2 für Dunkelheit ein")
    if var_light == "0":
        var_light = 0

    elif var_light == "1":
        var_light = 1

    elif var_light == "2":
        var_light = 2
    else:
        print("Nicht zulässig. Bitte noch einmal eingeben")
        continue
    break

while True:
    var_streetstatus = input(
        "Wie ist die Straßenoberfläche? Geben Sie bitte eine 0 für trocken, eine 1 für nass/feucht/schlüpfrig oder eine 2 für winterglatt ein")
    if var_streetstatus == "0":
        var_streetstatus = 0

    elif var_streetstatus == "1":
        var_streetstatus = 1

    elif var_streetstatus == "2":
        var_streetstatus = 2
    else:
        print("Nicht zulässig. Bitte noch einmal eingeben")
        continue
    break

var_year = 2022
var_month = 3
var_hour = 9
var_weekday = 4
var_Krad = 0
var_sonstige = 0
var_Gkfz = 0

input_variablenarray = []
input_variablenarray.append([var_year, var_month, var_hour, var_weekday, var_light, var_fahrrad, var_auto, var_fuß, var_Krad, var_Gkfz, var_sonstige, var_streetstatus])
prediction_utyp = modelRFC.predict(input_variablenarray)
prediction_uart = modelRFC_art.predict(input_variablenarray)
#print(prediction_utyp)
#print(prediction_uart)

while True:

    if prediction_utyp == 1:
        prediction_utyp = "Fahrunfall"

    elif prediction_utyp == 2:
        prediction_utyp = "Abbiegeunfall"

    elif prediction_utyp == 3:
        prediction_utyp = "Einbiegen / Kreuzen-Unfall"

    elif prediction_utyp == 4:
        prediction_utyp = "Überschreiten-Unfall"

    elif prediction_utyp == 5:
        prediction_utyp = "Unfall durch ruhenden Verkehr"

    elif prediction_utyp == 6:
        prediction_utyp = "Unfall im Längsverkehr"

    elif prediction_utyp == 7:
        prediction_utyp = "nicht definierten Unfall"

    else:
        print("keinen Unfall")
        continue
    break

while True:

    if prediction_uart == 1:
        prediction_uart = "mögliche Zusammenstöße mit anfahrenden/anhaltenden und ruhenden Fahrzeugen"

    elif prediction_uart == 2:
        prediction_uart = "mögliche Zusammenstöße mit vorausfahrenden / wartenden Fahrzeugen"

    elif prediction_uart == 3:
        prediction_uart = "mögliche Zusammenstöße mit seitlich in gleicher Richtung fahrenden Fahrzeugen"

    elif prediction_uart == 4:
        prediction_uart = "mögliche Zusammenstöße mit entgegenkommendem Fahrzeug"

    elif prediction_uart == 5:
        prediction_uart = "mögliche Zusammenstöße mit einbiegendem / kreuzendem Fahrzeug"

    elif prediction_uart == 6:
        prediction_uart = "mögliche Zusammenstöße zwischen Fahrzeug und Fußgänger"

    elif prediction_uart == 7:
        prediction_uart = "möglichen Aufprall auf Fahrbahnhindernis"

    elif prediction_uart == 8:
        prediction_uart = "Abkommen von Fahrbahn nach rechts"

    elif prediction_uart == 9:
        prediction_uart = "Abkommen von Fahrbahn nach links"

    elif prediction_uart == 0:
        prediction_uart = "Unfall anderer Art"

    else:
        print("keinen Unfall")
        continue
    break

warning_popup = "Bitte passen Sie besonders auf", prediction_uart, "auf. Zudem ist es sehr wahrscheinlich dass Sie in einen", prediction_utyp, "verwickelt werden könnten."
#warning_popup = str(warning_popup)
print(warning_popup)
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
    popup=warning_popup,
    icon=folium.Icon(color='red', icon='ok-sign'),
).add_to(route_map)
route_map.save('route.html')
