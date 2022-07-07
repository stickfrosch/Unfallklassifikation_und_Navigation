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

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

########## Lese Augsburg Unfälle ein #############
augsburg = pd.read_pickle("augsburg.pkl")
#print(augsburg)


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

    elif var_mobility_input == "2":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='walk')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_walk"].isin(route2)]

    elif var_mobility_input == "3":
        G = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='bike')
        orig = ox.distance.nearest_nodes(G, location.longitude, location.latitude)
        dest = ox.distance.nearest_nodes(G, endlocation.longitude, endlocation.latitude)
        route2 = ox.shortest_path(G, orig, dest, weight='travel_time')
        unfaelle_auf_route = augsburg[augsburg["unfallnodes_bike"].isin(route2)]

    else:
        print("Invalid Input. Bitte nochmal eingeben")
        continue
    break


Lon = unfaelle_auf_route['XGCSWGS84'].values[:]
Lat = unfaelle_auf_route['YGCSWGS84'].values[:]
ids = np.array(unfaelle_auf_route.loc[:,'UART'])
print(unfaelle_auf_route)
unfaelle_auf_route.to_pickle("unfaelle_auf_route.pkl")


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