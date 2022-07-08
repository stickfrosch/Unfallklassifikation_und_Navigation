import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import osmnx as ox
import networkx as nx
import folium
import webbrowser
import geopandas
#import geopy
from geopy.geocoders import Nominatim

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

########## Lese Augsburg Unfälle ein #############
augsburg = pd.read_pickle("bayern.pkl")
print(augsburg)

Lon = augsburg['XGCSWGS84'].values[:]
Lat = augsburg['YGCSWGS84'].values[:]

############ Erstelle Graphen für Auto, zu Fuß und Fahrrad anhand einer Adresse mithilfe osmnx Package ##########
G_drive = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='drive')
G_walk = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='walk')
G_bike = ox.graph.graph_from_address('Rathausplatz 2, Augsburg, Germany', dist=15000, network_type='bike')

print("Graphen berechnet")
############ Berechne nearest Node für alle Unfälle ##########
unfallnodes_drive = ox.distance.nearest_nodes(G_drive, Lon, Lat)
unfallnodes_walk = ox.distance.nearest_nodes(G_walk, Lon, Lat)
unfallnodes_bike = ox.distance.nearest_nodes(G_bike, Lon, Lat)
print("Näheste Knoten berechnet")
########### Auto ############
unfallnodes_drive = np.asarray(unfallnodes_drive)
augsburg["unfallnodes_drive"] = unfallnodes_drive

########### zu Fuß ############
unfallnodes_walk = np.asarray(unfallnodes_walk)
augsburg["unfallnodes_walk"] = unfallnodes_walk

########### Fahrrad ############
unfallnodes_bike = np.asarray(unfallnodes_bike)
augsburg["unfallnodes_bike"] = unfallnodes_bike

print("Unfallknoten für jeweiliges Fortbewegungsmittel berechnet")
print(augsburg.head())
augsburg = augsburg.apply(pd.to_numeric, errors='coerce')
augsburg.to_pickle("augsburg.pkl")
