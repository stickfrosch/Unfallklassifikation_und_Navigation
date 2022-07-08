import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import Neu_Deskriptiv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

########## Lese Augsburg Unfälle ein #############
augsburg = pd.read_pickle("augsburg.pkl")


########## Nutze Nominatim Geocoding Serivce um Adresse in Lat und Lon umzurechnen #############
locator = Nominatim(user_agent="myGeocoder")
location = locator.geocode("Rathaus, Augsburg, Germany")
endlocation = locator.geocode("Bürgeramt, Augsburg, Germany")

unfaelle_auf_route = pd.read_pickle("unfaelle_auf_route.pkl")

Lon = unfaelle_auf_route['XGCSWGS84'].values[:]
Lat = unfaelle_auf_route['YGCSWGS84'].values[:]
ids = np.array(unfaelle_auf_route.loc[:,:])

df2 = pd.DataFrame(ids)

df2 = df2.drop(df2.iloc[:, 35:],axis = 1)

df = Neu_Deskriptiv.df
df_for_finaldf = df.drop(df.iloc[:, 0:17],axis = 1)
df_for_finaldf = df_for_finaldf.apply( pd.to_numeric, errors='coerce' )
df = df.apply( pd.to_numeric, errors='coerce' )
print("df and df2 shape:")
print(df.shape)
print(df2.shape)



# Die routenspezifischen Unfälle zu 10% des Datensatzes vermehrfachen
rownumberdf2 = df2.shape[0]
rownumberdf = df.shape[0]

gewichtung = (0.1 * rownumberdf) / rownumberdf2
gewichtung = round(gewichtung)

newdf2 = pd.DataFrame(np.repeat(df2.values, gewichtung, axis=0))
newdf2.columns = df2.columns
print(newdf2.shape)


# Definieren der Spaltennamen
newdf2.columns = ['Land', 'RB', 'Kreis', 'VB', 'Gem', 'Gemeindename', 'Flaechekm2',
 'Einwohneranzahl', 'maennlich', 'weiblich', 'je km2', 'PLZ', 'FullAGS',
 'OBJECTID', 'ULAND', 'UREGBEZ', 'UKREIS', 'UGEMEINDE', 'UJAHR', 'UMONAT',
 'USTUNDE', 'UWOCHENTAG', 'UKATEGORIE', 'UART', 'UTYP1', 'ULICHTVERH', 'IstRad',
 'IstPKW', 'IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstige', 'STRZUSTAND',
 'XGCSWGS84', 'YGCSWGS84']
# Einige unwichtige Merkmale entfernen
newdf2 = newdf2.drop(newdf2.iloc[:, 0:17],axis = 1)
newdf2 = newdf2.apply( pd.to_numeric, errors='coerce' )

# die Dataframes für die routenspezifischen Vorhersagen zusammenführen
finaldf = pd.concat([df_for_finaldf, newdf2], ignore_index=True, axis=0)



#Die unterschiedlichen Versionen definieren:
Version1 = ['OBJECTID', 'ULAND', 'UREGBEZ', 'UKREIS','UGEMEINDE', 
                    'UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG', 
                    'ULICHTVERH', 'IstRad', 'IstPKW', 'IstFuss', 'IstKrad', 
                    'IstGkfz', 'IstSonstige', 'STRZUSTAND', 'XGCSWGS84', 
                    'YGCSWGS84']

Version2 = ['UMONAT', 'USTUNDE', 'UWOCHENTAG', 'ULICHTVERH', 'IstRad', 
                    'IstPKW', 'IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstige', 
                    'STRZUSTAND', 'XGCSWGS84', 'YGCSWGS84']

# UART und UTYP1 haben wegen unterschiedlichen Korrelationen auch unterschiedliche Version 3
UARTv3 = ['IstPKW', 'ULICHTVERH', 'IstRad', 'IstFuss', 'YGCSWGS84']
UTYP1v3 = ['STRZUSTAND', 'ULICHTVERH', 'IstRad', 'IstFuss', 'USTUNDE']

Version4 = ['UJAHR', 'UMONAT', 'USTUNDE', 'UWOCHENTAG', 
'ULICHTVERH', 'IstRad', 'IstPKW', 'IstFuss', 'IstKrad', 
'IstGkfz', 'IstSonstige', 'STRZUSTAND']


# Auswahl der jeweiligen Version und jeweiligen 
X = df[Version4]
#y = df[['UTYP1']]
y = df[['UART']]

# 70% Trainingset und 30% Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 


# Der Logistic Regression Klassifizierer
#LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_tr, y_tr)
#LR.predict(X_test)
#print(round(LR.score(X_test,y_test), 4))


# Decision Tree Klassifizierer, unterschiedliche Tiefen werden getestet
#rid = 1
#while rid <= 51:
#    DTC = DecisionTreeClassifier(random_state=0, max_depth=rid, criterion='entropy').fit(X_train, y_train)
#    DTC.predict(X_test)
#    print(round(DTC.score(X_test,y_test), 4))
#    rid += 5


# Testen der Tiefe von 21 bis 51 beim Random Forest Klassifizierer
rfc_max_depth_hist = []
rfc_score = []
rfc_nest_index = 1
max_depth = 1
while max_depth <= 21:
    RFC = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=max_depth).fit(X_train, y_train)
    RFC.predict(X_test)
    rfc_score.append(round(RFC.score(X_test,y_test), 4))
    rfc_max_depth_hist.append(max_depth)
    max_depth += 5

print(rfc_score)


# Den benutzerspezifischen Input geben und die Vorhersage machen
#new_input = [[0, 2, 1, 0, 12]]
#new_output = LR.predict(new_input)
#print(new_input, new_output)
#round(LR.score(X_test,y_test), 4)






