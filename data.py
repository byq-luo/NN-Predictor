import keras
import numpy
import pandas as pd
import datetime

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#-------------------------- Preprocesado de datos ---------------------------

#Lectura del fichero csv
df = pd.read_csv("/home/miriam/PycharmProjects/NNBPP/DATA_1.csv", delimiter=';', quotechar='"')

#Transformación de los tipos de campos object --> float64 / int64 --> float64
df['VELOCIDAD_VIENTO'] = pd.to_numeric(df['VELOCIDAD_VIENTO'], errors='coerce')
df['FECHA_HORA'] = pd.to_numeric(df['FECHA_HORA'], errors='coerce')
df['TEMERATURA_AIRE'] = pd.to_numeric(df['TEMERATURA_AIRE'], errors='coerce')
df['HUMEDAD_RRELATIVA'] = pd.to_numeric(df['HUMEDAD_RRELATIVA'], errors='coerce')
df['DIRECCION_VIENTO'] = pd.to_numeric(df['DIRECCION_VIENTO'], errors='coerce')
df['LONGITUD_VEHICULO'] = df.LONGITUD_VEHICULO.astype(float)
df['CARRIL_CIRCULACION'] = df.CARRIL_CIRCULACION.astype(float)
df['VELOCIDAD_VEHICULO'] = df.VELOCIDAD_VEHICULO.astype(float)
df['PESO_VEHICULO'] = df.PESO_VEHICULO.astype(float)
df['NUMERO_EJES'] = df.NUMERO_EJES.astype(float)

#Eliminación de los valores nulos e infinitos
df = df.fillna(df.median(axis=0))

#Creación de matrices
x = df.iloc[:, 1:14].values
y = df.iloc[:, 13].values

#Codificación de aquellos campos de texto
#Aquellos que tengan más campos que dos se codificarán de 0 a n-classes
labelenconder_X_1 = LabelEncoder()
labelenconder_X_2 = LabelEncoder()
labelenconder_X_3 = LabelEncoder()
labelenconder_X_4 = LabelEncoder()
labelenconder_Y = LabelEncoder()

#Intercambiar los valores de las columnas por sus nuevas codificaciones
x[:, 12] = labelenconder_X_1.fit_transform(x[:, 12])
x[:, 11] = labelenconder_X_2.fit_transform(x[:, 11])
x[:, 8] = labelenconder_X_2.fit_transform(x[:, 8])
x[:, 7] = labelenconder_X_2.fit_transform(x[:, 7])

y = labelenconder_Y.fit_transform(y)

#Se crean variables de tipo dummy para la codificación de aquellos campos que tengan valores superiores a 1
#Como es el caso de ESTADO_CARRETERA, de esta forma se crea una columna binaria por cada categoría
#
onehotenconder_X1 = OneHotEncoder(categorical_features=[11])
onehotenconder_X2 = OneHotEncoder(categorical_features=[8])
onehotenconder_X3 = OneHotEncoder(categorical_features=[7])

x = onehotenconder_X1.fit_transform(x).toarray()
x = onehotenconder_X2.fit_transform(x).toarray()
x = onehotenconder_X3.fit_transform(x).toarray()

#División del conjunto de datos para entrenar y testear
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Estandarización y escalado de los datos
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)

#-------------------------- Red Neuronal ---------------------------

classifier = Sequential()

# #Input layer: relu es una función de activación rectificadora
classifier.add(Dense(32, kernel_initializer='uniform', activation='relu', input_dim=25))

# #Hidden layers

# #Output layer: se usa la función de activación sigmoide para una salida de 0 / 1
classifier.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))


# #Compiling

#Se emplea el adam que es similar al sgd (descenso por el gradiente estocástico)
#La pérdida es binaria porque la clasficación es de dos, si fuera multic-clases sería categorical
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(x_train.shape)

# #Fitting
classifier.fit(x_train, y_train, batch_size=32, nb_epoch=100)

#Predicción
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# #Metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

