import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D,Activation, Dropout,Flatten,Dense,MaxPooling1D
from sklearn.svm import SVC
from keras.regularizers import l2
from sklearn.metrics import accuracy_score

heart_disease=pd.read_csv("dataset\heart_statlog_cleveland_hungary_final.csv")
print(heart_disease)
# preprocessing
# print(heart_disease.shape)
# print(heart_disease.info())
# print("null value sum",heart_disease.isnull().sum())
# print("duplicate row",heart_disease.duplicated())
x=heart_disease.drop(columns=['target'])
print(x.shape)
y=heart_disease['target']
print(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
print(X_scaled)
X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
print("new",X_reshaped)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape[1],1)
model =Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
# https://www.analyticsvidhya.com/blog/2021/06/build-an-image-classifier-with-svm/
model.add(Dense(1, kernel_regularizer=l2(0.01), activation="linear"))   

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# prediction=model.predict(X_train)
# print(prediction)
ypred=model.predict(X_test)
print(y_test.shape)
print(ypred.shape)
ypred_binary = np.round(ypred) 
print(ypred_binary)
accuracy = accuracy_score(y_test, ypred_binary)
print("Accuracy:", accuracy)


