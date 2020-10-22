from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#We import the libraries
file = 'Social_Network_Ads.csv'
df = pd.read_csv(file)
df = df.drop('User ID',axis = 1)
#We read the data file
#We drop unnecesary ID column, as it serves no purpose on the features of our dataset and model.
def gender_to_binary(gen):
    if gen == 'Male':
        return 0
    elif gen == 'Female':
        return 1
#We transform the gender data from strings to binary int values
df['Gender'] = df['Gender'].map(lambda x: gender_to_binary(x))
x = df.drop('Purchased',axis=1)
y = df['Purchased']
#We declare x and y arrays for training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
#We divide the data into train and test sets for our model
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#We scale the data
model = Sequential()
model.add(Dense(3,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#We create our deep learning neural network model
model.compile(loss='binary_crossentropy',optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss')
#We create an early stopping call to not overfit our model
model.fit(X_train,y_train,epochs=500,
            batch_size=4,callbacks=[early_stop],validation_data=(X_test,y_test))
#We train our model based on our early stopping call
predictions = model.predict_classes(X_test)
#We create our predictions based on the X_test dataset
print(classification_report(y_test,predictions))
#We print the classification report on our predictions and the true labels