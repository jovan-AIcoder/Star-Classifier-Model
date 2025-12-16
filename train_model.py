import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from tensorflow import keras
import joblib
# import data
df = pd.read_csv('Stars.csv')
# preprocess data
color_encoder = pp.LabelEncoder()
spectral_class_encoder = pp.LabelEncoder()
color_encoder.fit(df['Color'])
spectral_class_encoder.fit(df['Spectral_Class'])
df['Color_code'] = color_encoder.transform(df['Color'])
df['Spectral_Class_code'] = spectral_class_encoder.transform(df['Spectral_Class'])
df = df.drop(['Color','Spectral_Class'],axis=1)
X = df.drop(['Type'],axis=1)
y = df[['Type']]
scaler = pp.StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler,'scaler.pkl')
# model
model = keras.Sequential([
    keras.layers.Input(shape=(X_scaled.shape[1],)),
    keras.layers.Dense(64,activation='tanh'),
    keras.layers.Dense(32,activation='tanh'),
    keras.layers.Dense(16,activation='softplus'),
    keras.layers.Dense(len(np.unique(y)),activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Training and saving
model.fit(X_scaled,y,epochs=150,batch_size=24)
model.save('star_classifier_model.h5')
print("Model is saved successfully")