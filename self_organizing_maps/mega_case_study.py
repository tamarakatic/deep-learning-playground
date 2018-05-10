import numpy as np
from som import discover_frauds

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

frauds, dataset = discover_frauds()

customers = dataset.iloc[:, 1:].values

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

sc = StandardScaler()
customers = sc.fit_transform(customers)

classifier = Sequential()

classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
