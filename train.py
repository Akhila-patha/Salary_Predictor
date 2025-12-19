import pandas as pd
data = pd.read_csv("SelfPracticeSalary_Data.csv")
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
import joblib
joblib.dump(scaler,'scaler.pkl')
X=scaler.fit_transform(data[['YearsExperience']])
Y=scaler.fit_transform(data[['Salary']])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=100, batch_size=1, verbose=1)
model.save('salary_prediction_model.h5')
r=model.predict(X_test)
r=scaler.inverse_transform(r)
print(r)