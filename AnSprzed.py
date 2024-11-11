import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from keras.layers import Dropout

#1 ANALIZA
#PRZYGOTOWANIE DANYCH
# Ustawienia wyświetlania
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\Users\zales\Documents\projekty\sprzedaz\online_sales_dataset.csv")
numeric_col = df.select_dtypes(include=['number']).columns
df = df[(df[numeric_col] >= 0).all(axis=1)]
st.write("Przykładowe dane:", df.head())
df.dropna(inplace=True)

# SPRZEDAZ
# Analiza sprzedaży
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['Calkowity_przychod'] = df['Quantity'] * df['UnitPrice']
calkowity_przychod = df['Calkowity_przychod'].sum()
st.write("Całkowity przychód ze sprzedaży:", calkowity_przychod)

przychód_kategorie = df.groupby('Category')['Calkowity_przychod'].sum().sort_values(ascending=False)
st.write("Przychód wg kategorii:\n", przychód_kategorie)

przychód_kraje = df.groupby('Country')['Calkowity_przychod'].sum().sort_values(ascending=False)
st.write("Przychód wg krajów zakupu:\n", przychód_kraje)

# wykres czasowy przychody
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
przychody_wg_daty = df.groupby(df['InvoiceDate'].dt.date)['Calkowity_przychod'].sum()

st.title("Przychody ze sprzedaży w czsie")
fig,ax = plt.subplots()
przychody_wg_daty.plot(ax=ax, color = 'pink')
ax.set_title("Przychody ze sprzedazy w czasie")
ax.set_xlabel("Data")
ax.set_ylabel("Przychody")
st.pyplot(fig)

# wykres czasowy przychody(średnia)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Rok'] = df['InvoiceDate'].dt.year
średnia_rok = df.groupby('Rok')['Calkowity_przychod'].mean()

st.title('Średnie roczne przychody')
fig,ax = plt.subplots()
średnia_rok.plot(ax =ax, color = 'pink')
ax.set_title("średnie roczne przychody")
ax.set_xlabel("Data")
ax.set_ylabel("Średnie przychody")
st.pyplot(fig)

# najlepiej sprzedające się produkty

top_produkty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
st.title("Najlepiej sprzedające się produkty")
st.bar_chart(top_produkty)

#ZACHOWANIA KLIENTOW
#Segmentacja

df[calkowity_przychod] = df['Quantity'] * df['UnitPrice']
średnie_zamowienie = df.groupby('CustomerID')['Calkowity_przychod'].mean().reset_index()
średnie_zamowienie.columns = ['CustomerID', 'ŚrednieZamowienie']
df = df.merge(średnie_zamowienie, on='CustomerID', how= 'left')

segments = df.groupby(['PaymentMethod', 'Country'])['ŚrednieZamowienie'].mean().reset_index()
segments.columns = ['PaymentMethod', 'Country', 'MeanŚrednieZamowienie']
st.title("Segmentacja Klientow")
st.dataframe(segments)

fig, ax = plt.subplots(figsize = (12,8))
sns.barplot(data = segments, x = 'Country', y = 'MeanŚrednieZamowienie', hue = 'PaymentMethod', ax = ax )
ax.set_title('Mean value by payment method and country')
ax.set_xlabel('Kraj')
ax.set_ylabel('Wartość')
plt.xticks(rotation = 45)
st.pyplot(fig)

#zwroty

returns = df[df['ReturnStatus'] == 'Returned']
returns_segmenty = returns.groupby(['PaymentMethod', 'Country'])['InvoiceNo'].count().reset_index()
returns_segmenty.columns = ['PaymentMethod', 'Country', 'ReturnCount']
st.title("Analiza zwrotow")
st.dataframe(returns_segmenty)


#OGOLNE WYKRESY
# Wykres kategorie
st.subheader("Procentowy rozkład kategorii")
kategorie = df['Category'].value_counts(normalize=True) * 100
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(kategorie, labels=kategorie.index, autopct='%1.1f%%', startangle=90)
st.pyplot(fig1)

# Wykres płatności
st.subheader("Liczba płatności w różnych metodach")
suma_platnosci = df['PaymentMethod'].value_counts()
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.bar(suma_platnosci.index, suma_platnosci.values)
ax3.set_title('Liczba płatności')
ax3.set_xlabel('Metody')
ax3.set_ylabel('Liczba')
plt.xticks(rotation=45)
st.pyplot(fig3)

#2 MODEL LSTM

meble = df[df['Category'] == 'Furniture']
meble = meble[meble['Quantity'] > 0] 
meble['InvoiceDate'] = pd.to_datetime(meble['InvoiceDate'])
meble.set_index('InvoiceDate', inplace=True)
meble_sprzedaz = meble.resample('M')['Quantity'].sum()
meble_sprzedaz_diff = meble_sprzedaz.diff().dropna()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
meble_scaled = scaler.fit_transform(meble_sprzedaz_diff.values.reshape(-1, 1))

def create_sequences(data, seq_length):
    X,y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 12
X, y = create_sequences(meble_scaled, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential([
    LSTM(200, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(200, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(200, activation='relu'),
    Dropout(0.2),
    Dense(1)
])


model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X,y, epochs = 1000, batch_size = 32, verbose = 1)

X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size= 0.2, shuffle=False)
st.write(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

test_loss = model.evaluate(X_test, y_test)
print(f'Test loss:{test_loss}')

y_pred = model.predict(X_test)

plt.figure(figsize=(10,6))
plt.plot(y_test, label = 'Rzeczywiste')
plt.plot(y_pred, label = 'Przewidywane')
plt.legend()

st.pyplot(plt)

#ocena modelu
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
st.write(f'mse: {mse}')
st.write(f'mae:{mae}')

r2 = r2_score(y_test, y_pred)
st.write(f'r2: {r2}')
