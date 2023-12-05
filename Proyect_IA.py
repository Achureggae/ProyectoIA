import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

meses_dict = {
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}

file_path = 'C:\\Users\\achus\\OneDrive\\Escritorio\\pp\\IA\\datos_pet1.csv'  
data = pd.read_csv(file_path)

data['mes'] = data['fecha'].apply(lambda x: x.split('-')[0])  
data['año'] = data['fecha'].apply(lambda x: int(x.split('-')[1]))  
data['mes_numerico'] = data['mes'].map(meses_dict)

X = data[['mes_numerico', 'año']].values  
y = data['cantidad_pet'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación:', r2_score(y_test, y_pred))


meses_futuros = ['enero','febrero','marzo','abril','mayo','junio','julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
año_futuro = 2024

nuevos_meses = [[meses_dict[mes], año_futuro] for mes in meses_futuros]  
predicciones_futuras = model.predict(nuevos_meses)

meses_nombres = {v: k for k, v in meses_dict.items()}
meses_predicciones = [meses_nombres.get(mes, mes) for mes in predicciones_futuras]

resultados = list(zip(meses_futuros, predicciones_futuras, meses_predicciones))
print('Predicciones futuras de cantidad de PET en', año_futuro, ':')
for resultado in resultados:
    cantidad_redondeada = "{:.2f}".format(resultado[1])
    print(f'{resultado[0]} {año_futuro} - Cantidad: {cantidad_redondeada} Kgs')

promedio_por_mes = data.groupby('mes')['cantidad_pet'].mean().reset_index()
top_3_tendencia_mayor = promedio_por_mes.nlargest(3, 'cantidad_pet')

print('\nLos tres meses con tendencia a mayor cantidad de PET:')
for index, row in top_3_tendencia_mayor.iterrows():
    mes_nombre = row['mes']
    mes_numero = meses_dict.get(mes_nombre.lower(), -1)  
    if mes_numero != -1:
        cantidad_promedio = "{:.2f}".format(row["cantidad_pet"])  
        print(f'{mes_nombre.capitalize()} - Cantidad promedio: {cantidad_promedio} Kgs')
    else:
        print(f'Mes desconocido ({mes_nombre}) - Cantidad promedio: {row["cantidad_pet"]} Kgs')