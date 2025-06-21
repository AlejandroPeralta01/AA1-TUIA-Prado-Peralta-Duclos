import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MapeoRegiones(BaseEstimator, TransformerMixin):
    def __init__(self, df_ubicaciones_data):
        self.df_ubicaciones_data = df_ubicaciones_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, columna_ciudad='Location'):
        df_union = X.copy().merge(
            self.df_ubicaciones_data[['Location', 'regiones', 'lat', 'lon']],
            left_on=columna_ciudad,
            right_on='Location',
            how='left'
        )

        ciudades_invalidas = df_union[df_union['regiones'].isna()]
        if len(ciudades_invalidas) > 0:
            ciudades_no_encontradas = X.loc[ciudades_invalidas.index, columna_ciudad].tolist()
            raise ValueError(f"Ciudades no encontradas en el mapeo: {ciudades_no_encontradas}")

        df_union['regiones'] = df_union['regiones'].astype('object')
        return df_union.drop(['Location'], axis=1)

class EliminarTarget(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'RainTomorrow' in X_copy.columns:
            X_copy = X_copy.drop('RainTomorrow', axis=1)
        return X_copy

class CodificacionFecha(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        mes = X_copy['Date'].dt.month
        X_copy['Month'] = mes.astype(int)
        X_copy['month_sin'] = np.sin(2 * np.pi * mes / 12)
        X_copy['month_cos'] = np.cos(2 * np.pi * mes / 12)
        return X_copy.drop('Date', axis=1)

class CodificacionViento(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        direccion_angulo = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
            'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        for col in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:
            if col in X_copy.columns:
                grados = X_copy[col].map(direccion_angulo).astype(float)
                radianes = np.radians(grados)
                X_copy[f'{col}_sin'] = np.sin(radianes)
                X_copy[f'{col}_cos'] = np.cos(radianes)
                X_copy = X_copy.drop(col, axis=1)

        return X_copy

class CodificacionRainToday(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.dropna(subset=['RainToday'])

        if X_copy.empty:
            raise ValueError("RainToday es obligatorio para hacer predicciones")

        X_copy['RainToday'] = X_copy['RainToday'].map({'No': 0, 'Yes': 1})
        return X_copy

class ValidacionNulos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        umbral = X.shape[1] // 2
        filas_validas = X.notnull().sum(axis=1) > umbral
        filas_malas = (~filas_validas).sum()

        if filas_malas > 0:
            raise ValueError(f"{filas_malas} filas tienen más del 50% de valores nulos")

        return X

class ImputacionNulos(BaseEstimator, TransformerMixin):
    def __init__(self, media_datos, mediana_datos, criterio_media_list, criterio_mediana_list):
        self.media_datos = media_datos
        self.mediana_datos = mediana_datos
        self.criterio_media_list = criterio_media_list
        self.criterio_mediana_list = criterio_mediana_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        def imputar_con_tablas(df, columnas, tabla_imputacion):
            for col in columnas:
                for idx, row in df[df[col].isna()].iterrows():
                    clave = (row['regiones'], row['Month'])
                    if clave in tabla_imputacion.index:
                        valor = tabla_imputacion.loc[clave, col]
                        df.at[idx, col] = valor
            return df

        X_copy = imputar_con_tablas(X_copy, self.criterio_media_list, self.media_datos)
        X_copy = imputar_con_tablas(X_copy, self.criterio_mediana_list, self.mediana_datos)

        return X_copy.drop('Month', axis=1)

