import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
import numpy as np
from pycaret.classification import load_model

# ===== Funciones =====

def mapeo_regiones(data_nueva, columna_ciudad='Location'):
    """Mapea ciudades a regiones, latitud y longitud"""
    try:
        if columna_ciudad not in data_nueva.columns:
            raise ValueError(f"Columna '{columna_ciudad}' no encontrada")
        
        # Cargar ubicaciones
        df_ubicaciones = joblib.load('pkl/df_location.pkl')
        
        # Realizar merge
        df_union = data_nueva.copy().merge(
            df_ubicaciones[['Location', 'regiones', 'lat', 'lon']], 
            left_on=columna_ciudad,
            right_on='Location',
            how='left'
        )
        
        # Verificar ciudades no encontradas
        ciudades_invalidas = df_union[df_union['regiones'].isna()]
        if len(ciudades_invalidas) > 0:
            ciudades_no_encontradas = data_nueva.loc[ciudades_invalidas.index, columna_ciudad].tolist()
            raise ValueError(f"Ciudades no encontradas: {ciudades_no_encontradas}")
        
        # Limpiar y retornar
        df_union['regiones'] = df_union['regiones'].astype('object')
        return df_union.drop(['Location'], axis=1)
        
    except Exception as e:
        print(f"Error en mapeo_regiones: {e}")
        raise


def validacion_nulos(X):
    """Rechaza filas con más del 50% de valores faltantes"""
    try:
        umbral = X.shape[1] // 2
        filas_validas = X.notnull().sum(axis=1) > umbral
        filas_malas = (~filas_validas).sum()
        
        if filas_malas > 0:
            raise ValueError(f"{filas_malas} filas tienen más del 50% de nulos")
        
        return X
        
    except Exception as e:
        print(f"Error en validación de nulos: {e}")
        raise


def codificacion_fecha(X):
    """Extrae mes de Date y lo codifica con sin/cos para ciclicidad"""
    try:
        X_copy = X.copy()
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        
        # Extraer mes
        mes = X_copy['Date'].dt.month
        
        # Creamos Month para utilizar en una posterior imputación
        X_copy['Month'] = mes.astype(int)
        
        # Codificación
        X_copy['month_sin'] = np.sin(2 * np.pi * mes / 12)
        X_copy['month_cos'] = np.cos(2 * np.pi * mes / 12)
        
        return X_copy.drop('Date', axis=1)
        
    except Exception as e:
        print(f"Error en codificación de fecha: {e}")
        raise


def codificacion_viento(X):
    """Convierte direcciones de viento a coordenadas cíclicas sin/cos"""
    try:
        X_copy = X.copy()
        
        # Mapeo de direcciones a grados
        direccion_angulo = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        # Procesar columnas de viento
        columnas_viento = ['WindDir9am', 'WindDir3pm', 'WindGustDir']
        
        for col in columnas_viento:
            if col in X_copy.columns:
                grados = X_copy[col].map(direccion_angulo).astype(float)
                radianes = np.radians(grados)
                X_copy[f'{col}_sin'] = np.sin(radianes)
                X_copy[f'{col}_cos'] = np.cos(radianes)
        
        # Eliminar columnas originales
        columnas_a_eliminar = [col for col in columnas_viento if col in X_copy.columns]
        return X_copy.drop(columnas_a_eliminar, axis=1)
        
    except Exception as e:
        print(f"Error en codificación de viento: {e}")
        raise


def codificacion_rain_today(X):
    """Convierte RainToday de categórico a binario"""
    try:
        X_copy = X.copy()
        
        # Eliminar filas sin RainToday
        X_copy = X_copy.dropna(subset=['RainToday'])
        
        if X_copy.empty:
            raise ValueError("RainToday es obligatorio para la predicción")
        
        # Convertir a binario
        X_copy['RainToday'] = X_copy['RainToday'].map({'No': 0, 'Yes': 1})
        return X_copy
        
    except Exception as e:
        print(f"Error en codificación de RainToday: {e}")
        raise


def imputacion_nulos(X):
    """Imputa valores faltantes usando tablas de entrenamiento"""
    try:
        X_copy = X.copy()
        
        # Cargar imputadores
        media_tabla = joblib.load('pkl/media_imputacion.pkl')
        mediana_tabla = joblib.load('pkl/mediana_imputacion.pkl')
        criterio_media = joblib.load('pkl/criterio_media.pkl')
        criterio_mediana = joblib.load('pkl/criterio_mediana.pkl')
        
        def imputar_con_tablas(df, columnas, tabla_imputacion):
            for col in columnas:
                for idx, row in df[df[col].isna()].iterrows():
                    clave = (row['regiones'], row['Month'])
                    if clave in tabla_imputacion.index:
                        valor = tabla_imputacion.loc[clave, col]
                        df.at[idx, col] = valor
            return df
        
        # Imputar usando criterios de entrenamiento
        X_copy = imputar_con_tablas(X_copy, criterio_media, media_tabla)
        X_copy = imputar_con_tablas(X_copy, criterio_mediana, mediana_tabla)
        
        # Eliminar Month 
        X_copy = X_copy.drop('Month', axis=1)
        
        return X_copy
        
    except Exception as e:
        print(f"Error en imputación: {e}")
        raise


def eliminar_target(X):
    """Elimina la columna target si existe"""
    try:
        X_copy = X.copy()
        
        if 'RainTomorrow' in X_copy.columns:
            X_copy = X_copy.drop('RainTomorrow', axis=1)
        
        return X_copy
        
    except Exception as e:
        print(f"Error eliminando target: {e}")
        raise


# ===== WRAPPERS =====

def mapeo_wrapper(df):
    resultado = mapeo_regiones(df, 'Location')
    print("Mapeo de regiones completado")
    return resultado


def eliminar_target_wrapper(X):
    resultado = eliminar_target(X)
    print("Target eliminado")
    return resultado


def codificacion_fecha_wrapper(X):
    resultado = codificacion_fecha(X)
    print("Codificación de fechas completada")
    return resultado


def codificacion_viento_wrapper(X):
    resultado = codificacion_viento(X)
    print("Codificación de viento completada")
    return resultado


def codificacion_rain_wrapper(X):
    resultado = codificacion_rain_today(X)
    print("Codificación de RainToday completada")
    return resultado


def validacion_wrapper(X):
    resultado = validacion_nulos(X)
    print("Validación de nulos completada")
    return resultado


def imputacion_wrapper(X):
    resultado = imputacion_nulos(X)
    print("Imputación de nulos completada")
    return resultado



# ===== CARGAR MODELO =====

reglog_model = joblib.load('models/reg_log_balanced_model.pkl')
nn_model = joblib.load('models/nn_optuna_model.pkl') 
automl_model = load_model('models/automl_model')

# ===== ELEGIR MODELO =====

# Modelo a usar
modelo = reglog_model

# ===== PIPELINE =====

# Preprocesamiento final
preprocesamiento = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), make_column_selector(dtype_exclude='object')), 
        ('cat', OneHotEncoder(drop='first'), make_column_selector(dtype_include='object'))
    ]
)

# Pasos
steps = [
    ('mapeo', FunctionTransformer(mapeo_wrapper, validate=False)),
    ('eliminar_target', FunctionTransformer(eliminar_target_wrapper, validate=False)),
    ('codificacion_fecha', FunctionTransformer(codificacion_fecha_wrapper, validate=False)),
    ('codificacion_viento', FunctionTransformer(codificacion_viento_wrapper, validate=False)),
    ('codificacion_rain', FunctionTransformer(codificacion_rain_wrapper, validate=False)),  
    ('validacion_nulos', FunctionTransformer(validacion_wrapper, validate=False)),
    ('imputacion', FunctionTransformer(imputacion_wrapper, validate=False)),
    ('preprocesamiento', preprocesamiento),
    ('modelo', modelo)
]

pipeline = Pipeline(steps)

# ===== PREDICCIONES =====
# prediccion = pipeline.predict(datos_nuevos)
#print(f"{'Llueve' if prediccion[0] == 1 else 'No llueve'}")






