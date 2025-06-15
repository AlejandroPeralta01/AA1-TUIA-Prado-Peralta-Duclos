import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
import numpy as np



def mapeo_regiones(data_nueva, columna_ciudad='Location'):
    try:
        if columna_ciudad not in data_nueva.columns:
            raise ValueError(f"Columna '{columna_ciudad}' no encontrada")
        
        # Cargar df de ubicaciones
        df_ubicaciones = joblib.load('pkl/df_location.pkl')
        
        # Realizamos una copia para no modificar los originales
        data_copia = data_nueva.copy()
        ubicaciones_copia = df_ubicaciones.copy()
        
        # Merge 
        df_union = data_copia.merge(
            ubicaciones_copia[['Location', 'regiones', 'lat', 'lon']], 
            left_on=columna_ciudad,
            right_on='Location',
            how='left'
        )
        
        # Verificar valores faltantes

        ciudades_invalidas = df_union[df_union['regiones'].isna()]
        
        if len(ciudades_invalidas) > 0:
            ciudades_no_encontradas = data_copia.loc[ciudades_invalidas.index, columna_ciudad].tolist()
            print(f"Ciudades no encontradas: {ciudades_no_encontradas}")
            raise
        
        # Convertimos a object
        df_union['regiones'] = df_union['regiones'].astype('object')

        # Limpiar columnas duplicadas 
        df_final = df_union.drop(['Location'], axis=1)  # Eliminar la del merge
        
        return df_final
        
    except Exception as e:
        print(f"Error en mapeo_regiones: {e}")
        raise

#---------------------------------------------------------------------

def validacion_nulos(X):

    """Rechaza filas con más del 50% de valores faltantes"""
    
    try:
        umbral = X.shape[1] // 2  # 50% de las columnas
        filas_validas = X.notnull().sum(axis=1) > umbral
        filas_malas = (~filas_validas).sum()
        
        if filas_malas > 0:
            raise ValueError(f"{filas_malas} filas tienen más del 50% de nulos, por favor, ingresar datos válidos para hacer la predicción")
        
        return X
        
    except Exception as e:
        print(f"Error en validación de nulos: {e}")
        raise

#---------------------------------------------------------------------

def codificacion_fecha(X):
    """Extrae mes de Date y lo codifica con sin/cos para mantener ciclicidad"""
    
    try:
        X_copy = X.copy()
        
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        
        # Extraer mes y codificar
        mes = X_copy['Date'].dt.month

        X_copy['Month'] = mes.astype(int)

        X_copy['month_sin'] = np.sin(2 * np.pi * mes / 12)
        X_copy['month_cos'] = np.cos(2 * np.pi * mes / 12)
        
        # Eliminar Date
        X_copy = X_copy.drop('Date', axis=1)
        
        return X_copy
        
    except Exception as e:
        print(f"Error en codificación de fecha: {e}")
        raise

#---------------------------------------------------------------------

def codificacion_viento(X):
    """Convierte direcciones de viento a features cíclicas sin/cos"""
    
    try:

        X_copy = X.copy()
        
        # Diccionario de mapeo
        direccion_angulo = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        # Columnas a procesar
        columnas_viento = ['WindDir9am', 'WindDir3pm', 'WindGustDir']
        
        for col in columnas_viento:
            if col in X_copy.columns:

                # Convertir a grados
                grados = X_copy[col].map(direccion_angulo).astype(float)
                
                # Convertir a radianes y aplicar sin/cos
                radianes = np.radians(grados)
                X_copy[f'{col}_sin'] = np.sin(radianes)
                X_copy[f'{col}_cos'] = np.cos(radianes)
        
        # Eliminar columnas originales
        columnas_a_eliminar = [col for col in columnas_viento if col in X_copy.columns]
        X_copy = X_copy.drop(columnas_a_eliminar, axis=1)
        
        return X_copy
        
    except Exception as e:
        print(f"Error en codificación de viento: {e}")
        raise

#---------------------------------------------------------------------

def codificacion_rain_today(X):
    try:
        X_copy = X.copy()
        
        # Eliminamos filas sin RainToday
        X_copy = X_copy.dropna(subset=['RainToday'])
        
        # Verificamos si quedaron fiales
        if X_copy.empty:
            raise ValueError("No se pueden procesar los datos: RainToday es obligatorio")
        
        # Convertir a binario
        X_copy['RainToday'] = X_copy['RainToday'].map({'No': 0, 'Yes': 1})
        
        return X_copy
        
    except Exception as e:
        print(f"Error en codificación de RainToday: {e}")
        raise

#---------------------------------------------------------------------

def imputacion_nulos(X):
    """Imputa igual que en entrenamiento"""
    
    try:
        X_copy = X.copy()
        
        # Cargar tablas y criterios
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
        
        # Imputar igual que en entrenamiento
        X_copy = imputar_con_tablas(X_copy, criterio_media, media_tabla)
        X_copy = imputar_con_tablas(X_copy, criterio_mediana, mediana_tabla)

        if 'month_sin' in X_copy.columns:
            X_copy = X_copy.drop('Month', axis=1)

        return X_copy
        
    except Exception as e:
        print(f"Error en imputación: {e}")
        raise
#---------------------------------------------------------------------}

def eliminar_target(X):

    """Eliminar la columna target si existe"""
    
    try:
        X_copy = X.copy()
        
        if 'RainTomorrow' in X_copy.columns:
            X_copy = X_copy.drop('RainTomorrow', axis=1)
            print("RainTomorrow eliminada (nuestra target)")
        
        return X_copy
        
    except Exception as e:
        print(f"Error eliminando target: {e}")
        raise


# Wrappers
#---------------------------------------------------------------------

def mapeo_wrapper(df):
    return mapeo_regiones(df, 'Location')

#---------------------------------------------------------------------

def validacion_wrapper(X):
    return validacion_nulos(X)

#---------------------------------------------------------------------
def codificacion_fecha_wrapper(X):
    return codificacion_fecha(X)

#---------------------------------------------------------------------

def codificacion_viento_wrapper(X):
    return codificacion_viento(X)

#---------------------------------------------------------------------

def imputacion_wrapper(X):
    return imputacion_nulos(X)

#---------------------------------------------------------------------

def codificacion_rain_wrapper(X):
    return codificacion_rain_today(X)

#---------------------------------------------------------------------

def eliminar_target_wrapper(X):
    return eliminar_target(X)


preprocesamiento = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), make_column_selector(dtype_exclude='object')), 
        ('cat', OneHotEncoder(drop='first'), make_column_selector(dtype_include='object'))
    ]
)

# Definimos los pasos del Pipeline
steps = [
    ('mapeo', FunctionTransformer(mapeo_wrapper, validate=False)),
    ('eliminar_target', FunctionTransformer(eliminar_target_wrapper, validate=False)),
    ('codificacion_fecha', FunctionTransformer(codificacion_fecha_wrapper, validate=False)),
    ('codificacion_viento', FunctionTransformer(codificacion_viento_wrapper, validate=False)),
    ('codificacion_rain', FunctionTransformer(codificacion_rain_wrapper, validate=False)),  
    ('validacion_nulos', FunctionTransformer(validacion_wrapper, validate=False)),
    ('imputacion', FunctionTransformer(imputacion_wrapper, validate=False)),
    

    ('preprocesamiento', preprocesamiento)
    #('modelo', modelo)
]



pipeline = Pipeline(steps)




df = pd.read_csv('weatherAUS.csv', sep=',')

# Tomar una muestra pequeña para probar
df_sample = df.head(10).copy()
print("Datos originales:")
print(df_sample.shape)
print(df_sample.columns.tolist())

# Ejecutar transformaciones
try:
    resultado = pipeline.fit_transform(df_sample)
    print(f"\n✅ Pipeline exitoso!")
    print(f"Shape resultado: {resultado.shape}")
    print(f"Columnas finales: {resultado.columns.tolist()}")
    print(resultado.isna().sum())
    print(resultado.dtypes)

except Exception as e:
    print(f"❌ Error en pipeline: {e}")
