import joblib
import pandas as pd
import warnings
import logging
from transformadores import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.simplefilter('ignore')

# Cargamos Pipeline
pipeline = joblib.load('pipeline.pkl')
logger.info('Pipeline cargado')

# Procesamos
df_input = pd.read_csv('/files/input.csv')
logger.info('Datos cargados')

print(df_input.head())

# Predicciónes
output = pipeline.predict(df_input)
logger.info('Predicciones realizadas')

# Guardar resultados
pd.DataFrame(output, columns=['NEDV_predicted']).to_csv('/files/output.csv', index=False)
logger.info('Resultados guardados')
