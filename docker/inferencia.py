import joblib
import pandas as pd
import warnings
import logging
from sklearn.metrics import classification_report
from transformadores import *

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.simplefilter('ignore')

# Cargamos Pipeline
all_pipelines = joblib.load('all_pipelines.pkl')
logger.info('Pipeline cargado')

# Procesamos
df_input = pd.read_csv('/files/input.csv')
df_test = pd.read_csv('/files/test.csv')
logger.info('Datos cargados')

# Predicciones
output_reglog_cw = all_pipelines['reglog_cw'].predict(df_input)
#output_nn_optuna = all_pipelines['nn_optuna'].predict(df_input)
#output_automl = all_pipelines['automl'].predict(df_input)
logger.info('Predicciones realizadas')

# Guardar resultados
pd.DataFrame(output_reglog_cw, columns=['NEDV_predicted']).to_csv('/files/output_reglog_cw.csv', index=False)
#pd.DataFrame(output_nn_optuna, columns=['NEDV_predicted']).to_csv('/files/output_nn_optuna.csv', index=False)
#pd.DataFrame(output_automl, columns=['NEDV_predicted']).to_csv('/files/output_automl.csv', index=False)
logger.info('Resultados guardados')

# Métricas
logger.info(f'Classification Report RegLog:\n{classification_report(df_test, output_reglog_cw)}')
#logger.info(f'Classification Report NN:\n{classification_report(df_test, output_nn_optuna)}')
#logger.info(f'Classification Report AutoML:\n{classification_report(df_test, output_automl)}')
