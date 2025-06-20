# Predicción de Lluvia - Modelo en Docker

Modelo de machine learning para predecir lluvia del día siguiente basado en datos meteorológicos australianos.

## Requisitos

- Docker instalado
- Git para clonar el repositorio

## Instalación y Uso

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/AlejandroPeralta01/AA1-TUIA-Prado-Peralta-Duclos.git
cd AA1-TUIA-Prado-Peralta-Duclos
```

### Paso 2: Navegar al directorio Docker
```bash
cd docker
```

### Paso 3: Construir la imagen Docker
```bash
docker build -t rain-predictor .
```

### Paso 4: Ejecutar el modelo
```bash
# Linux/Mac
docker run --rm -v ${PWD}/files:/files rain-predictor

# Windows PowerShell
docker run --rm -v ${PWD}/files:/files rain-predictor

# Windows CMD
docker run --rm -v %cd%/files:/files rain-predictor
```

## Formato de Datos de Entrada

Los datos deben estar en el archivo `files/input.csv` con el siguiente formato:

### Columnas Obligatorias
- **Date**: Fecha en formato YYYY-MM-DD
- **Location**: Ciudad australiana válida (ver lista completa abajo)
- **RainToday**: Valores `Yes` o `No`

### Columnas Opcionales
Todas las demás columnas meteorológicas son opcionales y se imputarán automáticamente si faltan:

MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm

### Ejemplo de archivo input.csv
```csv
Date,Location,MinTemp,MaxTemp,Rainfall,RainToday
2024-01-01,Sydney,13.4,22.9,0.6,No
2024-01-02,Melbourne,15.2,25.1,0.0,No
2024-01-03,Brisbane,18.7,28.3,2.1,Yes
```

## Ciudades Soportadas

Adelaide, Albany, Albury, BadgerysCreek, Ballarat, Bendigo, Brisbane, Cairns, Canberra, Cobar, CoffsHarbour, Dartmoor, Darwin, GoldCoast, Hobart, Katherine, Launceston, Melbourne, MelbourneAirport, Mildura, Moree, MountGambier, MountGinini, Newcastle, Nhil, NorahHead, NorfolkIsland, Nuriootpa, PearceRAAF, Penrith, Perth, PerthAirport, Portland, Richmond, Sale, Sydney, SydneyAirport, Townsville, Tuggeranong, Uluru, WaggaWagga, Walpole, Watsonia, Williamtown, Witchcliffe, Wollongong, Woomera

## Resultados

Los resultados se generan automáticamente en `files/output.csv` con el formato:

```csv
NEDV_predicted
0
1
0
```

Donde:
- `0` = No lloverá mañana
- `1` = Lloverá mañana

## Estructura de Archivos

```
docker/
├── files/
│   ├── input.csv          # Datos de entrada (modificar aquí)
│   └── output.csv         # Resultados generados
├── Dockerfile
├── inferencia.py
├── pipeline.pkl
├── requirements.txt
└── transformadores.py
```

