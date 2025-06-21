FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY transformadores.py .    
COPY all_pipelines.pkl .          
COPY inferencia.py .         

RUN mkdir -p /files
CMD ["python", "inferencia.py"]