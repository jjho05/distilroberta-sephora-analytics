FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Comando oficial para ejecutar Shiny en Hugging Face Spaces (puerto 7860 es mandatorio)
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
