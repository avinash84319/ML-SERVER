FROM python:3.8-slim-buster

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "ml-server.py"]