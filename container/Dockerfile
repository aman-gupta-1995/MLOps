FROM python:3.12.6-slim-bullseye

WORKDIR /container1

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV FLASK_APP=api:api

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]