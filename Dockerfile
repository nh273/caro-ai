FROM python:3.7.10

WORKDIR /app
COPY Pipfile ./
COPY Pipfile.lock ./

RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile --dev
COPY . ./
