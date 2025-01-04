FROM python:3.10.13-slim-bookworm AS base

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ARG POETRY_VERSION=1.7.1

RUN pip install poetry==${POETRY_VERSION}

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV WORKING_DIR_PATH=/app
RUN mkdir $WORKING_DIR_PATH
WORKDIR ${WORKING_DIR_PATH}

COPY pyproject-poetry.toml poetry.lock ${WORKING_DIR_PATH}/
RUN mv pyproject-poetry.toml pyproject.toml
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

COPY src ${WORKING_DIR_PATH}/src
COPY app.py ${WORKING_DIR_PATH}/
COPY gunicorn.conf.py ${WORKING_DIR_PATH}/
COPY config.yml ${WORKING_DIR_PATH}/
COPY data ${WORKING_DIR_PATH}/data

EXPOSE 5000
ENTRYPOINT [ "poetry", "run", "gunicorn", "app:app"]
