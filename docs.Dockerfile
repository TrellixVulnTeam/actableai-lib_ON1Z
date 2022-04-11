FROM python:3.7 AS docs

WORKDIR /data

RUN apt-get update -y \
    && apt-get install -y build-essential libcurl4-openssl-dev r-base \
    && apt-get install -y libssl-dev libc-dev \
    && R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

COPY ./requirements.txt ./
COPY ./third_parties/autogluon ./third_parties/autogluon

RUN pip install --upgrade pip \
    && pip install --use-deprecated=legacy-resolver -r requirements.txt \
    && pip install --use-deprecated=legacy-resolver fbprophet==0.7.1 \
    && pip install sphinx==4.5.0 maisie-sphinx-theme==0.1.2

COPY . .

RUN pip install --no-deps -e .

RUN sphinx-apidoc -o ./docs actableai
RUN cd ./docs && make html


FROM nginx:alpine

COPY --from=docs /data/docs/_build/html/. /usr/share/nginx/html

EXPOSE 80
