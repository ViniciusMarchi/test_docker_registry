FROM python:3.12-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     software-properties-common

COPY docker/streamlit/run.sh /home/appuser/
RUN chmod +x /home/appuser/run.sh \
    && chown appuser:appuser /home/appuser/run.sh

USER appuser
WORKDIR /home/appuser

COPY docker/streamlit/requirements.txt app/requirements.txt
COPY src/ui/ app/
# COPY src/ui/streamlit_app.py app/streamlit_app.py

ENV VIRTUAL_ENV=/home/appuser/venv
RUN virtualenv ${VIRTUAL_ENV}
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r app/requirements.txt

EXPOSE 8501

ENTRYPOINT ["./run.sh"]