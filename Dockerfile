#
FROM python:3.9

#
WORKDIR /semantic_analysis

#RUN apt-get update && apt-get install -y ca-certificates
#COPY /lib/ZscalerRootCertificate-2048-SHA256.crt /usr/share/ca-certificates
#RUN update-ca-certificates

#
COPY ./config/requirements.txt /semantic_analysis/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /semantic_analysis/requirements.txt

#
COPY ./app /semantic_analysis/app

#
CMD ["uvicorn", "app.main:webapp", "--host", "0.0.0.0", "--port", "80"]