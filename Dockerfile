# base docker image
FROM python:3.9

# setting the name of working directory in docker image
WORKDIR /semantic_analysis

#RUN apt-get update && apt-get install -y ca-certificates
#COPY /lib/ZscalerRootCertificate-2048-SHA256.crt /usr/share/ca-certificates
#RUN update-ca-certificates

# copy requirements file from current directory to docker image
COPY ./config/requirements.txt /semantic_analysis/requirements.txt

# install requirements in docker image
RUN pip install --no-cache-dir --upgrade -r /semantic_analysis/requirements.txt

# copying code from current directory to image (/semantic_analysis/app) directory
COPY ./app /semantic_analysis/app

# initialise uvicorn server
CMD ["uvicorn", "app.main:webapp", "--host", "127.0.0.1", "--port", "8000"]