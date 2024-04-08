#!/bin/bash

LOCAL_DOCKER_URL="http://127.0.0.1:8000/"

if [ $(curl -X 'GET' "${LOCAL_DOCKER_URL}" -H 'accept: application/json' -o /dev/null -w '%{http_code}\n' -s) == "200" ]; then
  echo "Endpoint is working."
else
  echo "Endpoint is not reachable. Check your docker container or local server"
  exit 1
fi

echo $(realpath $(dirname $0))

#FILE=$(find ../ -name script.sh)
#
#if [ -f "$FILE" ]; then
#  DIR=$(dirname "$FILE")
#  echo "Changing directory to $DIR"
#  cd "$DIR"
#fi
echo "Changing directory to client"
cd "client"

echo "Choose option: 1) Simple Semantic Similarity (TYPE: simple) or 2) Detail Semantic Similarity (TYPE: dense) "
read  -p "Input " OPTION

echo "You have selected ${OPTION}"

PYTHON_SCRIPT="$(pwd)/run.py"
echo "Invoking api endpoint"


python $PYTHON_SCRIPT ${OPTION}