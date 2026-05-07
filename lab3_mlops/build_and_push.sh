
#!/bin/bash
DOCKER_USER="idealvista"
IMAGE="lms-sentiment-container"
SHA=$(git rev-parse --short HEAD)

echo "Building $DOCKER_USER/$IMAGE:$SHA"
docker build -t $DOCKER_USER/$IMAGE:$SHA -t $DOCKER_USER/$IMAGE:latest .
docker push $DOCKER_USER/$IMAGE:$SHA
docker push $DOCKER_USER/$IMAGE:latest
echo "ok, everything is fine"


