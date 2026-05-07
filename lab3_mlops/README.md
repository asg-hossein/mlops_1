# Lab3 

git clone

cd lab3_mlops

docker build -t lms-sentiment-container .

docker compose up -d

docker ps

# для теститрования

curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"text": "I love this one!"}'

# пример ответа

{"sentiment":"POSITIVE","score":0.98}
 
https://hub.docker.com/r/idealvista/lms-sentiment-container

docker stop lms-sentiment-container