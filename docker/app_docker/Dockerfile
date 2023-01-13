FROM python:3.7-slim
WORKDIR /app
COPY . .
COPY ./docker/app_docker/app_requirements.txt ./requirements.txt
RUN rm -r ./docker
RUN pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
EXPOSE 5002