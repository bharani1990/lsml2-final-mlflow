# MLflow usage with docker compose.
MLflow Tracking server with nginx as frontend, MLflow runs are stored in the backend with MYSQL. 
  - Toy tweet classifier model with flask web app. 
  - Containerized with docker-compose up.
  - Configure the SQL server with a .env file for more security. 
  - Build with ```docker-compose up --build -d```.
  
Project

- [x] 1. Project documentation

        Add this tomo
        
    1.1. design document

        Add this tomo
        
    1.2. run instructions (env, commands) env can be seen here - https://github.com/bharani1990/lsml2-final-mlflow/blob/main/.env
         
         docker-compose up --build -d
      
    1.3. architecture, losses, metrics

      architecture - 
        A MySQL database server,
        A MLflow server,
        A reverse proxy NGINX
      losses - 
        log_loss
      metrics - 
        accuracy

- [x] 2. Data set 
      is from this repo - https://github.com/sharmaroshan/Twitter-Sentiment-Analysis
- [x] 3. Model training code 

    3.1. Jupyter Notebook 
      is seen here - https://github.com/bharani1990/lsml2-final-mlflow/blob/main/app/task.ipynb

    3.2. MLFlow project
      is seen here - https://github.com/bharani1990/lsml2-final-mlflow/tree/main/mlflow

- [x] 4. Service deployment and usage instructions
    
    4.1. dockerfile or docker-compose file
      is seen here - https://github.com/bharani1990/lsml2-final-mlflow/blob/main/docker-compose.yml
    
    4.2. required services: databases
    is seen here - https://github.com/bharani1990/lsml2-final-mlflow/tree/main/nginx
      
    4.3. client for service
      is seen here - https://github.com/bharani1990/lsml2-final-mlflow/tree/main/app
     
    4.4. model
      a) Experiment was done on five different algo
         i) Logistic regression
         ii) SGD Classifier
         iii) XGBoost Classifier
         iv) Random Forest Classifier
         v) Decision Tree Classifier
      b) Random forest was the winner and it was pushed to production in MLFlow
      c) This was used for prediction also (eg can be see in the task.ipynb)
      d) model can be seen here - https://github.com/bharani1990/lsml2-final-mlflow/blob/main/app/best_pipe.pkl

