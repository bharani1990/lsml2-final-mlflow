FROM jupyter/scipy-notebook

# Copy requirements first to validate pip cache to prevent rebuild
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Create the environment:
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the other files
COPY . /app

# Run jupyter
RUN python _preprocess.py
RUN python _loaders.py

CMD ["python", "app.py"]