# Use Python 3.10 base image
FROM python:3.10-slim

# Avoid writing .pyc files and enable unbuffered output for logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /streamlit_app

# Copy all project files from the local machine to the container
COPY . /streamlit_app

# Upgrade pip and install all required dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run model.py first (to prepare the model) then start the Streamlit app
CMD bash -c "python src/model.py && streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501 --server.enableCORS=false"
