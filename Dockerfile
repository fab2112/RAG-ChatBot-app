# RAG-ChatBot-app

# Pull imagem base
FROM python:3.12-slim

# Ensures that Python output is immediately written to stdout/stderr and not buffered:
ENV PYTHONUNBUFFERED=1

# Create working directory 
RUN mkdir -p /home/app/src

# Set the working directory
WORKDIR /home/app/src

# Copy project
COPY src/ /home/app/src 
COPY requirements.txt /home/app/src

# Create a non-root user (appuser) and group (appgroup)
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Change ownership of the application directory to the new user
RUN chown -R appuser:appgroup /home/app/src

# Add the gunicorn installation directory to $PATH
ENV PATH="/home/app/src/venv/bin:${PATH}"

# Add missing system package
RUN apt-get update && \
    apt-get install -y libgssapi-krb5-2 && \
    rm -r /var/lib/apt/lists/*

# Set the timezone to America/Sao_Paulo
ENV TZ=America/Sao_Paulo
RUN apt-get update && \
    apt-get install -y tzdata curl && \
    rm -r /var/lib/apt/lists/*
RUN cp /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# Create a virtual environment, activate it as the appuser user and install the dependencies
RUN su appuser -c "python3 -m venv venv"
RUN su appuser -c ". venv/bin/activate && pip install --upgrade pip && pip install wheel && pip install -r requirements.txt"

# Return appuser
USER appuser

# Start application
ENTRYPOINT ["python", "app.py"]
