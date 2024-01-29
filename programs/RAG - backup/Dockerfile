# Use the official Python image from Docker Hub as the base image
FROM public.ecr.aws/lambda/python:3.11

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements.txt file into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

ADD faiss_index ${LAMBDA_TASK_ROOT}/faiss_index

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
COPY __init__.py ${LAMBDA_TASK_ROOT}

# # RUN echo "deb http://archive.debian.org/debian stretch main" > /etc/apt/sources.lists

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget \
#     nginx \
#     awscli \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/* \
#     && wget -qO - https://packages.grafana.com/gpg.key | apt-key add -

RUN python -m pip install --no-cache-dir -r requirements.txt

# RUN lambda handler
CMD ["lambda_function.lambda_handler"]
