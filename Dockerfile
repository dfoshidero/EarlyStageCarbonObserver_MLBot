# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM python:3.10 AS build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}

# Install the function's dependencies
RUN pip install --no-cache-dir --target ${FUNCTION_DIR} awslambdaric && \
    pip install --no-cache-dir --target ${FUNCTION_DIR} -r ${FUNCTION_DIR}/requirements.txt

# Set PYTHONPATH to include the target directory
ENV PYTHONPATH="${FUNCTION_DIR}:${PYTHONPATH}"

# Ensure additional dependencies are installed and downloaded
RUN python -m spacy download en_core_web_trf && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-mpnet-base-v2')" && \
    python ${FUNCTION_DIR}/download_nltk_data.py


# Use a slim version of the base Python image to reduce the final image size
FROM python:3.10-slim

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Set PYTHONPATH to include the target directory
ENV PYTHONPATH="${FUNCTION_DIR}:${PYTHONPATH}"

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
# Pass the name of the function handler as an argument to the runtime
CMD [ "app.lambda_handler" ]