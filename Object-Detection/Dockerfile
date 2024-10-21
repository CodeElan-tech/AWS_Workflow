FROM continuumio/miniconda3:latest
WORKDIR /app


COPY conda.yaml .
RUN conda env create -f conda.yaml

RUN echo "source activate object_detection_env" > ~/.bashrc
ENV PATH /opt/conda/envs/object_detection_env/bin:$PATH


COPY . .
EXPOSE 5000
CMD ["mlflow", "run", ".", "-e", "run_all"]
