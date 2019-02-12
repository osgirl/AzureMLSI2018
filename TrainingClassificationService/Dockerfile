FROM tensorflow/serving:latest
RUN mkdir /TrainingClassificationService
ADD ./requirements.txt ./TrainingClassificationService/requirements.txt
ADD ./install.sh ./TrainingClassificationService/install.sh

COPY . /TrainingClassificationService
WORKDIR /TrainingClassificationService
RUN ./install.sh
RUN CASS_DRIVER_NO_CYTHON=1 pip3 install -r ./requirements.txt
COPY ./models.py /usr/local/lib/python3.5/dist-packages/keras_vggface/
RUN ls .
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python3","TrainingClassificationService.py"]