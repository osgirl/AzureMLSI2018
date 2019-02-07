FROM tensorflow/serving:latest
RUN mkdir /InputService
ADD ./requirements.txt ./InputService/requirements.txt
ADD ./install.sh ./InputService/install.sh

COPY . /InputService
WORKDIR /InputService
RUN ./install.sh
RUN CASS_DRIVER_NO_CYTHON=1 pip3 install -r ./requirements.txt
COPY ./models.py /usr/local/lib/python3.5/dist-packages/keras_vggface/
RUN ls .
ENTRYPOINT ["python3"]
CMD ["InputService.py"]
