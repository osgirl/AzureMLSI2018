FROM kkdai/docker-python3-opencv-face_recognition
RUN mkdir /InputService
ADD ./requirements.txt ./InputService/requirements.txt
RUN CASS_DRIVER_NO_CYTHON=1 pip install -r ./InputService/requirements.txt

COPY . /InputService
RUN ls ./InputService
WORKDIR /InputService
ENTRYPOINT ["python"]
CMD ["InputService.py"]