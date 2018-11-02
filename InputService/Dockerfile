FROM python:3
COPY . /InputService
WORKDIR /InputService
RUN ls .
RUN CASS_DRIVER_NO_CYTHON=1 pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["InputService.py"]