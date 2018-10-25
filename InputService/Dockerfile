FROM python:3
COPY . /InputService
WORKDIR /InputService
RUN ls .
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["InputService.py"]