FROM python:3
COPY . /InitializationScript
WORKDIR /InitializationScript
RUN ls .
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
