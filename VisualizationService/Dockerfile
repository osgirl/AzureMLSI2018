FROM python:3
COPY . /VisualizationService
WORKDIR /VisualizationService
RUN ls .
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["VisualizationService.py"]
