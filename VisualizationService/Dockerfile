FROM python:3
COPY . /VisualizationService
WORKDIR /VisualizationService
RUN ls .
#CASS_DRIVER_NO_CYTHON speeds up compilation for testin
RUN CASS_DRIVER_NO_CYTHON=1 pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["VisualizationService.py"]
