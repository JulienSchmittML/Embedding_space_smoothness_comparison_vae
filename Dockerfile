FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY dash_app.py .
COPY dash_utils.py .
COPY model.py .
COPY utils.py .
COPY runs /workspace/runs

# Expose the port Dash will run on
EXPOSE 8050

CMD ["python", "dash_app.py"]