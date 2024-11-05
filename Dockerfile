FROM python:3.11 

WORKDIR   /code 

COPY ./requirements.txt /code/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install fastapi uvicorn joblib numpy scikit-learn

COPY . /code/app/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]