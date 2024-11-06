# Machine-learning-pipeline-deployment-using-Docker-and-FastAPI

This project is about deploying the trained machine learning pipeline using FastAPI and Docker. The ml pipeline which will be deployed is taken from my repository [Thierno774/Deploying_ML_with_fastAPI_Docker](https://github.com/Thierno774/Deploying_ML_with_fastAPI_Docker/blob/main/).

The ml pipeline includes a RandomForestClassifier for classifying the loan borrowers as **defaulted / not-defaulted**.

## Building API using FastAPI framework
The API code must be in **'api.py'** file within a directory **'app'** according to FastAPI guidelines.

Import the required packages
```python
# Imports for server
from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Optional

# App name
app = FastAPI(openapi_tags = [{
                    "name": "API for Iris classifier",
                    "description": "default functions"
}], 
                title = "API for Iris classifier", 
                description = "My owner API",
                contact = {
                        "name" : "Thierno BAH", 
                        "email" : "thiernosidybah232@gmail.com"
                        })

```

### Representing the loan data point
To represent a sample of loan details along with the data type of each atttribute, a class needs to be defined using the ```BaseModel``` from the pydantic library.
```python
# defining base class for Loan to represent a data point for predictions 
class IrisData(BaseModel): 
    sepal_length: Optional[float] = None 
    sepal_width: Optional[float] = None
    petal_length: Optional[float] = None
    petal_width: Optional[float] = None
```

### Loading the trained Machine learning pipeline
The trained machine learning pipeline needs to be loaded into memory, so it can be used for predictions in future.

One way is to load the machine learning pipeline during the startup of our ```Server```. To do this, the function needs to be decorated with ```@app.on_event("startup")```. This decorator ensures that the function loading the ml pipeline is triggered right when the Server starts.

The ml pipeline is stored in `app/models.py' directory.

```python
@app.on_event("startup")
def load_ml_pipeline():
    # loading the machine learning Models
    global RFC_pipeline
   model = joblib.load('app/iris_classifier.joblib')

```

### Server Endpoint for Prediction
Finally, an endpoint on our server handles the **prediction requests** and return the value predicted by our deployed ml pipeline.

The endpoint is **server/predict** with a **POST** operation. 

Finally, a JSON response is returned containing the prediction

```python
# Defining the function for handling the prediction requests, it will be run by ```/predict``` endpoint of server
# and expects an instance inference request of Loan class to make prediction
@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Classification API"}

@app.post("/predict")
async def predict_iris(data: IrisData)->dict:
    # Create a dictionary
    new_data = [[
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width,
            ]]
    
    # Make prediction
    class_idx = model.predict(new_data)[0]
    predictions_class = class_names[class_idx]
    ## Return predictions
    return {"prediction_class" : predictions_class}

```
### Server
As our API has been built, the Uvicorn Server can be use the API to serve the prediction requests. But for now, this server will be dockerized. And final predictions will be served by the Docker container.

## Dockerizing the Server
The Docker container will be run on localhost.
```
..
└── Base dir
    ├── app/
    │   ├── models.py (server code)
    │   └── iris_classifier.py (file containing the ML Model)
    ├── requirements.txt (Python dependencies)
    ├── loan-examples/ (loan examples to test the server)
    ├── README.md (this file)
    └── Dockerfile
```
## Creating the Dockerfile
Now in the base directory, a file is created ```Dockerfile``. The ```Dockerfile``` contain all the instructions required to build the docker image.

```Dockerfile
FROM python:3.11 

WORKDIR   /code 

COPY ./requirements.txt /code/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install fastapi uvicorn joblib numpy scikit-learn

COPY . /code/app/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Base Image
The `FROM` instruction allows to use a pre-existing image as base of our new docker image, instead of writing our docker image from the scratch. This allows the software in pre-existing image to be available in our new docker image. 

Other existing images, can be used as base image of our new docker image, but size of those is a lot heavier. So using the one mentioned, as it a great image for required task.

### Installing Dependencies
Now our docker image has environment with python installed, so the dependencies required for serving the inference requests need to be installed in our docker image.

The dependencies are written in requirements.txt file in our base dir. This file needs to be copied in our docker image `COPY requirements.txt .` and then the dependencies are installed by `RUN pip install -r requirements.txt`

### Exposing the port
Our server will listen to inference requests on port 8000.
```Dockerfile
EXPOSE 8000
```

### Copying our App into Docker image
Our app should be inside the docker image.
```Dockerfile
COPY . /code/app
```

### Spinning up the server
Dockers are efficient at carrying out single task. When a docker container is run, the `CMD` commands get executed only once. This is the command which will start our server by specifying the `host` and `post`, when a docker container created from our docker image is started.
```Dockerfile
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Build the Docker Image
Now in base dir, the Dockerfile is present. The Docker image is built using the `docker build` command:
```Dockerfile
docker build -t ml_pipeline:RFC
```
The `-t` flags is used for specifying the **name:tag** of docker image.

## Run the Docker Container
Now that the docker image is created. To run a docker container out of it:
```Dockerfile
docker container run -d --name container_name -p 80000:8000 ml_pipeline:RFC
```
The `-p 8000:8000` flag performs port mapping operations. The container and as well as local machine, has own set of ports. As our container is exposed on port 80, so it needs to be mapped to a port on local machine which is also 8000.
<p align="center">
  <img src="/other/images/1.png">
</p>

## Make Inference Requests to Dockerized Server
Now that our server is listening on port 80, a `POST` request can be made for predicting the class of loan.

The requests should contain the data in `JSON` format.
```JSON
{
"sepal_length":2.5 ,
"sepal_width": 3.5 ,
"petal_length" : 2 , 
"petal_width" : 1,
}
```
### FastAPI built-in Client
FastAPI has a built-in client to interact with the deployed server.
<p align="center">
  <img src="/other/images/3.png">
  <img src="/other/images/2.png">
</p>

### Using `curl` to send request
`curl` command can be used to send the inference request to deployed server.
```bash
curl -X POST http://localhost:80/predict \
    -d @./loan-examples/1.json \
    -H "Content-Type: application/json"
```
Three flags are used with `curl`:
`-X`: to specify the type of request like `POST`
`-d`: data to be sent with request
`-H`: header to specify the type of data sent with request

The directory `loan-examples` has 2 json files containing the loan samples for prediction, for testing the deployed dockerized server.
