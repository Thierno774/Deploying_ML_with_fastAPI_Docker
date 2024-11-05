from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Optional

## Load the model 
model = joblib.load('app/iris_classifier.joblib')

class_names = np.array(["setosa" ,"visicolor", "virginica"])

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



class IrisData(BaseModel): 
    sepal_length: Optional[float] = None 
    sepal_width: Optional[float] = None
    petal_length: Optional[float] = None
    petal_width: Optional[float] = None




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

    
