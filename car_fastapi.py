from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import joblib
import pandas as pd 

# Import the ml_model
with open('car_predictor.pkl','rb') as f:
    ml_model = joblib.load(f)

app = FastAPI()

# create pydantic model to validate.

class UserInput(BaseModel):
    Brand:Annotated[Literal['Hyundai', 'Volkswagen', 'Toyota', 'Honda', 'Maruti', 'Mahindra',
       'Tata', 'Kia'],Field(...,description=("Car Brand name"))]
    Car_Age: Annotated[int,Field(...,description=("Enter your car age"),gt=0,lt=15)]
    Kilometers_Driven:Annotated[int,Field(...,description=("How much car drove"))]
    Engine_CC:Annotated[int,Field(...,description=("How much is Engine power."))]
    Mileage_KMPL:Annotated[float,Field(...,description=("Enter car milage in KMPL"))]
    Fuel_Type:Annotated[Literal['Petrol', 'Diesel', 'CNG', 'Electric'],Field(...,description=("Enter fuel type."))]
    Transmission:Annotated[Literal['Manual', 'Automatic'],Field(...,description="Enter Transmission type.")]
    Previous_Owners:Annotated[int,Field(...,description=("Number of previous owner."))]
    Seats:Annotated[Literal[5,7],Field(...,description="Enter number of seats.")]
    Location:Annotated[Literal['Delhi', 'Pune', 'Kolkata', 'Hyderabad', 'Chennai', 'Durgapur',
       'Bengaluru', 'Mumbai'],Field(...,description=("Enter the city name from where the car from."))]
    

@app.post("/predict")
def predict_car_price(data:UserInput):

    input = pd.DataFrame([
        {
            "Brand":data.Brand,
            "Car_Age":data.Car_Age,
            "Kilometers_Driven":data.Kilometers_Driven,
            "Engine_CC":data.Engine_CC,
            "Mileage_KMPL":data.Mileage_KMPL,
            "Fuel_Type":data.Fuel_Type,
            "Transmission":data.Transmission,
            "Previous_Owners":data.Previous_Owners,
            "Seats":data.Seats,
            "Location":data.Location
        }
    ])

    prediction =  ml_model.predict(input)[0]

    return JSONResponse(status_code=200,content={'predicted_category':prediction})
