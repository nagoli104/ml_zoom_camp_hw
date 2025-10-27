import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


with open('pipeline_v1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

app = FastAPI(title="customer-churn-prediction")

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

def predict_single(customer):
    result = model.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer):
    


    prob = predict_single(customer.model_dump())

    return {"conversion_probability":float(prob), "convert":bool(prob >= 0.5)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)