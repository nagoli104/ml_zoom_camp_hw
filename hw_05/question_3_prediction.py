import pickle

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


lead ={
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

X = dv.transform([lead])

y_pred = model.predict_proba(X)[0, 1]
conversion = y_pred >= 0.5

print(f"Prediction:{y_pred}, Conversion: {conversion}")