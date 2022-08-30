import dill
import pandas
from fastapi import FastAPI
from pydantic import BaseModel
import glob
import os

path = os.environ.get('PROJECT_PATH', '.')

app = FastAPI()
files_path = os.path.join(f'{path}/data/models', '*')
files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)

with open(files[0], 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: object
    event_action: object
    client_id: object
    visit_date: object
    visit_time: object
    visit_number: float
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object


class Prediction(BaseModel):
    client_id: object
    result: int


@app.get('/status')
def status():
    return "Hello, i'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pandas.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    print(y[0])
    return {
        'id': form.client_id,
        'result': y[0]
    }


