from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
from prediction import *
import numpy

# Declaring our FastAPI instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get('/hw')
def hello_world():
    return "Hello World!"

@app.get('/np')
def numpy_version():
    return f"{numpy.__version__}"

@app.get('/hello')
def hello(name:str):
    return f"Hello {name}!"

@app.post('/api/predict')
async def predict_image(file:UploadFile = File(...)):
    image = read_image(await file.read())
    image = preprocess(image)

    conf_score, pred = predict(image)

    # print(f"Score: {conf_score}\tClass: {pred}")
    return f"Score: {conf_score}    Class: {pred}"
if __name__ == "__main__":
    uvicorn.run(app)
