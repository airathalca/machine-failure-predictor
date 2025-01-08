from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

from machine_failure.pipeline.prediction_pipeline import MachineClassifier, MachineData

app = FastAPI()
templates = Jinja2Templates(directory='templates')
origins = ["*"]
app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

model = MachineClassifier()

class DataForm:
  def __init__(self, request: Request):
    self.request: Request = request
    self.product_id: Optional[int] = None
    self.type: Optional[str] = None
    self.air_temperature: Optional[float] = None
    self.process_temperature: Optional[float] = None
    self.rotational_speed: Optional[int] = None
    self.torque: Optional[float] = None
    self.tool_wear: Optional[int] = None
    self.TWF: Optional[int] = None
    self.HDF: Optional[int] = None
    self.PWF: Optional[int] = None
    self.OSF: Optional[int] = None

  async def get_machine_data(self):
    form = await self.request.form()
    self.product_id = int(form.get("product_id"))
    self.type = form.get("type")
    self.air_temperature = float(form.get("air_temperature"))
    self.process_temperature = float(form.get("process_temperature"))
    self.rotational_speed = int(form.get("rotational_speed"))
    self.torque = float(form.get("torque"))
    self.tool_wear = int(form.get("tool_wear"))
    self.TWF = 1 if form.get("TWF") else 0
    self.HDF = 1 if form.get("HDF") else 0
    self.PWF = 1 if form.get("PWF") else 0
    self.OSF = 1 if form.get("OSF") else 0

@app.get("/", tags=["authentication"])
async def index(request: Request):
  return templates.TemplateResponse("index.html",{"request": request, "context": "Rendering"})

@app.post("/predict")
async def predictRouteClient(request: Request):
  try:
    form = DataForm(request)
    await form.get_machine_data()
    
    machine_data = MachineData(form.product_id, form.type, form.air_temperature, 
                              form.process_temperature, form.rotational_speed, form.torque, 
                              form.tool_wear, form.TWF, form.HDF, form.PWF, form.OSF)
    df = machine_data.convert_to_pandas()
    value = model.predict(df)[0]
    status = 'Machine Failure' if value == 1 else 'Machine is OK'

    return JSONResponse(content={"status": status})
      
  except Exception as e:
    return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)

if __name__ == "__main__":
  app_run(app, host='0.0.0.0', port=3000)