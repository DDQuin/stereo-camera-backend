from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic_core.core_schema import FieldValidationInfo
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from database import (
    add_photo,
    retrieve_photo,
    retrieve_photos,
    retrieve_photo_latest,
)
import cv2 as cv
import numpy as np
import base64
import re
import datetime
import os



### SCHEDULER SETUP ###

scheduler = BackgroundScheduler()
jobstores = {
    'default': MemoryJobStore()
}
executors = {
    'default': ThreadPoolExecutor(20),
    'processpool': ProcessPoolExecutor(5)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 3
}
scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc)
scheduler.start()


async def takeScreenshot(file1: str, file2: str):
    print("Taking camera photo - mock")
#    imgL = cv.imread(file1, cv.imread_grayscale)
#    imgr = cv.imread(file2, cv.imread_grayscale)
#    stereo = cv.stereobm.create(numdisparities=16, blocksize=15)
#    disparity = stereo.compute(imgl,imgr)
#    _, buffer = cv.imencode('.jpg', disparity)
#    img_l_text = base64.b64encode(imgl)
#    img_r_text = base64.b64encode(imgr)
#    jpg_as_text = base64.b64encode(buffer)
#    cv.imwrite("images/stereo.png", disparity)
#    new_photo = await add_photo({"brightness": app.params.brightness,
#                "saturation": app.params.saturation,
#                "contrast": app.params.contrast,
#                "timestamp": datetime.datetime.now(),
#                "stereo": jpg_as_text,
#                "left": img_l_text,
#                "right": img_r_text,
#               })
#    return disparity.shape
#

def setSchedule(times: List[str]):
    for job in scheduler.get_jobs():
        print(f'removing job {job}')
        job.remove()
    for time in times:
        print(f'adding schedule {time}')
        hour_min = time.split(":")
        job = scheduler.add_job(takeScreenshot, 'cron', args=("images/small.jpg", "images/small.jpg"), hour=int(hour_min[0]), minute=int(hour_min[1]))


### MODELS ###
class CameraParams(BaseModel):
    brightness: int = Field(ge = 0, le=99, title="The brightness of the camera")
    saturation: int = Field(ge = 0, le=99, title="The saturation of the camera")
    contrast: int = Field(ge = 0, le=99, title="The contrast of the camera")
    schedule: List[str] = Field(title="Schedule of camera")

    @field_validator('schedule')
    def correct(cls, v):
        for time in v:
            time_match = re.search(r'^(([0-1]?[0-9])|(2[0-3])):[0-5][0-9]$', time)
            if time_match == None:
                raise ValueError(f'time {time} did not match HH:MM format')
        if len(v) != len(set(v)):
            raise ValueError(f'There was atleast one duplicate time!')
        return v

class Photo(BaseModel):
    id: str
    timestamp: str
    brightness: int
    saturation: int
    contrast: int
    image: str
    left: Optional[str] = None
    right: Optional[str] = None

### APP SETUP ###

app = FastAPI()

app.params: CameraParams = CameraParams(brightness=2, saturation=0, contrast=0, schedule=["09:38", "09:39"])
setSchedule(app.params.schedule)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### ROUTES ###

@app.get("/parameters")
def read_params() -> CameraParams:
    return app.params

@app.put("/set_parameters")
def set_params(new_params: CameraParams) -> CameraParams:
    app.params = new_params
    setSchedule(app.params.schedule)
    return app.params

@app.get("/take_photo")
def take_photo():
    shape = takeScreenshot("images/tsukaba_l.png", "images/tsukaba_r.png")
    return {"Shape": shape}

@app.get("/get_latest_photo")
async def get_latest_photo() -> Photo:
    photo = await retrieve_photo_latest()
    
    if photo:
        return Photo(id=str(photo['_id']),
            brightness=photo['brightness'],
                      saturation=photo['saturation'],
                     contrast=photo['contrast'],
                     image=str(photo['stereo'].decode()),
                     timestamp=str(photo['timestamp']))
    raise HTTPException(status_code=404, detail=f'id {id} not found')

@app.post("/upload_photos")
async def upload_photos(files: list[UploadFile]):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Must be two files")
    for file in files:
        if "image" not in file.content_type:
            raise HTTPException(status_code=400, detail=f'File must be image, is currently {file.content_type}')
    bytes_image_l = await files[0].read()
    bytes_image_r = await files[1].read()
    img_l = cv.imdecode(np.frombuffer(bytes_image_l,np.uint8), cv.IMREAD_GRAYSCALE)
    img_r = cv.imdecode(np.frombuffer(bytes_image_r,np.uint8), cv.IMREAD_GRAYSCALE)
    if img_l.shape != img_r.shape:
            raise HTTPException(status_code=400, detail=f'images are not the same dimensions! {img_l.shape} and {img_r.shape}')
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img_l, img_r)
    _, buffer = cv.imencode('.jpg', disparity)
    img_l_text = base64.b64encode(bytes_image_l)
    img_r_text = base64.b64encode(bytes_image_r)
    jpg_as_text = base64.b64encode(buffer)
    cv.imwrite("images/output.png", disparity)
    new_photo = await add_photo({"brightness": app.params.brightness,
                "saturation": app.params.saturation,
                "contrast": app.params.contrast,
                "timestamp": datetime.datetime.now(),
                "stereo": jpg_as_text,
                "left": img_l_text,
                "right": img_r_text,
               })
    return {"frame": disparity.shape, "filenames": [file.filename for file in files], "content": [file.content_type for file in files]}



@app.get("/photo")
async def get_photo_by_id(id: str) -> Photo:
    photo = await retrieve_photo(id)
    if photo:
        return Photo(id=str(photo['_id']),
            brightness=photo['brightness'],
                      saturation=photo['saturation'],
                     contrast=photo['contrast'],
                     image=str(photo['stereo'].decode()),
                     left=str(photo['left'].decode()),
                     right=str(photo['right'].decode()),
                     timestamp=str(photo['timestamp']))
    raise HTTPException(status_code=404, detail=f'id {id} not found')


@app.get("/photos")
async def get_photos(offset: int, limit: int) -> List[Photo]:
    photos_dict = await retrieve_photos(offset, limit)
    photos = []

    for photo in photos_dict:
        photos.append(Photo(id=str(photo['_id']),
            brightness=photo['brightness'],
                      saturation=photo['saturation'],
                     contrast=photo['contrast'],
                     image=str(photo['stereo'].decode()),
                     timestamp=str(photo['timestamp'])))
    return photos

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}
