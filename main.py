from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from database import (
    add_photo,
    retrieve_photo,
    retrieve_photos,
    retrieve_photo_latest,
)
from stereo import (
    stereo_fusion,
    testStereo,
)
from model import (
    CameraParams,
    Photo,
)
from schedule import (
    startSchedule,
    setSchedule,
)
import cv2 as cv
import numpy as np
import base64
import datetime
import requests
import os

### HELPER FUNCTIONS ###

async def captureImage():
    print("Capturing image from ESP")
    response = requests.post(url=url, data=f"{app.params.saturation},{app.params.contrast},{app.params.brightness}")
    if response.status_code != 200:
        raise HTTPException("Something went wrong")
    bytes_image_l = response.content
    bytes_image_r = response.content
    await fuseAndUploadImages(bytes_image_l, bytes_image_r)
    print(response.headers)

async def fuseAndUploadImages(bytes_image_l: bytes, bytes_image_r: bytes):
    img_l = cv.imdecode(np.frombuffer(bytes_image_l,np.uint8), cv.IMREAD_COLOR)
    img_r = cv.imdecode(np.frombuffer(bytes_image_r,np.uint8), cv.IMREAD_COLOR)
    if img_l.shape != img_r.shape:
            raise HTTPException(status_code=400,
            detail=f'images are not the same dimensions{img_l.shape} and {img_r.shape}')
    result = stereo_fusion(img_l, img_r)
    _, buffer = cv.imencode('.jpg', result)
    img_l_text = base64.b64encode(bytes_image_l)
    img_r_text = base64.b64encode(bytes_image_r)
    jpg_as_text = base64.b64encode(buffer)
    cv.imwrite("images/stereo.png", result)
    new_photo = await add_photo({"brightness": app.params.brightness,
                "saturation": app.params.saturation,
                "contrast": app.params.contrast,
                "timestamp": datetime.datetime.now(),
                "stereo": jpg_as_text,
                "left": img_l_text,
                "right": img_r_text,
               })

### APP SETUP ###
    
url = os.getenv("MCC_URL", "http://localhost:8090/jpg")

startSchedule()

app = FastAPI()

app.params: CameraParams = CameraParams(brightness=2, saturation=0, contrast=0, schedule=["09:39"])
setSchedule(app.params.schedule, captureImage)


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### ROUTES ###

@app.get("/parameters", description="Get current parameters")
def read_params() -> CameraParams:
    return app.params

@app.put("/set_parameters", description="Set current parameters")
def set_params(new_params: CameraParams) -> CameraParams:
    app.params = new_params
    setSchedule(app.params.schedule)
    return app.params

@app.get("/take_photo", description="Make MCC capture image")
async def take_photo():
    await captureImage()
    return {"Shape": "sss"}

@app.get("/get_latest_photo", description="Return latest photo saved from db")
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

@app.post("/upload_photos", description="Test route to upload photos and perform stereo fusion")
async def upload_photos(files: list[UploadFile]):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Must be two files")
    for file in files:
        if "image" not in file.content_type:
            raise HTTPException(status_code=400,
            detail=f'File must be image, is currently {file.content_type}')
    bytes_image_l = await files[0].read()
    bytes_image_r = await files[1].read()
    await fuseAndUploadImages(bytes_image_l, bytes_image_r)



@app.get("/photo", description="Get photo from its ID")
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


@app.get("/photos", description="Get list of photos")
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


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}

@app.get("/test_stereo")
async def test_stereo():
    await testStereo()
