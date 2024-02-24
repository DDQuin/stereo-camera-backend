from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from database import (
    add_photo,
    retrieve_photo,
    retrieve_photos,
    retrieve_photo_latest,
    retrieve_config,
    set_config
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
import time

### HELPER FUNCTIONS ###

async def getConfig() -> CameraParams:
    config = await retrieve_config()
    configCam = CameraParams(
        brightness=config['brightness'],
        saturation=config['saturation'],
        contrast=config['contrast'],
        schedule=config['schedule'],
        special_effect=config['special_effect'],
        wb_mode=config['wb_mode'],
        ae_level=config['ae_level'],
        aec_value=config['aec_value'],
        agc_gain=config['agc_gain'],
        gainceiling=config['gainceiling'],
        lenc=config['lenc'],
        agc=config['agc'],
        aec=config['aec'],
        hmirror=config['hmirror'],
        vflip=config['vflip'],
        aec2=config['aec2'],
        bpc=config['bpc'],
        wpc=config['wpc']
                 )
    return configCam

async def saveConfig():
    await set_config({
        "brightness": app.params.brightness,
        "saturation": app.params.saturation,
        "contrast": app.params.contrast,
        "special_effect": app.params.special_effect,
        "wb_mode": app.params.wb_mode,
        "ae_level": app.params.ae_level,
        "aec_value": app.params.aec_value,
        "agc_gain": app.params.agc_gain,
        "gainceiling": app.params.gainceiling,
        "lenc": app.params.lenc,
        "agc": app.params.agc,
        "aec": app.params.aec,
        "hmirror": app.params.hmirror,
        "vflip": app.params.vflip,
        "aec2": app.params.aec2,
        "bpc": app.params.bpc,
        "wpc": app.params.wpc,
        "schedule": app.params.schedule
    })

async def setESPConfig():
    response = requests.post(url=f'{url}/config', json={
        "brightness": app.params.brightness,
        "saturation": app.params.saturation,
        "contrast": app.params.contrast
        "special_effect": app.params.special_effect,
        "wb_mode": app.params.wb_mode,
        "ae_level": app.params.ae_level,
        "aec_value": app.params.aec_value,
        "agc_gain": app.params.agc_gain,
        "gainceiling": app.params.gainceiling,
        "lenc": app.params.lenc,
        "agc": app.params.agc,
        "aec": app.params.aec,
        "hmirror": app.params.hmirror,
        "vflip": app.params.vflip,
        "aec2": app.params.aec2,
        "bpc": app.params.bpc,
        "wpc": app.params.wpc,
        })
    if response.status_code != 200:
        raise HTTPException("Something went wrong")

async def captureImage() -> Photo:
    print("Capturing image from ESP")
    response = requests.get(url=f'{url}/cam1')
    response2 = requests.get(url=f'{url}/cam2')
    if response.status_code != 200 or response2.status_code != 200:
        raise HTTPException("Something went wrong")
    bytes_image_l = response.content
    bytes_image_r = response2.content
    captured_photo = await fuseAndUploadImages(bytes_image_l, bytes_image_r)
    return captured_photo

async def fuseAndUploadImages(bytes_image_l: bytes, bytes_image_r: bytes) -> Photo:
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
    cv.imwrite("images/img_l.png", img_l)
    cv.imwrite("images/img_r.png", img_r)
    cv.imwrite("images/stereo.png", result)
    new_photo = await add_photo({"brightness": app.params.brightness,
                "saturation": app.params.saturation,
                "contrast": app.params.contrast,
                "timestamp": datetime.datetime.now(),
                "stereo": jpg_as_text,
                "left": img_l_text,
                "right": img_r_text,
               })
    return Photo(id=str(new_photo['_id']),
            brightness=new_photo['brightness'],
                      saturation=new_photo['saturation'],
                     contrast=new_photo['contrast'],
                     image=str(new_photo['stereo'].decode()),
                     timestamp=str(new_photo['timestamp']))

### APP SETUP ###
    
url = os.getenv("MCC_URL", "http://localhost:8090")
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event('startup')
async def app_startup():
    startSchedule()
    app.params = await getConfig()
    setSchedule(app.params.schedule, captureImage)

### ROUTES ###

@app.get("/parameters", description="Get current parameters")
async def read_params() -> CameraParams:
    return app.params 

@app.put("/set_parameters", description="Set current parameters")
async def set_params(new_params: CameraParams) -> CameraParams:
    app.params = new_params
    setSchedule(app.params.schedule, captureImage)
    await saveConfig()
    await setESPConfig()
    return app.params

@app.get("/take_photo", description="Make MCC capture image and receive it")
async def take_photo() -> Photo:
    captured_photo = await captureImage()
    return captured_photo

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
