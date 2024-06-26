from fastapi import FastAPI, HTTPException, UploadFile
from timeit import default_timer as timer
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from database import (
    add_photo,
    retrieve_backend,
    retrieve_photo,
    retrieve_photos,
    retrieve_photo_latest,
    retrieve_config,
    set_config,
    set_backend
)
from stereo import (
    getDimensionsBounding,
    testStereo,
    stereoFuse,
)
from model import (
    BackendValues,
    BoundingBox,
    CameraParams,
    ObjectDimensions,
    Photo,
)
from schedule import (
    getNextTime,
   # startSchedule,
    setSchedule,
)
import cv2 as cv
import numpy as np
import base64
import datetime
import requests
import os
import asyncio

### HELPER FUNCTIONS ###

async def getBackendValues() -> BackendValues:
    values = await retrieve_backend()
    print(type(values['next_wakeup']))
   # dt_object = datetime.datetime.strptime(values['next_wakeup'], '%Y-%m-%d %H:%M:%S')
    dt_object = values['next_wakeup']
    backend_values = BackendValues(
        next_wakeup=dt_object
    )
    return backend_values

async def saveBackendValues():
    await set_backend({
        "next_wakeup": app.backend.next_wakeup
    })

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
        wpc=config['wpc'],
        sleep=config['sleep'],
        sd_save=config['sd_save'],
        auto_sleep=config['auto_sleep'],
        low_light=config['low_light']
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
        "sleep": app.params.sleep,
        "schedule": app.params.schedule,
        "sd_save": app.params.sd_save,
        "auto_sleep": app.params.auto_sleep,
        "low_light": app.params.low_light
    })

async def setESPConfig():
    try:
        requests.post(url=f'{url}/config', json={
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
            "sd_save": app.params.sd_save    
        }, timeout=5)
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, "Couldn't connect to ESP")
    
async def setWUCConfig():
    try:
        headers = {'Content-Type': 'text/plain'}
        requests.post(url=f'{wuc}/sleep', data=f"{app.params.sleep}", timeout=5, headers=headers)
#        requests.get(url=f'{wuc}/exit', timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
        raise HTTPException(500, "Couldn't connect to WUC")


async def captureImageSched():
    print("capturing image")
    try:
        photo = await captureImage()
    except Exception as e:
        print("couldnt capture image since ESP disabled")
    seconds, dt = getNextTime(app.params.schedule)
    print(f"wuc Sleep for {seconds - 30}")
    print(f"next awake {dt}")
    if app.params.auto_sleep == True:
        print("Attempt sleep as auto sleep is on")
        try:
            requests.post(url=f'{wuc}/sleep', data=f"{seconds - 30}", timeout=5)
            requests.get(url=f'{wuc}/exit', timeout=5)
            app.backend.next_wakeup = dt
            await saveBackendValues()
        except Exception as e:
            print(e)
            print("Couldnt sleep or exit since WUC disabled")
    else:
        print("Skipping sleep as autosleep is off")


async def captureImage() -> Photo:
    try:
        # Need to take extra pictures at start if sd_save is on to fix light issues
        if app.params.sd_save == True:
            response = requests.get(url=f'{url}/pic', timeout=15)
            await asyncio.sleep(5)
            response = requests.get(url=f'{url}/pic', timeout=15)
            await asyncio.sleep(5)
        print(f'Capturing from {url}/pic')
        response = requests.get(url=f'{url}/pic', timeout=15)
        print("First pic taken")
        await asyncio.sleep(5)
        print("five seconds passed")
        response2 = requests.get(url=f'{url}/pic', timeout=15)
        print("Second pic taken")
        bytes_image_l = response.content
        bytes_image_r = response2.content
        captured_photo = await fuseAndUploadImages(bytes_image_l, bytes_image_r)
        return captured_photo
    except requests.exceptions.RequestException as e:
        print(e)
        raise HTTPException(500, "Couldn't connect to ESP")
    
    

async def fuseAndUploadImages(bytes_image_l: bytes, bytes_image_r: bytes) -> Photo:
    img_l = cv.imdecode(np.frombuffer(bytes_image_l,np.uint8), cv.IMREAD_COLOR)
    img_r = cv.imdecode(np.frombuffer(bytes_image_r,np.uint8), cv.IMREAD_COLOR)
    if img_l.shape != img_r.shape:
            raise HTTPException(status_code=400,
            detail=f'images are not the same dimensions{img_l.shape} and {img_r.shape}')
    result, disp, left_rect, right_rect = stereoFuse(img_l, img_r)
    _, buffer = cv.imencode('.jpg', result)
    _, left_rect_buf = cv.imencode('.jpg', left_rect)
    _, right_rect_buf = cv.imencode('.jpg', right_rect)
    # img_l_text = base64.b64encode(bytes_image_l)
    # img_r_text = base64.b64encode(bytes_image_r)
    img_l_text = base64.b64encode(left_rect_buf)
    img_r_text = base64.b64encode(right_rect_buf)
    jpg_as_text = base64.b64encode(buffer)
    # these writes arent needed for final
    cv.imwrite("images/img_l.png", img_l)
    cv.imwrite("images/img_r.png", img_r)
    cv.imwrite("images/stereo.png", result)
    new_photo = await add_photo({"brightness": app.params.brightness,
                "saturation": app.params.saturation,
                "contrast": app.params.contrast,
                "timestamp": datetime.datetime.utcnow(),
                "stereo": jpg_as_text,
                "left": img_l_text,
                "right": img_r_text,
               })
    return Photo(id=str(new_photo['_id']),
            brightness=new_photo['brightness'],
                      saturation=new_photo['saturation'],
                     contrast=new_photo['contrast'],
                     image=str(new_photo['stereo'].decode()),
                    left=str(new_photo['left'].decode()),
                    right=str(new_photo['right'].decode()),
                     timestamp=str(new_photo['timestamp']))

### APP SETUP ###
    
url = os.getenv("MCC_URL", "http://192.168.45.145:19520")
wuc = os.getenv("WUC_URL", "http://192.168.45.218:19520")

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
    #startSchedule()
    app.params = await getConfig()
    app.backend = await getBackendValues()
    setSchedule(app.params.schedule, captureImageSched)

### ROUTES ###

@app.get("/parameters", description="Get current params")
async def read_params() -> CameraParams:
    return app.params 

@app.get("/backend_values", description="Get current backend values")
async def read_backend_values() -> BackendValues:
    return app.backend 

@app.put("/set_backend_values", description="Set current backend values")
async def set_backend_values(new_params: BackendValues) -> BackendValues:
    app.backend = new_params
    await saveBackendValues()
    return app.backend

@app.put("/set_parameters", description="Set current parameters")
async def set_params(new_params: CameraParams) -> CameraParams:
    app.params = new_params
    setSchedule(app.params.schedule, captureImageSched)
    await saveConfig()
    await setWUCConfig()
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
                     left=str(photo['left'].decode()),
                     right=str(photo['right'].decode()),
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

@app.post("/get_object_dimensions", description="Get photo from its ID")
async def get_object_dimensions_(bounding_box: BoundingBox,id: str) -> ObjectDimensions:
    start = timer()
    photo = await retrieve_photo(id)
    if not photo:
        raise HTTPException(status_code=404, detail=f'id {id} not found')
    bytes_image_l = base64.decodebytes(photo['left'])
    bytes_image_r = base64.decodebytes(photo['right'])
    img_l = cv.imdecode(np.frombuffer(bytes_image_l,np.uint8), cv.IMREAD_COLOR)
    img_r = cv.imdecode(np.frombuffer(bytes_image_r,np.uint8), cv.IMREAD_COLOR)
    dimensions = getDimensionsBounding(img_l, img_r, bounding_box)
    end = timer()
    print(end-start)
    return dimensions 

@app.get("/wuc_sleep", description="Make WUC sleep for set time")
async def wuc_sleep() -> datetime.datetime:
    try:
        requests.post(url=f'{wuc}/sleep', data=f"{app.params.sleep}", timeout=5)
        requests.get(url=f'{wuc}/exit', timeout=5)
        dtutc = datetime.datetime.utcnow()
        dtutc = dtutc + datetime.timedelta(seconds=app.params.sleep)
        app.backend.next_wakeup = dtutc
        print(f"Next wakeup after exit is {dtutc}")
        await saveBackendValues()
        return dtutc
    except Exception as e:
        print(e)
        print("Couldnt sleep or exit since WUC disabled")
        raise HTTPException(500, "Couldn't connect to ESP")


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}

@app.get("/test_stereo")
async def test_stereo():
    await testStereo()
