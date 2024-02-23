import motor.motor_asyncio
from bson.objectid import ObjectId
from typing import List, Optional
import os

mongo = os.getenv("MONGO", "ss")
client = motor.motor_asyncio.AsyncIOMotorClient(mongo)

database = client.photos

photo_collection = database.get_collection("photos_collection")
config_collection = database.get_collection("config_collection")

# Retrieve all photos present in the d0atabase
async def retrieve_config() -> dict:
    configs = []
    async for config in config_collection.find():
        configs.append(config)
    return configs[0]

async def set_config(config_data: dict):
    # Delete config data
    await config_collection.drop()
    config = await config_collection.insert_one(config_data)
    new_config = await config_collection.find_one({"_id": config.inserted_id})
    return new_config

# Retrieve all photos present in the database
async def retrieve_photos():
    photos = []
    async for photo in photo_collection.find():
        photos.append(photo)
    return photos


# Add a new photo into to the database
async def add_photo(photo_data: dict) -> dict:
    photo = await photo_collection.insert_one(photo_data)
    new_photo = await photo_collection.find_one({"_id": photo.inserted_id})
    return new_photo


# Retrieve a photo with a matching ID
async def retrieve_photo(id: str) -> dict:
    if not ObjectId.is_valid(id):
        return
    photo = await photo_collection.find_one({"_id": ObjectId(id)})
    if photo:
        return photo
    
# Retrieve latest photo
async def retrieve_photo_latest() -> dict:
    photos =  []
    async for photo in photo_collection.find().limit(1).sort({"_id": -1}):
        photos.append(photo)
    if photo:
        return photo

async def retrieve_photos(offset: int, limit: int) -> List[dict]:
    photos =  []
    async for photo in photo_collection.find().skip(offset).limit(limit).sort({"_id": -1}):
        photos.append(photo)
    return photos
