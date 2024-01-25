from pydantic_core.core_schema import FieldValidationInfo
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional
import re

class CameraParams(BaseModel):
    brightness: int = Field(ge = -2, le=2, title="The brightness of the camera")
    saturation: int = Field(ge = -2, le=2, title="The saturation of the camera")
    contrast: int = Field(ge = -2, le=2, title="The contrast of the camera")
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