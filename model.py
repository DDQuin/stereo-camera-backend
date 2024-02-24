from pydantic_core.core_schema import FieldValidationInfo
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional
import re

class CameraParams(BaseModel):
    brightness: int = Field(ge = -2, le=2, title="The brightness of the camera")
    saturation: int = Field(ge = -2, le=2, title="The saturation of the camera")
    contrast: int = Field(ge = -2, le=2, title="The contrast of the camera")
    special_effect: int = Field(ge = 0, le=6, title="Special effect")
    wb_mode: int = Field(ge = 0, le=4, title="WB Mode")
    ae_level: int = Field(ge = -2, le=2, title="AE level")
    aec_value: int = Field(ge = 0, le=1200, title="AEC value")
    agc_gain: int = Field(ge = 0, le=30, title="AGC Gain")
    gainceiling: int = Field(ge = 0, le=6, title="Gain Ceiling")
    lenc: bool = Field(title="Lenc")
    agc: bool = Field(title="Agc")
    aec: bool = Field(title="Aec")
    hmirror: bool = Field(title="Hmirror")
    vflip: bool = Field(title="VFlip")
    aec2: bool = Field(title="Aec2")
    bpc: bool = Field(title="Bpc")
    wpc: bool = Field(title="Wpc")
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
