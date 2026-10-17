"""Strict shared input and response contracts."""
import math
import unicodedata
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .data_pipeline import ALL_FEATURES, CATEGORY_OPTIONS

class PatientInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    age: int = Field(ge=1, le=120)
    sex: int
    cp: int
    trestbps: float = Field(ge=60, le=220)
    chol: float = Field(ge=50, le=605)
    fbs: int
    restecg: int
    thalach: float = Field(ge=50, le=250)
    exang: int
    oldpeak: float = Field(ge=0, le=6.5)
    slope: int
    ca: int = Field(ge=0, le=3)
    thal: int
    patient_id: str = Field(default='Anonymous', max_length=64)

    @field_validator(*ALL_FEATURES, mode='before')
    @classmethod
    def valid_number(cls, value, info):
        if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('A finite numeric value is required')
        if info.field_name in CATEGORY_OPTIONS and (type(value) is not int or value not in CATEGORY_OPTIONS[info.field_name]):
            raise ValueError('An exact supported integer category is required')
        return value

    @field_validator('patient_id')
    @classmethod
    def normalize_identifier(cls, value):
        value = unicodedata.normalize('NFKC',value).strip()
        if any(unicodedata.category(c).startswith('C') for c in value) or len(value)>64:
            raise ValueError('Report ID contains control characters or exceeds 64 characters')
        return value or 'Anonymous'

def domain_warnings(record, support):
    warnings = [f'{key} is outside observed training support ({bounds[0]} to {bounds[1]}).'
                for key,bounds in support.items() if not bounds[0]<=record[key]<=bounds[1]]
    if not 29<=record['age']<=77:
        warnings.insert(0,'Age is outside this adult research cohort (29-77); scoring is withheld.')
    return warnings

class Factor(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    field: str
    feature: str
    value: str
    impact: float

class Explanation(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    status: Literal['available','unavailable']
    target: str
    scale: Literal['probability'] = 'probability'
    method: str
    base_value: float | None = None
    contributions: list[Factor] = []
    residual: float | None = None
    reconstruction_error: float | None = None
    note: str

class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    artifact_id: str
    probability: float | None = Field(default=None,ge=0,le=1)
    prediction: Literal[0,1] | None = None
    prediction_label: str
    threshold: float = Field(gt=0,lt=1)
    calibration: str
    domain_status: Literal['within_observed_ranges','outside_observed_ranges','abstained']
    warnings: list[str]
    explanation: Explanation

UNITS = {'age':'years','trestbps':'mmHg','chol':'mg/dL','thalach':'beats/min','oldpeak':'ST depression','ca':'vessels'}
