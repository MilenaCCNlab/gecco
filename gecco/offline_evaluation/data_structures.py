from pydantic import BaseModel, Field
from typing import Dict, Optional

class FitResult(BaseModel):
    params: Dict[str, float]
    nll: float
    bic: Optional[float] = None
    aic: Optional[float] = None
    success: bool = True
    info: Dict = Field(default_factory=dict)
