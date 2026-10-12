from fastapi import APIRouter

from app.schemas import TriageRequest, TriageResponse
from app.services.triage import run_triage

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("", response_model=TriageResponse)
async def triage(body: TriageRequest) -> TriageResponse:
    result = await run_triage(body.model_dump())
    return TriageResponse(**result)
