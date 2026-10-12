from fastapi import APIRouter, HTTPException

from app.schemas import MealRequest, MealResponse
from app.services.vision import analyse_meal

router = APIRouter(prefix="/meal", tags=["nutrition"])


@router.post("/analyse", response_model=MealResponse)
async def analyse(body: MealRequest) -> MealResponse:
    try:
        result = await analyse_meal(body.image, body.condition_ids, body.note)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return MealResponse(**{k: v for k, v in result.items() if k in MealResponse.model_fields})
