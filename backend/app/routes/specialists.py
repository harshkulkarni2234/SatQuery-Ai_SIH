from fastapi import APIRouter

from app.services.registry import list_specialists

router = APIRouter(tags=["specialists"])


@router.get("/specialists")
def get_specialists():
    return [spec.model_dump() for spec in list_specialists()]
