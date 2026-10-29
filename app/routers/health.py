"""Application health endpoint."""

from fastapi import APIRouter

from app.helpers.responses import APIResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=APIResponse[None],
    summary="Check API availability",
    response_description="The API is running; database connectivity is not checked.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"message": "Service is healthy", "data": None}
                }
            }
        }
    },
)
async def health() -> APIResponse[None]:
    return APIResponse[None](message="Service is healthy")
