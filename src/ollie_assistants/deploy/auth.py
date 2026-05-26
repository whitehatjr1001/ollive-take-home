from dataclasses import dataclass

from fastapi import HTTPException, status


@dataclass(frozen=True)
class BearerTokenValidator:
    expected_token: str | None

    def validate(self, authorization_header: str | None) -> None:
        if not self.expected_token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OSS_BEARER_TOKEN is not configured",
            )
        expected = f"Bearer {self.expected_token}"
        if authorization_header != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
            )
