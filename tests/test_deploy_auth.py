import pytest
from fastapi import HTTPException

from ollie_assistants.deploy.auth import BearerTokenValidator


def test_bearer_token_validator_accepts_matching_token() -> None:
    BearerTokenValidator("secret").validate("Bearer secret")


def test_bearer_token_validator_rejects_missing_token() -> None:
    with pytest.raises(HTTPException) as exc:
        BearerTokenValidator("secret").validate(None)
    assert exc.value.status_code == 401
