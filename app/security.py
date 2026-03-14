import os, re

from fastapi import HTTPException

HF_TOKEN_PATTERN = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def resolve_hf_token(
    form_token: str,
    authorization_header: str | None,
    x_hf_token_header: str | None,
) -> str:
    auth_token = (authorization_header or "").strip()
    if auth_token.lower().startswith("bearer "):
        auth_token = auth_token[7:].strip()

    token_candidates = [
        auth_token,
        (x_hf_token_header or "").strip(),
        (form_token or "").strip(),
        (os.getenv("HUGGINGFACE_API_KEY") or "").strip(),
        (os.getenv("HF_TOKEN") or "").strip(),
        (os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip(),
    ]

    token = next((candidate for candidate in token_candidates if candidate), "")
    if not token:
        raise HTTPException(status_code=401, detail="Missing Hugging Face token")

    if not HF_TOKEN_PATTERN.match(token):
        raise HTTPException(status_code=401, detail="Invalid Hugging Face token format")

    return token
