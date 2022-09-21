from pydantic.main import BaseModel
import random


class GenerateRequest(BaseModel):
    image_id: str
    prompt: str
    initial_image_path: str | None = None
    ddim_steps: float | None = 50
    ddim_eta: float | None = 0.0
    verbose: bool | None = False
    count: int | None = 1
    scale: float | None = 5.0
    channels: int | None = 4
    width: int | None = 512
    height: int | None = 512
    down_sample_factor: int | None = 8
    start_code: str | None = None
    seed: int | None = 64
    ckpt: str | None = "models/ldm/stable-diffusion-v1/model.ckpt"
