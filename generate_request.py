from pydantic.main import BaseModel
from typing import Union


class GenerateRequest(BaseModel):
    image_id: str
    prompt: str
    initial_image_path: Union[str, None] = None
    ddim_steps: float | Union[str, None] = 50
    ddim_eta: float | Union[str, None] = 0.0
    verbose: bool | Union[bool, None] = False
    count: int | Union[int, None] = 1
    scale: float | Union[float, None] = 5.0
    channels: int | Union[int, None] = 4
    width: int | Union[int, None] = 512
    height: int | Union[int, None] = 512
    down_sample_factor: int | Union[int, None] = 8
    start_code: str | Union[str, None] = None
    seed: int | Union[str, int] = 64
    ckpt: str | Union[str, None] = "models/ldm/stable-diffusion-v1/model.ckpt"
