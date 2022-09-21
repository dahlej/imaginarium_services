from pydantic.main import BaseModel
from typing import Union


class GenerateRequest(BaseModel):
    image_id: str
    prompt: str
    initial_image_path: Union[str, None] = None
    ddim_steps: Union[float, None] = 50
    ddim_eta: Union[float, None] = 0.0
    verbose: Union[bool, None] = False
    count: Union[int, None] = 1
    scale: Union[float, None] = 5.0
    channels: Union[int, None] = 4
    width: Union[int, None] = 512
    height: Union[int, None] = 512
    down_sample_factor: Union[int, None] = 8
    start_code: Union[str, None] = None
    seed: Union[str, int] = 64
    ckpt: Union[str, None] = "models/ldm/stable-diffusion-v1/model.ckpt"
