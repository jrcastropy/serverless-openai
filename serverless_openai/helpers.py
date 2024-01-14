from enum import Enum
from pydantic import AfterValidator, BaseModel, HttpUrl, validator, TypeAdapter, ValidationError
from typing import List, Optional, Union
from typing_extensions import Annotated
import base64, cv2, requests, json, uuid
import numpy as np

HttpUrlString = Annotated[HttpUrl, AfterValidator(lambda v: str(v))]

class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
    def __repr__(self):
      return self.value

class Roles(str, ExtendedEnum):
    user: str = 'user'
    system: str = 'system'
    assistant: str = 'assistant'

class TextCompletionModels(str, ExtendedEnum):
    gpt4_1106 : str = "gpt-4-1106-preview"
    gpt4 : str = "gpt-4"
    gpt35_turbo_1106 : str = "gpt-3.5-turbo-1106"
    gpt35_turbo_16k : str = "gpt-3.5-turbo-16k"
    gpt35_turbo : str = "gpt-3.5-turbo"

class Message(BaseModel):
    role: Roles
    content: str

class Messages(BaseModel):
    messages: List[Message] = []

class ImageCreationModels(str, ExtendedEnum):
    dalle_2 : str = "dall-e-2"
    dalle_3 : str = "dall-e-3"

class VisionModels(str, ExtendedEnum):
    gpt4_vision : str = "gpt-4-vision-preview"

class VisionMessage(BaseModel):
    text: str
    role: Roles = Roles.user
    image: str

    @validator('image', always=True)
    def check_url(cls, v, values):
        try:
            TypeAdapter(HttpUrl).validate_python(v)
            return v
        except ValidationError:
            return encode_image(v)

class EmbeddingModels(str, ExtendedEnum):
    ada2 : str = "text-embedding-ada-002"

class EmbeddingPrompts(BaseModel):
    prompt: Union[str, List[str]]

class Similarity(BaseModel):
    vector: List[List[float]]
    matrix: List[List[float]]

def download_img(img_url):
    req = requests.get(
        url=img_url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    arr = np.asarray(bytearray(req.content), dtype=np.uint8)
    img_np = cv2.imdecode(arr, -1) # 'Load it as it is'
    cv_im = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return cv_im
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"

def crop_image(img_np, dir='images'):
    curr_h = 0
    max_h = 4096
    img_b64_list = []
    while True:
        fn = f"{dir}/{uuid.uuid4()}.png"
        crop_img = img_np[curr_h:max_h, :]
        curr_h = max_h
        max_h += max_h
        if not crop_img.shape[0]:
            break
        cv2.imwrite(fn, crop_img)
        b64_img = encode_image(fn)
        img_b64_list.append(b64_img)
    return img_b64_list