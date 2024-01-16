from enum import Enum
from pydantic import AfterValidator, BaseModel, HttpUrl, validator, TypeAdapter, ValidationError
from typing import List, Optional, Union
from typing_extensions import Annotated
import base64, cv2, requests, json, uuid, os
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

class OpenAIResults(BaseModel):
    result: Union[str, bool, dict, list] = None
    result_json: dict = None

def save_npimage(
        filename: str, 
        img_np: np.array,
        dir: str = "saved_images"
    ) -> str:
    os.makedirs(dir, exist_ok=True)
    cv2.imwrite(f"{dir}/{filename}", img_np)
    return f"{dir}/{filename}"

def urlimage_to_np(
        img_url: str,
        save_image: bool=False,
        filename: str="test.png"
    ) -> np.array:
    req = requests.get(
        url=img_url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    arr = np.asarray(bytearray(req.content), dtype=np.uint8)
    img_np = cv2.imdecode(arr, -1) # 'Load it as it is'
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if save_image:
        saved_image = save_npimage(filename, img_np)
    return img_np
    
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    
def b64_to_np(
        base64_image: str,
        save_image: bool = False,
        filename: str="test.png"
    ) -> np.array:
    base64_image =  base64_image.replace('data:image/jpeg;base64,', "")
    image_data = base64.b64decode(base64_image)

    # Convert the raw image data to a numpy array
    np_arr = np.frombuffer(image_data, np.uint8)

    # Decode the numpy array into an image
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if save_image:
        saved_image = save_npimage(filename, image_np)
    return image_np

def crop_image(
        img_np: np.array, 
        max_h: int = 4086,
    ) -> list:
    curr_h = 0
    img_b64_list = []
    while True:
        fn = f"{uuid.uuid4()}.png"
        crop_img = img_np[curr_h:max_h, :]
        curr_h = max_h
        max_h += max_h
        if not crop_img.shape[0]:
            break
        saved_image = save_npimage(fn, crop_img)
        b64_img = encode_image(saved_image)
        img_b64_list.append(b64_img)
    return img_b64_list

def cosine_similarity(
        sim: Similarity,
        data_list: List[str],
        get_scores: bool = False
    ) -> list:
    vector = np.array(sim.vector)
    matrix = np.array(sim.matrix)
    scores = (np.sum(vector*matrix,axis=1) / ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )
    if get_scores:
        return scores
    result_list = get_similarity_result(scores, data_list)
    return result_list

def get_similarity_result(
        scores: list, 
        all_data: list, 
        topn: int = 5
    ) -> list:
    idx = (-scores).argsort()
    result_list = []
    for ix in idx[:topn]:
        p = all_data[ix]
        scr = scores[ix]
        result_list.append(p)
    return result_list