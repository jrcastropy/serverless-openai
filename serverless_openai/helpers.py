from enum import Enum
from pydantic import AfterValidator, BaseModel, HttpUrl, validator, TypeAdapter, ValidationError
from typing import List, Optional, Union
from typing_extensions import Annotated
import base64, cv2, requests, uuid, os, tiktoken
import numpy as np

HttpUrlString = Annotated[HttpUrl, AfterValidator(lambda v: str(v))]

tokenizer = tiktoken.get_encoding("cl100k_base")

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
    gpt4_turbo : str = "gpt-4-turbo-preview"
    gpt4_0125 : str = "gpt-4-0125-preview"
    gpt35_turbo_0125: str = "gpt-3.5-turbo-0125"
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
    image: Union[str, np.ndarray]
    role: Roles = Roles.user
    
    @validator('image', always=True)
    def check_url(cls, v, values):
        if isinstance(v, str):
            try:
                TypeAdapter(HttpUrl).validate_python(v)
                return v
            except ValidationError:
                return encode_image(v)
        elif isinstance(v, list):
            vlist = []
            for vi in v:
                try:
                    TypeAdapter(HttpUrl).validate_python(v)
                    vlist.append(vi)
                except ValidationError:
                    vlist.append(encode_image(vi))
            return vlist
    class Config:
        arbitrary_types_allowed = True

class EmbeddingModels(str, ExtendedEnum):
    te_ada2 : str = "text-embedding-ada-002"
    te3_small: str = "text-embedding-3-small"
    te3_large: str = "text-embedding-3-large"

class EmbeddingPrompts(BaseModel):
    prompt: Union[str, List[str]]

class ModerationModels(str, ExtendedEnum):
    moderation_stable : str = "text-moderation-stable"
    moderation_latest : str = "text-moderation-latest"

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

def np_to_b64(
        img_np: np.array
    ) -> str:
    _, buffer = cv2.imencode('.png', img_np)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

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
        b64_img = np_to_b64(crop_img)
        img_b64_list.append(b64_img)
    return img_b64_list

def cosine_similarity(
        sim: Similarity,
        data_list: List[str],
        topn: int = 5
    ) -> dict:
    vector = np.array(sim.vector)
    matrix = np.array(sim.matrix)
    scores = (np.sum(vector*matrix,axis=1) / ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )
    res_dict = get_similarity_result(scores, data_list, topn=topn)
    return res_dict

def get_similarity_result(
        scores: list, 
        all_data: list, 
        topn: int = 5
    ) -> dict:
    idx = (-scores).argsort()
    result_list = []
    scores_llist = []
    for ix in idx[:topn]:
        p = all_data[ix]
        score = scores[ix]
        scores_llist.append(score)
        result_list.append(p)
    return {
        "result_list": result_list,
        "scores_llist": scores_llist
    }

def get_token_count(
        text: str
    ) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(tokenizer.encode(text))
    return num_tokens

def limit_tokens(
        text: str,
        chunk_size: int = 5000
    ) -> str:
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return False
    # Tokenize the text
    tokens = tokenizer.encode(text, disallowed_special=())
    chunk = tokens[:chunk_size]
    chunk_text = tokenizer.decode(chunk)
    return chunk_text

def checkoutput_init(
        result: dict, 
        tools: dict,
        apply_min_item: bool = False
    ) ->  bool:
    params = tools[0]['function']['parameters']
    return checkoutput(result, params, apply_min_item=apply_min_item)

def checkoutput(
        res: dict, 
        params: dict,
        apply_min_item: bool = False
    ) -> bool:
    jtype = params['type']
    # print("RES:", res)
    # print("TYPE:", jtype)
    if jtype == 'object':
        required = params['required']
        properties = params['properties']
        for req in required:
            # print("REQUIRED:", req)
            if req not in res:
                print(f"Requirement {req} is not fulfilled")
                return False
            p = properties[req]
            r = res[req]
            ret_val = checkoutput(r, p, apply_min_item=apply_min_item)
            if not ret_val:
                return ret_val
    elif jtype == 'array':
        items = params['items']
        if apply_min_item:
            minItems = 1
            if 'minItems' in params:
                minItems = params['minItems']
            if len(res) < minItems:
                print(f"Minimum items of {minItems} not acquired only {len(res)} were filled")
                return False
        if not len(res):
            print(f"This array is empty: {res}")
            return False
        for r in res:
            ret_val = checkoutput(r, items, apply_min_item=apply_min_item)
            if not ret_val:
                return ret_val
    else:
        if res:
            if isinstance(res, str):
                if res.strip() == "":
                    print(f"result was an empty string")
                    return False
        else:
            print(f"Failed in running the results")
            return False
    return True