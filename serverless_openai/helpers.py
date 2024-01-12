from enum import Enum
from pydantic import AfterValidator, BaseModel, HttpUrl, validator
from typing import List, Optional, Union
from typing_extensions import Annotated
import base64

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"

class VisionMessage(BaseModel):
    text: str
    role: Roles = Roles.user
    image_url: Optional[HttpUrlString] = None
    image_path: Optional[str] = None

    @validator('image_path', always=True)
    def check_url(cls, v, values):
        if v:
            return encode_image(v)
        elif not values.get('image_url') and not v:
            raise ValueError('either a or b is required')
        return v
    


# class VisionMessages(BaseModel):
#     messages: List[Message] = []

# class ResultUsage(BaseModel):
#     prompt_tokens: int
#     completion_tokens: int
#     total_tokens: int

# class ResultMessage(BaseModel):
#     role: str
#     content: str

# class Results(BaseModel):
#     usage: ResultUsage
#     message: ResultMessage