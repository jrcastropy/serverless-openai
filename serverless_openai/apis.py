import requests, json
from typing import Optional, List, Union
from pydantic import BaseModel
from serverless_openai.helpers import *

class OpenAIAPI(BaseModel):
    api_key: str
    org_key: Optional[str] = None
    text_models: List[str] = TextCompletionModels
    imagecreation_models: List[str] = ImageCreationModels
    completion_url: str = "https://api.openai.com/v1/chat/completions"
    imagecreation_url: str = "https://api.openai.com/v1/images/generations"

    class Config:  
        use_enum_values = True
    
    def chat_completion(
            self, 
            messages: Messages,
            model: Union[TextCompletionModels, str] = TextCompletionModels.gpt4_1106,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            frequency_penalty: Optional[float] = 0,
            # logit_bias: Union[str, None] = None,
            logit_bias: Optional[str] = None,
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            max_tokens: Optional[int] = None,
            n: Optional[int] = None,
            presence_penalty: Optional[float] = 0,
            response_format: Optional[dict] = None,
            seed: Optional[int] = None,
            stop: Optional[list] = None,
            stream: Optional[bool] = False,
            top_p: Optional[int] = 1
        ):
        messages = messages.model_dump()['messages']
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.org_key:
            headers["OpenAI-Organization"] = self.org_key
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "top_p": top_p
        }
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=headers, json=data, timeout=timeout).json()
                # print(res)
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return message['content']
            except Exception as e:
                print("ERROR:", e)
        return False
    
    def tools(
            self, 
            messages: Messages,
            tools: List[dict],
            tool_choice: str,
            model: Union[TextCompletionModels, str] = TextCompletionModels.gpt4_1106,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            frequency_penalty: Optional[float] = 0,
            # logit_bias: Union[str, None] = None,
            logit_bias: Optional[str] = None,
            logprobs: Optional[bool] = False,
            top_logprobs: Optional[int] = None,
            max_tokens: Optional[int] = None,
            n: Optional[int] = None,
            presence_penalty: Optional[float] = 0,
            response_format: Optional[dict] = {"type": "json_object"},
            seed: Optional[int] = None,
            stop: Optional[list] = None,
            stream: Optional[bool] = False,
            top_p: Optional[int] = 1,
            
        ):
        messages = messages.model_dump()['messages']
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.org_key:
            headers["OpenAI-Organization"] = self.org_key
        # Number between -2.0 and 2.0.
        # Positive values penalize new tokens based on whether they appear
        # in the text so far, increasing the model's
        # likelihood to talk about new topics.
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tool_choice}},
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "n": n,
            "presence_penalty": presence_penalty,
            "response_format": response_format,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "top_p": top_p
        }
        for _ in range(5):
            try:
                results = requests.post(self.completion_url, headers=headers, json=data, timeout=timeout).json()
                print(results)
                if 'choices' in results:
                    res = results['choices'][0]['message']
                    res_json = json.loads(res['tool_calls'][0]['function']['arguments'], strict=False)
                    return res_json
                else:
                    print("RESULTS:", results)
            except Exception as e:
                print("REQ POST ERROR:", e)
        return False
    
    def dall_e(
            self,
            prompt: str,
            model: Union[ImageCreationModels, str] = ImageCreationModels.dalle_3,
            n: int = 1,
            size: str = "1024x1024",
            timeout: int = 500
        ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.org_key:
            headers["OpenAI-Organization"] = self.org_key
        
        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        results = requests.post(self.imagecreation_url, headers=headers, json=data, timeout=timeout).json()
        return results
    
    def vision(
            self,
            messages: VisionMessage,
            model: Union[VisionModels, str] = VisionModels.gpt4_vision,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            max_tokens: Optional[int] = 1024,
        ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        if self.org_key:
            headers["OpenAI-Organization"] = self.org_key

        newm = [
            {
                "role": messages.role,
                "content": [
                    {
                        "type": "text",
                        "text": messages.text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": messages.image_url if messages.image_url else messages.image_path,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]        
        data = {
            "model": model,
            "messages": newm,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=headers, json=data, timeout=timeout).json()
                print(res)
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return message['content']
            except Exception as e:
                print("ERROR:", e)
        return False