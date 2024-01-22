from typing import Optional, List, Union
from pydantic import BaseModel
from serverless_openai.helpers import *
from bs4 import BeautifulSoup
import re, json

class OpenAIAPI(BaseModel):
    api_key: str
    org_key: Optional[str] = None
    headers: Optional[dict] = None
    text_models: List[str] = TextCompletionModels
    imagecreation_models: List[str] = ImageCreationModels
    completion_url: HttpUrlString = "https://api.openai.com/v1/chat/completions"
    imagecreation_url: HttpUrlString = "https://api.openai.com/v1/images/generations"
    embeddings_url: HttpUrlString = "https://api.openai.com/v1/embeddings"
    

    @validator('headers', always=True)
    def check_url(cls, v, values):
        if not v:
            h: dict = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {values['api_key']}"
            }
            if values['org_key']:
                h["OpenAI-Organization"] = values['org_key']
            return h
        return v


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
        ) -> OpenAIResults:
        messages = messages.model_dump()['messages']
        
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
        res = {}
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return OpenAIResults(result=message['content'], result_json=res)
            except Exception as e:
                print("ERROR:", e)
        
        return OpenAIResults(result=False, result_json=res)
    
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
        ) -> OpenAIResults:
        messages = messages.model_dump()['messages']

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

        results = {}
        for _ in range(5):
            try:
                results = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'choices' in results:
                    res = results['choices'][0]['message']
                    res_json = json.loads(res['tool_calls'][0]['function']['arguments'], strict=False)
                    return OpenAIResults(result=res_json, result_json=results)
                else:
                    print("RESULTS:", results)
            except Exception as e:
                print("REQ POST ERROR:", e)
        return OpenAIResults(result=False, result_json=results)
    
    def dall_e(
            self,
            prompt: str,
            model: Union[ImageCreationModels, str] = ImageCreationModels.dalle_3,
            n: int = 1,
            size: str = "1024x1024",
            timeout: int = 500,
            tries: int = 5
        ) -> OpenAIResults:

        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }

        results = {}
        for _ in range(tries):
            results = requests.post(self.imagecreation_url, headers=self.headers, json=data, timeout=timeout).json()
            print(results)
            return OpenAIResults(result=results['data'][0]['url'], result_json=results)
        return OpenAIResults(result=False, result_json=results)
    
    def vision(
            self,
            messages: VisionMessage,
            model: Union[VisionModels, str] = VisionModels.gpt4_vision,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            max_tokens: Optional[int] = 1024,
        ) -> OpenAIResults:

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
                            "url": messages.image,
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
        res = {}
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return OpenAIResults(result=message['content'], result_json=res)
            except Exception as e:
                print("ERROR:", e)
        return OpenAIResults(result=False, result_json=res)
    
    def vision_longimage(
            self,
            messages: VisionMessage,
            model: Union[VisionModels, str] = VisionModels.gpt4_vision,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            max_tokens: Optional[int] = 1024,
        ) -> OpenAIResults:

        if isinstance(messages.image, str):
            if 'data:image/jpeg;base64' in messages.image:
                image_np = b64_to_np(messages.image)
            else:
                image_np = urlimage_to_np(messages.image)
        elif isinstance(messages.image, np.ndarray):
            image_np = messages.image
        img_b64_list = crop_image(image_np)
        newm = [
            {
                "role": messages.role,
                "content": [
                    {
                        "type": "text",
                        "text": messages.text
                    },
                ]
            }
        ]
        for b64 in img_b64_list:
            newm[0]['content'].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": b64,
                        "detail": "high"
                    }
                }
            )

        data = {
            "model": model,
            "messages": newm,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        res = {}
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return OpenAIResults(result=message['content'], result_json=res)
            except Exception as e:
                print("ERROR:", e)
        return OpenAIResults(result=False, result_json=res)

    def embeddings(
            self,
            prompt: EmbeddingPrompts,
            model: EmbeddingModels = EmbeddingModels.ada2,
            tries: int = 5,
            timeout: int = 500,
        ) -> OpenAIResults:
        data = {
            "model": model,
            "input": prompt,
        }

        res = {}
        for _ in range(tries):
            try:
                res = requests.post(self.embeddings_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'data' in res:
                    return OpenAIResults(result=[x['embedding'] for x in res['data']], result_json=res)
            except Exception as e:
                print("ERROR:", e)
        return OpenAIResults(result=False, result_json=res)
    
class ScrapingBeeAPI(BaseModel):
    api_key: str
    scrape_url: HttpUrlString = "https://app.scrapingbee.com/api/v1"

    def scrape(
            self,
            url: str,
            get_links: bool = False,
            block_resources: bool = True,
            render_js: bool = True,
            wait_browser: str = 'load',
            premium_proxy: bool = False,
            stealth_proxy: bool = True,
            block_ads: bool = True,
            js_scenario: dict = {
                "instructions": [
                    {"evaluate": 'document.querySelectorAll("style").forEach(function(e){e.remove();});'}
                ]
            },
            extract_rules: str = None
        ) -> dict:

        params = {
            'api_key': self.api_key,
            'url': url,
            "block_resources": block_resources,
            'render_js': render_js,
            'wait_browser': wait_browser,
            'premium_proxy': premium_proxy,
            'stealth_proxy': stealth_proxy,
            'block_ads': block_ads,
            'block_resources': block_resources,
            "js_scenario": json.dumps(js_scenario),
        }
        if extract_rules:
            params['extract_rules'] = extract_rules

        resp = requests.get(self.scrape_url, params=params)
        soup = BeautifulSoup(resp.content, 'html.parser')
        body = soup.get_text()
        body = re.sub('\n+', '\n', body).strip()
        result = {
            'body': body
        }
        if get_links:
            all_links = []
            params['extract_rules'] = '''{"all_links":{"selector":"a@href", "type":"list"}}'''

            all_links = requests.get(self.scrape_url, params=params).json()['all_links']
            all_links = list(dict.fromkeys(all_links))
            all_links = [x for x in all_links if x is not None]
            result['links'] = all_links
        return result
    
    def screenshot_website(
            self,
            url: str,
            screenshot_full_page: bool = True,
        ) ->  dict:
        params = {
            "api_key": self.api_key,
            "url": url,
            "screenshot": "True",
            "screenshot_full_page": screenshot_full_page,
            'wait_browser': 'load',
        }
        res = requests.get(self.scrape_url, params=params)
        if res.ok:
            arr = np.asarray(bytearray(res.content), dtype=np.uint8)
            img_np = cv2.imdecode(arr, -1) # 'Load it as it is'
            return {'image': img_np, 'status': True}
        return {'image': False, 'status': False}