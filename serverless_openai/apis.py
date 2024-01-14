from typing import Optional, List, Union
from pydantic import BaseModel
from serverless_openai.helpers import *
from bs4 import BeautifulSoup
import re

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
        ) -> str:
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
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
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
        ) -> dict:
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
        for _ in range(5):
            try:
                results = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
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
        ) -> dict:

        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        results = requests.post(self.imagecreation_url, headers=self.headers, json=data, timeout=timeout).json()
        return results
    
    def vision(
            self,
            messages: VisionMessage,
            model: Union[VisionModels, str] = VisionModels.gpt4_vision,
            tries: int = 5,
            timeout: int = 500,
            temperature: Optional[float] = 1,
            max_tokens: Optional[int] = 1024,
        ) -> str:

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
        for _ in range(tries):
            try:
                res = requests.post(self.completion_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'choices' in res:
                    message = res['choices'][0]['message']
                    return message['content']
            except Exception as e:
                print("ERROR:", e)
        return False

    def embeddings(
            self,
            prompt: EmbeddingPrompts,
            model: EmbeddingModels = EmbeddingModels.ada2,
            tries: int = 5,
            timeout: int = 500,
        ) -> list:
        data = {
            "model": model,
            "input": prompt,
        }
        for _ in range(tries):
            try:
                res = requests.post(self.embeddings_url, headers=self.headers, json=data, timeout=timeout).json()
                if 'data' in res:
                    return [x['embedding'] for x in res['data']]
            except Exception as e:
                print("ERROR:", e)
        return False
    
    def cosine_similarity(
            self, 
            sim: Similarity,
            data_list: List[str],
            get_scores: bool = False
        ) -> list:
        vector = np.array(sim.vector)
        matrix = np.array(sim.matrix)
        scores = (np.sum(vector*matrix,axis=1) / ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )
        if get_scores:
            return scores
        result_list = self.get_similarity_result(scores, data_list)
        return result_list
    
    def get_similarity_result(
            self,
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