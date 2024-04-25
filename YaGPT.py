from langchain.embeddings.base import Embeddings
import time
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain

class YaGPTEmbeddings(Embeddings):
    """
        Класс для работы с эмбеддингами модели YaGPT.
        """

    def __init__(self,folder_id,api_key,sleep_interval=1):
        """
        Инициализация объекта YaGPTEmbeddings.

        Args:
            folder_id (str): Идентификатор папки в Yandex Cloud.
            api_key (str): API ключ Yandex Cloud.
            sleep_interval (int): Интервал между запросами к API (по умолчанию 1 секунда).
                """
        self.folder_id = folder_id
        self.api_key = api_key
        self.sleep_interval = sleep_interval
        self.headers = { 
                        "Authorization" : f"Api-key {api_key}",
                        "x-folder-id" : folder_id }
        
    def embed_document(self, text):
        """
        Получает эмбеддинг документа.

        Args:
            text (str): Текст документа.

        Returns:
            list: Эмбеддинг документа.
        """
        j = {
          "model" : "general:embedding",
          "embedding_type" : "EMBEDDING_TYPE_DOCUMENT",
          "text": text
        }
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                            json=j,headers=self.headers)
        vec = res.json()['embedding']
        return vec

    def embed_documents(self, texts, chunk_size = 0):
        """
        Получает эмбеддинги для списка документов.

        Args:
            texts (List[str]): Список текстов документов.
            chunk_size (int): Размер чанка для отправки запросов (по умолчанию 0).

        Returns:
            list: Список эмбеддингов для каждого документа.
                """
        res = []
        for x in texts:
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res
        
    def embed_query(self, text):
        """
        Получает эмбеддинг запроса.

        Args:
            text (str): Текст запроса.

        Returns:
            list: Эмбеддинг запроса.
                """
        j = {
          "model" : "general:embedding",
          "embedding_type" : "EMBEDDING_TYPE_QUERY",
          "text": text
        }
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/embedding",
                            json=j,headers=self.headers)
        vec = res.json()['embedding']
        time.sleep(self.sleep_interval)
        return vec
    


class YandexLLM(langchain.llms.base.LLM):
    """
    Класс для работы с моделью Yandex Language Models (YaGPT).
    """
    api_key: str = None
    iam_token: str = None
    folder_id: str
    max_tokens : int = 1500
    temperature : float = 1
    instruction_text : str = None

    @property
    def _llm_type(self) -> str:
        return "yagpt"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """
        Вызывает модель для генерации текста на основе переданного промпта.

        Args:
            prompt (str): Текст промпта.
            stop (List[str]): Список стоп-слов для остановки генерации (не используется).
            run_manager (CallbackManagerForLLMRun): Менеджер коллбэков для выполнения.

        Returns:
            str: Сгенерированный текст.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        headers = { "x-folder-id" : self.folder_id }
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        if self.api_key:
            headers["Authorization"] = f"Api-key {self.api_key}"
        req = {
          "model": "general",
          "instruction_text": self.instruction_text,
          "request_text": prompt,
          "generation_options": {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
          }
        }
        res = requests.post("https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
          headers=headers, json=req).json()
        return res['result']['alternatives'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Получает идентифицирующие параметры."""
        return {"max_tokens": self.max_tokens, "temperature" : self.temperature }
