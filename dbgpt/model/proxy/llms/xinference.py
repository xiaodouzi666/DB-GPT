import logging
from concurrent.futures import Executor
from typing import Iterator, Optional

from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

logger = logging.getLogger(__name__)

def xinference_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=4096
):
    client: XinferenceLLMClient = model.proxy_llm_client
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    for r in client.sync_generate_stream(request):
        yield r

class XinferenceLLMClient(ProxyLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        model_alias: Optional[str] = "xinference_proxyllm",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        if not model:
            model = "default_model"
        if not host:
            host = "http://localhost:9997"  # Default hosting URL, change as necessary
        self._model = model
        self._host = host

        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )

    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "XinferenceLLMClient":
        return cls(
            model=model_params.proxyllm_backend,
            host=model_params.proxy_server_url,
            model_alias=model_params.model_name,
            context_length=model_params.max_context_size,
            executor=default_executor,
        )

    @property
    def default_model(self) -> str:
        return self._model

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        try:
            from xinference import Client
        except ImportError as e:
            raise ImportError(
                "Could not import xinference package. "
                "Please install xinference with `pip install xinference`"
            ) from e
        request = self.local_covert_message(request, message_converter)
        messages = request.to_common_messages()

        client = Client(self._host)
        try:
            stream = client.chat(
                model=self._model,
                messages=messages,
                stream=True,
            )
            content = ""
            for chunk in stream:
                content += chunk["message"]["content"]
                yield ModelOutput(text=content, error_code=0)
        except Exception as e:  # Modify according to specific exceptions xinference might raise
            yield ModelOutput(
                text=f"Error during response generation: {str(e)}",
                error_code=-1,
            )
