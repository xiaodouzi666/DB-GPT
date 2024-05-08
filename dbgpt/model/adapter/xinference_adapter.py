import logging
from dbgpt.model.adapter.base import LLMModelAdapter
from dbgpt.model.adapter.template import ConversationAdapter, PromptType

# import xinference apis
try:
    from xinference.api import XinferenceConversation, register_conv_template
except ImportError as exc:
    raise ImportError("Could not import xinference. Please install it.") from exc

logger = logging.getLogger(__name__)

class XinferenceConversationAdapter(ConversationAdapter):
    """The conversation adapter for Xinference."""

    def __init__(self, conv: XinferenceConversation):
        self._conv = conv

    @property
    def prompt_type(self) -> PromptType:
        return PromptType.XINFER

    @property
    def roles(self) -> Tuple[str]:
        return self._conv.roles

    @property
    def sep(self) -> Optional[str]:
        return self._conv.sep

    @property
    def stop_str(self) -> str:
        return self._conv.stop_str

    def get_prompt(self) -> str:
        """Get the prompt string."""
        return self._conv.get_prompt()

    def append_message(self, role: str, message: str) -> None:
        """Append a new message."""
        self._conv.append_message(role, message)

    def update_last_message(self, message: str) -> None:
        """Update the last output."""
        self._conv.update_last_message(message)

    def copy(self) -> "ConversationAdapter":
        """Create a copy of the conversation."""
        return XinferenceConversationAdapter(self._conv.copy())

class XinferenceLLMModelAdapterWrapper(LLMModelAdapter):
    """Wrapper for Xinference LLM model adapter."""

    def __init__(self, adapter):
        self._adapter = adapter

    def new_adapter(self, **kwargs) -> "LLMModelAdapter":
        return XinferenceLLMModelAdapterWrapper(self._adapter)

    def load(self, model_path: str, from_pretrained_kwargs: dict):
        return self._adapter.load_model(model_path, from_pretrained_kwargs)

    def get_generate_stream_function(self, model, model_path: str):
        from xinference.api import get_generate_stream_function
        return get_generate_stream_function(model, model_path)

    def get_default_conv_template(
        self, model_name: str, model_path: str
    ) -> Optional[ConversationAdapter]:
        conv_template = self._adapter.get_default_conv_template(model_path)
        return XinferenceConversationAdapter(conv_template) if conv_template else None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._adapter.__class__.__module__}.{self._adapter.__class__.__name__})"

# Register the conversation template if necessary
register_conv_template(
    XinferenceConversation(
        roles=("User", "AI"),
        messages=(),
        stop_str="</s>",
    ),
    override=True,
)
