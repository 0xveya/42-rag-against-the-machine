"""Tree-sitter language registry."""

from rag_against_the_machine.indexing.languages.config import (
    LanguageChunkConfig,
)
from rag_against_the_machine.indexing.languages.registry import (
    LANGUAGE_CONFIGS,
    file_type_for_extension,
    get_config_by_name,
    ignored_directory_names_for_file_type,
    get_language_config,
    supported_extensions,
    supported_file_types,
)

__all__ = [
    "LANGUAGE_CONFIGS",
    "LanguageChunkConfig",
    "file_type_for_extension",
    "get_config_by_name",
    "ignored_directory_names_for_file_type",
    "get_language_config",
    "supported_extensions",
    "supported_file_types",
]
