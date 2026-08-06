"""Configuration model for Tree-sitter code chunking."""

from dataclasses import dataclass

from tree_sitter import Language


@dataclass(frozen=True, slots=True)
class LanguageChunkConfig:
    """Describe syntax-aware chunking for one language.

    Attributes:
        name: Canonical internal language name.
        language: Loaded Tree-sitter grammar.
        file_types: Values accepted from SourceFile.file_type.
        extensions: File extensions belonging to the language.
        structural_nodes: Top-level nodes used as chunk boundaries.
        recursive_nodes: Oversized nodes that may be split using named children.
    """

    name: str
    language: Language
    file_types: frozenset[str]
    extensions: frozenset[str]
    structural_nodes: frozenset[str]
    recursive_nodes: frozenset[str]
