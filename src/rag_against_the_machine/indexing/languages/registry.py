"""Registry for supported Tree-sitter languages."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from rag_against_the_machine.indexing.languages.config import (
    LanguageChunkConfig,
)
from rag_against_the_machine.indexing.languages.grammars import (
    C_LANGUAGE,
    CPP_LANGUAGE,
    GO_LANGUAGE,
    JAVASCRIPT_LANGUAGE,
    PYTHON_LANGUAGE,
    RUST_LANGUAGE,
    TSX_LANGUAGE,
    TYPESCRIPT_LANGUAGE,
    ZIG_LANGUAGE,
)

PYTHON_CONFIG = LanguageChunkConfig(
    name="python",
    language=PYTHON_LANGUAGE,
    file_types=frozenset({
        "python",
        "py",
    }),
    extensions=frozenset({
        ".py",
        ".pyi",
    }),
    structural_nodes=frozenset({
        "function_definition",
        "class_definition",
        "decorated_definition",
    }),
    recursive_nodes=frozenset({
        "class_definition",
        "decorated_definition",
    }),
)


JAVASCRIPT_CONFIG = LanguageChunkConfig(
    name="javascript",
    language=JAVASCRIPT_LANGUAGE,
    file_types=frozenset({
        "javascript",
        "js",
        "jsx",
    }),
    extensions=frozenset({
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
    }),
    structural_nodes=frozenset({
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
    }),
    recursive_nodes=frozenset({
        "class_declaration",
        "export_statement",
    }),
)


TYPESCRIPT_CONFIG = LanguageChunkConfig(
    name="typescript",
    language=TYPESCRIPT_LANGUAGE,
    file_types=frozenset({
        "typescript",
        "ts",
    }),
    extensions=frozenset({
        ".ts",
        ".mts",
        ".cts",
    }),
    structural_nodes=frozenset({
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "internal_module",
        "ambient_declaration",
        "lexical_declaration",
        "export_statement",
    }),
    recursive_nodes=frozenset({
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "internal_module",
        "ambient_declaration",
        "export_statement",
    }),
)


TSX_CONFIG = LanguageChunkConfig(
    name="tsx",
    language=TSX_LANGUAGE,
    file_types=frozenset({
        "tsx",
    }),
    extensions=frozenset({
        ".tsx",
    }),
    structural_nodes=TYPESCRIPT_CONFIG.structural_nodes,
    recursive_nodes=TYPESCRIPT_CONFIG.recursive_nodes,
)


C_CONFIG = LanguageChunkConfig(
    name="c",
    language=C_LANGUAGE,
    file_types=frozenset({
        "c",
    }),
    extensions=frozenset({
        ".c",
        ".h",
    }),
    structural_nodes=frozenset({
        "function_definition",
        "declaration",
        "type_definition",
        "linkage_specification",
        "preproc_def",
        "preproc_function_def",
        "preproc_if",
        "preproc_ifdef",
    }),
    recursive_nodes=frozenset({
        "function_definition",
        "declaration",
        "type_definition",
        "linkage_specification",
        "preproc_if",
        "preproc_ifdef",
    }),
)


CPP_CONFIG = LanguageChunkConfig(
    name="cpp",
    language=CPP_LANGUAGE,
    file_types=frozenset({
        "cpp",
        "c++",
        "cplusplus",
    }),
    extensions=frozenset({
        ".cc",
        ".cpp",
        ".cxx",
        ".hh",
        ".hpp",
        ".hxx",
        ".ipp",
        ".tpp",
    }),
    structural_nodes=frozenset({
        "function_definition",
        "declaration",
        "template_declaration",
        "template_instantiation",
        "namespace_definition",
        "namespace_alias_definition",
        "linkage_specification",
        "type_definition",
        "alias_declaration",
        "concept_definition",
        "module_declaration",
        "preproc_def",
        "preproc_function_def",
        "preproc_if",
        "preproc_ifdef",
    }),
    recursive_nodes=frozenset({
        "function_definition",
        "declaration",
        "template_declaration",
        "namespace_definition",
        "linkage_specification",
        "preproc_if",
        "preproc_ifdef",
    }),
)


RUST_CONFIG = LanguageChunkConfig(
    name="rust",
    language=RUST_LANGUAGE,
    file_types=frozenset({
        "rust",
        "rs",
    }),
    extensions=frozenset({
        ".rs",
    }),
    structural_nodes=frozenset({
        "function_item",
        "function_signature_item",
        "struct_item",
        "enum_item",
        "union_item",
        "trait_item",
        "impl_item",
        "mod_item",
        "type_item",
        "const_item",
        "static_item",
        "macro_definition",
        "foreign_mod_item",
    }),
    recursive_nodes=frozenset({
        "impl_item",
        "trait_item",
        "mod_item",
        "foreign_mod_item",
    }),
)


GO_CONFIG = LanguageChunkConfig(
    name="go",
    language=GO_LANGUAGE,
    file_types=frozenset({
        "go",
        "golang",
    }),
    extensions=frozenset({
        ".go",
    }),
    structural_nodes=frozenset({
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "const_declaration",
        "var_declaration",
    }),
    recursive_nodes=frozenset({
        "type_declaration",
        "const_declaration",
        "var_declaration",
    }),
)


ZIG_CONFIG = LanguageChunkConfig(
    name="zig",
    language=ZIG_LANGUAGE,
    file_types=frozenset({
        "zig",
    }),
    extensions=frozenset({
        ".zig",
    }),
    structural_nodes=frozenset({
        "function_declaration",
        "test_declaration",
        "variable_declaration",
        "comptime_declaration",
    }),
    recursive_nodes=frozenset({
        "function_declaration",
        "test_declaration",
        "variable_declaration",
        "comptime_declaration",
        "struct_declaration",
        "enum_declaration",
        "union_declaration",
        "opaque_declaration",
    }),
)


LANGUAGE_CONFIGS: tuple[LanguageChunkConfig, ...] = (
    PYTHON_CONFIG,
    JAVASCRIPT_CONFIG,
    TYPESCRIPT_CONFIG,
    TSX_CONFIG,
    C_CONFIG,
    CPP_CONFIG,
    RUST_CONFIG,
    GO_CONFIG,
    ZIG_CONFIG,
)


def _build_file_type_registry(
    configs: tuple[LanguageChunkConfig, ...],
) -> Mapping[str, LanguageChunkConfig]:
    """Build and validate the file-type registry."""
    registry: dict[str, LanguageChunkConfig] = {}

    for config in configs:
        for file_type in config.file_types:
            normalized = file_type.casefold()

            existing = registry.get(normalized)

            if existing is not None and existing is not config:
                raise ValueError(
                    "Duplicate Tree-sitter file type "
                    f"{file_type!r}: "
                    f"{existing.name!r} and {config.name!r}"
                )

            registry[normalized] = config

    return MappingProxyType(registry)


def _build_extension_registry(
    configs: tuple[LanguageChunkConfig, ...],
) -> Mapping[str, LanguageChunkConfig]:
    """Build and validate the extension registry."""
    registry: dict[str, LanguageChunkConfig] = {}

    for config in configs:
        for extension in config.extensions:
            normalized = extension.casefold()

            if not normalized.startswith("."):
                raise ValueError(f"Extension must begin with '.': {extension!r}")

            existing = registry.get(normalized)

            if existing is not None and existing is not config:
                raise ValueError(
                    "Duplicate Tree-sitter extension "
                    f"{extension!r}: "
                    f"{existing.name!r} and {config.name!r}"
                )

            registry[normalized] = config

    return MappingProxyType(registry)


CONFIG_BY_FILE_TYPE = _build_file_type_registry(LANGUAGE_CONFIGS)

CONFIG_BY_EXTENSION = _build_extension_registry(LANGUAGE_CONFIGS)

# Generated directories that should only be ignored for the languages that
# commonly create them.  Keeping these here prevents discovery from applying,
# for example, Rust's ``target`` rule to unrelated source trees.
_LANGUAGE_IGNORED_DIRECTORIES: Mapping[str, frozenset[str]] = MappingProxyType({
    "python": frozenset({"__pycache__"}),
    "javascript": frozenset({"node_modules"}),
    "typescript": frozenset({"node_modules"}),
    "tsx": frozenset({"node_modules"}),
    "rust": frozenset({"target"}),
    "c": frozenset({"build"}),
    "cpp": frozenset({"build"}),
    "zig": frozenset({"zig-cache", ".zig-cache", "zig-out"}),
})


def ignored_directory_names_for_file_type(file_type: str) -> frozenset[str]:
    """Return generated-directory names for one canonical file type."""
    config = CONFIG_BY_FILE_TYPE.get(file_type.casefold().strip())
    if config is None:
        return frozenset()
    return _LANGUAGE_IGNORED_DIRECTORIES.get(config.name, frozenset())


def file_type_for_extension(extension: str) -> str | None:
    """Return the canonical file type for an extension."""
    config = CONFIG_BY_EXTENSION.get(extension.casefold())
    return None if config is None else config.name


def get_language_config(
    file_type: str,
    file_path: str,
) -> LanguageChunkConfig | None:
    """Resolve the Tree-sitter configuration for a source file.

    Explicit file type takes priority over extension detection.

    Args:
        file_type: Normalized or user-provided file type.
        file_path: Source path used for extension detection.

    Returns:
        Matching language configuration, or None when unsupported.
    """
    normalized_file_type = file_type.casefold().strip()

    by_file_type = CONFIG_BY_FILE_TYPE.get(normalized_file_type)

    if by_file_type is not None:
        return by_file_type

    extension = Path(file_path).suffix.casefold()
    return CONFIG_BY_EXTENSION.get(extension)


def get_config_by_name(
    language_name: str,
) -> LanguageChunkConfig | None:
    """Find a configuration by its canonical language name."""
    normalized_name = language_name.casefold().strip()

    for config in LANGUAGE_CONFIGS:
        if config.name == normalized_name:
            return config

    return None


def supported_extensions() -> frozenset[str]:
    """Return all extensions supported by Tree-sitter chunking."""
    return frozenset(CONFIG_BY_EXTENSION)


def supported_file_types() -> frozenset[str]:
    """Return all accepted SourceFile file-type values."""
    return frozenset(CONFIG_BY_FILE_TYPE)
