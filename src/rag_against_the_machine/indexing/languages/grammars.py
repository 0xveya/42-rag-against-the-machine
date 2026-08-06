"""Load installed Tree-sitter grammar packages."""

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
import tree_sitter_zig
from tree_sitter import Language

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
JAVASCRIPT_LANGUAGE = Language(tree_sitter_javascript.language())

TYPESCRIPT_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
TSX_LANGUAGE = Language(tree_sitter_typescript.language_tsx())

C_LANGUAGE = Language(tree_sitter_c.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
RUST_LANGUAGE = Language(tree_sitter_rust.language())
GO_LANGUAGE = Language(tree_sitter_go.language())
ZIG_LANGUAGE = Language(tree_sitter_zig.language())
