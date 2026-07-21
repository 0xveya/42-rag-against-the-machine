"""Command line argument parser."""

# i really need unit tests for this

import sys
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

from .errors import CliError, Diagnostic, Err, Ok, Result

T = TypeVar("T")

HelpRenderer = Callable[["Parser"], None]


def CliErr(error: CliError, diagnostic: Diagnostic | None = None) -> Err[CliError]:
    """Helper to automatically bake module defaults into every error."""
    return Err(
        error=error,
        diagnostic=diagnostic,
        namespace="cli::parser",
        context_msg="Command line syntax analysis failed",
    )


def arg(
    help: str = "",
    positional: bool = False,
    choices: list[Any] | None = None,
    default: Any = MISSING,
    default_factory: Any = MISSING,
) -> Any:
    """Helper to define dataclass fields for CLI arguments without nested metadata dicts."""
    metadata = {
        "help": help,
        "positional": positional,
        "choices": choices or [],
    }
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")

    if default is not MISSING:
        return field(default=default, metadata=metadata)
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(metadata=metadata)


class HelpMenuStyle(Enum):
    """Enumerate the available help menu styles."""

    FANCY = auto()
    VANILLA = auto()
    CUSTOM = auto()


class Action(Enum):
    """Actions for the Arg class."""

    STORE = "store"
    STORE_TRUE = "store_true"
    APPEND = "append"


def _resolve_type(typ: Any) -> Any:
    import types
    import typing

    origin = get_origin(typ)
    UnionTypes = (types.UnionType, typing.Union) if hasattr(types, "UnionType") else (typing.Union,)
    if origin in UnionTypes:
        args = [t for t in get_args(typ) if t is not type(None)]
        if len(args) == 1:
            return _resolve_type(args[0])
    return typ


def _collect_args(
    schema: type[Any],
    prefix: str = "",
    parent_required: bool = True,
    parent_default: Any = None,
) -> list["Arg"]:
    args_list = []
    type_hints = get_type_hints(schema)
    for f in fields(schema):
        typ = type_hints.get(f.name, f.type)
        resolved_typ = _resolve_type(typ)

        required = parent_required and (f.default is MISSING and f.default_factory is MISSING)

        default = None
        if parent_default is not None:
            default = getattr(parent_default, f.name, None)
        else:
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:
                default = f.default_factory()

        is_positional = f.metadata.get("positional", False)

        if is_dataclass(resolved_typ):
            nested_args = _collect_args(
                cast(type[Any], resolved_typ),
                prefix=f"{prefix}{f.name}.",
                parent_required=required,
                parent_default=default,
            )
            args_list.extend(nested_args)
        else:
            action = Action.STORE
            arg_type = str
            if resolved_typ is bool:
                action = Action.STORE_TRUE
            elif get_origin(resolved_typ) is list:
                action = Action.APPEND
                arg_type = cast(type[Any], get_args(resolved_typ)[0])
            else:
                origin = get_origin(resolved_typ)
                if origin is not None:
                    args = [t for t in get_args(resolved_typ) if t is not type(None)]
                    if len(args) == 1:
                        arg_type = cast(type[Any], args[0])
                else:
                    arg_type = cast(type[Any], resolved_typ)

            choices = f.metadata.get("choices", [])
            help_text = f.metadata.get("help", "")

            args_list.append(
                Arg(
                    name=f"{prefix}{f.name}",
                    help=help_text,
                    default=default,
                    required=required,
                    arg_type=arg_type,
                    choices=choices,
                    action=action,
                    positional=is_positional,
                )
            )
    return args_list


def _instantiate_schema(
    schema: type[Any],
    values: dict[str, Any],
    explicitly_provided: set[str],
    prefix: str = "",
) -> Any:
    type_hints = get_type_hints(schema)
    args_dict = {}
    for f in fields(schema):
        typ = type_hints.get(f.name, f.type)
        resolved_typ = _resolve_type(typ)
        key = f"{prefix}{f.name}"
        if is_dataclass(resolved_typ):
            any_provided = any(k.startswith(f"{key}.") for k in explicitly_provided)
            if any_provided:
                args_dict[f.name] = _instantiate_schema(
                    cast(type[Any], resolved_typ), values, explicitly_provided, f"{key}."
                )
            else:
                if f.default is not MISSING:
                    args_dict[f.name] = f.default
                elif f.default_factory is not MISSING:
                    args_dict[f.name] = f.default_factory()
                else:
                    args_dict[f.name] = _instantiate_schema(
                        cast(type[Any], resolved_typ), values, explicitly_provided, f"{key}."
                    )
        else:
            args_dict[f.name] = values.get(key)
    return schema(**args_dict)


@dataclass
class Arg:
    """Argument definition for the Parser class."""

    name: str
    help: str
    default: Any = None
    required: bool = False
    arg_type: type[Any] = str
    choices: list[str] = field(default_factory=list)
    action: Action = Action.STORE
    positional: bool = False


@dataclass
class Parser:
    """Command line argument parser."""

    description: str
    args: list[Arg] = field(default_factory=list)
    help_flag: str = "help"
    style: HelpMenuStyle = HelpMenuStyle.FANCY
    help_renderer: HelpRenderer | None = None

    def add(
        self,
        name: str,
        help: str,
        default: Any = None,
        required: bool = False,
        arg_type: type[Any] = str,
        choices: list[str] | None = None,
        action: Action = Action.STORE,
        positional: bool = False,
    ) -> None:
        """Add an argument to the parser."""
        self.args.append(
            Arg(
                name=name,
                help=help,
                default=default,
                required=required,
                arg_type=arg_type,
                choices=choices or [],
                action=action,
                positional=positional,
            )
        )

    def parse(self, argv: list[str] | None = None) -> Result[dict[str, object], CliError]:
        """Parse the command line arguments."""
        if argv is None:
            argv = sys.argv[1:]

        if f"--{self.help_flag}" in argv or f"-{self.help_flag[0]}" in argv:
            self.help()
            sys.exit(0)

        raw_cmd_string = " ".join(argv)
        possible = {a.name for a in self.args}
        possible.add(self.help_flag)

        parsed_tokens: list[tuple[str, str | None]] = []
        positional_values: list[str] = []
        it = iter(argv)

        current_file = argv[0] if argv else "cli"
        for token in it:
            if token.startswith("-"):
                name = token.lstrip("-")
                if name not in possible:
                    col_start = raw_cmd_string.find(token)
                    col_end = col_start + len(token)
                    best_match, dist = _find_best_string_match(name, possible)
                    if dist is not None and best_match and dist <= 2:
                        help_msg = f"Unknown argument: --{name}. Did you mean '--{best_match}'?"
                    else:
                        help_msg = f"Unknown argument: --{name}"
                    diag = Diagnostic(
                        filename=current_file,
                        line_num=1,
                        line_text=raw_cmd_string,
                        col_start=max(0, col_start),
                        col_end=max(0, col_end),
                        help_msg=help_msg,
                    )
                    return CliErr(CliError.UNKNOWN_ARGUMENT, diagnostic=diag)

                arg_def = next(a for a in self.args if a.name == name)

                if arg_def.action == Action.STORE_TRUE:
                    parsed_tokens.append((name, None))
                else:
                    col_start = raw_cmd_string.find(token)
                    col_end = col_start + len(token)
                    try:
                        val = next(it)
                        if val.startswith("-"):
                            diag = Diagnostic(
                                filename=current_file,
                                line_num=1,
                                line_text=raw_cmd_string,
                                col_start=max(0, col_start),
                                col_end=max(0, col_end),
                                help_msg=(f"Option --{name} requires an argument"),
                            )
                            return CliErr(
                                CliError.MISSING_ARGUMENT_VALUE,
                                diag,
                            )
                    except StopIteration:
                        diag = Diagnostic(
                            filename=current_file,
                            line_num=1,
                            line_text=raw_cmd_string,
                            col_start=max(0, col_start),
                            col_end=max(0, col_end),
                            help_msg=f"Option --{name} requires an argument",
                        )
                        return CliErr(CliError.MISSING_ARGUMENT_VALUE, diag)

                    parsed_tokens.append((name, val))
            else:
                positional_values.append(token)

        positional_args = [a for a in self.args if getattr(a, "positional", False)]
        seen_names = {name for name, _ in parsed_tokens}
        unbound_positional_args = [a for a in positional_args if a.name not in seen_names]

        val_idx = 0
        for i, arg in enumerate(unbound_positional_args):
            if val_idx >= len(positional_values):
                break

            remaining_vals = len(positional_values) - val_idx
            remaining_args_count = len(unbound_positional_args) - i - 1

            if arg.action == Action.APPEND:
                consume_count = remaining_vals - remaining_args_count
                if consume_count < 0:
                    consume_count = 0
                consumed = positional_values[val_idx : val_idx + consume_count]
                val_idx += consume_count
                for c_val in consumed:
                    parsed_tokens.append((arg.name, c_val))
            else:
                c_val = positional_values[val_idx]
                val_idx += 1
                parsed_tokens.append((arg.name, c_val))

        if val_idx < len(positional_values):
            unexpected_token = positional_values[val_idx]
            col_start = raw_cmd_string.find(unexpected_token)
            col_end = col_start + len(unexpected_token)
            diag = Diagnostic(
                filename=current_file,
                line_num=1,
                line_text=raw_cmd_string,
                col_start=max(0, col_start),
                col_end=max(0, col_end),
                help_msg=f"Unexpected positional argument: {unexpected_token}",
            )
            return CliErr(CliError.UNKNOWN_ARGUMENT, diag)

        seen = {name for name, _ in parsed_tokens}
        self.explicitly_provided = seen

        missing = [a.name for a in self.args if a.required and a.name not in seen]
        if missing:
            col_start = raw_cmd_string.find(f"--{' '.join(missing)}")
            if col_start == -1:
                col_start = raw_cmd_string.find(" ".join(missing))
            col_end = col_start + len(f"--{' '.join(missing)}")
            diag = Diagnostic(
                filename=current_file,
                line_num=1,
                line_text=raw_cmd_string,
                col_start=max(0, col_start),
                col_end=max(0, col_end),
                help_msg=(f"Missing required arguments: {', '.join(f'--{m}' for m in missing)}"),
            )
            return CliErr(CliError.MISSING_REQUIRED_ARGUMENT, diag)

        append_started = set()
        result = {a.name: a.default for a in self.args}
        for name, token_val in parsed_tokens:
            arg = next(a for a in self.args if a.name == name)

            if arg.action == Action.STORE_TRUE:
                result[arg.name] = True
            elif arg.action == Action.APPEND:
                try:
                    coerced = arg.arg_type(token_val)
                    if arg.name not in append_started:
                        result[arg.name] = []
                        append_started.add(arg.name)
                    result[arg.name].append(coerced)
                except (ValueError, TypeError):
                    val_str = str(token_val) if token_val is not None else ""
                    col_start = raw_cmd_string.rfind(val_str)
                    col_end = col_start + len(val_str) if col_start != -1 else 0
                    diag = Diagnostic(
                        filename=current_file,
                        line_num=1,
                        line_text=raw_cmd_string,
                        col_start=max(0, col_start),
                        col_end=max(0, col_end),
                        help_msg=(
                            f"Invalid value for argument '{arg.name}': '{token_val}'. "
                            f"Expected type {arg.arg_type.__name__}."
                        ),
                    )
                    return CliErr(CliError.INVALID_ARGUMENT_TYPE, diag)
            else:
                if arg.choices and token_val not in arg.choices:
                    val_str = str(token_val) if token_val is not None else ""
                    col_start = raw_cmd_string.rfind(val_str)
                    col_end = col_start + len(val_str) if col_start != -1 else 0
                    best_match, dist = _find_best_string_match(val_str, arg.choices)
                    if dist is not None and best_match and dist <= 2:
                        help_msg = (
                            f"{arg.name}: '{token_val}' not in {arg.choices}. "
                            f"Did you mean '{best_match}'?"
                        )
                    else:
                        help_msg = f"{arg.name}: '{token_val}' not in {arg.choices}"
                    diag = Diagnostic(
                        filename=current_file,
                        line_num=1,
                        line_text=raw_cmd_string,
                        col_start=max(0, col_start),
                        col_end=max(0, col_end),
                        help_msg=help_msg,
                    )
                    return CliErr(CliError.INVALID_CHOICE, diag)

                try:
                    result[arg.name] = arg.arg_type(token_val)
                except (ValueError, TypeError):
                    val_str = str(token_val) if token_val is not None else ""
                    col_start = raw_cmd_string.rfind(val_str)
                    col_end = col_start + len(val_str) if col_start != -1 else 0
                    diag = Diagnostic(
                        filename=current_file,
                        line_num=1,
                        line_text=raw_cmd_string,
                        col_start=max(0, col_start),
                        col_end=max(0, col_end),
                        help_msg=(
                            f"Invalid value for argument '{arg.name}': '{token_val}'. "
                            f"Expected type {arg.arg_type.__name__}."
                        ),
                    )
                    return CliErr(CliError.INVALID_ARGUMENT_TYPE, diag)

        return Ok(result)

    @classmethod
    def from_dataclass(
        cls,
        schema: type[Any],
        description: str = "",
    ) -> "Parser":
        if not is_dataclass(schema):
            raise TypeError("schema must be a dataclass")

        parser = cls(description=description)
        for arg in _collect_args(schema):
            parser.args.append(arg)
        return parser

    def parse_into(
        self,
        schema: type[T],
        argv: list[str] | None = None,
    ) -> Result[T, CliError]:
        result = self.parse(argv)

        match result:
            case Err() as err:
                return cast(Err[CliError], err)

            case Ok(values):
                instantiated = _instantiate_schema(
                    schema, values, getattr(self, "explicitly_provided", set())
                )
                return Ok(instantiated)

    def help(self) -> None:
        if self.help_renderer is not None:
            self.help_renderer(self)
            return

        match self.style:
            case HelpMenuStyle.FANCY:
                self._help_fancy()

            case HelpMenuStyle.VANILLA:
                self._help_vanilla()

    def _help_fancy(self) -> None:
        """Prints the help menu in fancy style."""
        BOLD = "\033[1m"
        BLUE = "\033[1;34m"
        CYAN = "\033[1;36m"
        RESET = "\033[0m"

        print(f"{BOLD}llm_router::help{RESET}\n")

        positional_args = [a for a in self.args if getattr(a, "positional", False)]
        pos_usage = " ".join(f"[{a.name}]" for a in positional_args)
        usage_line = "llm-router [options]"
        if pos_usage:
            usage_line += f" {pos_usage}"

        print(f" {BLUE}╭─{RESET} Usage")
        print(f" {BLUE}│{RESET}")
        print(f" {BLUE}│{RESET}  {usage_line}")
        print(f" {BLUE}│{RESET}")
        print(f" {BLUE}╰─▶{RESET} Required arguments\n")

        for arg in self.args:
            if arg.required:
                arg_label = (
                    f"{arg.name} (or --{arg.name})"
                    if getattr(arg, "positional", False)
                    else f"--{arg.name}"
                )
                print(f"     {CYAN}{arg_label}{RESET}")
                print(f"         {arg.help}")
                print()

        print(f"\n {BLUE}╭─{RESET} Optional arguments")
        print(f" {BLUE}│{RESET}")

        for arg in self.args:
            if not arg.required:
                arg_label = (
                    f"{arg.name} (or --{arg.name})"
                    if getattr(arg, "positional", False)
                    else f"--{arg.name}"
                )
                print(f" {BLUE}│{RESET}  {CYAN}{arg_label}{RESET}")

                if arg.help:
                    print(f" {BLUE}│{RESET}      {arg.help}")

                if arg.default is not None:
                    print(f" {BLUE}│{RESET}      default: {arg.default}")

                print(f" {BLUE}│{RESET}")

        print(f" {BLUE}╰─▶{RESET} {CYAN}--{self.help_flag}{RESET}\n     Show this message")

    def _help_vanilla(self) -> None:
        """Prints the help menu in vanilla style."""
        print(self.description)
        print()

        print("Options:")
        for a in self.args:
            req = "(required)" if a.required else f"(default: {a.default})"
            arg_label = (
                f"{a.name} (or --{a.name})" if getattr(a, "positional", False) else f"--{a.name}"
            )

            print(f"  {arg_label:<24} {a.help} {req}")

        print(f"\n  --{self.help_flag:<24} Show this help message")


class Command:
    """Cobra-like subcommand and group support."""

    def __init__(
        self,
        name: str,
        short: str = "",
        long: str = "",
        example: str = "",
        schema: type[Any] | None = None,
        run: Callable[[Any], Any] | None = None,
    ):
        self.name = name
        self.short = short
        self.long = long
        self.example = example
        self.schema = schema
        self.run_func = run
        self.commands: dict[str, Command] = {}
        self.parent: Command | None = None

    def add_command(self, cmd: "Command") -> "Command":
        cmd.parent = self
        self.commands[cmd.name] = cmd
        return cmd

    def _resolve_command(self, argv: list[str]) -> tuple["Command", list[str]]:
        current = self
        args = list(argv)
        while args:
            token = args[0]
            if token.startswith("-"):
                break
            if token in current.commands:
                current = current.commands[token]
                args.pop(0)
            else:
                break
        return current, args

    def execute(self, argv: list[str] | None = None) -> Result[Any, CliError]:
        """Parse arguments, resolve subcommand, instantiate schema, and execute it."""
        if argv is None:
            argv = sys.argv[1:]

        # Check for help flag anywhere in argv
        help_flag = "help"
        has_help = f"--{help_flag}" in argv or f"-{help_flag[0]}" in argv

        target_cmd, remaining_argv = self._resolve_command(argv)

        if has_help:
            target_cmd.help()
            sys.exit(0)

        if target_cmd.commands and not target_cmd.schema:
            # It's a command group with no schema, so we require a subcommand
            target_cmd.help()
            raw_cmd_string = " ".join(argv)
            diag = Diagnostic(
                filename=argv[0] if argv else "cli",
                line_num=1,
                line_text=raw_cmd_string,
                col_start=0,
                col_end=len(raw_cmd_string),
                help_msg=f"Command '{target_cmd.name}' requires a subcommand.",
            )
            return CliErr(CliError.MISSING_REQUIRED_ARGUMENT, diag)

        if target_cmd.schema:
            desc = target_cmd.long or target_cmd.short
            parser = Parser.from_dataclass(target_cmd.schema, description=desc)
            res = parser.parse_into(target_cmd.schema, remaining_argv)
            match res:
                case Err() as err:
                    return err
                case Ok(instance):
                    if target_cmd.run_func is not None:
                        run_res = target_cmd.run_func(instance)
                    elif hasattr(instance, "run") and callable(instance.run):
                        run_res = instance.run()
                    else:
                        return Ok(instance)

                    if isinstance(run_res, (Ok, Err)):
                        return run_res
                    return Ok(run_res)
        else:
            if target_cmd.run_func is not None:
                run_res = target_cmd.run_func(None)
                if isinstance(run_res, (Ok, Err)):
                    return run_res
                return Ok(run_res)
            return Ok(None)

    def help(self) -> None:
        """Prints the help menu for this command/group."""
        path = []
        curr: Command | None = self
        while curr:
            path.append(curr.name)
            curr = curr.parent
        cmd_path = " ".join(reversed(path))

        BOLD = "\033[1m"
        BLUE = "\033[1;34m"
        CYAN = "\033[1;36m"
        RESET = "\033[0m"

        print(f"{BOLD}{cmd_path}::help{RESET}\n")

        desc = self.long or self.short
        if desc:
            print(f" {desc}\n")

        if self.example:
            print(f" {BOLD}Examples:{RESET}")
            print(f"   {self.example}\n")

        usage_str = f" {BLUE}╭─{RESET} Usage\n"
        usage_str += f" {BLUE}│{RESET}\n"
        if self.commands:
            usage_str += f" {BLUE}│{RESET}  {cmd_path} [command]"
            if self.schema:
                usage_str += " [options]"
        else:
            usage_str += f" {BLUE}│{RESET}  {cmd_path} [options]"
        usage_str += f"\n {BLUE}│{RESET}\n"

        if self.commands:
            usage_str += f" {BLUE}╰─▶{RESET} Available Commands\n"
            print(usage_str)
            for name, cmd in self.commands.items():
                print(f"     {CYAN}{name:<12}{RESET} {cmd.short or cmd.long}")
            print()
        else:
            usage_str += f" {BLUE}╰─▶{RESET} Options\n"
            print(usage_str)

        if self.schema:
            desc_str = self.long or self.short
            parser = Parser.from_dataclass(self.schema, description=desc_str)
            for arg in parser.args:
                arg_label = (
                    f"{arg.name} (or --{arg.name})"
                    if getattr(arg, "positional", False)
                    else f"--{arg.name}"
                )
                print(f"     {CYAN}{arg_label}{RESET}")
                print(f"         {arg.help}")
                if arg.default is not None:
                    print(f"         default: {arg.default}")
                print()

            print(f"     {CYAN}--help{RESET}\n         Show this message")
        else:
            print(f"     {CYAN}--help{RESET}\n         Show this message")


@dataclass
class TestArgs:
    name: str = field(metadata={"help": "Your name"})

    verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose"},
    )

    tag: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Add a tag",
            "action": Action.APPEND,
        },
    )

    format: str = field(
        default="json",
        metadata={
            "help": "Output format",
            "choices": ["json", "csv", "xml"],
        },
    )


def main() -> None:
    """testing main dont use

    uv run python -m src.cli_fw \
        --name abc \
        --tag erm \
        --tag foo \
        --format json \
        --verbose
    """

    p = Parser.from_dataclass(
        TestArgs,
        description="My test CLI",
    )

    if len(sys.argv) == 1:
        sys.exit(67)

    result = p.parse_into(TestArgs)

    match result:
        case Ok() as ok:
            args = cast(TestArgs, ok.value)
            print(args)
            print(f"name    = {args.name}")
            print(f"verbose = {args.verbose}")
            print(f"tag     = {args.tag}")
            print(f"format  = {args.format}")

        case Err() as fallback:
            fallback.print_diagnostic()
            sys.exit(1)


def _levenshteinRecursive(str1: str, str2: str, len_str1: int, len_str2: int) -> int:
    """Calculates the Levenshtein distance between two strings"""
    s1 = str1[:len_str1]
    s2 = str2[:len_str2]
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def _find_best_string_match(
    target: str, valid_choices: list[str] | set[str]
) -> tuple[str | None, int | None]:
    """
    Finds the closest string match using Levenshtein distance with tie-breaks

    tie-breaking does those rules
    no
    one
    cares
    they are tho
    1. lower levenshtein distance
    2. longer shared common prefix
    3. closer absolute string length differences
    4. lexicographical :nerdemoji: ahh word fallback
    """
    if not valid_choices:
        return None, None

    def _common_prefix_len(s1: str, s2: str) -> int:
        count = 0
        for c1, c2 in zip(s1, s2, strict=False):
            if c1 != c2:
                break
            count += 1
        return count

    best_match, dist = min(
        (
            (
                choice,
                _levenshteinRecursive(target, choice, len(target), len(choice)),
            )
            for choice in valid_choices
        ),
        key=lambda item: (
            item[1],  # levenshtein distance
            -_common_prefix_len(target, item[0]),  # prefer common starting prefix
            abs(len(target) - len(item[0])),  # prefer closer string length
            item[0],  # alphabetical sorting
        ),
    )
    return best_match, dist


if __name__ == "__main__":
    main()
