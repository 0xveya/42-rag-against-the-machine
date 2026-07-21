"""Unit tests for the CLI argument parser."""

from dataclasses import dataclass, field
from typing import Any, cast

from rag_against_the_machine.cli_fw import Action, Command, Parser, arg
from rag_against_the_machine.errors import CliError, Err, Ok


@dataclass
class RobustTestArgs:
    name: str = field(metadata={"help": "The resource name"})
    verbose: bool = field(default=False, metadata={"help": "Enable logging"})
    tags: list[str] = field(
        default_factory=list,
        metadata={"help": "Metadata tags", "action": Action.APPEND},
    )
    format: str = field(
        default="json",
        metadata={"help": "Output format", "choices": ["json", "csv", "xml"]},
    )
    limit: int = field(default=10, metadata={"help": "Max items to fetch"})


@dataclass
class MinimalArgs:
    required_flag: str


def assert_has_help_msg(result: Err[CliError], substring: str, invert: bool = False) -> None:
    """Helper to check help message substring."""
    assert result.diagnostic is not None, "Expected a diagnostic object, but got None"
    assert result.diagnostic.help_msg is not None, "Expected help_msg to be a string, but got None"
    if invert:
        assert substring not in result.diagnostic.help_msg
    else:
        assert substring in result.diagnostic.help_msg


def test_parse_success_all_fields():
    """Verify parsing succeeds when all argument types are well-formed."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = [
        "--name",
        "prod-cluster",
        "--verbose",
        "--tags",
        "infra",
        "--tags",
        "database",
        "--format",
        "csv",
        "--limit",
        "50",
    ]

    result = parser.parse(argv)

    assert isinstance(result, Ok)
    data = cast(Ok[dict[str, Any]], result).value
    assert data["name"] == "prod-cluster"
    assert data["verbose"] is True
    assert data["tags"] == ["infra", "database"]
    assert data["format"] == "csv"
    assert data["limit"] == 50


def test_parse_success_defaults():
    """Verify default fallback values match dataclass factory defaults."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "dev-box"]

    result = parser.parse(argv)

    assert isinstance(result, Ok)
    data = cast(Ok[dict[str, Any]], result).value
    assert data["name"] == "dev-box"
    assert data["verbose"] is False
    assert data["tags"] == []
    assert data["format"] == "json"
    assert data["limit"] == 10


def test_missing_required_argument():
    """Should fail if a required field is omitted."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--verbose", "--limit", "5"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.MISSING_REQUIRED_ARGUMENT
    assert_has_help_msg(cast(Err[CliError], result), "Missing required arguments: --name")


def test_missing_argument_value():
    """Flags requiring a value should fail if given trailing flag or hit EOL."""
    parser = Parser.from_dataclass(RobustTestArgs)

    # Trailing value missing completely
    result_eol = parser.parse(["--name"])
    assert isinstance(result_eol, Err)
    assert result_eol.error == CliError.MISSING_ARGUMENT_VALUE
    assert_has_help_msg(cast(Err[CliError], result_eol), "Option --name requires an argument")

    # Next item is another flag instead of a value
    result_flag = parser.parse(["--name", "--verbose"])
    assert isinstance(result_flag, Err)
    assert result_flag.error == CliError.MISSING_ARGUMENT_VALUE


def test_unexpected_positional_argument():
    """Unregistered raw strings should be handled as invalid positional args."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "test", "stray_positional_value"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.UNKNOWN_ARGUMENT
    assert_has_help_msg(cast(Err[CliError], result), "Unexpected positional argument")


def test_invalid_choice_fuzzy_suggestion():
    """Fuzzy engine suggests match if distance <= 2."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "app", "--format", "jjson"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.INVALID_CHOICE
    assert_has_help_msg(cast(Err[CliError], result), "Did you mean 'json'?")


def test_invalid_choice_no_suggestion():
    """Fuzzy engine shouldn't guess wild choices if distance is far."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "app", "--format", "parquet"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.INVALID_CHOICE
    assert_has_help_msg(cast(Err[CliError], result), "Did you mean", invert=True)


def test_unknown_argument_typo_suggestion():
    """Fuzzy engine should match argument keys when flags contain typos."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--namme", "typo-fix"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.UNKNOWN_ARGUMENT
    assert_has_help_msg(
        cast(Err[CliError], result),
        "Unknown argument: --namme. Did you mean '--name'?",
    )


def test_parse_into_object():
    """Ensure parse_into maps parsed dicts back to requested structures."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "hydrated-obj", "--limit", "100"]

    result = parser.parse_into(RobustTestArgs, argv)

    assert isinstance(result, Ok)
    args_obj = result.value
    assert isinstance(args_obj, RobustTestArgs)
    assert args_obj.name == "hydrated-obj"
    assert args_obj.limit == 100


def test_parse_into_type_coercion_error():
    """Parser must forward data casting issues if conversion logic breaks."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--name", "coercion-fail", "--limit", "not-an-int"]

    result = parser.parse(argv)
    assert isinstance(result, Err)
    assert result.error == CliError.INVALID_ARGUMENT_TYPE
    assert_has_help_msg(
        cast(Err[CliError], result),
        "Invalid value for argument 'limit': 'not-an-int'. Expected type int.",
    )


def test_unknown_argument_help_typo_suggestion():
    """Fuzzy engine should match help key when help flag contains typos."""
    parser = Parser.from_dataclass(RobustTestArgs)
    argv = ["--hhelp"]

    result = parser.parse(argv)

    assert isinstance(result, Err)
    assert result.error == CliError.UNKNOWN_ARGUMENT
    assert_has_help_msg(
        cast(Err[CliError], result),
        "Unknown argument: --hhelp. Did you mean '--help'?",
    )


@dataclass
class PositionalTestArgs:
    first: str = field(metadata={"positional": True, "help": "First arg"})
    second: int = field(metadata={"positional": True, "help": "Second arg"})
    flag: bool = field(default=False, metadata={"help": "A boolean flag"})


def test_positional_arguments():
    # Test standard positional argument parsing
    parser = Parser.from_dataclass(PositionalTestArgs)

    # 1. As purely positional values
    res = parser.parse_into(PositionalTestArgs, ["val1", "42"])
    assert isinstance(res, Ok)
    assert res.value.first == "val1"
    assert res.value.second == 42
    assert res.value.flag is False

    # 2. Mixed with flags
    res = parser.parse_into(PositionalTestArgs, ["--flag", "val1", "42"])
    assert isinstance(res, Ok)
    assert res.value.first == "val1"
    assert res.value.second == 42
    assert res.value.flag is True

    # 3. Positional args provided as flags
    res = parser.parse_into(PositionalTestArgs, ["--first", "val1", "--second", "100"])
    assert isinstance(res, Ok)
    assert res.value.first == "val1"
    assert res.value.second == 100

    # 4. Mixed: one positional value, one flag
    res = parser.parse_into(PositionalTestArgs, ["--second", "100", "val1"])
    assert isinstance(res, Ok)
    assert res.value.first == "val1"
    assert res.value.second == 100

    # 5. Too many positional values
    res_err = parser.parse(["val1", "42", "extra_val"])
    assert isinstance(res_err, Err)
    assert res_err.error == CliError.UNKNOWN_ARGUMENT
    assert_has_help_msg(
        cast(Err[CliError], res_err),
        "Unexpected positional argument: extra_val",
    )


@dataclass
class ChildConfig:
    host: str = field(metadata={"help": "The host"})
    port: int = field(default=8080, metadata={"help": "The port"})


@dataclass
class ParentConfig:
    title: str = field(metadata={"help": "App title"})
    child: ChildConfig = field(metadata={"help": "Nested config"})
    debug: bool = field(default=False, metadata={"help": "Debug mode"})


def test_nested_dataclasses():
    parser = Parser.from_dataclass(ParentConfig)

    # 1. Parsing all fields including nested ones
    res = parser.parse_into(
        ParentConfig,
        [
            "--title",
            "MyApp",
            "--child.host",
            "localhost",
            "--child.port",
            "9000",
            "--debug",
        ],
    )
    assert isinstance(res, Ok)
    assert res.value.title == "MyApp"
    assert res.value.child.host == "localhost"
    assert res.value.child.port == 9000
    assert res.value.debug is True

    # 2. Nested field defaults used
    res = parser.parse_into(ParentConfig, ["--title", "MyApp", "--child.host", "localhost"])
    assert isinstance(res, Ok)
    assert res.value.child.port == 8080

    # 3. Missing nested required field fails
    res_err = parser.parse(["--title", "MyApp"])
    assert isinstance(res_err, Err)
    assert res_err.error == CliError.MISSING_REQUIRED_ARGUMENT
    assert_has_help_msg(cast(Err[CliError], res_err), "Missing required arguments: --child.host")


def test_command_subcommands():
    # 1. Create a root command
    root = Command("app", "Root CLI application")

    # 2. Create a group command
    config = Command("config", "Configuration management")
    root.add_command(config)

    # 3. Create subcommands
    @dataclass
    class SetCmd:
        key: str = field(metadata={"help": "The config key"})
        value: str = field(metadata={"help": "The config value"})

        def run(self) -> str:
            return f"Set {self.key} = {self.value}"

    set_cmd = Command("set", "Set a config value", schema=SetCmd)
    config.add_command(set_cmd)

    @dataclass
    class GreetCmd:
        name: str = field(metadata={"positional": True, "help": "Name of user"})
        shout: bool = field(default=False, metadata={"help": "Shout the greeting"})

    def run_greet(args: GreetCmd) -> str:
        msg = f"Hello {args.name}"
        return msg.upper() if args.shout else msg

    greet_cmd = Command("greet", "Greet a user", schema=GreetCmd, run=run_greet)
    root.add_command(greet_cmd)

    # Test executing a simple subcommand
    res = root.execute(["greet", "Veya"])
    assert isinstance(res, Ok)
    assert res.value == "Hello Veya"

    # Test with flags
    res = root.execute(["greet", "Veya", "--shout"])
    assert isinstance(res, Ok)
    assert res.value == "HELLO VEYA"

    # Test executing nested subcommand
    res = root.execute(["config", "set", "--key", "theme", "--value", "dark"])
    assert isinstance(res, Ok)
    assert res.value == "Set theme = dark"

    # Test executing a group without a command (should fail and show help)
    res_err = root.execute(["config"])
    assert isinstance(res_err, Err)
    assert res_err.error == CliError.MISSING_REQUIRED_ARGUMENT
    assert_has_help_msg(cast(Err[CliError], res_err), "Command 'config' requires a subcommand.")


def test_arg_helper_and_nargs():
    @dataclass
    class CopyArgs:
        sources: list[str] = arg(positional=True, help="Source files")
        dest: str = arg(positional=True, help="Destination directory")
        verbose: bool = arg(default=False, help="Verbose output")

    parser = Parser.from_dataclass(CopyArgs)

    # Test N-args positional mapping
    res = parser.parse_into(CopyArgs, ["file1.txt", "file2.txt", "file3.txt", "/tmp", "--verbose"])
    assert isinstance(res, Ok)
    assert res.value.sources == ["file1.txt", "file2.txt", "file3.txt"]
    assert res.value.dest == "/tmp"
    assert res.value.verbose is True

    # Test Command with short/long/example help attributes
    cmd = Command(
        "cp",
        short="Copy files",
        long="Copies source files to a destination directory",
        example="cp file1.txt file2.txt /tmp",
        schema=CopyArgs,
    )
    assert cmd.short == "Copy files"
    assert cmd.long == "Copies source files to a destination directory"
    assert cmd.example == "cp file1.txt file2.txt /tmp"
