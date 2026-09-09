"""The Click that Typer drives.

Typer 0.26 vendored Click into ``typer._click`` and dropped its dependency on
the standalone package. A command built by ``typer.main.get_command`` therefore
raises ``typer._click`` exceptions — which ``except click.UsageError`` does not
catch when ``click`` is the separately installed one — and its parameters are
not instances of ``click.Argument``. Both were silent behaviour changes: unknown
options started printing a traceback instead of a usage error.

So the CLI and the docs generator import their Click from here, and this module
resolves whichever copy Typer is actually using. The fallback keeps working on
Typer < 0.26, which drove the standalone package.

Only the names bioaccx needs are re-exported. There is deliberately no ``Group``
or ``Argument``: the vendored copy has neither (Typer supplies ``TyperGroup``
and ``TyperArgument``), so a group is recognised by having ``.commands`` and an
argument by ``param.param_type_name == "argument"``.
"""

from __future__ import annotations

try:  # Typer >= 0.26 — the vendored copy, and the only one Typer raises from.
    from typer import Abort as Abort
    from typer import Exit as Exit
    from typer._click.core import Command as Command
    from typer._click.core import Context as Context
    from typer._click.core import Parameter as Parameter
    from typer._click.exceptions import ClickException as ClickException
    from typer._click.exceptions import UsageError as UsageError
    from typer._click.globals import get_current_context as get_current_context
except ImportError:  # pragma: no cover - Typer < 0.26
    from click.core import Command as Command
    from click.core import Context as Context
    from click.core import Parameter as Parameter
    from click.exceptions import Abort as Abort
    from click.exceptions import ClickException as ClickException
    from click.exceptions import Exit as Exit
    from click.exceptions import UsageError as UsageError
    from click.globals import get_current_context as get_current_context


def is_group(command: Command) -> bool:
    """True when *command* holds subcommands."""
    return hasattr(command, "commands")


def is_argument(param: Parameter) -> bool:
    """True for a positional argument, false for an option."""
    return param.param_type_name == "argument"
