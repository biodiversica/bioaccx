"""Translation of the pre-subcommand flag interface into subcommand argv.

Before subcommands, every mode was a top-level flag on one command
(``bioaccx cfg.yaml --dataset --no-split``).  Scripts and documentation in the
wild still use that form, so it keeps working: :func:`is_legacy` recognises it
and :func:`translate` rewrites it into the equivalent subcommand invocation,
which :mod:`bioaccx.cli` then dispatches normally.

Everything here exists only to serve those older invocations — including the
cross-flag validation that subcommands make unnecessary, since a subcommand
simply cannot be handed a flag belonging to another mode.  When the legacy
form is eventually dropped, this whole module goes with it and nothing in
``cli.py`` needs untangling.
"""
from __future__ import annotations

import argparse

#: Subcommand names, so a first argument naming one is never treated as legacy.
SUBCOMMANDS = frozenset({
    "train", "validate", "dataset", "embeddings",
    "merge", "extract-head", "convert-head", "registry", "gui",
})

#: Flags that only ever existed on the old interface. Their presence marks an
#: invocation as legacy even when it starts with an option rather than a path.
_LEGACY_FLAGS = frozenset({
    "--validate", "--dataset", "--embeddings", "--registry",
    "--convert-head", "--convert_head",
    "--merge", "--extract-head", "--extract_head",
    "--no-split", "--backbone", "--embed-dim",
})


def is_legacy(argv: list[str]) -> bool:
    """True when *argv* uses the old flag interface rather than a subcommand.

    A first argument that is not an option decides on its own: a subcommand
    name is new-style, anything else is the old bare-config form.  Only when
    the first argument *is* an option do the legacy flags get a say, so that
    ``merge HEAD --backbone 0xbb00`` is correctly read as new-style even though
    ``--backbone`` also appears in old invocations.

    >>> is_legacy(["cfg.yaml"])
    True
    >>> is_legacy(["train", "cfg.yaml"])
    False
    >>> is_legacy(["merge", "head.onnx", "--backbone", "0xbb00"])
    False
    >>> is_legacy(["--registry"])
    True
    >>> is_legacy(["--help"])
    False
    >>> is_legacy([])
    False
    """
    if not argv:
        return False
    first = argv[0]
    if not first.startswith("-"):
        return first not in SUBCOMMANDS
    return any(arg in _LEGACY_FLAGS for arg in argv)


def _parser() -> argparse.ArgumentParser:
    """The old parser, kept intact so legacy invocations fail exactly as before.

    It never prints help — ``--help`` routes to the subcommand interface — so
    it carries no description or epilog, only the arguments needed to read an
    old command line and the errors that rejected bad combinations.
    """
    parser = argparse.ArgumentParser(prog="bioaccx", add_help=False)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--registry", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--merge", nargs="?", const=True, metavar="HEAD")
    parser.add_argument("--extract-head", "--extract_head", dest="extract_head",
                        nargs="?", const=True, metavar="MODEL")
    parser.add_argument("--convert-head", "--convert_head", dest="convert_head",
                        metavar="HEAD")
    parser.add_argument("-o", "--out", metavar="PATH")
    parser.add_argument("--backbone", metavar="ID|PATH")
    parser.add_argument("--embed-dim", type=int, metavar="N")
    parser.add_argument("--labels", metavar="PATH")
    parser.add_argument("--format", choices=("onnx", "tflite", "both"), default="both")
    return parser


def translate(argv: list[str]) -> list[str]:
    """Rewrite a legacy command line as its subcommand equivalent.

    Rejects the flag combinations the old interface never allowed, with the
    same messages and exit code 2, then returns argv in subcommand form.

    >>> translate(["cfg.yaml"])
    ['train', 'cfg.yaml']
    >>> translate(["cfg.yaml", "--dataset", "--no-split"])
    ['dataset', 'cfg.yaml', '--no-split']
    >>> translate(["--registry"])
    ['registry']
    """
    parser = _parser()
    args = parser.parse_args(argv)

    if args.no_split and not args.dataset:
        parser.error("--no-split is only valid together with --dataset")

    # `--merge HEAD` (a string) takes its inputs from the command line;
    # a bare `--merge` (True) reads them from the config file instead.
    cli_merge = isinstance(args.merge, str)
    cli_extract = isinstance(args.extract_head, str)

    if args.out and not (args.convert_head or cli_merge or cli_extract):
        parser.error(
            "-o/--out is only valid with --convert-head, or with --merge / --extract-head "
            "when their inputs are given on the command line"
        )
    if (args.backbone or args.embed_dim or args.labels) and not (cli_merge or cli_extract):
        parser.error(
            "--backbone / --embed-dim / --labels are only valid with --merge <HEAD> or "
            "--extract-head <MODEL>; with a config file these come from the config"
        )
    if args.config and (cli_merge or cli_extract):
        parser.error(
            "give either a config file or the command-line inputs, not both — "
            "the config already carries the head/model path and the output location"
        )

    out = ["-o", args.out] if args.out else []

    if args.convert_head:
        return ["convert-head", args.convert_head, *out]
    if cli_merge:
        backbone = ["--backbone", args.backbone] if args.backbone else []
        embed_dim = ["--embed-dim", str(args.embed_dim)] if args.embed_dim else []
        return ["merge", args.merge, *backbone, *embed_dim, *out]
    if cli_extract:
        backbone = ["--backbone", args.backbone] if args.backbone else []
        embed_dim = ["--embed-dim", str(args.embed_dim)] if args.embed_dim else []
        labels = ["--labels", args.labels] if args.labels else []
        return ["extract-head", args.extract_head, *backbone, *embed_dim, *labels,
                "--format", args.format, *out]

    if args.registry:
        return ["registry"]

    if args.config is None:
        parser.error(
            "a config file is required unless --registry / --convert-head is used, or "
            "--merge / --extract-head are given their inputs directly"
        )

    if args.validate:
        return ["validate", args.config]
    if args.merge:
        return ["merge", args.config]
    if args.extract_head:
        return ["extract-head", args.config]
    if args.dataset:
        return ["dataset", args.config, *(["--no-split"] if args.no_split else [])]
    if args.embeddings:
        return ["embeddings", args.config]
    return ["train", args.config]
