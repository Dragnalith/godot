#!/usr/bin/env python3
"""
Inspect SCons action logs (JSONL from SCONS_ACTION_LOG) and related build metadata.

If ``BUILD_GRAPH_ACTION_FILE`` is set to the path of ``actions.jsonl`` (or several paths
joined with ``os.pathsep``), subcommands may omit JSONL path arguments on the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal

# Leading Python-style call: FuncName( ... ) — name is the action.
_FUNC_CALL_RE = re.compile(r"^([A-Za-z_][\w]*)\s*\(")

# C/C++ translation-unit suffixes traced from static libs for ``cc_library``.
_CC_CXX_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})

# If set, default JSONL path(s) when none are passed on the command line (see resolve_jsonl_files).
ENV_BUILD_GRAPH_ACTION_FILE = "BUILD_GRAPH_ACTION_FILE"

Executor = Literal["function", "cmd"]


def fatal_error(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def check(condition: bool, msg: str) -> None:
    if not condition:
        fatal_error(msg)


def resolve_jsonl_files(cli_paths: list[Path]) -> list[Path]:
    """
    Use CLI paths when non-empty; otherwise ``BUILD_GRAPH_ACTION_FILE`` (one or more paths
    separated by ``os.pathsep``).
    """
    if cli_paths:
        return cli_paths
    raw = os.environ.get(ENV_BUILD_GRAPH_ACTION_FILE, "")
    if not raw.strip():
        fatal_error(
            f"pass path(s) to actions.jsonl on the command line, or set {ENV_BUILD_GRAPH_ACTION_FILE}"
        )
    parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    if not parts:
        fatal_error(
            f"{ENV_BUILD_GRAPH_ACTION_FILE} is set but empty after splitting on path separator"
        )
    return [Path(p) for p in parts]


def _first_logical_cmd_line(cmd: str) -> str | None:
    """First non-empty line after stripping leading os.chdir(...) lines."""
    text = cmd.strip()
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    while lines and lines[0].startswith("os.chdir("):
        lines.pop(0)
    return lines[0] if lines else None


def extract_action_key(cmd: str | None) -> str | None:
    """
    Derive a short action identifier from a logged cmd string (same as ``list-kind`` keys).

    - If cmd looks like ``Name(...)``, return ``Name``.
    - Otherwise treat the first non-empty line as a shell command and return its
      first whitespace-delimited token (args[0]).
    """
    if cmd is None or not isinstance(cmd, str):
        return None
    first = _first_logical_cmd_line(cmd)
    if not first:
        return None
    m = _FUNC_CALL_RE.match(first)
    if m:
        return m.group(1)
    parts = first.split(None, 1)
    return parts[0] if parts else None


def executor_from_cmd(cmd: str | None) -> Executor:
    if cmd is None or not isinstance(cmd, str):
        return "cmd"
    first = _first_logical_cmd_line(cmd)
    if not first:
        return "cmd"
    return "function" if _FUNC_CALL_RE.match(first) else "cmd"


def norm_target_path(p: str | Path) -> Path:
    """Stable Path key for indexing and lookup (path need not exist)."""
    return Path(os.path.normpath(str(p)))


def first_output_artifact(obj: dict) -> str:
    """First logged target path for this action, or empty if none."""
    raw = obj.get("targets")
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return ""


def output_extension_token(path_str: str) -> str:
    """
    Single extension bucket for a target path: final suffix only (lowercased), e.g. '.obj', '.h'.
    Paths with no suffix (e.g. directory 'bin') use '(none)'.
    """
    suf = Path(path_str).suffix
    return suf.lower() if suf else "(none)"


def collect_extensions_for_row(obj: dict) -> set[str]:
    out: set[str] = set()
    raw = obj.get("targets")
    if not isinstance(raw, list):
        return out
    for t in raw:
        out.add(output_extension_token(str(t)))
    return out


def format_extension_list(exts: set[str]) -> str:
    if not exts:
        return "-"
    return ",".join(sorted(exts))


class Action:
    """One row from actions.jsonl (parsed).

    ``command_kind`` is the same key as ``list-kind`` / ``extract_action_key(cmd)``.
    """

    __slots__ = ("command_kind", "targets", "sources", "cmd", "executor")

    def __init__(
        self,
        command_kind: str | None,
        targets: list[Path],
        sources: list[Path],
        cmd: str,
        executor: Executor,
    ) -> None:
        self.command_kind = command_kind
        self.targets = targets
        self.sources = sources
        self.cmd = cmd
        self.executor = executor


class Graph:
    """
    Build graph from actions.jsonl.

    - ``actions``: ordered list of :class:`Action` (``targets``, ``sources``, ``executor``, …).
    - ``targets_to_action_indices``: each output path -> indices into ``actions`` that list it.
    - ``command_kind_to_action_indices``: ``list-kind`` command-kind key -> indices into ``actions``.

    Use :class:`BazelGraph` for a Bazel-oriented view (``CcLibrary`` / :class:`CompileSet`).
    """

    __slots__ = (
        "actions",
        "targets_to_action_indices",
        "command_kind_to_action_indices",
    )

    def __init__(
        self,
        actions: list[Action],
        targets_to_action_indices: dict[Path, list[int]],
        command_kind_to_action_indices: dict[str, list[int]],
    ) -> None:
        self.actions = actions
        self.targets_to_action_indices = targets_to_action_indices
        self.command_kind_to_action_indices = command_kind_to_action_indices

    @classmethod
    def from_jsonl(cls, jsonl_paths: list[Path]) -> Graph:
        actions: list[Action] = []
        targets_to_action_indices: dict[Path, list[int]] = defaultdict(list)
        command_kind_to_action_indices: dict[str, list[int]] = defaultdict(list)

        for path in jsonl_paths:
            check(path.is_file(), f"not a file: {path}")
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "cmd" not in obj:
                        continue
                    cmd = obj.get("cmd")
                    cmd_str = cmd if isinstance(cmd, str) else ""
                    ex = executor_from_cmd(cmd_str if cmd_str else None)
                    ck = extract_action_key(cmd_str if cmd_str else None)
                    raw_targets = obj.get("targets")
                    tlist: list[Path] = []
                    if isinstance(raw_targets, list):
                        for t in raw_targets:
                            tlist.append(norm_target_path(str(t)))
                    raw_sources = obj.get("sources")
                    slist: list[Path] = []
                    if isinstance(raw_sources, list):
                        for s in raw_sources:
                            slist.append(norm_target_path(str(s)))
                    a = Action(
                        command_kind=ck,
                        targets=tlist,
                        sources=slist,
                        cmd=cmd_str,
                        executor=ex,
                    )
                    idx = len(actions)
                    actions.append(a)
                    for tp in tlist:
                        targets_to_action_indices[tp].append(idx)
                    if ck:
                        command_kind_to_action_indices[ck].append(idx)

        return cls(
            actions,
            dict(targets_to_action_indices),
            dict(command_kind_to_action_indices),
        )


def format_action_block(
    a: Action,
    *,
    show_sources: bool = True,
    show_cmd: bool = True,
) -> str:
    lines = []
    lines.append(
        f"kind: {a.command_kind if a.command_kind is not None else '(none)'}"
    )
    lines.append(f"executor: {a.executor}")
    if a.targets:
        lines.append("targets:")
        for t in a.targets:
            lines.append(f"  {t}")
    else:
        lines.append("targets: (none)")
    if show_sources:
        if a.sources:
            lines.append("sources:")
            for s in a.sources:
                lines.append(f"  {s}")
        else:
            lines.append("sources: (none)")
    if show_cmd:
        lines.append("cmd:")
        for ln in a.cmd.splitlines() or [""]:
            lines.append(f"  {ln}")
    return "\n".join(lines)


def cmd_target(
    graph: Graph,
    target_path: str,
    *,
    show_sources: bool,
    show_cmd: bool,
) -> None:
    key = norm_target_path(target_path)
    if key not in graph.targets_to_action_indices or not graph.targets_to_action_indices[key]:
        fatal_error(f"no action found for target: {target_path}")
    indices = graph.targets_to_action_indices[key]
    for i, idx in enumerate(indices):
        if len(indices) > 1:
            print(f"--- action index {idx} ---")
        print(
            format_action_block(
                graph.actions[idx],
                show_sources=show_sources,
                show_cmd=show_cmd,
            )
        )
        if i < len(indices) - 1:
            print()


def _collect_direct_sources(graph: Graph, action_indices: list[int]) -> list[Path]:
    """Union of sources for these actions, first-seen order preserved."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for idx in action_indices:
        for s in graph.actions[idx].sources:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
    return ordered


def _is_cc_cxx_source(p: Path) -> bool:
    return p.suffix.lower() in _CC_CXX_SOURCE_SUFFIXES


def _is_obj_artifact(p: Path) -> bool:
    return p.suffix.lower() == ".obj"


def _collect_cpp_closure_and_obj_map(
    graph: Graph, root: Path
) -> tuple[set[Path], dict[Path, Path]]:
    """
    Transitive set of ``.c`` / ``.cc`` / ``.cpp`` / ``.cxx`` sources plus
    ``source -> .obj`` for the object file built from that source (first edge seen along
    the walk from ``root``).
    """
    out: set[Path] = set()
    src_to_obj: dict[Path, Path] = {}
    stack_set: set[Path] = set()

    def walk(p: Path) -> None:
        if p in stack_set:
            return
        if _is_cc_cxx_source(p):
            out.add(p)
        indices = graph.targets_to_action_indices.get(p)
        if not indices:
            return
        stack_set.add(p)
        try:
            for s in _collect_direct_sources(graph, indices):
                if _is_cc_cxx_source(s):
                    out.add(s)
                    if _is_obj_artifact(p):
                        src_to_obj.setdefault(s, p)
                if s in graph.targets_to_action_indices:
                    walk(s)
        finally:
            stack_set.remove(p)

    walk(norm_target_path(root))
    return out, src_to_obj


def collect_cpp_closure(graph: Graph, root: Path) -> set[Path]:
    """
    All ``.c`` / ``.cc`` / ``.cpp`` / ``.cxx`` files reachable from ``root`` by following
    logged edges: direct sources, and any source that is itself a build target (e.g.
    ``.obj`` -> source).
    """
    src_paths, _ = _collect_cpp_closure_and_obj_map(graph, root)
    return src_paths


def _cmd_string_for_argv_split(cmd: str) -> str:
    """Join non-empty lines, skipping leading ``os.chdir(...)`` lines (same idea as first logical cmd)."""
    chunks: list[str] = []
    for ln in cmd.splitlines():
        s = ln.strip()
        if not s or s.startswith("os.chdir("):
            continue
        chunks.append(s)
    return " ".join(chunks)


def _split_cmd_to_args_raw(cmd: str) -> list[str]:
    """Split a logged cmd into argv tokens (order preserved)."""
    text = _cmd_string_for_argv_split(cmd).strip()
    if not text:
        return []
    posix = os.name == "posix"
    try:
        return shlex.split(text, posix=posix)
    except ValueError:
        return text.split()


def _path_key_for_arg_match(s: str) -> str:
    """Normalize a path-like argv token for comparison (case-folded, ``/`` separators)."""
    return os.path.normpath(s).replace("\\", "/").lower()


def _compile_path_match_keys(p: Path) -> set[str]:
    q = norm_target_path(p)
    keys = {_path_key_for_arg_match(str(q)), _path_key_for_arg_match(q.as_posix())}
    return {k for k in keys if k}


def _is_msvc_d1_d2_switch_token(t: str) -> bool:
    """
    True for MSVC ``/d1`` / ``/d2`` internal switches (e.g. ``/d2archSSE42``), which must
    not be treated as ``/D`` preprocessor defines.
    """
    if len(t) < 3 or t[0] != "/":
        return False
    if t[1].lower() != "d":
        return False
    if t[2] not in "12":
        return False
    # ``/d1``, ``/d2``, ``/d2archSSE42``, …
    return len(t) == 3 or t[3].isalpha() or t[3] == "_"


def _split_defines_and_includes_from_argv(
    tokens: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Partition argv into (remaining, defines, includes).

    Handles MSVC-style ``/D``, ``/I`` (and glued ``/Dname``, ``/Ipath``) and common
    ``-D`` / ``-I`` forms. ``/d1`` / ``/d2`` codegen switches stay in ``remaining``.
    Flag matching is case-insensitive for the switch letter; macro and path strings
    keep their original spelling.
    """
    remaining: list[str] = []
    defines: list[str] = []
    includes: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        low = t.lower()
        if _is_msvc_d1_d2_switch_token(t):
            remaining.append(t)
            i += 1
            continue
        # Other MSVC ``/d*`` switches that are not ``/D`` macros (e.g. ``/doc``).
        if low.startswith("/doc") or low.startswith("/diagnostics"):
            remaining.append(t)
            i += 1
            continue
        if low in ("/d", "-d"):
            if i + 1 < n:
                defines.append(tokens[i + 1])
                i += 2
            else:
                remaining.append(t)
                i += 1
            continue
        if (low.startswith("/d") or low.startswith("-d")) and len(low) > 2:
            defines.append(t[2:])
            i += 1
            continue
        if low in ("/i", "-i"):
            if i + 1 < n:
                includes.append(tokens[i + 1])
                i += 2
            else:
                remaining.append(t)
                i += 1
            continue
        if (low.startswith("/i") or low.startswith("-i")) and len(low) > 2:
            includes.append(t[2:])
            i += 1
            continue
        remaining.append(t)
        i += 1
    return remaining, defines, includes


def _filter_compile_argv_tokens(tokens: list[str], src: Path, obj: Path) -> list[str]:
    """
    Drop the primary translation-unit path (``.c`` / ``.cc`` / ``.cpp`` / ``.cxx``), the
    ``.obj`` path, and MSVC ``/Fo`` output arguments (``/Fopath`` or ``/Fo`` + next
    token) so remaining tokens are compiler flags only.
    """
    src_keys = _compile_path_match_keys(src)
    obj_keys = _compile_path_match_keys(obj)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        tk = _path_key_for_arg_match(t)
        if tk in src_keys or tk in obj_keys:
            i += 1
            continue
        low = t.lower()
        if low.startswith("/fo") and len(low) > 3:
            i += 1
            continue
        if low in ("/fo", "-fo"):
            i += 2 if i + 1 < n else 1
            continue
        out.append(t)
        i += 1
    return out


def compile_args_for_cc_library(cmd: str, src: Path, obj: Path) -> list[str]:
    """Argv tokens for ``cc_library`` compare: split, drop src/obj/``/Fo``, sort lexicographically."""
    raw = _split_cmd_to_args_raw(cmd)
    filtered = _filter_compile_argv_tokens(raw, src, obj)
    return sorted(filtered)


def compile_partition_for_cc_library(
    cmd: str, src: Path, obj: Path
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Like :func:`compile_args_for_cc_library`, but split preprocessor defines and include
    paths out for separate attributes. Grouping key is
    ``(sorted(remaining), sorted(defines), sorted(includes))``.
    """
    raw = _split_cmd_to_args_raw(cmd)
    filtered = _filter_compile_argv_tokens(raw, src, obj)
    rem, defs, incs = _split_defines_and_includes_from_argv(filtered)
    return (tuple(sorted(rem)), tuple(sorted(defs)), tuple(sorted(incs)))


def _source_dirs_overlap_equal_or_ancestor(a: Path, b: Path) -> bool:
    """
    True if ``a`` and ``b`` denote the same directory, or one is a proper ancestor of
    the other (parent/child). Paths are compared after :func:`os.path.normpath`.
    """
    an = Path(os.path.normpath(str(a)))
    bn = Path(os.path.normpath(str(b)))
    if an == bn:
        return True
    try:
        an.relative_to(bn)
        return True
    except ValueError:
        pass
    try:
        bn.relative_to(an)
        return True
    except ValueError:
        return False


def deepest_common_parent_dir(paths: Iterable[Path]) -> Path | None:
    """
    Deepest directory that is a parent of every path (lowest common ancestor of
    ``path.parent``). Empty input returns ``None``.
    """
    parents: list[tuple[str, ...]] = []
    for p in paths:
        par = p.parent
        if par == p:
            parts: tuple[str, ...] = ()
        else:
            parts = par.parts
        parents.append(parts)
    if not parents:
        return None
    common = list(parents[0])
    for parts in parents[1:]:
        i = 0
        n = min(len(common), len(parts))
        while i < n and common[i] == parts[i]:
            i += 1
        common = common[:i]
    if not common:
        return Path(".")
    return Path(*common)


def _bucket_dir_under_common_parent(source_file: Path, common_parent: Path) -> Path:
    """
    The directory at ``common_parent`` or one level below it that contains ``source_file``
    (via ``source_file.parent`` chain). Used to list disjoint-ish subfolders when several
    :class:`CompileSet` share the same deepest common parent.
    """
    d = source_file.parent
    if d == common_parent:
        return common_parent
    while d.parent != common_parent:
        d = d.parent
    return d


def _finalize_compile_set_source_dirs(compile_sets: list[CompileSet]) -> None:
    """
    Set :attr:`CompileSet.source_dirs` for each set in one :class:`CcLibrary`.

    Normally ``source_dirs`` is ``(package,)`` where ``package`` is the deepest
    directory containing every C/C++ source in the set. If two or more compile sets that have
    a real compile signature (``argv_signature`` not ``None``) share the same
    ``package``, each of those sets gets the sorted unique list of per-file bucket
    dirs under that parent (see :func:`_bucket_dir_under_common_parent`). Lists are
    computed independently per set — no cross-set intersection. Sets with no compile
    action (``argv_signature is None``) always use ``(package,)`` and do not affect
    collision detection.
    """
    pkgs = [
        cs.package
        for cs in compile_sets
        if cs.argv_signature is not None
    ]
    cnt = Counter(p for p in pkgs if p is not None)
    colliding = {p for p, n in cnt.items() if n > 1}

    for cs in compile_sets:
        P = cs.package
        if P is None:
            cs.source_dirs = ()
        elif cs.argv_signature is None:
            cs.source_dirs = (P,) if P else ()
        elif P in colliding:
            buckets = {_bucket_dir_under_common_parent(s, P) for s in cs.sources}
            cs.source_dirs = tuple(sorted(buckets, key=_cpp_sort_key))
        else:
            cs.source_dirs = (P,)


def _cpp_sort_key(p: Path) -> str:
    return str(p).replace("\\", "/").lower()


class CompileSet:
    """
    ``.c`` / ``.cc`` / ``.cpp`` / ``.cxx`` sources that share one filtered compile argv
    signature (Bazel-oriented grouping).

    ``sources`` are sorted paths. ``argv_signature`` is the remaining compile flags (no
    ``/D`` / ``/I`` tokens), sorted, or ``None`` when no compile action was found in the
    log for those sources. ``defines`` and ``includes`` hold extracted preprocessor
    defines and include directory paths (sorted).

    ``package`` is the deepest directory that contains every path in ``sources`` (same as
    :func:`deepest_common_parent_dir`); several compile sets in one library may share the
    same ``package``.

    ``source_dirs`` is filled by :func:`_finalize_compile_set_source_dirs`: the deepest
    folder spanning all sources, or when another compile set in the same library shares
    that folder, a sorted tuple of per-set subfolders under it (see module helper).
    """

    __slots__ = ("sources", "argv_signature", "defines", "includes", "package", "source_dirs")

    def __init__(
        self,
        sources: list[Path],
        argv_signature: tuple[str, ...] | None,
        defines: tuple[str, ...],
        includes: tuple[str, ...],
        *,
        package: Path | None = None,
        source_dirs: tuple[Path, ...] = (),
    ) -> None:
        self.sources = sources
        self.argv_signature = argv_signature
        self.defines = defines
        self.includes = includes
        self.package = package if package is not None else deepest_common_parent_dir(sources)
        self.source_dirs = source_dirs

    @property
    def package_path(self) -> Path | None:
        """Alias for :attr:`package` (historical name)."""
        return self.package


class CcLibrary:
    """One static ``.lib`` and its compile-set partition."""

    __slots__ = ("path", "compile_sets")

    def __init__(self, path: Path, compile_sets: list[CompileSet]) -> None:
        self.path = path
        self.compile_sets = compile_sets


def _bazel_graph_common_compile_metadata(
    cc_libraries: dict[Path, CcLibrary],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Return ``(flags, defines)`` where each tuple lists values common to every
    :class:`CompileSet` under ``cc_libraries`` (set intersection, then sorted).

    Include directories are not aggregated here — they stay on each :class:`CompileSet`
    only.

    ``flags`` mirrors ``argv_signature`` tokens. If any compile set has
    ``argv_signature is None``, ``flags`` is empty — not all sets share a known argv tail.
    """
    all_sets: list[CompileSet] = [
        cs for cc in cc_libraries.values() for cs in cc.compile_sets
    ]
    if not all_sets:
        return (), ()

    d_common: frozenset[str] | None = None
    for cs in all_sets:
        d_common = frozenset(cs.defines) if d_common is None else d_common & frozenset(cs.defines)
    defines = tuple(sorted(d_common or ()))

    if any(cs.argv_signature is None for cs in all_sets):
        flags: tuple[str, ...] = ()
    else:
        f_common: frozenset[str] | None = None
        for cs in all_sets:
            assert cs.argv_signature is not None
            sig = cs.argv_signature
            f_common = frozenset(sig) if f_common is None else f_common & frozenset(sig)
        flags = tuple(sorted(f_common or ()))

    return flags, defines


class BazelGraph:
    """
    Bazel-oriented view derived from :class:`Graph`: ``CcLibrary`` nodes keyed by ``.lib``
    path (normalized).

    ``flags`` and ``defines`` are the sorted set-intersection of the same fields across
    **every** :class:`CompileSet` in :attr:`cc_libraries` (see
    :func:`_bazel_graph_common_compile_metadata`). Include paths are only on each
    :class:`CompileSet`.
    """

    __slots__ = ("cc_libraries", "flags", "defines")

    def __init__(self, cc_libraries: dict[Path, CcLibrary]) -> None:
        self.cc_libraries = dict(cc_libraries)
        self.flags, self.defines = _bazel_graph_common_compile_metadata(self.cc_libraries)

    @classmethod
    def from_graph(cls, graph: Graph, lib_paths: list[Path]) -> BazelGraph:
        m = {lib: _cc_library_from_graph(graph, lib) for lib in lib_paths}
        return cls(m)


def verify_bazel_graph(
    bg: BazelGraph,
    *,
    verify_disjoint_source_dirs: bool = False,
) -> list[str]:
    """
    Structural checks on a built :class:`BazelGraph`. Returns human-readable problem lines
    (empty if all checks pass). The ``cc_library`` subcommand prints them to stderr only
    when the list is non-empty.

    If ``verify_disjoint_source_dirs`` is true, each :class:`CcLibrary` is also checked so
    that no two ``source_dirs`` paths (from any compile set, including two entries in the
    same set) are the same directory or in a parent/child relationship. This is stricter
    than :func:`_finalize_compile_set_source_dirs` may currently emit (it can list both a
    common parent and a subdirectory); use the flag when validating a Bazel-oriented layout.
    """
    problems: list[str] = []
    for lib_path in sorted(bg.cc_libraries.keys(), key=_cpp_sort_key):
        cc = bg.cc_libraries[lib_path]
        if not cc.compile_sets:
            problems.append(
                f"CcLibrary has no CompileSet: {lib_path.as_posix()}"
            )
            continue
        if not verify_disjoint_source_dirs:
            continue
        lip = lib_path.as_posix()
        indexed_dirs: list[tuple[int, Path]] = [
            (si, d) for si, cs in enumerate(cc.compile_sets) for d in cs.source_dirs
        ]
        for i in range(len(indexed_dirs)):
            si, pi = indexed_dirs[i]
            for j in range(i + 1, len(indexed_dirs)):
                sj, pj = indexed_dirs[j]
                if not _source_dirs_overlap_equal_or_ancestor(pi, pj):
                    continue
                a = pi.as_posix()
                b = pj.as_posix()
                if si == sj:
                    problems.append(
                        f"CcLibrary source_dirs overlap within compile_set[{si}] "
                        f"({lip}): {a!r} vs {b!r}"
                    )
                else:
                    problems.append(
                        f"CcLibrary source_dirs overlap compile_set[{si}] vs "
                        f"compile_set[{sj}] ({lip}): {a!r} vs {b!r}"
                    )
    return problems


def _cc_library_from_graph(graph: Graph, lib: Path) -> CcLibrary:
    src_paths, src_to_obj = _collect_cpp_closure_and_obj_map(graph, lib)
    by_sig: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], list[Path]
    ] = defaultdict(list)
    missing_compile: list[Path] = []

    for src in sorted(src_paths, key=_cpp_sort_key):
        obj = src_to_obj.get(src)
        if obj is None:
            missing_compile.append(src)
            continue
        idxs = graph.targets_to_action_indices.get(obj)
        if not idxs:
            missing_compile.append(src)
            continue
        cmd_s = graph.actions[idxs[0]].cmd
        rem, defs, incs = compile_partition_for_cc_library(cmd_s, src, obj)
        by_sig[(rem, defs, incs)].append(src)

    compile_set_entries: list[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], list[Path]]
    ] = []
    for sig_triple, paths in by_sig.items():
        rem, defs, incs = sig_triple
        compile_set_entries.append(
            (rem, defs, incs, sorted(paths, key=_cpp_sort_key))
        )
    compile_set_entries.sort(
        key=lambda item: (
            _cpp_sort_key(item[3][0]) if item[3] else "",
            item[0],
            item[1],
            item[2],
        )
    )

    compile_sets = [
        CompileSet(
            sources=paths,
            argv_signature=rem,
            defines=defs,
            includes=incs,
        )
        for rem, defs, incs, paths in compile_set_entries
    ]
    if missing_compile:
        compile_sets.append(
            CompileSet(
                sources=sorted(missing_compile, key=_cpp_sort_key),
                argv_signature=None,
                defines=(),
                includes=(),
            )
        )

    _finalize_compile_set_source_dirs(compile_sets)
    return CcLibrary(path=lib, compile_sets=compile_sets)


def _resolve_cc_library_lib_paths(graph: Graph, lib_path: str | None) -> list[Path]:
    """``*.lib`` targets for ``cc_library`` / ``BUILD`` (all libs if ``lib_path`` is None)."""
    if lib_path is not None:
        key = norm_target_path(lib_path)
        if key.suffix.lower() != ".lib":
            fatal_error(f"not a *.lib path: {lib_path!r}")
        if key not in graph.targets_to_action_indices or not graph.targets_to_action_indices[key]:
            fatal_error(f"no action found for library target: {lib_path}")
        return [key]
    return sorted(
        (p for p in graph.targets_to_action_indices if p.suffix.lower() == ".lib"),
        key=lambda x: str(x).replace("\\", "/").lower(),
    )


def _print_bazel_graph_header(bg: BazelGraph) -> None:
    """Shared ``bazel_graph:`` / common flags+defines block ending with ``---``."""
    print("bazel_graph:")
    print("  flags:")
    if not bg.flags:
        print("    (none)")
    else:
        for a in bg.flags:
            print(f"    {a}")
    print("  defines:")
    if not bg.defines:
        print("    (none)")
    else:
        for d in bg.defines:
            print(f"    {d}")
    print("---")


def _emit_cc_library_compile_set(
    cs: CompileSet,
    *,
    show_sources: bool,
    show_cmd: bool,
    only_includes: bool,
    common_flags: frozenset[str],
    common_defines: frozenset[str],
    from_lib: Path | None = None,
    show_package_field: bool = True,
) -> None:
    """Print one ``compile_set`` block (indented under ``lib`` or under ``package``)."""
    b1 = "  "
    b2 = "    "
    b3 = "      "
    paths_in_set = cs.sources
    argv_sig = cs.argv_signature
    print(f"{b1}compile_set:")
    if from_lib is not None:
        print(f"{b2}lib: {from_lib}")
    if only_includes:
        print(f"{b2}includes:")
        if not cs.includes:
            print(f"{b3}(none)")
        else:
            for inc in cs.includes:
                print(f"{b3}{inc}")
        return
    if show_package_field:
        print(f"{b2}package:")
        if cs.package is None:
            print(f"{b3}(none)")
        else:
            print(f"{b3}{cs.package.as_posix()}")
    print(f"{b2}src_count: {len(paths_in_set)}")
    print(f"{b2}source_dirs:")
    if not cs.source_dirs:
        print(f"{b3}(none)")
    else:
        for d in cs.source_dirs:
            print(f"{b3}{d.as_posix()}")
    if show_sources:
        print(f"{b2}srcs:")
        for c in paths_in_set:
            print(f"{b3}{c.as_posix()}")
    print(f"{b2}defines:")
    defines_rest = tuple(d for d in cs.defines if d not in common_defines)
    if not defines_rest:
        print(f"{b3}(none)")
    else:
        for d in defines_rest:
            print(f"{b3}{d}")
    print(f"{b2}includes:")
    if not cs.includes:
        print(f"{b3}(none)")
    else:
        for inc in cs.includes:
            print(f"{b3}{inc}")
    if show_cmd:
        print(f"{b2}flags:")
        if argv_sig is None or not argv_sig:
            print(f"{b3}(none)")
        else:
            flags_rest = tuple(a for a in argv_sig if a not in common_flags)
            if not flags_rest:
                print(f"{b3}(none)")
            else:
                for a in flags_rest:
                    print(f"{b3}{a}")


def print_bazel_graph_cc_libraries(
    bg: BazelGraph,
    *,
    show_sources: bool,
    show_cmd: bool,
    only_includes: bool = False,
) -> None:
    """Text report for ``cc_library`` CLI, driven only by :class:`BazelGraph`."""
    _print_bazel_graph_header(bg)
    common_flags = frozenset(bg.flags)
    common_defines = frozenset(bg.defines)
    for lib_path in sorted(bg.cc_libraries.keys(), key=_cpp_sort_key):
        cc = bg.cc_libraries[lib_path]
        print("---")
        print(f"lib: {cc.path}")
        if not cc.compile_sets:
            print("  compile_set: (none)")
            continue
        for cs in cc.compile_sets:
            _emit_cc_library_compile_set(
                cs,
                show_sources=show_sources,
                show_cmd=show_cmd,
                only_includes=only_includes,
                common_flags=common_flags,
                common_defines=common_defines,
                from_lib=None,
                show_package_field=True,
            )


def _package_path_sort_key(p: Path | None) -> str:
    """Sort key so ``None`` (no package) sorts last."""
    if p is None:
        return "\xff"
    return _cpp_sort_key(p)


def print_bazel_build_files(
    bg: BazelGraph,
    *,
    show_sources: bool,
    show_cmd: bool,
    only_includes: bool = False,
) -> None:
    """
    Text report for ``BUILD`` CLI: one logical ``BUILD`` file per distinct
    :attr:`CompileSet.package`, listing every compile set that belongs to that package.
    """
    _print_bazel_graph_header(bg)
    common_flags = frozenset(bg.flags)
    common_defines = frozenset(bg.defines)

    by_package: dict[Path | None, list[tuple[Path, CompileSet]]] = defaultdict(list)
    for lib_path in sorted(bg.cc_libraries.keys(), key=_cpp_sort_key):
        cc = bg.cc_libraries[lib_path]
        for cs in cc.compile_sets:
            by_package[cs.package].append((cc.path, cs))

    for pkg in sorted(by_package.keys(), key=_package_path_sort_key):
        entries = by_package[pkg]

        print("---")
        if pkg is None:
            print("build_file: (no_package)/BUILD")
            print("package:")
            print("  (none)")
        else:
            print(f"build_file: {pkg.as_posix()}/BUILD")
            print(f"package: {pkg.as_posix()}")
        if not entries:
            print("  compile_set: (none)")
            continue
        for lib_p, cs in entries:
            _emit_cc_library_compile_set(
                cs,
                show_sources=show_sources,
                show_cmd=show_cmd,
                only_includes=only_includes,
                common_flags=common_flags,
                common_defines=common_defines,
                from_lib=lib_p,
                show_package_field=False,
            )


def cmd_deps(
    graph: Graph,
    target_path: str,
    *,
    show_sources: bool,
) -> None:
    """
    Print the target, then each direct source indented, recursively for sources that are
    themselves build outputs (appear as targets in the graph).

    """
    key = norm_target_path(target_path)
    if key not in graph.targets_to_action_indices or not graph.targets_to_action_indices[key]:
        fatal_error(f"no action found for target: {target_path}")

    def walk(p: Path, depth: int, stack: set[Path]) -> None:
        prefix = "  " * depth
        if p in stack:
            print(f"{prefix}{p}  (cycle)")
            return
        print(f"{prefix}{p}")
        indices = graph.targets_to_action_indices.get(p)
        if not indices:
            return
        if not show_sources:
            return
        stack.add(p)
        try:
            for s in _collect_direct_sources(graph, indices):
                walk(s, depth + 1, stack)
        finally:
            stack.remove(p)

    walk(key, 0, set())


def cmd_kind(
    graph: Graph,
    command_kind: str,
    n: int,
    *,
    show_sources: bool,
    show_cmd: bool,
) -> None:
    if (
        command_kind not in graph.command_kind_to_action_indices
        or not graph.command_kind_to_action_indices[command_kind]
    ):
        fatal_error(f"no action found for command kind: {command_kind!r}")
    indices = graph.command_kind_to_action_indices[command_kind]
    pick = indices[:n]
    for i, idx in enumerate(pick):
        if len(pick) > 1:
            print(f"--- action index {idx} ---")
        print(
            format_action_block(
                graph.actions[idx],
                show_sources=show_sources,
                show_cmd=show_cmd,
            )
        )
        if i < len(pick) - 1:
            print()


def cmd_cc_library(
    graph: Graph,
    *,
    lib_path: str | None,
    show_sources: bool,
    show_cmd: bool,
    only_includes: bool = False,
    verify_disjoint_source_dirs: bool = False,
) -> None:
    """
    Resolve ``*.lib`` targets, build a :class:`BazelGraph`, and print via
    :func:`print_bazel_graph_cc_libraries`.
    """
    libs = _resolve_cc_library_lib_paths(graph, lib_path)
    bg = BazelGraph.from_graph(graph, lib_paths=libs)
    vproblems = verify_bazel_graph(
        bg, verify_disjoint_source_dirs=verify_disjoint_source_dirs
    )
    if vproblems:
        print("bazel_graph_verification:", file=sys.stderr)
        for line in vproblems:
            print(f"  FAIL  {line}", file=sys.stderr)
    print_bazel_graph_cc_libraries(
        bg,
        show_sources=show_sources,
        show_cmd=show_cmd,
        only_includes=only_includes,
    )


def cmd_build(
    graph: Graph,
    *,
    lib_path: str | None,
    show_sources: bool,
    show_cmd: bool,
    only_includes: bool = False,
    verify_disjoint_source_dirs: bool = False,
) -> None:
    """
    Resolve ``*.lib`` targets, build a :class:`BazelGraph`, and print logical ``BUILD``
    file layout via :func:`print_bazel_build_files`.
    """
    libs = _resolve_cc_library_lib_paths(graph, lib_path)
    bg = BazelGraph.from_graph(graph, lib_paths=libs)
    vproblems = verify_bazel_graph(
        bg, verify_disjoint_source_dirs=verify_disjoint_source_dirs
    )
    if vproblems:
        print("bazel_graph_verification:", file=sys.stderr)
        for line in vproblems:
            print(f"  FAIL  {line}", file=sys.stderr)
    print_bazel_build_files(
        bg,
        show_sources=show_sources,
        show_cmd=show_cmd,
        only_includes=only_includes,
    )


def cmd_list_kind(
    jsonl_paths: list[Path],
    *,
    show_count: bool,
    executor_filter: Executor | None,
    show_targets: bool,
    show_extensions: bool,
) -> None:
    tally: Counter[str] = Counter()
    example_out: dict[str, str] = {}
    executor_by_kind: dict[str, Executor] = {}
    exts_by_kind: dict[str, set[str]] = {}
    for path in jsonl_paths:
        check(path.is_file(), f"not a file: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "cmd" not in obj:
                    continue
                key = extract_action_key(obj.get("cmd"))
                if not key:
                    continue
                tally[key] += 1
                if key not in example_out:
                    ex = first_output_artifact(obj)
                    example_out[key] = ex if ex else "-"
                if key not in executor_by_kind:
                    cmd_s = obj.get("cmd")
                    executor_by_kind[key] = executor_from_cmd(
                        cmd_s if isinstance(cmd_s, str) else None
                    )
                row_exts = collect_extensions_for_row(obj)
                if key not in exts_by_kind:
                    exts_by_kind[key] = set()
                exts_by_kind[key].update(row_exts)

    rows = tally.most_common()
    if executor_filter is not None:
        rows = [
            (name, num)
            for name, num in rows
            if executor_by_kind.get(name, "cmd") == executor_filter
        ]

    for name, num in rows:
        parts: list[str] = []
        if show_count:
            parts.append(str(num))
        parts.append(name)
        if show_targets:
            parts.append(example_out.get(name, "-"))
        if show_extensions:
            parts.append(format_extension_list(exts_by_kind.get(name, set())))
        print("\t".join(parts))


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="more diagnostic output (reserved for future subcommands)",
    )
    print_action = argparse.ArgumentParser(add_help=False)
    print_action.add_argument(
        "--no-cmd",
        action="store_true",
        help=(
            "omit cmd from target/kind; for cc_library omit flags inside each compile_set; ignored by deps"
        ),
    )
    print_action.add_argument(
        "--no-sources",
        action="store_true",
        help=(
            "omit sources from target/kind; for deps, only the root target line; "
            "for cc_library, omit srcs inside each compile_set"
        ),
    )
    parser = argparse.ArgumentParser(
        description="Analyze Godot/SCons build action logs and related artifacts.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _list_kind_epilog = """
Kinds (how cmd is turned into one output key):

  [function]  First line matches Name( ... )  -> key is Name.
  [cli]       Otherwise first line is a shell command -> key is argv[0] (first token).
  [chdir]     Leading os.chdir(...) lines are skipped; then [function] or [cli] rules apply.

Default output: one column, kind names only. Rows are always ordered by occurrence count
descending (most common first), whether or not --count is printed.

Optional columns (tab-separated, in order): --count, kind (always), --targets, --extensions.
  --count       print occurrence count column.
  --targets     one example output path per kind (first targets[0] seen for that kind).
  --extensions  comma-sorted distinct final suffixes for that kind ('(none)' if no dot suffix).

Filter (omit to list all kinds):
  --executor cmd       only kinds whose action is executed as a shell/tool (argv0), not ``Name(...)``.
  --executor function only kinds whose action is a Python function call (first line ``Name(...)``).

""".replace("<TAB>", "\t")

    p_list_kind = sub.add_parser(
        "list-kind",
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="list distinct command kinds from cmd strings (optional columns)",
        description=(
            "Read actions.jsonl; each row with cmd yields a kind key (function name or CLI argv0). "
            "By default print one column: kind, ordered by occurrence count (most common first). "
            "Optional --executor cmd|function filters by Action.executor. "
            "Add --count, --targets, and/or --extensions for extra tab-separated columns."
        ),
        epilog=_list_kind_epilog,
    )
    p_list_kind.add_argument(
        "jsonl_files",
        nargs="*",
        type=Path,
        help=(
            "path(s) to actions.jsonl (optional if "
            f"{ENV_BUILD_GRAPH_ACTION_FILE} is set; multiple paths in the env separated by "
            f"{os.pathsep!r})"
        ),
    )
    p_list_kind.add_argument(
        "--count",
        action="store_true",
        help="include occurrence count column (rows are always sorted by count descending)",
    )
    p_list_kind.add_argument(
        "--executor",
        choices=["cmd", "function"],
        default=None,
        metavar="KIND",
        help='only list kinds with this executor: "cmd" or "function" (omit for all)',
    )
    p_list_kind.add_argument(
        "--targets",
        action="store_true",
        help="include example target column (first targets[0] per kind)",
    )
    p_list_kind.add_argument(
        "--extensions",
        action="store_true",
        help="include distinct output-suffix column per kind",
    )

    p_target = sub.add_parser(
        "target",
        parents=[common, print_action],
        help="print action(s) that list a given output target path",
    )
    p_target.add_argument(
        "jsonl_files",
        nargs="*",
        type=Path,
        help=(
            "path(s) to actions.jsonl (optional if "
            f"{ENV_BUILD_GRAPH_ACTION_FILE} is set)"
        ),
    )
    p_target.add_argument(
        "target_path",
        type=str,
        help="build output path as in the log (e.g. bin\\\\obj\\\\foo.obj)",
    )

    p_deps = sub.add_parser(
        "deps",
        parents=[common, print_action],
        help="print transitive source dependency tree for a target (indented)",
    )
    p_deps.add_argument(
        "jsonl_files",
        nargs="*",
        type=Path,
        help=(
            "path(s) to actions.jsonl (optional if "
            f"{ENV_BUILD_GRAPH_ACTION_FILE} is set)"
        ),
    )
    p_deps.add_argument(
        "target_path",
        type=str,
        help="build output path as in the log (e.g. bin\\\\obj\\\\foo.obj)",
    )

    p_cc_library = sub.add_parser(
        "cc_library",
        parents=[common, print_action],
        help=(
            "per *.lib: partition transitive C/C++ sources (.c, .cc, .cpp, .cxx) into "
            "compile_set groups (same filtered argv); optional trailing path selects one .lib"
        ),
    )
    p_cc_library.add_argument(
        "paths",
        nargs="*",
        type=str,
        metavar="PATH",
        help=(
            "optional JSONL path(s); if the last path ends with .lib it names a single library "
            f"(other paths are JSONL). JSONL optional if {ENV_BUILD_GRAPH_ACTION_FILE} is set."
        ),
    )
    p_cc_library.add_argument(
        "--only-includes",
        action="store_true",
        help=(
            "for each compile_set, print only the includes list (omit src_count, source_dirs, "
            "srcs, defines, flags); overrides --no-sources / --no-cmd for compile_set output"
        ),
    )
    p_cc_library.add_argument(
        "--verify-source-dirs-disjoint",
        action="store_true",
        help=(
            "after building the graph, fail verification if any two source_dirs paths for "
            "the same .lib are equal or in a parent/child directory relation (any compile_set)"
        ),
    )

    p_build = sub.add_parser(
        "BUILD",
        parents=[common, print_action],
        help=(
            "group compile sets by package path; print one logical BUILD file per package "
            f"(same JSONL / optional trailing .lib rules as cc_library; {ENV_BUILD_GRAPH_ACTION_FILE} applies)"
        ),
    )
    p_build.add_argument(
        "paths",
        nargs="*",
        type=str,
        metavar="PATH",
        help=(
            "optional JSONL path(s); if the last path ends with .lib it names a single library "
            f"(other paths are JSONL). JSONL optional if {ENV_BUILD_GRAPH_ACTION_FILE} is set."
        ),
    )
    p_build.add_argument(
        "--only-includes",
        action="store_true",
        help="same as cc_library --only-includes",
    )
    p_build.add_argument(
        "--verify-source-dirs-disjoint",
        action="store_true",
        help="same as cc_library --verify-source-dirs-disjoint",
    )

    p_kind = sub.add_parser(
        "kind",
        parents=[common, print_action],
        help="print up to N actions for a list-kind command-kind key (same key rules as list-kind)",
    )
    p_kind.add_argument(
        "jsonl_files",
        nargs="*",
        type=Path,
        help=(
            "path(s) to actions.jsonl (optional if "
            f"{ENV_BUILD_GRAPH_ACTION_FILE} is set)"
        ),
    )
    p_kind.add_argument(
        "command_kind",
        type=str,
        help="kind string from list-kind (e.g. cl, build_rd_headers, run)",
    )
    p_kind.add_argument(
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="number of actions to print (first N in log order for that kind; default: 1)",
    )

    args = parser.parse_args()
    if args.command == "cc_library":
        raw = list(args.paths)
        if raw and raw[-1].lower().endswith(".lib"):
            lib_path_arg: str | None = raw[-1]
            jsonl_for_cc = resolve_jsonl_files([Path(p) for p in raw[:-1]])
        else:
            lib_path_arg = None
            jsonl_for_cc = resolve_jsonl_files([Path(p) for p in raw])
        g = Graph.from_jsonl(jsonl_for_cc)
        cmd_cc_library(
            g,
            lib_path=lib_path_arg,
            show_sources=not args.no_sources,
            show_cmd=not args.no_cmd,
            only_includes=args.only_includes,
            verify_disjoint_source_dirs=args.verify_source_dirs_disjoint,
        )
    elif args.command == "BUILD":
        raw = list(args.paths)
        if raw and raw[-1].lower().endswith(".lib"):
            lib_path_build: str | None = raw[-1]
            jsonl_for_build = resolve_jsonl_files([Path(p) for p in raw[:-1]])
        else:
            lib_path_build = None
            jsonl_for_build = resolve_jsonl_files([Path(p) for p in raw])
        g = Graph.from_jsonl(jsonl_for_build)
        cmd_build(
            g,
            lib_path=lib_path_build,
            show_sources=not args.no_sources,
            show_cmd=not args.no_cmd,
            only_includes=args.only_includes,
            verify_disjoint_source_dirs=args.verify_source_dirs_disjoint,
        )
    else:
        jsonl_files = resolve_jsonl_files(args.jsonl_files)
        if args.command == "list-kind":
            cmd_list_kind(
                jsonl_files,
                show_count=args.count,
                executor_filter=args.executor,
                show_targets=args.targets,
                show_extensions=args.extensions,
            )
        elif args.command == "target":
            g = Graph.from_jsonl(jsonl_files)
            cmd_target(
                g,
                args.target_path,
                show_sources=not args.no_sources,
                show_cmd=not args.no_cmd,
            )
        elif args.command == "deps":
            g = Graph.from_jsonl(jsonl_files)
            cmd_deps(g, args.target_path, show_sources=not args.no_sources)
        elif args.command == "kind":
            g = Graph.from_jsonl(jsonl_files)
            check(args.n >= 1, "-n must be >= 1")
            cmd_kind(
                g,
                args.command_kind,
                args.n,
                show_sources=not args.no_sources,
                show_cmd=not args.no_cmd,
            )
        else:
            fatal_error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
