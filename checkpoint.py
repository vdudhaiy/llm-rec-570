"""
Resume support for the expensive pipeline steps.

Steps 1 and 2 make one network call per movie: roughly 30 minutes against TMDB and
over ten hours against the LLM endpoint. Neither should ever redo work it has
already paid for, and neither should lose work if the process dies partway.

Two on-disk formats are involved:

* **Delimited** (`basic_descriptions.txt`, `recommendation_driven_descriptions.txt`)
  - entries separated by ``\\n::\\n``, positional: entry *i* belongs to movie *i*.
* **DAT rows** (`movies_desc.dat`, `movies_enhanced_desc_combined.dat`)
  - one line per movie, ``movieId::title::year::genres::description``, keyed by id.

Both readers tolerate a file that was cut off mid-write, and both writers are
atomic (write to a temp file, then replace) so an interrupted checkpoint cannot
corrupt the work already banked.
"""

import logging
import os

logger = logging.getLogger(__name__)

DELIMITER = "\n::\n"

# Markers written in place of a description when a call was made and came back
# unusable. These are distinct from *blank*, which means no call was ever made.
FAILURE_MARKERS = {"[ERROR]", "NOT_FOUND", "NOT FOUND", "ERROR FETCHING"}

# Anything that is not a real description.
PLACEHOLDERS = FAILURE_MARKERS | {""}

# How many completed items to accumulate before flushing to disk. Each item costs
# seconds of network time, so flushing often is essentially free insurance.
CHECKPOINT_EVERY = 10


def is_blank(text):
    """
    True if `text` holds nothing at all.

    For the delimited format this means a checkpoint reserved the slot but the
    call had not happened yet, so the entry has never actually been attempted.

    Args:
        text: Description string to inspect (may be None)

    Returns:
        bool: True when the entry is empty
    """
    return text is None or not str(text).strip()


def is_failure(text):
    """
    True if `text` is an explicit failure marker written by a completed attempt.

    Args:
        text: Description string to inspect (may be None)

    Returns:
        bool: True when a call was made and returned nothing usable
    """
    return text is not None and str(text).strip() in FAILURE_MARKERS


def is_placeholder(text):
    """
    True if `text` is not a real description, whether blank or an error marker.

    Args:
        text: Description string to inspect (may be None)

    Returns:
        bool: True when the entry is unusable as a description
    """
    return is_blank(text) or is_failure(text)


def _atomic_write(path, payload):
    """
    Write `payload` to `path` via a temp file so a crash cannot truncate the target.

    Args:
        path: Destination file path
        payload: Full file contents to write
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Delimited entry files
# ---------------------------------------------------------------------------

def load_entries(path):
    """
    Read a ``\\n::\\n``-delimited description file.

    A complete file ends with the delimiter, so the split always leaves a trailing
    fragment: either an empty string (clean) or a half-written entry (crash). Both
    are discarded, which is what makes an interrupted run safe to resume.

    Args:
        path: Path to the descriptions file

    Returns:
        list[str]: One entry per movie, in file order. Empty list if absent.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        return []

    parts = raw.split(DELIMITER)
    entries = [p.strip() for p in parts[:-1]]

    if not raw.endswith(DELIMITER):
        logger.warning(
            f"{path} ends mid-entry (previous run was interrupted); "
            f"discarding the incomplete final entry"
        )

    return entries


def save_entries(path, entries):
    """
    Write entries back in ``\\n::\\n``-delimited form.

    Args:
        path: Destination file path
        entries: List of description strings
    """
    _atomic_write(path, "".join(f"{e}{DELIMITER}" for e in entries))


# ---------------------------------------------------------------------------
# DAT row files
# ---------------------------------------------------------------------------

def load_dat_rows(path):
    """
    Read a ``movieId::title::year::genres::description`` file into a dict.

    Args:
        path: Path to the .dat file

    Returns:
        dict: movieId (str) -> description (str). Empty dict if absent.
    """
    if not os.path.exists(path):
        return {}

    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Cap the split at 5: a synopsis may legitimately contain "::".
            parts = line.split("::", 4)
            if len(parts) < 5:
                logger.warning(f"{path}:{line_no} has {len(parts)} fields, expected 5; skipping")
                continue
            rows[parts[0].strip()] = parts[4]

    return rows


def save_dat_rows(path, movies, descriptions):
    """
    Write movies out in DAT form, one line each.

    Newlines inside a description are collapsed to spaces: the format is
    line-oriented, so an embedded newline would silently split one movie into two
    unparseable rows.

    Args:
        path: Destination file path
        movies: Iterable of dicts with movieId/title/year/genres keys
        descriptions: dict of movieId (str) -> description
    """
    lines = []
    for mov in movies:
        mid = str(mov["movieId"])
        desc = str(descriptions.get(mid, "")).replace("\r", " ").replace("\n", " ")
        lines.append(f"{mid}::{mov['title']}::{mov['year']}::{mov['genres']}::{desc}\n")
    _atomic_write(path, "".join(lines))


# ---------------------------------------------------------------------------
# Work planning
# ---------------------------------------------------------------------------

def plan_work(existing, total, force=False, retry_failed=True, label="items",
              blank_is_pending=True):
    """
    Decide which positional indices still need to be generated, and log the plan.

    Three states are distinguished, because conflating them either wastes API
    calls or silently declares an interrupted run finished:

    * **absent** (past the end of the list, or None) - never attempted, always redo
    * **blank** - meaning depends on the format, see `blank_is_pending`
    * **failure marker** - attempted and failed, redo only when `retry_failed`

    Args:
        existing: Entries already on disk; None marks an absent entry
        total: How many entries the finished file should have
        force: Regenerate everything, ignoring what is already there
        retry_failed: Also redo entries that carry a failure marker
        label: Noun used in the log line
        blank_is_pending: True for the delimited format, where a checkpoint
            reserves every slot up front so a blank means "not generated yet".
            False for DAT rows, where a row only exists once the movie has been
            attempted, so a blank means "attempted, came back empty".

    Returns:
        tuple: (list of indices to generate, working list of length `total`)
    """
    if force:
        logger.info(f"--force: regenerating all {total} {label} from scratch")
        return list(range(total)), [""] * total

    working = [("" if e is None else e) for e in existing[:total]]
    working += [""] * max(0, total - len(working))

    todo = []
    never_run = 0
    retried = 0

    for i in range(total):
        entry = existing[i] if i < len(existing) else None

        if entry is None:
            todo.append(i)
            never_run += 1
        elif is_blank(entry):
            if blank_is_pending:
                todo.append(i)
                never_run += 1
            elif retry_failed:
                todo.append(i)
                retried += 1
        elif is_failure(entry):
            if retry_failed:
                todo.append(i)
                retried += 1

    done = total - len(todo)
    if not todo:
        logger.info(f"All {total} {label} are already done")
    elif done:
        msg = f"Resuming: {done}/{total} {label} already done, {len(todo)} to go"
        if retried:
            msg += f" ({never_run} never generated, {retried} previously-failed being retried)"
        logger.info(msg)
    else:
        logger.info(f"Generating all {total} {label}")

    return todo, working
