"""
End-to-end skip/resume behaviour, with every network call stubbed out.

Nothing here contacts TMDB or the LLM endpoint: _ask_llm and the TMDB helpers are
replaced with fakes that record what they were asked for. What we assert is which
movies the pipeline *decides* to work on.
"""
import logging, os, shutil, sys, tempfile
logging.basicConfig(level=logging.INFO, format="      %(message)s")

import main, prompting, fetching_descriptions

results = []
def check(name, cond):
    results.append(cond)
    print(f"  [{'PASS' if cond else '**FAIL**'}] {name}")

work = tempfile.mkdtemp()
shutil.copy("movies_desc.dat", work)
shutil.copy("basic_descriptions.txt", work)
os.makedirs(os.path.join(work, "movielens-1m"))
shutil.copy("movielens-1m/movies.dat", os.path.join(work, "movielens-1m"))
os.chdir(work)

TOTAL = sum(1 for l in open("movies_desc.dat", encoding="utf-8") if l.strip())

# How many rows currently hold no usable description. Derived, not hardcoded:
# the pipeline fills these in over time, so a fixed number goes stale.
from checkpoint import load_dat_rows, is_placeholder
PLACEHOLDER_ROWS = sum(1 for v in load_dat_rows("movies_desc.dat").values() if is_placeholder(v))

# Make the basic-descriptions fixture partial regardless of the live file's state,
# so the resume assertions below are stable as the real pipeline progresses.
_all = [p for p in open("basic_descriptions.txt", encoding="utf-8").read().split("\n::\n") if p.strip()]
PREEXISTING = min(440, len(_all))
open("basic_descriptions.txt", "w", encoding="utf-8").write(
    "".join(f"{e}\n::\n" for e in _all[:PREEXISTING]))

# --- stub every outbound call -------------------------------------------------
asked = []
prompting._ask_llm = lambda prompt: (asked.append(prompt), "STUB SUMMARY")[1]
prompting.RATE_LIMIT_DELAY = 0

fetched = []
fetching_descriptions.get_tmdb_id = lambda title: (fetched.append(title), 999)[1]
fetching_descriptions.fetch_description = lambda mid: "STUB DESCRIPTION"
fetching_descriptions.TMDB_API_KEY = "stub"
fetching_descriptions._require_api_key = lambda: None

movies = prompting.read_movies_desc()

print(f"\nStarting state: movies_desc.dat = {TOTAL} movies, "
      f"basic_descriptions.txt = {len(open('basic_descriptions.txt', encoding='utf-8').read().split(chr(10)+'::'+chr(10)))-1} entries")

print("\n1. Pipeline sees the partial basic file as INCOMPLETE (the old code called it done)")
pipe = main.Pipeline(model_type="B", force=False)
check("_entries_complete(basic) is False", pipe._entries_complete("basic_descriptions.txt", TOTAL) is False)
check("_descriptions_complete(movies_desc) is True", pipe._descriptions_complete("movies_desc.dat", TOTAL) is True)

print("\n2. Resuming the basic pass only calls the LLM for what is missing")
asked.clear()
prompting.generateBasicDescriptions(movies, output_file="basic_descriptions.txt")
check(f"made {len(asked)} calls, not {TOTAL}", len(asked) == TOTAL - PREEXISTING)
check("file now complete", len(prompting.load_entries("basic_descriptions.txt")) == TOTAL)

print("\n3. Running it AGAIN is free")
asked.clear()
prompting.generateBasicDescriptions(movies, output_file="basic_descriptions.txt")
check("0 API calls on a completed file", len(asked) == 0)

print("\n4. Pipeline now reports the pass as complete")
check("_entries_complete(basic) is True", pipe._entries_complete("basic_descriptions.txt", TOTAL) is True)

print("\n5. --force regenerates everything")
asked.clear()
prompting.generateBasicDescriptions(movies, output_file="basic_descriptions.txt", force=True)
check(f"{TOTAL} API calls with force=True", len(asked) == TOTAL)

print("\n6. An interrupted run keeps its partial progress")
calls = {"n": 0}
def flaky(prompt):
    calls["n"] += 1
    if calls["n"] > 25:
        raise KeyboardInterrupt("simulated Ctrl-C")
    return f"SUMMARY {calls['n']}"
prompting._ask_llm = flaky
os.remove("basic_descriptions.txt")
try:
    prompting.generateBasicDescriptions(movies, output_file="basic_descriptions.txt")
except KeyboardInterrupt:
    pass
from checkpoint import is_blank
banked = prompting.load_entries("basic_descriptions.txt")
real = [e for e in banked if not is_blank(e)]
check(f"{len(real)} real entries survived the interrupt (old code saved 0)", len(real) == 25)
check("banked entries are real content", banked[0] == "SUMMARY 1")

print("\n6b. The interrupted file is recognised as INCOMPLETE, not done")
check("_entries_complete is False despite 3883 slots",
      pipe._entries_complete("basic_descriptions.txt", TOTAL) is False)
prompting._ask_llm = lambda p: (asked.append(p), "RESUMED")[1]
asked.clear()
prompting.generateBasicDescriptions(movies, output_file="basic_descriptions.txt", retry_failed=False)
check(f"resume filled the {TOTAL - 25} blanks even without --retry-failed",
      len(asked) == TOTAL - 25)

print("\n7. TMDB fetch resumes by movie id and skips what it has")
prompting._ask_llm = lambda p: "STUB"
fetched.clear()
fetching_descriptions.generate_descriptions(force=False, retry_failed=False)
check("0 TMDB lookups when every movie already has a description", len(fetched) == 0)

fetched.clear()
fetching_descriptions.generate_descriptions(force=False, retry_failed=True)
check(f"{len(fetched)} lookups with --retry-failed "
      f"(matches the {PLACEHOLDER_ROWS} unusable rows in movies_desc.dat)",
      len(fetched) == PLACEHOLDER_ROWS)

fetched.clear()
fetching_descriptions.generate_descriptions(force=True)
check(f"{TOTAL} lookups with --force", len(fetched) == TOTAL)

os.chdir(os.path.dirname(os.path.abspath(main.__file__)))
shutil.rmtree(work, ignore_errors=True)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
