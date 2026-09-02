"""Verify resume + --force behaviour without making a single API call."""
import logging, os, shutil, sys, tempfile
logging.basicConfig(level=logging.INFO, format="      %(message)s")

from checkpoint import (load_entries, save_entries, load_dat_rows, save_dat_rows,
                        plan_work, is_placeholder)

ok = lambda c: "PASS" if c else "**FAIL**"
results = []
def check(name, cond):
    results.append(cond)
    print(f"  [{ok(cond)}] {name}")

tmp = tempfile.mkdtemp()

print("\n1. The real description files parse cleanly")
# Smoke-test the readers against live data, without asserting counts that the
# pipeline legitimately changes as it makes progress.
entries = load_entries("basic_descriptions.txt")
movie_total = sum(1 for l in open("movies_desc.dat", encoding="utf-8") if l.strip())
check(f"basic_descriptions.txt parses ({len(entries)} entries)", isinstance(entries, list))
check(f"never more entries than movies ({len(entries)} <= {movie_total})",
      len(entries) <= movie_total)
check("movies_desc.dat is non-empty", movie_total > 0)

print("\n2. A rerun skips the work already paid for")
# Synthetic partial run: PARTIAL done, one of them a failed attempt.
PARTIAL = 441
partial = [f"description {i}" for i in range(PARTIAL)]
partial[17] = "[ERROR]"
todo, working = plan_work(partial, movie_total, force=False, retry_failed=True,
                          label="basic descriptions")
expected = movie_total - PARTIAL + 1  # everything missing, plus the one retry
check(f"only {len(todo)} of {movie_total} queued ({PARTIAL} done, 1 [ERROR] retried)",
      len(todo) == expected)
check("already-done entries preserved in the working list",
      working[0] == partial[0] and working[PARTIAL - 1] == partial[PARTIAL - 1])
check("the [ERROR] entry is queued for retry", 17 in todo)
check("a completed entry is NOT queued", 18 not in todo)
saved_hours = (movie_total - len(todo)) * 5.0 / 3600
print(f"      -> skips {movie_total - len(todo)} calls, about {saved_hours:.2f}h of API time")

print("\n2b. Without --retry-failed the [ERROR] entry is left alone")
todo_nr, _ = plan_work(partial, movie_total, force=False, retry_failed=False,
                       label="basic descriptions")
check("failed entry not re-requested", 17 not in todo_nr)
check(f"{len(todo_nr)} queued, one fewer than with retry", len(todo_nr) == expected - 1)

print("\n3. --force queues everything")
todo_f, working_f = plan_work(partial, movie_total, force=True, label="basic descriptions")
check("all movies queued", len(todo_f) == movie_total)
check("existing content discarded", all(e == "" for e in working_f))

print("\n4. A file cut off mid-write loses only the partial entry")
truncated = os.path.join(tmp, "partial.txt")
open(truncated, "w", encoding="utf-8").write("alpha\n::\nbeta\n::\ngamma-was-being-writ")
recovered = load_entries(truncated)
check("complete entries kept, incomplete tail dropped", recovered == ["alpha", "beta"])

print("\n5. Checkpoint writes are atomic")
target = os.path.join(tmp, "atomic.txt")
save_entries(target, ["one", "two"])
check("no .tmp file left behind", not os.path.exists(target + ".tmp"))
check("round-trips exactly", load_entries(target) == ["one", "two"])

print("\n6. DAT rows survive descriptions containing '::' and newlines")
movies = [{"movieId": "1", "title": "A", "year": "1995", "genres": "X"},
          {"movieId": "2", "title": "B", "year": "1996", "genres": "Y"}]
descs = {"1": "has :: colons in it", "2": "has\na newline"}
datf = os.path.join(tmp, "rows.dat")
save_dat_rows(datf, movies, descs)
back = load_dat_rows(datf)
check("'::' inside a description preserved", back["1"] == "has :: colons in it")
check("newline collapsed, row not split", back["2"] == "has a newline" and len(back) == 2)

print("\n7. Failure markers are treated as unfinished work")
check("[ERROR] / NOT_FOUND / blank all count as todo",
      all(is_placeholder(v) for v in ["[ERROR]", "NOT_FOUND", "ERROR FETCHING", "", "  "]))
check("real text does not", not is_placeholder("Woody leads Andy's toys."))

print("\n8. Fully-complete input means zero work")
complete = ["desc %d" % i for i in range(movie_total)]
todo_c, _ = plan_work(complete, movie_total, force=False, label="basic descriptions")
check("nothing queued when the file is complete", todo_c == [])

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
