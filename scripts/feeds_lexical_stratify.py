"""Item 5: feeds lexical-bias stratification.

Split the 2000 test queries by whether the query text contains feeds-type
strings (URL, absolute path, file name with extension, UUID, hash), then
write two --task-ids files for scripts/aggregate_main_results.py to compute
R-GCN − seed FS@2048 per group with 95% CI.
"""

import json
import pathlib
import re

TASKS = pathlib.Path(
    "data/processed/datasets/isetrace/a729232bf650444a89572949522d8941/tasks.json"
)
OUT_IDENT = pathlib.Path("/tmp/feeds_ident_tasks.txt")
OUT_NOIDENT = pathlib.Path("/tmp/feeds_noident_tasks.txt")

# Feeds edges are built from exact typed-string reuse of URLs, absolute paths,
# file names, UUIDs, and hashes. A query that mentions such a string could be
# lexical-bias-linked to feeds-connected evidence.
PATTERN = re.compile(
    r"https?://\S+"  # URL
    r"|ftp://\S+"  # FTP URL
    r"|/(?:[\w.-]+/){1,}[\w.-]+"  # absolute path
    r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # UUID
    r"|[0-9a-fA-F]{32,}"  # hash (md5/sha)
    r"|[\w.-]+\.[A-Za-z0-9]{2,4}(?=/|\s|$|[,);:.\"])",  # filename.ext
    re.IGNORECASE,
)

rows = json.loads(TASKS.read_text())
ident, noident = [], []
for r in rows:
    q = r.get("query_text") or ""
    (ident if PATTERN.search(q) else noident).append(r["task_id"])

OUT_IDENT.write_text("\n".join(ident))
OUT_NOIDENT.write_text("\n".join(noident))
print(f"identifier group: {len(ident)}")
print(f"no-identifier group: {len(noident)}")
print(
    "sample ident:",
    [next(r["query_text"][:80] for r in rows if r["task_id"] == t) for t in ident[:3]],
)
print(
    "sample noident:",
    [
        next(r["query_text"][:80] for r in rows if r["task_id"] == t)
        for t in noident[:3]
    ],
)
