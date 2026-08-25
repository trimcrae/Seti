"""What do Rubin and NOIRLab themselves say about whether the telescope is observing?

Runner-only: every source below is egress-blocked in the sandbox, which is why
the frozen-frontier diagnosis had to lean on press coverage instead of the
observatory's own words.  On the runner they are all reachable, so the repository
can hold the primary record rather than a summary of a summary.

WHAT IS FETCHED, AND WHY EACH
-----------------------------
* **The Rubin community forum (Discourse).**  ``community.lsst.org`` serves JSON
  at ``/t/<id>.json`` (every post in a thread, with timestamps) and
  ``/latest.json`` (newest topics site-wide).  This is where the observatory
  posts operational status, and the JSON carries a real ``created_at`` per post,
  so "when was this last updated" is a fact rather than an inference from page
  text.  ``/latest.json`` is fetched as well so a NEW status thread --- one whose
  id nobody here knows yet --- still shows up.
* **The NOIRLab storm announcement** --- the AURA-wide operational statement
  covering Cerro Pachón access, power and water.
* **Rubin's live-status page** --- recorded even though it is JavaScript-driven
  and may yield little text: a 200 with no usable content is itself worth
  knowing, and is not the same as an outage.

The extracted text is **evidence, not instruction**.  It is other people's
writing, fetched verbatim and stored; nothing here acts on it.  The keyword flags
are a reading aid for a human and are deliberately reported as counted phrases
rather than as a verdict --- the verdict on whether data is flowing comes from
``rubin_outage_check.py``, which asks the alert stream instead of asking a page.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib
import re

import requests

FORUM = "https://community.lsst.org"

# Threads known to carry operational status.  Ids, not slugs: Discourse resolves
# a topic by id alone, so a renamed thread still resolves.
FORUM_TOPICS = {
    12397: "Rubin Observatory Status",
    12295: "Winter Storm in Chile",
    12533: "The recovery continues",
}

PAGES = {
    "noirlab_storm_announcement":
        "https://noirlab.edu/science/news/announcements/sci26037",
    "rubin_live_status":
        "https://rubinobservatory.org/for-scientists/rubin-101/live-status",
    "lsst_news": "https://www.lsst.org/news",
}

# Phrases that would mark a return to sky, and phrases that would mark continued
# closure.  Counted and reported side by side rather than reduced to a verdict:
# a recovery post and a closure post use much of the same vocabulary, and a
# script that scored them would be guessing where a human can just read.
RESUMED_PHRASES = [
    "back on sky", "back on the sky", "on-sky", "on sky", "resumed observing",
    "resume observing", "resumed operations", "observations have resumed",
    "returned to service", "first night back", "science operations resumed",
    "power restored", "water restored", "access restored", "road is open",
]
CLOSED_PHRASES = [
    "inaccessible", "without water", "without power", "no water", "no power",
    "closed", "suspended", "shut down", "shutdown", "not yet able",
    "remains challenging", "recovery continues", "several weeks",
]

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def strip_html(raw: str) -> str:
    """HTML to readable text, without pulling in a parser dependency.

    Block tags become newlines first --- collapsing them into spaces would run
    a heading into the paragraph below it and make the dated status lines, which
    are the whole point of the fetch, much harder to read.
    """
    txt = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    txt = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)\s*/?>", "\n", txt)
    txt = _TAG.sub(" ", txt)
    txt = html.unescape(txt)
    txt = _WS.sub(" ", txt)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", txt).strip()


def phrase_hits(text: str, phrases: list[str]) -> dict[str, int]:
    low = text.lower()
    return {p: low.count(p) for p in phrases if p in low}


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_topic(topic_id: int, timeout: float, max_posts: int = 12) -> dict:
    """One forum thread: its newest posts, with their real timestamps."""
    url = f"{FORUM}/t/{topic_id}.json"
    rec: dict = {"url": url, "topic_id": topic_id}
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"Accept": "application/json"})
        rec["status"] = resp.status_code
        if resp.status_code != 200:
            rec["text_head"] = (resp.text or "")[:400]
            return rec
        data = resp.json()
        rec["title"] = data.get("title")
        rec["created_at"] = data.get("created_at")
        rec["last_posted_at"] = data.get("last_posted_at")
        posts = (data.get("post_stream") or {}).get("posts") or []
        rec["n_posts"] = len(posts)
        out = []
        for p in posts[-max_posts:]:
            body = strip_html(p.get("cooked") or "")
            out.append({"created_at": p.get("created_at"),
                        "username": p.get("username"),
                        "text": body[:4000]})
        rec["posts"] = out
        joined = "\n".join(p["text"] for p in out)
        rec["resumed_phrases"] = phrase_hits(joined, RESUMED_PHRASES)
        rec["closed_phrases"] = phrase_hits(joined, CLOSED_PHRASES)
    except Exception as exc:                                      # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return rec


def fetch_latest_topics(timeout: float, n: int = 30) -> dict:
    """Newest forum topics site-wide --- catches a status thread we do not know."""
    url = f"{FORUM}/latest.json"
    rec: dict = {"url": url}
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"Accept": "application/json"})
        rec["status"] = resp.status_code
        if resp.status_code != 200:
            rec["text_head"] = (resp.text or "")[:400]
            return rec
        topics = ((resp.json().get("topic_list") or {}).get("topics") or [])[:n]
        rec["topics"] = [{"id": t.get("id"), "title": t.get("title"),
                          "created_at": t.get("created_at"),
                          "last_posted_at": t.get("last_posted_at"),
                          "url": f"{FORUM}/t/{t.get('slug')}/{t.get('id')}"}
                         for t in topics]
    except Exception as exc:                                      # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return rec


def fetch_page(url: str, timeout: float) -> dict:
    rec: dict = {"url": url}
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "seti-research/1.0"})
        rec["status"] = resp.status_code
        text = strip_html(resp.text or "")
        rec["chars"] = len(text)
        rec["text"] = text[:12000]
        rec["resumed_phrases"] = phrase_hits(text, RESUMED_PHRASES)
        rec["closed_phrases"] = phrase_hits(text, CLOSED_PHRASES)
    except Exception as exc:                                      # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/rubin_outage")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec: dict = {"fetched_at_utc": _utc(), "forum_topics": {}, "pages": {}}

    for tid, name in FORUM_TOPICS.items():
        print(f"[status] forum topic {tid} ({name}) ...", flush=True)
        rec["forum_topics"][str(tid)] = fetch_topic(tid, args.timeout)

    print("[status] forum /latest.json ...", flush=True)
    rec["forum_latest"] = fetch_latest_topics(args.timeout)

    for name, url in PAGES.items():
        print(f"[status] {name} ...", flush=True)
        rec["pages"][name] = fetch_page(url, args.timeout)

    path = out_dir / "status_pages.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"[status] wrote {path}")

    # Digest: the newest dated post from each thread, which is what a human
    # actually wants out of this and is otherwise buried in a 12,000-char blob.
    print("\n=== newest forum posts ===")
    for tid, t in rec["forum_topics"].items():
        if t.get("error") or not t.get("posts"):
            print(f"[{tid}] {t.get('error') or 'no posts'} (status {t.get('status')})")
            continue
        last = t["posts"][-1]
        print(f"\n[{tid}] {t.get('title')} --- last post {last.get('created_at')} "
              f"by {last.get('username')}")
        print(last["text"][:1500])
        if t.get("resumed_phrases"):
            print("  RESUMED-ish phrases:", t["resumed_phrases"])
        if t.get("closed_phrases"):
            print("  CLOSED-ish phrases:", t["closed_phrases"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
