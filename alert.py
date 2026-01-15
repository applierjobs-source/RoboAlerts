import argparse
import json
import html as html_lib
import math
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
from urllib.parse import quote_plus, urlparse, parse_qs
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


ACTOR_ID = "ow5loPc1VwudoP5vY"
DEFAULT_QUERY = "RoboTaxi"
DEFAULT_PHRASES = [
    "robotaxi with no driver",
    "took a driverless robotaxi",
    "robotaxi no human",
    "robotaxi empty front seat",
    "AV ride with no operator",
    "autonomous ride no safety",
    "unsupervised",
]

TEXT_FIELDS = [
    "fullText",
    "text",
    "content",
    "tweetText",
    "rawContent",
    "body",
]
ID_FIELDS = [
    "id",
    "tweetId",
    "statusId",
    "tweet_id",
    "status_id",
    "url",
]
TIME_FIELDS = [
    "created_at",
    "createdAt",
    "date_posted",
    "datePosted",
    "timestamp",
]

_CACHE_LOCK = threading.Lock()
_LATEST_CACHE: Dict[str, Any] = {
    "items": [],
    "updated_at": None,
    "last_error": None,
    "last_no_new": None,
}
ARCHIVE_FILE = "archive.jsonl"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_actor_input(query: str, max_items: int) -> Dict[str, Any]:
    search_url = os.getenv("APIFY_SEARCH_URL")
    if not search_url:
        encoded = quote_plus(query)
        search_url = f"https://twitter.com/search?q={encoded}&f=live"
    return {
        "url": search_url,
        "searchTerms": [query],
        "maxItems": max_items,
        "includeReplies": False,
        "includeRetweets": False,
    }


def run_apify_actor(token: str, actor_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
    response = requests.post(url, params={"token": token}, json=actor_input, timeout=180)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Apify request failed ({response.status_code}): {response.text.strip()}"
        )
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Apify response was not a list of dataset items.")
    return data


def fetch_x_tweets(
    bearer_token: str,
    query: str,
    max_results: int,
    since_id: Optional[str],
    include_replies: bool,
    include_retweets: bool,
    start_time: Optional[str],
) -> List[Dict[str, Any]]:
    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    query_parts = [query]
    if not include_retweets:
        query_parts.append("-is:retweet")
    if not include_replies:
        query_parts.append("-is:reply")
    base_query = " ".join(query_parts).strip()

    per_page = max(10, min(100, max_results))
    next_token: Optional[str] = None
    collected: List[Dict[str, Any]] = []
    pages = 0
    max_pages = max(1, (max_results + per_page - 1) // per_page)
    max_pages = min(5, max_pages)

    while pages < max_pages and len(collected) < max_results:
        params: Dict[str, Any] = {
            "query": base_query,
            "max_results": per_page,
            "tweet.fields": "created_at,author_id",
        }
        if since_id:
            params["since_id"] = since_id
        if start_time:
            params["start_time"] = start_time
        if next_token:
            params["next_token"] = next_token
        response = requests.get(url, headers=headers, params=params, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(
                f"X API request failed ({response.status_code}): {response.text.strip()}"
            )
        payload = response.json()
        tweets = payload.get("data", [])
        if not isinstance(tweets, list):
            break
        collected.extend(tweet for tweet in tweets if isinstance(tweet, dict))
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        next_token = meta.get("next_token")
        if not next_token:
            break
        pages += 1

    for tweet in collected:
        if tweet.get("id"):
            tweet["url"] = f"https://x.com/i/web/status/{tweet['id']}"
    return collected[:max_results]


def extract_first(item: Dict[str, Any], fields: Iterable[str]) -> Optional[str]:
    for field in fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_text(item: Dict[str, Any]) -> Optional[str]:
    return extract_first(item, TEXT_FIELDS)


def extract_id(item: Dict[str, Any]) -> Optional[str]:
    return extract_first(item, ID_FIELDS)


def extract_timestamp(item: Dict[str, Any]) -> Optional[str]:
    return extract_first(item, TIME_FIELDS)


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def append_archive(path: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    with _CACHE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_archive(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with _CACHE_LOCK:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    return entries


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def fetch_embeddings(api_key: str, texts: List[str]) -> List[List[float]]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": "text-embedding-3-small", "input": texts}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI embeddings failed ({response.status_code}): {response.text.strip()}"
        )
    data = response.json()
    embeddings = [item["embedding"] for item in data.get("data", [])]
    if len(embeddings) != len(texts):
        raise RuntimeError("OpenAI embeddings returned unexpected length.")
    return embeddings


def score_texts(
    api_key: str,
    tweets: List[str],
    phrase_texts: List[str],
) -> List[Tuple[str, float]]:
    phrase_embeddings = fetch_embeddings(api_key, phrase_texts)
    tweet_embeddings = fetch_embeddings(api_key, tweets)
    scored: List[Tuple[str, float]] = []
    for text, embedding in zip(tweets, tweet_embeddings):
        max_score = max(
            cosine_similarity(embedding, phrase_embedding)
            for phrase_embedding in phrase_embeddings
        )
        scored.append((text, max_score))
    return scored


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"seen_ids": [], "last_seen_id": None, "last_fetch_time": None}
    data = load_json(path)
    if not isinstance(data, dict):
        return {"seen_ids": [], "last_seen_id": None, "last_fetch_time": None}
    seen_ids = data.get("seen_ids", [])
    if not isinstance(seen_ids, list):
        seen_ids = []
    last_seen_id = data.get("last_seen_id")
    if not isinstance(last_seen_id, str):
        last_seen_id = None
    last_fetch_time = data.get("last_fetch_time")
    if not isinstance(last_fetch_time, (int, float)):
        last_fetch_time = None
    return {
        "seen_ids": seen_ids,
        "last_seen_id": last_seen_id,
        "last_fetch_time": last_fetch_time,
    }


def filter_new_items(
    items: List[Dict[str, Any]], seen_ids: List[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    new_items: List[Dict[str, Any]] = []
    new_ids: List[str] = []
    for item in items:
        tweet_id = extract_id(item)
        if tweet_id and tweet_id in seen_ids:
            continue
        new_items.append(item)
        if tweet_id:
            new_ids.append(tweet_id)
    return new_items, new_ids


def max_tweet_id(items: List[Dict[str, Any]]) -> Optional[str]:
    max_id: Optional[int] = None
    for item in items:
        raw_id = extract_id(item)
        if not raw_id:
            continue
        try:
            value = int(raw_id)
        except ValueError:
            continue
        if max_id is None or value > max_id:
            max_id = value
    return str(max_id) if max_id is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboTaxi Twitter alert evaluator.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search keyword.")
    parser.add_argument(
        "--max-items", type=int, default=100, help="Max items per fetch."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold.",
    )
    parser.add_argument(
        "--state-file",
        default="state.json",
        help="Path to state file for seen tweet IDs.",
    )
    parser.add_argument(
        "--actor-input",
        default=None,
        help="Path to custom actor input JSON.",
    )
    parser.add_argument(
        "--x-query",
        default=None,
        help="Override X API search query (defaults to --query).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI scoring, just list tweet texts.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between runs (default: 60).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit.",
    )
    return parser.parse_args()


def load_actor_input(args: argparse.Namespace) -> Dict[str, Any]:
    env_input = os.getenv("APIFY_INPUT_JSON")
    if env_input:
        try:
            parsed = json.loads(env_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"APIFY_INPUT_JSON is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("APIFY_INPUT_JSON must be a JSON object.")
        return parsed
    if args.actor_input:
        return load_json(args.actor_input)
    return build_actor_input(args.query, args.max_items)


def run_once(args: argparse.Namespace, apify_token: str, openai_key: Optional[str]) -> int:
    x_bearer = os.getenv("X_BEARER_TOKEN")
    include_replies_env = os.getenv("X_INCLUDE_REPLIES")
    include_retweets_env = os.getenv("X_INCLUDE_RETWEETS")
    include_replies = (
        include_replies_env.lower() in ("1", "true", "yes")
        if include_replies_env is not None
        else True
    )
    include_retweets = (
        include_retweets_env.lower() in ("1", "true", "yes")
        if include_retweets_env is not None
        else True
    )
    backfill_minutes_env = os.getenv("X_BACKFILL_MINUTES", "10")
    try:
        backfill_minutes = max(0, int(backfill_minutes_env))
    except ValueError:
        backfill_minutes = 10
    source = "x-api" if x_bearer else "apify"

    try:
        if source == "x-api":
            query = args.x_query or args.query
            state = load_state(args.state_file)
            start_time = None
            if backfill_minutes:
                start_time = datetime.utcfromtimestamp(
                    time.time() - backfill_minutes * 60
                ).isoformat(timespec="seconds") + "Z"
            items = fetch_x_tweets(
                x_bearer,
                query,
                args.max_items,
                state.get("last_seen_id"),
                include_replies,
                include_retweets,
                start_time,
            )
        else:
            actor_input = load_actor_input(args)
            items = run_apify_actor(apify_token, actor_input)
    except RuntimeError as exc:
        print(f"{'X API' if source == 'x-api' else 'Apify'} error: {exc}", file=sys.stderr)
        with _CACHE_LOCK:
            _LATEST_CACHE["last_error"] = str(exc)
        return 1
    except ValueError as exc:
        print(f"Apify input error: {exc}", file=sys.stderr)
        with _CACHE_LOCK:
            _LATEST_CACHE["last_error"] = str(exc)
        return 1
    with _CACHE_LOCK:
        _LATEST_CACHE["last_error"] = None
    items = sorted(
        items,
        key=lambda item: parse_timestamp(extract_timestamp(item)) or datetime.min,
        reverse=True,
    )
    state = load_state(args.state_file)
    new_items, new_ids = filter_new_items(items, state["seen_ids"])
    if source == "x-api":
        latest_id = max_tweet_id(items)
        if latest_id:
            state["last_seen_id"] = latest_id
        state["last_fetch_time"] = time.time()

    if not new_items:
        print("No new tweets found.")
        with _CACHE_LOCK:
            _LATEST_CACHE["updated_at"] = time.time()
            _LATEST_CACHE["last_no_new"] = time.time()
        save_json(args.state_file, state)
        return 0

    pairs: List[Tuple[Dict[str, Any], str]] = []
    for item in items:
        text = extract_text(item)
        if text:
            pairs.append((item, text))
    texts = [text for _, text in pairs]
    if not texts:
        print("No tweet text found in results.")
        with _CACHE_LOCK:
            _LATEST_CACHE["items"] = []
            _LATEST_CACHE["updated_at"] = time.time()
        return 0

    if args.dry_run:
        print("Dry run: listing tweet texts.")
        for _, text in pairs:
            print(f"- {text}")
        with _CACHE_LOCK:
            _LATEST_CACHE["items"] = [
                {
                    "text": text,
                    "score": None,
                    "match": None,
                    "url": extract_first(item, ["url"]),
                    "timestamp": extract_timestamp(item),
                }
                for item, text in pairs
            ]
            _LATEST_CACHE["updated_at"] = time.time()
        archive_entries = []
        saved_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        for item, text in pairs:
            entry_id = extract_id(item) or text
            archive_entries.append(
                {
                    "id": entry_id,
                    "text": text,
                    "url": extract_first(item, ["url"]),
                    "timestamp": extract_timestamp(item),
                    "score": None,
                    "match": None,
                    "source": source,
                    "saved_at": saved_at,
                }
            )
        append_archive(ARCHIVE_FILE, archive_entries)
        return 0

    scored = score_texts(openai_key, texts, DEFAULT_PHRASES)
    matched = [(text, score) for text, score in scored if score >= args.threshold]

    with _CACHE_LOCK:
        _LATEST_CACHE["items"] = [
            {
                "text": text,
                "score": score,
                "match": score >= args.threshold,
                "url": extract_first(item, ["url"]),
                "timestamp": extract_timestamp(item),
            }
            for (text, score), (item, _) in zip(scored, pairs)
        ]
        _LATEST_CACHE["updated_at"] = time.time()
        _LATEST_CACHE["last_no_new"] = None

    if matched:
        print("Alerts:")
        for text, score in sorted(matched, key=lambda entry: entry[1], reverse=True):
            print(f"[{score:.3f}] {text}")
    else:
        print("No alerts triggered.")

    if new_ids:
        state["seen_ids"].extend(new_ids)
        save_json(args.state_file, state)
        entry_map: Dict[str, Dict[str, Any]] = {}
        saved_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        for (text, score), (item, _) in zip(scored, pairs):
            entry_id = extract_id(item) or text
            entry_map[entry_id] = {
                "id": entry_id,
                "text": text,
                "url": extract_first(item, ["url"]),
                "timestamp": extract_timestamp(item),
                "score": score,
                "match": score >= args.threshold,
                "source": source,
                "saved_at": saved_at,
            }
        archive_entries = []
        for item in new_items:
            entry_id = extract_id(item) or extract_text(item) or ""
            entry = entry_map.get(entry_id)
            if entry:
                archive_entries.append(entry)
        append_archive(ARCHIVE_FILE, archive_entries)

    return 0


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return

        if parsed.path != "/":
            self.send_response(404)
            self.end_headers()
            return

        with _CACHE_LOCK:
            updated_at = _LATEST_CACHE.get("updated_at")
            last_error = _LATEST_CACHE.get("last_error")
            last_no_new = _LATEST_CACHE.get("last_no_new")

        query = parse_qs(parsed.query)
        page = int(query.get("page", ["1"])[0] or 1)
        page_size = int(query.get("page_size", ["50"])[0] or 50)
        page = max(1, page)
        page_size = max(1, min(200, page_size))

        entries = load_archive(ARCHIVE_FILE)
        entries = sorted(
            entries,
            key=lambda entry: parse_timestamp(entry.get("timestamp"))
            or parse_timestamp(entry.get("saved_at"))
            or datetime.min,
            reverse=True,
        )
        total = len(entries)
        start = (page - 1) * page_size
        end = start + page_size
        items = entries[start:end]

        timestamp = "never"
        if isinstance(updated_at, (int, float)):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(updated_at))
        no_new_timestamp = None
        if isinstance(last_no_new, (int, float)):
            no_new_timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime(last_no_new)
            )

        rows = []
        for entry in items:
            text = html_lib.escape(entry.get("text", ""))
            score = entry.get("score")
            match = entry.get("match")
            url = entry.get("url")
            timestamp = html_lib.escape(entry.get("timestamp") or "")
            if not timestamp:
                timestamp = html_lib.escape(entry.get("saved_at") or "")
            if url:
                safe_url = html_lib.escape(url, quote=True)
                text = f"<a href='{safe_url}' target='_blank' rel='noopener noreferrer'>{text}</a>"
            if score is None:
                verdict = "Pending"
            else:
                verdict = "Match" if match else "No match"
            score_label = "" if score is None else f"{score:.3f}"
            rows.append(
                f"<tr><td class='tweet'>{text}</td><td class='timestamp'>{timestamp}</td><td class='result'>{verdict} {score_label}</td></tr>"
            )
        rows_html = "\n".join(rows) if rows else "<tr><td colspan='3'>No data yet.</td></tr>"

        total_pages = max(1, (total + page_size - 1) // page_size)
        prev_page = page - 1
        next_page = page + 1
        nav_links = []
        if prev_page >= 1:
            nav_links.append(
                f"<a href='/?page={prev_page}&page_size={page_size}'>Previous</a>"
            )
        if next_page <= total_pages:
            nav_links.append(
                f"<a href='/?page={next_page}&page_size={page_size}'>Next</a>"
            )
        nav_html = " | ".join(nav_links) if nav_links else ""

        error_html = ""
        if last_error:
            error_html = f"<div class='error'>Last error: {last_error}</div>"
        no_new_html = ""
        if no_new_timestamp:
            no_new_html = f"<div class='meta'>No new tweets found at {no_new_timestamp}</div>"

        html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>RoboAlerts</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      h1 {{ margin-bottom: 8px; }}
      .meta {{ color: #555; margin-bottom: 16px; }}
      .error {{ color: #b00020; margin-bottom: 12px; }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
      th {{ background: #f5f5f5; text-align: left; }}
      .tweet {{ width: 60%; }}
      .timestamp {{ width: 20%; white-space: nowrap; color: #555; }}
      .result {{ width: 20%; white-space: nowrap; }}
    </style>
  </head>
  <body>
    <h1>RoboTaxi Alerts</h1>
    <div class="meta">Last updated: {timestamp}</div>
    {error_html}
    {no_new_html}
    <div class="meta">Showing {start + 1 if total else 0}-{min(end, total)} of {total} total</div>
    <div class="meta">{nav_html}</div>
    <table>
      <thead>
        <tr>
          <th>Latest RoboTaxi Tweets</th>
          <th>Timestamp</th>
          <th>Match Result</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_status_server(port: int) -> HTTPServer:
    server = HTTPServer(("", port), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    args = parse_args()
    apify_token = os.getenv("APIFY_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    x_bearer = os.getenv("X_BEARER_TOKEN")

    if not apify_token and not x_bearer:
        print("Missing APIFY_TOKEN or X_BEARER_TOKEN environment variable.", file=sys.stderr)
        return 2
    if not openai_key and not args.dry_run:
        print("Missing OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 2

    port_value = os.getenv("PORT")
    if port_value:
        try:
            port = int(port_value)
        except ValueError:
            print(f"Invalid PORT value: {port_value}", file=sys.stderr)
            return 2
        start_status_server(port)

    if args.once:
        return run_once(args, apify_token, openai_key)

    while True:
        run_once(args, apify_token, openai_key)
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

