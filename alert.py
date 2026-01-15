import argparse
import json
import math
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote_plus
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
        return {"seen_ids": []}
    data = load_json(path)
    if not isinstance(data, dict):
        return {"seen_ids": []}
    seen_ids = data.get("seen_ids", [])
    if not isinstance(seen_ids, list):
        seen_ids = []
    return {"seen_ids": seen_ids}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboTaxi Twitter alert evaluator.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search keyword.")
    parser.add_argument(
        "--max-items", type=int, default=50, help="Max items for Apify actor."
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


def run_once(args: argparse.Namespace, apify_token: str, openai_key: Optional[str]) -> int:
    if args.actor_input:
        actor_input = load_json(args.actor_input)
    else:
        actor_input = build_actor_input(args.query, args.max_items)

    try:
        items = run_apify_actor(apify_token, actor_input)
    except RuntimeError as exc:
        print(f"Apify error: {exc}", file=sys.stderr)
        return 1
    state = load_state(args.state_file)
    new_items, new_ids = filter_new_items(items, state["seen_ids"])

    if not new_items:
        print("No new tweets found.")
        return 0

    texts = [extract_text(item) for item in new_items]
    texts = [text for text in texts if text]
    if not texts:
        print("No tweet text found in Apify results.")
        return 0

    if args.dry_run:
        print("Dry run: listing tweet texts.")
        for text in texts:
            print(f"- {text}")
        return 0

    scored = score_texts(openai_key, texts, DEFAULT_PHRASES)
    matched = [(text, score) for text, score in scored if score >= args.threshold]

    if matched:
        print("Alerts:")
        for text, score in sorted(matched, key=lambda entry: entry[1], reverse=True):
            print(f"[{score:.3f}] {text}")
    else:
        print("No alerts triggered.")

    if new_ids:
        state["seen_ids"].extend(new_ids)
        save_json(args.state_file, state)

    return 0


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"RoboAlerts is running.\n")

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

    if not apify_token:
        print("Missing APIFY_TOKEN environment variable.", file=sys.stderr)
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

