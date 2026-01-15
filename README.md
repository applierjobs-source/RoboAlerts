# RoboTaxi Twitter Alert

This program runs an Apify Twitter/X scraper with the keyword "RoboTaxi" and
uses the OpenAI embeddings API to check if each tweet is similar to phrases
about driverless robotaxi rides.

## Setup

1. Create and export your API keys:

```shell
export APIFY_TOKEN="your_apify_token"
export OPENAI_API_KEY="your_openai_key"
```

2. Install dependencies:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. (Optional) Customize the actor input JSON:

```shell
cp actor_input.example.json actor_input.json
```

Edit `actor_input.json` to match the actor's exact input schema if needed.
If the actor requires a URL, set `url` (example included).

## Run

Default run (loops every 60 seconds):

```shell
python alert.py
```

Set a custom search URL (if required by the actor):

```shell
export APIFY_SEARCH_URL="https://twitter.com/search?q=RoboTaxi&f=live"
```

Override actor input JSON (use the exact schema from Apify Console):

```shell
export APIFY_INPUT_JSON='{"url":"https://twitter.com/search?q=RoboTaxi&f=live","maxItems":50,"includeReplies":false,"includeRetweets":false}'
```

With custom input:

```shell
python alert.py --actor-input actor_input.json
```

Dry run (no OpenAI call):

```shell
python alert.py --dry-run
```

Run once and exit:

```shell
python alert.py --once
```

Custom interval:

```shell
python alert.py --interval 300
```

## Email alerts (SendGrid)

Set these environment variables:

```shell
export SENDGRID_API_KEY="your_sendgrid_api_key"
export EMAIL_FROM="verified-sender@yourdomain.com"
export EMAIL_TO="zacharrow3@gmail.com,johndavidarrow@gmail.com"
```

## SMS alerts (Twilio)

Set these environment variables:

```shell
export TWILIO_ACCOUNT_SID="your_twilio_account_sid"
export TWILIO_AUTH_TOKEN="your_twilio_auth_token"
export TWILIO_FROM_NUMBER="+15551234567"
export TWILIO_TO_NUMBERS="+15126363628,+15126269167"
```

Web status endpoint (Railway):

- When `PORT` is set, the app serves `http://<host>/` and `/health`.

## Notes

- The Apify actor is `ow5loPc1VwudoP5vY`.
- Alerts are triggered when cosine similarity exceeds the threshold (default
  `0.78`).
- A `state.json` file is created to avoid re-processing tweets.

Reference: https://console.apify.com/actors/ow5loPc1VwudoP5vY/input

