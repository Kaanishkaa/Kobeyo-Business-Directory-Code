# Webpage Checker for Business Leads

**Verify the availability and usability of career or business pages** listed in the CSV file from supabase - prod. It checks whether the page:
- Is valid (non-broken URL)
- Returns a proper HTTP status code (not 404, 403, etc.)
- Loads correctly in a browser (i.e., not stuck or blocked by CAPTCHA)
- Is not a blank page or misleading redirect

##  Code Flow

```
main.py
├── Reads the input CSV: data/manager_jobs_rows.csv
├── Extracts the 'channel_website' column
├── Filters first 100 non-null URLs ----> This is done main.py line 24  (remove head for working on all businesses)
├── For each URL:
│   ├── 1. Validates URL format
│   ├── 2. Checks HTTP status with httpx
│   └── 3. If OK, loads page in headless browser with Playwright:
│       ├── Waits for page to load fully
│       ├── Checks for:
│       │   ├── CAPTCHA
│       │   ├── Empty content
│       │   └── 'Not Found' text
│
└── Saves output CSV to: data/checked_results.csv
```

---

## File Descriptions

| File/Folder                | Description |
|----------------------------|-------------|
| `main.py`                  | Script that runs the checks and saves result |
| `ombined_checker.py` | Contains all helper functions (httpx, Playwright, etc.) |
| `data/manager_jobs_rows.csv`  | Input CSV with original business data |
| `data/checked_results.csv`    | Output CSV with verdicts |
| `requirements.txt`        | Python dependencies |

---

## Output Columns Explained

| Column         | Description |
|----------------|-------------|
| `Original URL` | The original URL from the CSV in prod |
| `Status Code`  | HTTP response code (e.g., 200, 404) |
| `Final URL`    | Actual URL after redirects |
| `Verdict`      | Human-readable status (OK, Not Found, CAPTCHA, etc.) |

---

## What Does This Mean?

> `https://jobs.harborfreight.com/,200,https://jobs.harborfreight.com/,BrowserError: Timeout 30000ms exceeded.`

This means:
- HTTP request succeeded (`200 OK`)
- Playwright tried to load the page using a headless browser
- It **didn't finish loading in 30 seconds** (`timeout=30000ms`)
- Possibly stuck loading resources, large JS, or blocked regionally

---

## ✅ Installation & Setup

```bash
pip install -r requirements.txt
python -m playwright install
```

Then run:

```bash
python main.py
```
