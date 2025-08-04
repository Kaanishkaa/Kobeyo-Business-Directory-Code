import pandas as pd
from combined_checker import (
    validate_url_format,
    check_http_status,
    check_with_browser
)

INPUT_CSV = "data/manager_jobs_rows.csv"
OUTPUT_CSV = "data/checked_results.csv"

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if 'channel_website' not in df.columns:
        print("❌ 'channel_website' column not found in CSV.")
        return

    #################################################################################
    # Read the README for this
    df = df[df['channel_website'].notnull()].head(100).copy()

    results = []

    for idx, url in enumerate(df['channel_website']):
        print(f"[{idx+1}/{len(df)}] Checking: {url}")

        if not validate_url_format(url):
            results.append((url, 'Invalid URL', None, None))
            continue

        http_res = check_http_status(url)

        if http_res['status_code'] and http_res['status_code'] < 400:
            verdict = check_with_browser(http_res['final_url'])
        else:
            verdict = http_res['error'] or f"HTTP {http_res['status_code']}"

        results.append((
            url,
            http_res['status_code'],
            http_res['final_url'],
            verdict
        ))

    results_df = pd.DataFrame(results, columns=[
        'Original URL', 'Status Code', 'Final URL', 'Verdict'
    ])

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Completed. Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
