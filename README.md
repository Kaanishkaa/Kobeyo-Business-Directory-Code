

# Kobeyo Business Directory Code

## Updates by Kanishka (June 11)

### Changes made:

- `process_urls()` now filters out broken URLs completely.
- Only working URLs are included in the final CSV.
- Excludes third-party job sites (Indeed, ZipRecruiter, LinkedIn Jobs, etc.).
- Captures only internal career pages with application forms.
- Prioritizes pages with career-related keywords in the URL or content.
- Detects product/services pages using keywords like:  
  `products`, `services`, `solutions`, `industries served`.
- Classifies emails into categories:
  - **HR Emails:** `hr@`, `hiring@`, `recruiting@`, `talent@`, `jobs@`, `careers@`
  - **General Emails:** `info@`, `contact@`, `hello@`, `support@`, `admin@`, and personal emails
  - **Sales Emails:** `sales@`, `business@`, `partnerships@`, `marketing@`
- URLs are shortened to a maximum of 80 characters.


## Updates by Dharini (June 12)

### Changes made:

- Merged both code into single file
- Scrapes up to **50 businesses per location**
    Method `def search_places(query, location, max_results=50)` is updated for this
- Supports **multiple business categories and cities** 
- Extracts and find only 1 HR email address --> Still needs refining for personal address
    Method `def extract_emails(text, soup=None)` is defined for this
- Removed the sales email logic and will be implementing the same logic as HR emails
- Finds social media profiles: Instagram, Facebook, and X (Twitter)
    Method `def extract_socials(soup)` is defined for this
- Methods for crawling and processing a business were updated while merging
- Saves clean, structured data to CSV


# Development Summary: Building a Scalable Data Enrichment Pipeline

**Date:** 2024-05-24
**Objective:** To evolve a series of standalone data scraping and analysis scripts into a single, cohesive, scalable, and fault-tolerant data processing pipeline.

## Overview of Accomplishments

Today, we successfully transformed a collection of individual scripts into a professional-grade, three-stage automated pipeline. We migrated technologies, refactored the entire architecture for scalability, and implemented several layers of robustness to handle real-world web scraping challenges.

---

## I. Initial Scraper Enhancement & Technology Migration

Our first step was to move beyond basic scraping and adopt more powerful, modern tools.

### 1. From Single-Page Scraper to Full-Site Crawler
- **Initial State:** A script that could scrape a single URL.
- **Enhancement:** We implemented a full-site crawler capable of:
  - Starting from a single URL.
  - Discovering all internal links on a page.
  - Intelligently staying within the same website domain.
  - Maintaining a queue of pages to visit and a record of visited pages to avoid loops.

### 2. Technology Migration: Selenium to Playwright
- **Problem:** Selenium, while effective, requires manual driver management and can be "flaky" due to timing issues.
- **Solution:** We migrated the entire web automation engine to **Playwright**.
  - **Auto-Waits:** Eliminated the need for `time.sleep()` by leveraging Playwright's intelligent auto-waiting capabilities, making the scraper more reliable.
  - **Simplified Setup:** Removed the complexity of managing `chromedriver` versions.
  - **Modern API:** Adopted the more modern and efficient `async` API for better performance.

---

## II. Major Architectural Refactoring for Scalability

The most significant achievement was redesigning the entire workflow from a single script into a decoupled, multi-stage pipeline.

### 1. Designed a 3-Stage Pipeline Architecture
We broke the monolithic process into three logical, independent stages:

- **`Stage 1: Discovery`**
  - **Input:** User search query (e.g., "bakeries in los angeles").
  - **Action:** Uses Google Places API to find relevant businesses with websites.
  - **Output:** A clean CSV file (`discovery_results.csv`) of businesses to be processed.

- **`Stage 2: Enrichment`**
  - **Input:** The list of businesses from Stage 1.
  - **Action:** The core concurrent engine that processes multiple websites in parallel. For each site, it performs:
    1. Full-site Crawling
    2. Email Address Extraction
    3. Aggregate Text Collection
    4. AI-based Skill Tagging
    5. Careers Page Discovery
  - **Output:** A checkpointed CSV file (`enriched_results.csv`) with all new data.

- **`Stage 3: Finalization`**
  - **Action:** Merges the newly enriched data with the main historical results file (`final_classified_businesses.csv`), preventing duplicates and preserving work across runs.

### 2. Implemented a Modular File Structure
- **Problem:** A single, large file is hard to maintain and debug.
- **Solution:** We organized the code into a professional structure:
  - `config.py`: For all settings, API keys, and configurations.
  - `modules.py`: Contains all the "worker" logic (Discovery and Enrichment classes).
  - `main.py`: The main orchestrator that runs the pipeline stages in order.

---

## III. Robustness & Fault-Tolerance Enhancements

To prepare the pipeline for long-running, large-scale jobs, we added several critical features to handle real-world failures.

### 1. Concurrency for Speed
- Implemented `asyncio.Semaphore` to process up to **10 websites in parallel**, dramatically reducing the total runtime of the enrichment stage.

### 2. Checkpointing for Data Safety
- **Problem:** A long-running process could be interrupted, losing all progress.
- **Solution:** The enrichment stage now **saves the result for each website immediately** after it's processed. If the pipeline is stopped and restarted, it automatically skips already-completed work.

### 3. Per-Task Timeouts
- **Problem:** A single slow or broken website could stall the entire pipeline.
- **Solution:** Each website enrichment task is now wrapped in an `asyncio.wait_for` with a **2-minute timeout**. If a site takes too long, the pipeline gives up on it, logs an error, and moves on to the next.

### 4. Automatic Retries
- Integrated the `tenacity` library to automatically **retry failed network requests**, making the scraper resilient to temporary network glitches or server errors.

### 5. Ethical Scraping Controls
- Implemented logic to read and respect a website's `robots.txt` file.
- Made this feature a configurable flag (`RESPECT_ROBOTS_TXT`) to allow for "aggressive" scraping when needed, while defaulting to polite behavior.
- Added a configurable request delay to avoid overwhelming servers.

---

## IV. Systematic Debugging and Logic Refinement

Throughout the process, we identified and fixed several bugs and logic errors:
- **`Asyncio` Event Loop Errors:** Resolved two separate errors (`sync_playwright` in a running loop and `asyncio.run()` in a running loop) by correctly converting the entire project to the `async` API and using `await main()` in the notebook environment.
- **`ModuleNotFoundError`:** Fixed import issues related to the new multi-file structure by correctly adding the project directory to `sys.path`.
- **`NameError`:** Corrected missing imports (`StrOutputParser`) and replaced non-standard functions (`exit()`) with proper control flow.
- **False-Positive Email Scraping:** Refined the email regex and added post-filtering to prevent parts of image filenames (e.g., `image@2x.jpg`) from being incorrectly identified as emails.
- **Faulty Keyword Matching:** Upgraded the business group discovery logic to correctly parse comma-separated keywords and match them against the user query, fixing a critical failure in the first stage of the pipeline.
- **Process Stalls & Timeouts:** Diagnosed `CancelledError` as a notebook environment timeout and implemented the checkpointing/per-task timeout solution.

This iterative process of building, testing, and refining has resulted in a powerful and reliable data processing system.