

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