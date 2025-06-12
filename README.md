

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
