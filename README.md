# Kobeyo-Business-Directory-Code

## Code updates by Kanishka-
Updated code- June 11th
Changes made- 
process_urls() function filters out broken URLs completely
Only working URLs are included in the final CSV
Excludes third-party job sites (Indeed, ZipRecruiter, LinkedIn Jobs, etc.)
Only captures internal careers pages with application forms
Prioritizes pages with career-related keywords in URL or content
Products/Services Page Detection-Looks for keywords like "products", "services", "solutions", "industries served"
HR Emails: hr@, hiring@, recruiting@, talent@, jobs@, careers@
General Emails: info@, contact@, hello@, support@, admin@ + personal emails
Sales Emails: sales@, business@, partnerships@, marketing@
URLs shortened to 80 characters max
