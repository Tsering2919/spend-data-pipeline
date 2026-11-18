# Decisions & Assumptions

Based on the data inconsistencies and the task brief, I implemented the following logic:

1. **Missing Tax Rates:** If an invoice tool specifies a gross amount but the VAT rate column is blank or invalid, a default rate of 15% is automatically applied to calculate net spend.
2. **Non-Procurement Exclusion:** I defined a regex pattern (`tax|intercompany|royalt|partner|financ`) to flag and exclude non-procurement spend based on keywords found within the line description.
3. **Foreign Exchange Fallback:** If a non-USD invoice line in Company B lacks an explicit `currency_rate`, the pipeline extracts the `YYYY-MM` from the invoice date and merges the corresponding historical exchange rate from `fx_rates_usd.csv`. If no historical rate is available, it defaults to 1.0 as a safety measure.
4. **Mixed Date Formats:** Because dates appear in multiple conflicting formats across systems, Pandas is configured to parse them dynamically with `format='mixed'` and coerce impossible dates to `NaT` (Not a Time).
5. **Status Exclusions:** Only records explicitly carrying a status of 'approved' or 'paid' are included. All others are flagged with `NOT_APPROVED`.