###############################################################################
# 0. Imports ──────────────────────────────────────────────────────────────────
# We bring in two foundational libraries:
#   • pandas (aliased as pd)  – best-of-breed “spreadsheet-in-Python” toolkit.
#                              It reads the CSV and lets us query/manipulate
#                              the table easily.
#   • numpy  (aliased as np)  – provides fast, memory-efficient n-dimensional
#                              arrays (tensors) that most ML frameworks accept.
###############################################################################
import pandas as pd
import numpy as np


###############################################################################
# 1. Load the CSV ─────────────────────────────────────────────────────────────
# Read the raw migration-flow file into a DataFrame called `df`.
# Each row in that CSV is expected to contain:
#   migration_month , country_from , country_to , num_migrants
# Example row:  2019-01,  EE,  AU,  42
###############################################################################
df = pd.read_csv("international_migration_flow.csv", keep_default_na = False)


###############################################################################
# 2. Drop rows that have missing critical fields ─────────────────────────────
# • `needed` lists the four columns we MUST have to map a row into the tensor.
# • `df.dropna(subset=needed)` throws away any record where at least one of
#   those columns is NaN/blank, preventing downstream KeyErrors or bad indices.
###############################################################################
needed = ["migration_month", "country_from", "country_to", "num_migrants"]
df = df.dropna(subset=needed)


###############################################################################
# 3. Build lookup tables that turn labels → integer indices ───────────────────
#
# — 3a. months  —
#     • `df["migration_month"].unique()` collects every distinct YYYY-MM label
#       found in the file.
#     • `sorted(...)` puts them in chronological order (string sort is fine
#       because the format is ISO-like).
#
# — 3b. countries —
#     • We need ONE alphabetically sorted list that includes every country
#       code that appears as either a sender (country_from) or receiver
#       (country_to).  Concatenating the two columns, dropping NaNs, then
#       taking `unique()` does the trick.
#
# — 3c. Dictionaries  —
#     • `month2idx` converts a month label (e.g. "2020-05") into an integer
#       0 … (M-1) where M is the number of months.
#     • `country2idx` converts an ISO-3 country code (e.g. "EE") into an
#       integer 0 … (C-1) where C is the number of distinct countries.
#       Both axes share the same country index list, so origin and destination
#       align perfectly.
###############################################################################
months = sorted(df["migration_month"].unique())

countries = sorted(
    pd.concat([df["country_from"], df["country_to"]]).dropna().unique()
)

month2idx   = {m: i for i, m in enumerate(months)}      # "2019-01" → 0
country2idx = {c: i for i, c in enumerate(countries)}   # "AD" → 0, "AE" → 1, …


###############################################################################
# 4. Pre-allocate the 3-D tensor ──────────────────────────────────────────────
# Shape breakdown:
#   axis-0 (len(months))  – time slices          → 48 for 2019-01 … 2022-12
#   axis-1 (len(countries)) – origin countries   → C (e.g. 180)
#   axis-2 (len(countries)) – destination countries → C
#
# We fill it with zeros so any origin-destination pair that never appears in
# the data remains 0 (meaning “no recorded migration”).
# `dtype=np.int32` keeps memory use modest while easily covering migrant counts.
###############################################################################
data = np.zeros(
    (len(months), len(countries), len(countries)),
    dtype=np.int32
)


###############################################################################
# 5. Populate the tensor row-by-row ───────────────────────────────────────────
# `df.iterrows()` yields one Series per CSV row.
# For each row we:
#   • translate the month/country labels into integer indices (mi, fi, ti).
#   • write the integer migrant count into the corresponding tensor cell.
#
# Variable name mnemonics:
#   mi  – **m**onth **i**ndex      (0 … 47)
#   fi  – **f**rom-country **i**ndex   (0 … C-1)
#   ti  – **t**o-country **i**ndex     (0 … C-1)
###############################################################################
for _, row in df.iterrows():
    mi = month2idx[row["migration_month"]]    # which 2-D slice along axis-0
    fi = country2idx[row["country_from"]]     # which row   along axis-1
    ti = country2idx[row["country_to"]]       # which col   along axis-2
    data[mi, fi, ti] = int(row["num_migrants"])   # write the migrant count


###############################################################################
# 6. Sanity check (optional but good practice) ────────────────────────────────
# Example: How many people migrated from EE (Estonia) to AU (Australia)
# during the last month in the dataset (index 47, which maps to "2022-12")?
###############################################################################
fi, ti = country2idx["EE"], country2idx["AU"]
print(data[47, fi, ti])       # expected to print the integer count
print(data[47, fi, fi])
print(data[47, ti, fi])



###############################################################################
# Test to make sure all 181 countries are present

import pandas as pd

# assuming your DataFrame is called df
# and the two columns are named 'countries_from' and 'countries_to'

# count distinct “from” countries
distinct_from = df['country_from'].nunique()
print(f"Distinct countries in country_from: {distinct_from}")

# count distinct “to” countries
distinct_to = df['country_to'].nunique()
print(f"Distinct countries in countries_to:   {distinct_to}")
###############################################################################




###############################################################################
# 7. End-of-script marker ─────────────────────────────────────────────────────
###############################################################################
print("CODE COMPLETED")
