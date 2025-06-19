###############################################################################
# 0. Imports — pandas for DataFrame I/O; numpy for compact n‑D arrays
###############################################################################


import pandas as pd
import numpy as np


###############################################################################
# 1. Load the CSV — migration_month,country_from,country_to,num_migrants
###############################################################################


df = pd.read_csv("international_migration_flow.csv", keep_default_na=False)


###############################################################################
# 2. Drop any row missing a required field
###############################################################################


needed = ["migration_month", "country_from", "country_to", "num_migrants"]
df = df.dropna(subset=needed)


###############################################################################
# 3. Build lookup tables
#    • months      – sorted unique YYYY‑MM labels
#    • countries   – sorted unique ISO codes from both origin & destination
#    • month2idx / country2idx map labels → integer indices
###############################################################################


months = sorted(df["migration_month"].unique())

countries = sorted(
    pd.concat([df["country_from"], df["country_to"]]).dropna().unique()
)

month2idx   = {m: i for i, m in enumerate(months)}
country2idx = {c: i for i, c in enumerate(countries)}


###############################################################################
# 4. Pre‑allocate 3‑D tensor  [months × countries × countries], fill with zeros
###############################################################################


data = np.zeros(
    (len(months), len(countries), len(countries)),
    dtype=np.int32
)

###############################################################################
# 5. Populate the tensor row‑by‑row
###############################################################################


for _, row in df.iterrows():
    mi = month2idx[row["migration_month"]]
    fi = country2idx[row["country_from"]]
    ti = country2idx[row["country_to"]]
    data[mi, fi, ti] = int(row["num_migrants"])


###############################################################################
# 6. Check if code is running properly — EE→AU count in last month (index 47)
###############################################################################


fi, ti = country2idx["EE"], country2idx["AU"]
print(data[47, fi, ti])
print(data[47, fi, fi])
print(data[47, ti, fi])


###############################################################################
# Test to make sure all expected countries are present
###############################################################################


distinct_from = df['country_from'].nunique()
print(f"Distinct countries in country_from: {distinct_from}")

distinct_to = df['country_to'].nunique()
print(f"Distinct countries in country_to:   {distinct_to}")


###############################################################################
# 7. Save tensor to .npy file
###############################################################################

output_path = "migration_tensor.npy"
np.save(output_path, data)
print(f"Tensor saved to {output_path}")

# (optional) now define your loader function
def load_migration_tensor(filepath: str = "migration_tensor.npy") -> np.ndarray:
    """
    Load and return the 3-D migration tensor from a .npy file.
    """
    return np.load(filepath)


###############################################################################
# 8. Make country and date lookup tables
###############################################################################


# 1) Build & save the date‐index table
months = sorted(df["migration_month"].unique())
date_idx = pd.DataFrame({
    "date":    months,
    "index":   range(len(months))
})
date_idx.to_csv("dateindex.csv", index=False)
print(f"Saved dateindex.csv with {len(months)} entries")

# 2) Build & save the country‐index table
countries = sorted(
    pd.concat([df["country_from"], df["country_to"]]).dropna().unique()
)
country_idx = pd.DataFrame({
    "country": countries,
    "index":   range(len(countries))
})
country_idx.to_csv("countryindex.csv", index=False)
print(f"Saved countryindex.csv with {len(countries)} entries")

# … your code that builds & saves dateindex.csv and countryindex.csv …

print(f"Saved dateindex.csv with {len(months)} entries")
print(f"Saved countryindex.csv with {len(countries)} entries")


###############################################################################
# End‑of‑script marker
###############################################################################


print("CODE COMPLETED")







