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
# Binary percentile function generator
###############################################################################


def threshold_by_percentile_per_month(weights: np.ndarray, percentile: float) -> np.ndarray:
    """
    For each month (axis 0), compute the percentile cutoff over that month's
    [n_countries × n_countries] slice, and binarize edges ≥ that cutoff.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n_months, n_countries, n_countries).
    percentile : float
        Percentile in [0, 100].

    Returns
    -------
    np.ndarray
        Same shape, dtype int8, with 1 for edges in the top (100 - percentile)% 
        of that month, else 0.
    """
    n_months = weights.shape[0]
    binary = np.zeros_like(weights, dtype=np.int8) # np.zeros_like creates an array of the same shape and type

    for m in range(n_months):
        # Flatten the m-th month slice to find its percentile cutoff
        cutoff = np.percentile(weights[m].ravel(), percentile)
        # Threshold that slice
        binary[m] = (weights[m] >= cutoff).astype(np.int8)

    return binary

# Example use:
data = np.load("migration_tensor.npy")  

# Pick percentile
pct = 90.0  

# 3) Call the function:
binary_tensor = threshold_by_percentile_per_month(data, pct)

# 4) Inspect the result:
print("Shape:", binary_tensor.shape)             # → (48, n_countries, n_countries)
print("Month 0: kept edges =", int(binary_tensor[0].sum()), "out of", data.shape[1]**2)

# 5) (Optional) Save it for your MCMC pipeline:
np.save(f"binary_by_month_top_{100-int(pct)}pct.npy", binary_tensor)


###############################################################################
# End‑of‑script marker
###############################################################################


print("CODE COMPLETED")