"""Filter functions for model experiments G / M / K.

Each function takes a raw startup DataFrame and returns a filtered DataFrame.
- filter_M: uses the existing data engineering pipeline filter (~160k startups)
- filter_K: custom filter to be defined (most restrictive, closest to real input)
"""


def filter_M(df):
    """Replicate the data pipeline filter:
    - only operating companies as of 2023 (dc_status == 'operating')
    - founded 2014 or later
    - at least one founder (founder_count > 0)
    """
    n = len(df)

    if "dc_status" in df.columns:
        df = df[df["dc_status"] == "operating"]
        print(f"  filter_M dc_status==operating: {n - len(df):,} removed → {len(df):,} remaining")
        n = len(df)

    df = df[df["founded_on_year"] >= 2014].reset_index(drop=True)
    print(f"  filter_M founded_on_year>=2014: {n - len(df):,} removed → {len(df):,} remaining")
    n = len(df)

    from src.data_engineering.filtering import filtering
    df = filtering(df).reset_index(drop=True)
    print(f"  filter_M founder_count>0: {n - len(df):,} removed → {len(df):,} remaining")
    return df


EUROPEAN_COUNTRY_CODES = {
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CYP", "CZE",
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GEO", "GRC", "HRV",
    "HUN", "IRL", "ISL", "ITA", "LIE", "LTU", "LUX", "LVA", "MDA", "MKD",
    "MLT", "MNE", "NLD", "NOR", "POL", "PRT", "ROM", "RUS", "SRB", "SVK",
    "SVN", "SWE", "TUR", "UKR", "IMN", "GGY", "JEY", "GIB", "SMR", "ALA",
    "FRO", "SJM",
}


def filter_K(df):
    """Filter M + European countries only + founded >= 2020 + min 1 funding round."""
    df = filter_M(df)
    n = len(df)

    df = df[df["country_code"].isin(EUROPEAN_COUNTRY_CODES)].reset_index(drop=True)
    print(f"  filter_K Europe only: {n - len(df):,} removed → {len(df):,} remaining")
    n = len(df)

    df = df[df["founded_on_year"] > 2020].reset_index(drop=True)
    print(f"  filter_K founded_on_year>2020: {n - len(df):,} removed → {len(df):,} remaining")
    n = len(df)

    df = df[df["num_funding_rounds"] >= 1].reset_index(drop=True)
    print(f"  filter_K num_funding_rounds>=1: {n - len(df):,} removed → {len(df):,} remaining")
    return df
