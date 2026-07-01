# src/etl/dataframe/common/constants.py
"""Shared configuration constants for all DataFrame implementations."""

# Boolean truthy/falsy values (lowercase)
TRUTHY_VALUES = ('y', 'yes', 't', 'true', 'on', '1')
FALSY_VALUES = ('n', 'no', 'f', 'false', 'off', '0')
ALL_BOOLEAN_VALUES = TRUTHY_VALUES + FALSY_VALUES

# Characters stripped from strings before numeric parsing ("$1,000" -> "1000",
# "85%" -> "85"). Every backend consumes this list, translated to its own
# replace mechanism.
NUMERIC_CLEANUP_CHARS = (',', '$', '%')

# Recognized date/timestamp formats, ordered by specificity (first match wins).
# One canonical set, spelled per engine: (chrono/strftime for Polars, Java
# pattern for Spark). A None on the Polars side marks a Spark-only format —
# offset-carrying formats parse to tz-aware values, which Polars cannot
# coalesce with the naive results of the other formats.
DATE_FORMAT_PAIRS = (
    (None, "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"),
    ("%Y-%m-%dT%H:%M:%S%.f", "yyyy-MM-dd'T'HH:mm:ss.SSS"),
    (None, "yyyy-MM-dd'T'HH:mm:ssXXX"),
    ("%Y-%m-%dT%H:%M:%S", "yyyy-MM-dd'T'HH:mm:ss"),
    ("%Y-%m-%d %H:%M:%S%.f", "yyyy-MM-dd HH:mm:ss.SSS"),
    ("%Y-%m-%d %H:%M:%S", "yyyy-MM-dd HH:mm:ss"),
    ("%Y-%m-%d", "yyyy-MM-dd"),
    ("%m/%d/%Y %H:%M:%S", "MM/dd/yyyy HH:mm:ss"),
    ("%m/%d/%Y", "MM/dd/yyyy"),
    ("%m-%d-%Y", "MM-dd-yyyy"),
    ("%d/%m/%Y", "dd/MM/yyyy"),
    ("%d-%m-%Y", "dd-MM-yyyy"),
    ("%Y/%m/%d", "yyyy/MM/dd"),
    ("%Y%m%d", "yyyyMMdd"),
    ("%b %d, %Y", "MMM dd, yyyy"),
    ("%d %b %Y", "dd MMM yyyy"),
    ("%B %d, %Y", "MMMM dd, yyyy"),
)

POLARS_DATE_FORMATS = tuple(p for p, _ in DATE_FORMAT_PAIRS if p is not None)
SPARK_DATE_FORMATS = tuple(s for _, s in DATE_FORMAT_PAIRS)
