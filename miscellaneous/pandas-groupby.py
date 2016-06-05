import pandas as pd
# Some groupby pandas recipes



# One can groupby using an index that has non-unique values (using the level=0
# flag). Amazing!
redundant_index = [1, 2, 3, 1, 2, 3]
s = pd.Series([1, 2, 3, 10, 20, 30], redundant_index)
grouped = s.groupby(level=0)

# Add a groupby level=0 on a dataset with redundant tms_gmt to
# deduplicate it => use mean() or first() as the aggregation function.
