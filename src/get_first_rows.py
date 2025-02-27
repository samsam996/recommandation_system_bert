import pandas as pd


def get_first_rows(path1, path2, rows_to_load):

    # Set the chunk size and the total number of rows to load
    chunksize = 1000

    # Read the first 1000 rows from each JSONL file
    df1_reader = pd.read_json(path1, lines=True, chunksize=chunksize)
    df2_reader = pd.read_json(path2, lines=True, chunksize=chunksize)

    # Initialize empty lists to hold the rows
    df1_sample = []
    df2_sample = []

    # Process the first chunk and take only the required number of rows
    for chunk in df1_reader:
        df1_sample.append(chunk)
        rows_to_load -= len(chunk)
        if rows_to_load <= 0:
            break

    # Reset to load df2 (if needed)
    for chunk in df2_reader:
        df2_sample.append(chunk)
        rows_to_load -= len(chunk)
        if rows_to_load <= 0:
            break

    # Concatenate the chunks into DataFrames
    df1_sample = pd.concat(df1_sample, ignore_index=True)
    df2_sample = pd.concat(df2_sample, ignore_index=True)

    return df1_sample, df2_sample
