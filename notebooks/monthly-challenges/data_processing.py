import pandas as pd


# Data processing utility functions


def drop_nan_columns(input_df, qty):
    df = input_df.copy()
    # This variable sets the threshold for NaN dropping
    threshold = qty * (df.shape[0])
    return df.dropna(thresh=threshold, axis=1)


def drop_features(input_df):
    """
    Drop useless, redundant and leaky (i.e. that contain information about
    the targets) features.
    """
    df = input_df.copy()
    features = ['desc', 'url', 'id', 'member_id', 'funded_amnt', 'funded_amnt_inv',
                'grade', 'sub_grade', 'emp_title', 'issue_d', 'zip_code',
                'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
                'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
                'last_pymnt_amnt']
    return df.drop(features, axis=1)


def extract_targets(input_df):
    """
    Extract targets labels and map them to numerical values:
    Fully paid -> 1
    Charged Off -> 0
    """
    df = input_df.copy()
    targets_labels = ['Fully Paid', 'Charged Off']
    targets_values = [1, 0]
    return (df.loc[lambda df: df.loan_status.isin(targets_labels), :]
              .replace(targets_labels, targets_values))


def drop_single_values_columns(input_df):
    df = input_df.copy()
    # .nunique returns 0 if the only available values are NaNs
    drop_columns = [col for col in df.columns if df[col].nunique() == 1]

    return df.drop(drop_columns, axis=1)


def process_loans_data(input_file='LoanStats3a.csv', qty=0.5):
    """
    Process the loans data by applying the previously defined functions.
    Save the resulting dataframe to a CSV.
    """
    output_file = 'processed_loans_data_' + str(100 * qty) + '.csv'

    return (pd.read_csv(input_file, skiprows=1)
              .pipe(drop_nan_columns, qty=qty)
              .pipe(drop_features)
              .pipe(extract_targets)
              .pipe(drop_single_values_columns)
              .to_csv(output_file, index=False))


if __name__ == '__main__':
    process_loans_data()
