import pandas as pd


# Data processing utility functions


def process_loan_data(input_df, qty=0.5):
    df = input_df.copy()
    # This variable sets the threshold for NaN dropping
    threshold = qty * (df.shape[0])
    output_file = 'cleaned_loans_data_' + str(100 * qty) + '.csv'
    return (df.dropna(thresh=threshold, axis=1)
              .drop(['desc', 'url'], axis=1)
              .to_csv(output_file, index=False))


def drop_features(input_df):
    """
    Drop redundant and leaky (i.e. that contain information about the targets)
    features.
    """
    df = input_df.copy()
    features = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv',
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
    targets_labels = ['Fully Paid', 'Charged Off']
    targets_values = [1, 0]
    return (df.loc[lambda df: df.loan_status.isin(targets_labels), :]
              .replace([targets_labels, targets_values))

# Process the data

(pd.read_csv('LoanStats3a.csv', skiprows=1)
   .pipe(process_loan_data))


loans_df = (pd.read_csv('cleaned_loans_data_50.0.csv')
              .pipe(drop_features)
              .pipe(extract_targets))
