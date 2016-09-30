import pandas as pd


def process_loan_data(input_df, qty=0.5):
    df = input_df.copy()
    # This variable sets the threshold for NaN dropping
    threshold = qty * (df.shape[0])
    output_file = 'cleaned_loans_data_' + str(100 * qty) + '.csv'
    return (df.dropna(thresh=threshold, axis=1)
              .drop(['desc', 'url'], axis=1)
              .to_csv(output_file, index=False))


# Process the data

(pd.read_csv('LoanStats3a.csv', skiprows=1)
   .pipe(process_loan_data))
