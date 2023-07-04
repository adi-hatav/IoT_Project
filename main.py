import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def build_classifier(X_train,y_train,X_test,y_test, window_size):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier on the testing set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy} window size {window_size}")


def transform_data(files, window_num):
    # Create an empty list to store the all DataFrames
    combined_df = pd.DataFrame(columns=column_names)
    for file in files:
        # Read the CSV file into a DataFrames
        file_details = pd.read_csv(file, nrows=3)
        activity_type = file_details.iloc[1, 1]
        file_data = pd.read_csv(file, skiprows=5)
        # find the window size according to data frequency and window_num
        window_size = int(window_num / (float(file_data.iloc[2, 0]) - float(file_data.iloc[1, 0])))
        df = pd.DataFrame()  # Dataframe for the data after preprocess
        for column in file_data.columns:
            # group every "window size" records in the "column" column into one record containing the noise
            file_data[column] = file_data[column].astype(float)
            diff_df = (file_data[column] - file_data[column].shift()).abs()
            df[column] = diff_df.rolling(window=window_size, min_periods=1, center=True).sum()
        df['Activity'] = activity_type
        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat([combined_df, df])
    combined_df1 = combined_df.dropna()
    X = combined_df1.drop('Activity', axis=1)  # Features
    y = combined_df1['Activity']  # Labels
    return X,y


def activity_type_detector(csv_files, column_names, window_num):
    # split the files to train and test
    train_set, test_set = train_test_split(csv_files, test_size=0.2, random_state=15)
    X_train, y_train = transform_data(train_set, window_num)
    X_test, y_test = transform_data(test_set, window_num)
    build_classifier(X_train,y_train,X_test,y_test, window_num)


def extract_params(file):
    file_details = pd.read_csv(file, nrows=3)
    activity_type = file_details.iloc[1, 1]
    steps = int(file_details.iloc[2, 1])
    file_data = pd.read_csv(file, skiprows=5)
    peaks_num = 0
    for axis in range(1, 4):
        # find the mean of the current axis
        axis_file_data = pd.DataFrame(file_data, columns=column_names).dropna().to_numpy().T[axis]
        mean = np.mean(axis_file_data)
        # add the number of passes above mean to peaks_num
        for i in range(len(axis_file_data) - 1):
            if (float(axis_file_data[i]) < mean < float(axis_file_data[i + 1])) or \
                    (float(axis_file_data[i]) > mean > float(axis_file_data[i + 1])):
                peaks_num += 1
    return activity_type, steps, file_data, peaks_num


def find_best_param(df):
    # find the best scalar to multiply the column with that minimize the RMSE
    best_param = None
    min_difference = np.inf

    # Iterate over x values
    for x in np.arange(0.1, 2, 0.01):
        multiplied_values = df['peaks'] * x
        difference = (((multiplied_values - df['steps']) ** 2).sum()) ** 0.5
        # Update minimum difference and best x value
        if difference < min_difference:
            min_difference = difference
            best_param = x
    return best_param


def count_steps(csv_files, regularization, threshold):
    # empty Dataframes for running and walking
    running_df = pd.DataFrame(columns=['steps', 'peaks'])
    walking_df = pd.DataFrame(columns=['steps', 'peaks'])
    # split the files to train and test
    train_set, test_set = train_test_split(csv_files, test_size=0.2, random_state=42)
    for file in train_set:
        # Read the CSV file and extract params
        activity_type, steps, file_data, peaks_num = extract_params(file)
        new_row = {'steps': steps, 'peaks': peaks_num / 3}
        if activity_type == 'Running':
            running_df = running_df.append(new_row, ignore_index=True)
        else:
            walking_df = walking_df.append(new_row, ignore_index=True)
    # find the best scalar to multiply the column with that minimize the RMSE
    running_parma = find_best_param(running_df)
    walking_param = find_best_param(walking_df)
    steps_dif = []
    for file in test_set:
        # Read the CSV file and extract params
        activity_type, steps, file_data, peaks_num = extract_params(file)
        frequency = (float(file_data.iloc[2, 0]) - float(file_data.iloc[1, 0]))
        # if the records file frequency is below the threshold
        if frequency < threshold:
            reg = regularization
        else:
            reg = 1
        if activity_type == 'Running':
            print(
                f"type : {activity_type}, step number = {steps},"
                f" predicted steps = {peaks_num / 3 * running_parma * reg}")
            steps_dif.append((abs(steps - peaks_num / 3 * running_parma * reg)) ** 2)
        else:
            print(
                f"type : {activity_type}, step number = {steps},"
                f" predicted steps = {peaks_num / 3 * walking_param * reg}")
            steps_dif.append((abs(steps - peaks_num / 3 * walking_param * reg)) ** 2)
    print(f"RMSE : {(np.sum(np.array(steps_dif)) ** 0.5)}")


if __name__ == '__main__':
    directory_path = 'fixed_data_set/'
    # get a list of all CSV file paths in the directory
    csv_files = glob.glob(directory_path + '*.csv')
    column_names = ['Time [sec]', 'ACC X', 'ACC Y', 'ACC Z']
    activity_type_detector(csv_files, column_names, 0.8)
    count_steps(csv_files, 0.82, 0.07)
