import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
from sklearn.tree import DecisionTreeClassifier


def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	return df.drop(columns="fnlwgt")


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return len(df)


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum()


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	return list(df.columns[df.isnull().any()])


# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
	education = df['education']
	bachelors_masters = [x for x in education if x == "Bachelors" or x == "Masters"]
	return round(len(bachelors_masters)/len(education) * 100, 1)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	return df.dropna()


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
	df_temp = df.drop(columns="class")
	one_hot = OneHotEncoder(handle_unknown='ignore')
	one_hot_encoded = one_hot.fit_transform(df_temp).toarray()
	attribute_names = one_hot.get_feature_names_out(column_names(df_temp))
	one_hot_df = pd.DataFrame(one_hot_encoded, columns=attribute_names)
	return one_hot_df


# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	label_enc = LabelEncoder()
	return pd.DataFrame(label_enc.fit_transform(df['class']), columns=['class'])


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(X_train, y_train)
	return clf.predict(X_train)


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	correct_predictions = [i == j for i, j in zip(y_pred, y_true)]
	return len(correct_predictions)/len(y_true) * 100


if __name__ == "__main__":
	df = read_csv_1('./data/adult.csv')
	print(num_rows(df))
	print(column_names(df))
	print(missing_values(df))
	print(columns_with_missing_values(df))
	print(bachelors_masters_percentage(df))
	print(len(data_frame_without_missing_values(df)))
	x_train = one_hot_encoding(df)
	y_train = label_encoding(df)
	y_pred = dt_predict(x_train, y_train)
	print(dt_error_rate(y_pred, y_train))

