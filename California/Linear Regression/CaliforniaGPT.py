import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for adding combined attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def fetch_housing_data():
    return pd.read_csv(
        "https://media.geeksforgeeks.org/wp-content/uploads/20240319120216/housing.csv")

def set_income_category(housing):
    # set income category based on median income
    housing["income_cat"] = pd.cut(housing["median_income"], 
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])
    return housing

def get_strat_train_test_dataset(housing):
    # stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    housing_split = split.split(housing, housing["income_cat"])
    # get train and test dataset
    for train_index, test_index in housing_split:
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]
        
    # drop income_category from train and test dataset
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    return train_set, test_set

def split_train_and_label_set(train_set):
    # drop median_house_value from training data
    housing_tr = train_set.drop("median_house_value", axis=1)
    # create a new dataframe with median_house_value
    housing_labels = train_set["median_house_value"].copy()
    return housing_tr, housing_labels

def get_rmse(housing_labels, predicted_data):
    # get mean squared error to analyse prediction error
    mse = mean_squared_error(housing_labels, predicted_data)
    rmse = np.sqrt(mse)
    return rmse

def transformation_pipeline():
    # pipeline execution
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
    return num_pipeline

def complete_pipeline(num_pipeline, num_attribs, cat_attribs):
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    return full_pipeline

# Main execution
if __name__ == "__main__":
    # get complete housing data
    housing = fetch_housing_data()
    print("Data loaded, shape:", housing.shape)
    
    # Add income category and perform stratified split
    housing = set_income_category(housing)
    train_set, test_set = get_strat_train_test_dataset(housing)
    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))
    
    # Split features and labels
    housing_train, housing_train_labels = split_train_and_label_set(train_set)
    housing_test, housing_test_labels = split_train_and_label_set(test_set)
    
    # Create a simple model using just median_income to compare
    print("\n=== Simple Model (Only Median Income) ===")
    simple_features = housing_train[['median_income']].copy()
    lin_reg_simple = LinearRegression()
    lin_reg_simple.fit(simple_features, housing_train_labels)
    
    # Evaluate simple model
    simple_predictions = lin_reg_simple.predict(simple_features)
    simple_rmse = get_rmse(housing_train_labels, simple_predictions)
    print("Simple model training RMSE:", simple_rmse)
    
    # Test simple model on test data
    simple_test_features = housing_test[['median_income']].copy()
    simple_test_predictions = lin_reg_simple.predict(simple_test_features)
    simple_test_rmse = get_rmse(housing_test_labels, simple_test_predictions)
    print("Simple model test RMSE:", simple_test_rmse)
    
    # Create a more complete preprocessing pipeline
    print("\n=== Complete Model (All Features) ===")
    num_attribs = list(housing_train.select_dtypes(include=[np.number]).columns)
    cat_attribs = ["ocean_proximity"]
    
    # Create and apply pipeline
    num_pipeline = transformation_pipeline()
    full_pipeline = complete_pipeline(num_pipeline, num_attribs, cat_attribs)
    
    # Transform training data
    housing_train_prepared = full_pipeline.fit_transform(housing_train)
    print("Preprocessed training data shape:", housing_train_prepared.shape)
    
    # Train complete model
    lin_reg_full = LinearRegression()
    lin_reg_full.fit(housing_train_prepared, housing_train_labels)
    
    # Evaluate on training data
    full_predictions = lin_reg_full.predict(housing_train_prepared)
    full_rmse = get_rmse(housing_train_labels, full_predictions)
    print("Complete model training RMSE:", full_rmse)
    
    # Transform test data and evaluate
    housing_test_prepared = full_pipeline.transform(housing_test)
    full_test_predictions = lin_reg_full.predict(housing_test_prepared)
    full_test_rmse = get_rmse(housing_test_labels, full_test_predictions)
    print("Complete model test RMSE:", full_test_rmse)
    
    # Generate visualization of data
    print("\n=== Creating visualization ===")
    plt.figure(figsize=(10, 7))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=housing["population"]/100, label="population",
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.title("California Housing Prices")
    plt.savefig("california_housing_map.png")
    print("Visualization saved as california_housing_map.png")