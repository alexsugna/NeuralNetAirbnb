"""
This file contains useful functions for cleaning the airbnb dataset.

Alex Angus, John Dale

November 13, 2019
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_cleaned_data(excel_file, features):
    """
    This function reads an excel spreadsheet of data via pandas dataframe and 
    returns an unnormalized array of feature data and and unnormalized array of 
    target values.
    
    params:
        excel_file: the excel file containing data
        
        features: a list of strings specifying the features we want to consider
                  in our model
    
    returns:
        X: numpy array of feature values
        y: numpy array of targets
        
    """
    data = pd.read_csv(excel_file, index_col=None, low_memory=False)            #read excel file into dataframe

    df = data[features].iloc[:11341].replace('[\$,]', '', regex=True)                        #eliminate dollar signs
    
    cancellation_policy_dict = {'flexible' : 0.0, 'moderate' : 1.0,             #rank cancellation policies
                                'luxury_moderate': 2.0,
                                'strict_14_with_grace_period': 3.0,
                                'super_strict_30': 4.0, 
                                'super_strict_60': 5.0, 'strict': 6.0} 
    
    instant_bookable_dict = {"t": 1, "f" : 0}
    
    df = df.replace({"cancellation_policy": cancellation_policy_dict}) 
    df = df.replace({"instant_bookable": instant_bookable_dict})
            
    fill_values = {"host_listings_count": df["host_listings_count"].mean(),     #fill NaN values with...
                   'bathrooms': 0.0, 'bedrooms': 0.0, 'security_deposit': 0.0,
                   "extra_people":0.0, "minimum_nights": 0.0, "maximum_nights": 0.0,
                   "availability_90": 0.0, "amenities": 0.0, 
                   "number_of_reviews": 0.0,"review_scores_rating": -1.0}
    
    df.amenities = [len(item) for item in df.amenities]                         #count the amenities
    
    df = df.fillna(value=fill_values)                                           #fill NaNs
    
    df.security_deposit = pd.to_numeric(df.security_deposit)                    #convert from string to float
    df.extra_people = pd.to_numeric(df.extra_people)
    df.price = pd.to_numeric(df.price)    
    
    X = np.array(df.drop('price', axis=1))                                      #split into features and labels
    y = np.array(df["price"])
   
    return X, y

def normalize(X, y=None):
    """
    Normalize our feature data such that the mean is 0 and the standard
    deviation is 1.
    
    X: feature array
    y: target array (optional)
    
    returns X_normalized: the normalized version of X
    """
    scaler_X = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    
    if type(y) == np.array:
        scaler_y = StandardScaler()
        y_normalized = scaler_y.fit_transform(y)
        return X_normalized, y_normalized
    
    else:
        return X_normalized


def main():
    
    excel_file = 'reduced_listings.xlsx'
    features = ["host_listings_count","bathrooms", "bedrooms", 
                "security_deposit","extra_people", 'minimum_nights',
                'maximum_nights', 'availability_90','amenities', 
                'number_of_reviews', 'review_scores_rating',
                'instant_bookable','cancellation_policy', "price"]
    X, y = get_cleaned_data(excel_file, features)
    X_normalized = normalize(X, y)
    print(X_normalized)
    
if __name__ == "__main__":
    main()
        