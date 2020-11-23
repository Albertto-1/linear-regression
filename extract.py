import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# * MAIN FUNCTION
def getDataXandY(dataset="insurance.csv"):
    path = Path('datasets')
    data = pd.read_csv(path / dataset)
    if (dataset=="insurance.csv"):
        return getInsuranceXandY(data)
    elif (dataset=="carprice.csv"):
        return getCarpriceXandY(data)

# * DATASET TRANSFORM FUNCTIONS
def getInsuranceXandY(data):
    # One-hot encoding
    transformed_data = oneHotEncoding(data, ['sex', 'smoker', 'region'])
    ## Log transform for charges (y)
    transformed_data['charges'] = np.log(transformed_data['charges'])
    X = transformed_data.drop('charges',axis=1)
    y = transformed_data['charges']
    return X,y

def getCarpriceXandY(data):
    # Drop trash
    transformed_data = data.drop('car_ID', axis=1)
    transformed_data = transformed_data.drop('symboling', axis=1)
    transformed_data = transformed_data.drop('CarName', axis=1)
    transformed_data = transformed_data.drop('citympg', axis=1)
    transformed_data = transformed_data.drop('highwaympg', axis=1)
    transformed_data = transformed_data.drop('peakrpm', axis=1)
    transformed_data = transformed_data.drop('carheight', axis=1)
    transformed_data = transformed_data.drop('stroke', axis=1)
    transformed_data = transformed_data.drop('compressionratio', axis=1)
    transformed_data = transformed_data.drop('carbody', axis=1)
    transformed_data = transformed_data.drop('drivewheel', axis=1)
    transformed_data = transformed_data.drop('fuelsystem', axis=1)
    transformed_data = transformed_data.drop('cylindernumber', axis=1)
    transformed_data = transformed_data.drop('enginelocation', axis=1)
    # One-hot encoding
    transformed_data = oneHotEncoding(transformed_data, [
                'fueltype',
                'aspiration',
                'doornumber',
                'enginetype'
    ])
    ## Log transform for price (y)
    transformed_data['price'] = np.log(transformed_data['price'])
    X = transformed_data.drop('price',axis=1)
    y = transformed_data['price']
    # return transformed_data
    return X,y


# * UTIL FUNCTIONS
# Plot the data correlation
def plotCorrelation(data):
    sns.heatmap(data.corr(), cmap = 'Wistia', annot= True);
    plt.show()

def oneHotEncoding(data, categorical_columns):
    return pd.get_dummies(
        data = data,
        prefix = 'C',
        prefix_sep = '_',
        columns = categorical_columns,
        dtype='int8')
