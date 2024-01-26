
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import OutlierHandler


def test_weathersit_variable_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config.weathersit_var,  
    )
    print("testing weathersit variable.....")
    print(sample_input_data.shape)
    print(sample_input_data.loc[12230,'weathersit'])
    assert np.isnan(sample_input_data.loc[12230,'weathersit'])

    print("going to fit and transform....")
    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    print("final assert...")
    # Then
    assert subject.loc[12230,'weathersit'] == 'Clear'


def test_temp_variable_outliers(sample_input_data):
    # Given
    outliers = OutlierHandler(
        variables=config.model_config.temp_var,  
    )
    df = sample_input_data.copy()
    test_temp_val = df.loc[12230, 'temp']
    print('test_temp_val: ',test_temp_val)
    
    
    q1 = df.describe()['temp'].loc['25%']
    q3 = df.describe()['temp'].loc['75%']
    iqr = q3 - q1
    lowerbound = q1 - ( 1.5 * iqr)
    upperbound = q3 + ( 1.5 * iqr)
    print(q1, q3, iqr, lowerbound, upperbound)

    print("testing temp variable.....")
    sample_input_data.loc[12230,'temp'] = lowerbound - 10

    assert lowerbound < sample_input_data.loc[12230,'temp'] or upperbound > sample_input_data.loc[12230,'temp']
    print("going to fit and transform....")
    # When
    subject = outliers.fit(sample_input_data).transform(sample_input_data)

    print("final assert...")
    # Then
    assert lowerbound < subject.loc[12230, 'temp'] or upperbound > subject.loc[12230, 'temp']