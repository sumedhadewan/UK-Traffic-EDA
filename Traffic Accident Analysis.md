## EDA of 1.5 million UK traffic accidents

This project is analysis on U.K accidents data from year 2005 to 2014. Data is from <a href='https://www.kaggle.com/datasets/daveianhickey/2000-16-traffic-flow-england-scotland-wales'> Kaggle</a>


It is always important to be attentive while on road driving to avoid road accidents. But are there times should we be extra vigilant ? We will explore this dataset to look into trends in reported accidents and casualties. 

### Dataset


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df1= pd.read_csv('dataset/traffic_accidents/accidents_2005_to_2007.csv')
df2= pd.read_csv('dataset/traffic_accidents/accidents_2009_to_2011.csv')
df3= pd.read_csv('dataset/traffic_accidents/accidents_2012_to_2014.csv')

```

    <ipython-input-2-c6fa97a23550>:1: DtypeWarning: Columns (31) have mixed types. Specify dtype option on import or set low_memory=False.
      df1= pd.read_csv('dataset/traffic_accidents/accidents_2005_to_2007.csv')
    <ipython-input-2-c6fa97a23550>:3: DtypeWarning: Columns (31) have mixed types. Specify dtype option on import or set low_memory=False.
      df3= pd.read_csv('dataset/traffic_accidents/accidents_2012_to_2014.csv')
    


```python
dataset_ori = pd.concat([df1,df2,df3])
dataset_ori.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1504150 entries, 0 to 464696
    Data columns (total 33 columns):
     #   Column                                       Non-Null Count    Dtype  
    ---  ------                                       --------------    -----  
     0   Accident_Index                               1504150 non-null  object 
     1   Location_Easting_OSGR                        1504049 non-null  float64
     2   Location_Northing_OSGR                       1504049 non-null  float64
     3   Longitude                                    1504049 non-null  float64
     4   Latitude                                     1504049 non-null  float64
     5   Police_Force                                 1504150 non-null  int64  
     6   Accident_Severity                            1504150 non-null  int64  
     7   Number_of_Vehicles                           1504150 non-null  int64  
     8   Number_of_Casualties                         1504150 non-null  int64  
     9   Date                                         1504150 non-null  object 
     10  Day_of_Week                                  1504150 non-null  int64  
     11  Time                                         1504033 non-null  object 
     12  Local_Authority_(District)                   1504150 non-null  int64  
     13  Local_Authority_(Highway)                    1504150 non-null  object 
     14  1st_Road_Class                               1504150 non-null  int64  
     15  1st_Road_Number                              1504150 non-null  int64  
     16  Road_Type                                    1504150 non-null  object 
     17  Speed_limit                                  1504150 non-null  int64  
     18  Junction_Detail                              0 non-null        float64
     19  Junction_Control                             901315 non-null   object 
     20  2nd_Road_Class                               1504150 non-null  int64  
     21  2nd_Road_Number                              1504150 non-null  int64  
     22  Pedestrian_Crossing-Human_Control            1504133 non-null  object 
     23  Pedestrian_Crossing-Physical_Facilities      1504116 non-null  object 
     24  Light_Conditions                             1504150 non-null  object 
     25  Weather_Conditions                           1504024 non-null  object 
     26  Road_Surface_Conditions                      1502192 non-null  object 
     27  Special_Conditions_at_Site                   1504135 non-null  object 
     28  Carriageway_Hazards                          1504121 non-null  object 
     29  Urban_or_Rural_Area                          1504150 non-null  int64  
     30  Did_Police_Officer_Attend_Scene_of_Accident  1501228 non-null  object 
     31  LSOA_of_Accident_Location                    1395912 non-null  object 
     32  Year                                         1504150 non-null  int64  
    dtypes: float64(5), int64(13), object(15)
    memory usage: 390.2+ MB
    

Making a **copy** of original dataset


```python
dataset = dataset_ori.copy()
```


```python
# Force Notebook to Display all columns
pd.set_option('display.max_columns', None)

```

#### Data from 2008 is missing.


```python
print('Traffic accident records are between {} and {}'.format(dataset['Date'].min(),dataset['Date'].max()))
```

    Traffic accident records are between 01/01/2005 and 31/12/2014
    


```python
dataset['Year'].unique()
```




    array([2005, 2006, 2007, 2009, 2010, 2011, 2012, 2013, 2014], dtype=int64)



### Data Cleaning

We do the following before checking duplictaed rows

* convert `Date` column to correct datatype datetime
* sort rows by `Date` column


```python
# Convert date in string format to datetime format
# Sort rows by Date Column in ascending order
dataset['Date'] = pd.to_datetime(dataset['Date'],format='%d/%m/%Y')
dataset.sort_values('Date',inplace= True)
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accident_Index</th>
      <th>Location_Easting_OSGR</th>
      <th>Location_Northing_OSGR</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Police_Force</th>
      <th>Accident_Severity</th>
      <th>Number_of_Vehicles</th>
      <th>Number_of_Casualties</th>
      <th>Date</th>
      <th>Day_of_Week</th>
      <th>Time</th>
      <th>Local_Authority_(District)</th>
      <th>Local_Authority_(Highway)</th>
      <th>1st_Road_Class</th>
      <th>1st_Road_Number</th>
      <th>Road_Type</th>
      <th>Speed_limit</th>
      <th>Junction_Detail</th>
      <th>Junction_Control</th>
      <th>2nd_Road_Class</th>
      <th>2nd_Road_Number</th>
      <th>Pedestrian_Crossing-Human_Control</th>
      <th>Pedestrian_Crossing-Physical_Facilities</th>
      <th>Light_Conditions</th>
      <th>Weather_Conditions</th>
      <th>Road_Surface_Conditions</th>
      <th>Special_Conditions_at_Site</th>
      <th>Carriageway_Hazards</th>
      <th>Urban_or_Rural_Area</th>
      <th>Did_Police_Officer_Attend_Scene_of_Accident</th>
      <th>LSOA_of_Accident_Location</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41863</th>
      <td>200506F917897</td>
      <td>382690.0</td>
      <td>398830.0</td>
      <td>-2.262343</td>
      <td>53.485888</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2005-01-01</td>
      <td>7</td>
      <td>15:33</td>
      <td>107</td>
      <td>E08000006</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Raining with high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01005607</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>98027</th>
      <td>200530D000007</td>
      <td>437460.0</td>
      <td>331310.0</td>
      <td>-1.444839</td>
      <td>52.877937</td>
      <td>30</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2005-01-01</td>
      <td>7</td>
      <td>15:20</td>
      <td>323</td>
      <td>E06000015</td>
      <td>3</td>
      <td>514</td>
      <td>Single carriageway</td>
      <td>40</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Raining without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>No</td>
      <td>E01013518</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>142953</th>
      <td>2005445ZD0001</td>
      <td>458550.0</td>
      <td>91960.0</td>
      <td>-1.171887</td>
      <td>50.724253</td>
      <td>44</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2005-01-01</td>
      <td>7</td>
      <td>01:20</td>
      <td>505</td>
      <td>E06000046</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01017342</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>195494</th>
      <td>200597LA70101</td>
      <td>229600.0</td>
      <td>682260.0</td>
      <td>-4.734319</td>
      <td>56.002963</td>
      <td>97</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2005-01-01</td>
      <td>7</td>
      <td>01:00</td>
      <td>913</td>
      <td>S12000035</td>
      <td>3</td>
      <td>814</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Automatic traffic signal</td>
      <td>4</td>
      <td>832</td>
      <td>None within 50 metres</td>
      <td>non-junction pedestrian crossing</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>142147</th>
      <td>2005445HF0001</td>
      <td>456800.0</td>
      <td>106200.0</td>
      <td>-1.194479</td>
      <td>50.852471</td>
      <td>44</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2005-01-01</td>
      <td>7</td>
      <td>19:40</td>
      <td>492</td>
      <td>E10000014</td>
      <td>3</td>
      <td>27</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Automatic traffic signal</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01022735</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>350257</th>
      <td>2.01E+12</td>
      <td>341037.0</td>
      <td>391726.0</td>
      <td>-2.888685</td>
      <td>53.419015</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2014-12-31</td>
      <td>4</td>
      <td>09:25</td>
      <td>91</td>
      <td>E08000012</td>
      <td>6</td>
      <td>0</td>
      <td>Dual carriageway</td>
      <td>40</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>3</td>
      <td>57</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01006571</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>393363</th>
      <td>201431B310014</td>
      <td>481903.0</td>
      <td>353979.0</td>
      <td>-0.778854</td>
      <td>53.076750</td>
      <td>31</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2014-12-31</td>
      <td>4</td>
      <td>16:25</td>
      <td>345</td>
      <td>E10000024</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>40</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>Central refuge</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01028292</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>434263</th>
      <td>2.01E+12</td>
      <td>637290.0</td>
      <td>150920.0</td>
      <td>1.395878</td>
      <td>51.208098</td>
      <td>46</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2014-12-31</td>
      <td>4</td>
      <td>19:40</td>
      <td>533</td>
      <td>E10000016</td>
      <td>6</td>
      <td>3315</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01024250</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>395116</th>
      <td>2.01E+12</td>
      <td>527328.0</td>
      <td>335121.0</td>
      <td>-0.108524</td>
      <td>52.898434</td>
      <td>32</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2014-12-31</td>
      <td>4</td>
      <td>13:41</td>
      <td>350</td>
      <td>E10000019</td>
      <td>3</td>
      <td>17</td>
      <td>Single carriageway</td>
      <td>60</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>4</td>
      <td>1397</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>E01026017</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>434237</th>
      <td>2.01E+12</td>
      <td>576090.0</td>
      <td>158410.0</td>
      <td>0.524431</td>
      <td>51.297524</td>
      <td>46</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2014-12-31</td>
      <td>4</td>
      <td>11:40</td>
      <td>536</td>
      <td>E10000016</td>
      <td>1</td>
      <td>20</td>
      <td>Slip road</td>
      <td>70</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>E01024335</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>1504150 rows × 33 columns</p>
</div>



##### Checking for **duplicated rows**

We will drop duplicated row before dropping column because there is a chance dropping columns before dropping duplicated rows may increase  no of duplicated rows.


```python
print('There are {} duplicated rows in dataset .Thats {} %'.format(dataset.duplicated().sum(),round(34155/dataset.shape[0]*100,)))
```

    There are 34155 duplicated rows in dataset .Thats 2 %
    


```python
dataset[dataset.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accident_Index</th>
      <th>Location_Easting_OSGR</th>
      <th>Location_Northing_OSGR</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Police_Force</th>
      <th>Accident_Severity</th>
      <th>Number_of_Vehicles</th>
      <th>Number_of_Casualties</th>
      <th>Date</th>
      <th>Day_of_Week</th>
      <th>Time</th>
      <th>Local_Authority_(District)</th>
      <th>Local_Authority_(Highway)</th>
      <th>1st_Road_Class</th>
      <th>1st_Road_Number</th>
      <th>Road_Type</th>
      <th>Speed_limit</th>
      <th>Junction_Detail</th>
      <th>Junction_Control</th>
      <th>2nd_Road_Class</th>
      <th>2nd_Road_Number</th>
      <th>Pedestrian_Crossing-Human_Control</th>
      <th>Pedestrian_Crossing-Physical_Facilities</th>
      <th>Light_Conditions</th>
      <th>Weather_Conditions</th>
      <th>Road_Surface_Conditions</th>
      <th>Special_Conditions_at_Site</th>
      <th>Carriageway_Hazards</th>
      <th>Urban_or_Rural_Area</th>
      <th>Did_Police_Officer_Attend_Scene_of_Accident</th>
      <th>LSOA_of_Accident_Location</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58934</th>
      <td>2.01E+12</td>
      <td>503250.0</td>
      <td>487860.0</td>
      <td>-0.415801</td>
      <td>54.275973</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2005-02-04</td>
      <td>6</td>
      <td>13:40</td>
      <td>186</td>
      <td>E10000023</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>3</td>
      <td>64</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01027826</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>147881</th>
      <td>2.01E+12</td>
      <td>577820.0</td>
      <td>167870.0</td>
      <td>0.553941</td>
      <td>51.381965</td>
      <td>46</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2005-05-18</td>
      <td>4</td>
      <td>16:10</td>
      <td>544</td>
      <td>E06000035</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01016046</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>151435</th>
      <td>2.01E+12</td>
      <td>628620.0</td>
      <td>139310.0</td>
      <td>1.264598</td>
      <td>51.107403</td>
      <td>46</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2005-05-19</td>
      <td>5</td>
      <td>18:15</td>
      <td>533</td>
      <td>E10000016</td>
      <td>3</td>
      <td>20</td>
      <td>Dual carriageway</td>
      <td>70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fog or mist</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>E01024249</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>50483</th>
      <td>2.00913E+12</td>
      <td>403470.0</td>
      <td>416890.0</td>
      <td>-1.948984</td>
      <td>53.648498</td>
      <td>13</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2009-01-06</td>
      <td>3</td>
      <td>16:35</td>
      <td>202</td>
      <td>E08000033</td>
      <td>3</td>
      <td>672</td>
      <td>Single carriageway</td>
      <td>50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>E01010955</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>461108</th>
      <td>2.01193E+12</td>
      <td>298760.0</td>
      <td>724660.0</td>
      <td>-3.642056</td>
      <td>56.403354</td>
      <td>93</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2011-01-07</td>
      <td>6</td>
      <td>10:40</td>
      <td>934</td>
      <td>S12000024</td>
      <td>3</td>
      <td>85</td>
      <td>Single carriageway</td>
      <td>60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29855</th>
      <td>201205AA06642</td>
      <td>330460.0</td>
      <td>394040.0</td>
      <td>-3.048320</td>
      <td>53.438521</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2012-12-31</td>
      <td>2</td>
      <td>13:45</td>
      <td>95</td>
      <td>E08000015</td>
      <td>6</td>
      <td>0</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Raining without high winds</td>
      <td>Wet/Damp</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01007237</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>28928</th>
      <td>201204FG12105</td>
      <td>386448.0</td>
      <td>438546.0</td>
      <td>-2.207454</td>
      <td>53.842978</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2012-12-31</td>
      <td>2</td>
      <td>16:53</td>
      <td>77</td>
      <td>E10000017</td>
      <td>3</td>
      <td>6068</td>
      <td>Single carriageway</td>
      <td>30</td>
      <td>NaN</td>
      <td>Giveway or uncontrolled</td>
      <td>6</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkness: Street lights present and lit</td>
      <td>Fine without high winds</td>
      <td>Wet/Damp</td>
      <td>Permanent sign or marking defective or obscured</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01025181</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>310912</th>
      <td>2.01E+12</td>
      <td>346241.0</td>
      <td>744344.0</td>
      <td>-2.876901</td>
      <td>56.587975</td>
      <td>93</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2013-01-30</td>
      <td>4</td>
      <td>09:00</td>
      <td>912</td>
      <td>S12000041</td>
      <td>4</td>
      <td>9127</td>
      <td>Single carriageway</td>
      <td>60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>2</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>392226</th>
      <td>2.01E+12</td>
      <td>434640.0</td>
      <td>378410.0</td>
      <td>-1.481677</td>
      <td>53.301500</td>
      <td>30</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2014-11-25</td>
      <td>3</td>
      <td>06:40</td>
      <td>327</td>
      <td>E10000007</td>
      <td>3</td>
      <td>61</td>
      <td>Dual carriageway</td>
      <td>70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkeness: No street lighting</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01019795</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>392225</th>
      <td>2.01E+12</td>
      <td>434640.0</td>
      <td>378410.0</td>
      <td>-1.481677</td>
      <td>53.301500</td>
      <td>30</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2014-11-25</td>
      <td>3</td>
      <td>06:40</td>
      <td>327</td>
      <td>E10000007</td>
      <td>3</td>
      <td>61</td>
      <td>Dual carriageway</td>
      <td>70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>0</td>
      <td>None within 50 metres</td>
      <td>No physical crossing within 50 meters</td>
      <td>Darkeness: No street lighting</td>
      <td>Fine without high winds</td>
      <td>Frost/Ice</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>Yes</td>
      <td>E01019795</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>34155 rows × 33 columns</p>
</div>



All values in Row 392226,392225 are the same suggesting rows are duplicated.


```python
#Dropping duplicated rows
dataset.drop_duplicates(inplace= True)
print(dataset.shape)
```

    (1469995, 33)
    

#### Dropping **columns**


```python
print('No of Columns : ',len(dataset.columns))
print('\n')
print(dataset.columns)
```

    No of Columns :  33
    
    
    Index(['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
           'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity',
           'Number_of_Vehicles', 'Number_of_Casualties', 'Date', 'Day_of_Week',
           'Time', 'Local_Authority_(District)', 'Local_Authority_(Highway)',
           '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
           'Junction_Detail', 'Junction_Control', '2nd_Road_Class',
           '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',
           'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
           'Weather_Conditions', 'Road_Surface_Conditions',
           'Special_Conditions_at_Site', 'Carriageway_Hazards',
           'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
           'LSOA_of_Accident_Location', 'Year'],
          dtype='object')
    


```python
# Dropping 20 columns
dataset.drop(['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
       'Longitude', 'Latitude','Local_Authority_(District)', 'Local_Authority_(Highway)',
             '1st_Road_Class', '1st_Road_Number','Junction_Detail', 'Junction_Control',
             '2nd_Road_Class','2nd_Road_Number','Pedestrian_Crossing-Human_Control',
       'Special_Conditions_at_Site', 'Carriageway_Hazards','Day_of_Week',
       'Urban_or_Rural_Area','Did_Police_Officer_Attend_Scene_of_Accident','LSOA_of_Accident_Location'],axis=1,inplace=True)
print('No of Columns after dropping : ',len(dataset.columns))
```

    No of Columns after dropping :  13
    

#### Data Types

`Accident_Severity`from Int to String data type Because we will use this column in categorizing.


```python
print('Before : {} Datatype'.format(dataset['Accident_Severity'].dtype))
dataset['Accident_Severity'] = dataset['Accident_Severity'].astype(str)
print('After : {} Datatype'.format(dataset['Accident_Severity'].dtype))
```

    Before : int64 Datatype
    After : object Datatype
    


```python
#convert Time column to datetime datatype and to extract hour
print('Before : {} Datatype'.format(dataset['Time'].dtype))
dataset['Time'] = pd.to_datetime(dataset['Time'])
print('After : {} Datatype'.format(dataset['Time'].dtype))
```

    Before : object Datatype
    After : datetime64[ns] Datatype
    

#### New columns

Adding `Hour` and `Day_of_week` columns


```python
dataset['Hour_day'] = dataset['Time'].dt.hour
# dataset['Hour_day'] = dataset['Hour_day'].astype(int)
dataset['Day_of_week'] = dataset['Date'].dt.day_name()
print(dataset[['Hour_day','Day_of_week']][:3])
print(dataset[['Hour_day','Day_of_week']].dtypes)
```

            Hour_day Day_of_week
    41863       15.0    Saturday
    98027       15.0    Saturday
    142953       1.0    Saturday
    Hour_day       float64
    Day_of_week     object
    dtype: object
    

##### Checking for null values


```python
percentage_null = dataset.isnull().sum()*100/dataset.shape[0]
percentage_null = percentage_null.to_frame()
percentage_null.rename(columns={0 :'Null values (%)' },inplace= True)
percentage_null
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Null values (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Police_Force</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Accident_Severity</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Number_of_Vehicles</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Number_of_Casualties</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Date</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Time</th>
      <td>0.007959</td>
    </tr>
    <tr>
      <th>Road_Type</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Speed_limit</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Pedestrian_Crossing-Physical_Facilities</th>
      <td>0.002313</td>
    </tr>
    <tr>
      <th>Light_Conditions</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Weather_Conditions</th>
      <td>0.008571</td>
    </tr>
    <tr>
      <th>Road_Surface_Conditions</th>
      <td>0.132313</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Hour_day</th>
      <td>0.007959</td>
    </tr>
    <tr>
      <th>Day_of_week</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} rows with null values. Thats {} %. Therefore we are dropping rows with null values'.format(dataset.isnull().sum().sum(),round(2188*100/dataset.shape[0],2)))
```

    There are 2339 rows with null values. Thats 0.15 %. Therefore we are dropping rows with null values
    


```python
dataset.dropna(inplace= True)
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1467786 entries, 41863 to 434237
    Data columns (total 15 columns):
     #   Column                                   Non-Null Count    Dtype         
    ---  ------                                   --------------    -----         
     0   Police_Force                             1467786 non-null  int64         
     1   Accident_Severity                        1467786 non-null  object        
     2   Number_of_Vehicles                       1467786 non-null  int64         
     3   Number_of_Casualties                     1467786 non-null  int64         
     4   Date                                     1467786 non-null  datetime64[ns]
     5   Time                                     1467786 non-null  datetime64[ns]
     6   Road_Type                                1467786 non-null  object        
     7   Speed_limit                              1467786 non-null  int64         
     8   Pedestrian_Crossing-Physical_Facilities  1467786 non-null  object        
     9   Light_Conditions                         1467786 non-null  object        
     10  Weather_Conditions                       1467786 non-null  object        
     11  Road_Surface_Conditions                  1467786 non-null  object        
     12  Year                                     1467786 non-null  int64         
     13  Hour_day                                 1467786 non-null  float64       
     14  Day_of_week                              1467786 non-null  object        
    dtypes: datetime64[ns](2), float64(1), int64(5), object(7)
    memory usage: 179.2+ MB
    

##### Checking unique values


```python
for col_name in ['Accident_Severity','Day_of_week','Road_Type','Speed_limit','Light_Conditions','Weather_Conditions','Year']:
    print('Column Name : {}'.format(col_name))
    print('Unique values in this column : {}'.format(dataset[col_name].unique()))
    print(dataset[col_name].value_counts(dropna=False)) #important to sent dropna= False
    print('===================================')
    print('\n')
```

    Column Name : Accident_Severity
    Unique values in this column : ['3' '2' '1']
    3    1250075
    2     198687
    1      19024
    Name: Accident_Severity, dtype: int64
    ===================================
    
    
    Column Name : Day_of_week
    Unique values in this column : ['Saturday' 'Sunday' 'Monday' 'Tuesday' 'Wednesday' 'Thursday' 'Friday']
    Friday       241233
    Wednesday    220838
    Thursday     220755
    Tuesday      218686
    Monday       208526
    Saturday     196658
    Sunday       161090
    Name: Day_of_week, dtype: int64
    ===================================
    
    
    Column Name : Road_Type
    Unique values in this column : ['Single carriageway' 'Dual carriageway' 'One way street' 'Slip road'
     'Roundabout' 'Unknown']
    Single carriageway    1099001
    Dual carriageway       216689
    Roundabout              98286
    One way street          30269
    Slip road               15341
    Unknown                  8200
    Name: Road_Type, dtype: int64
    ===================================
    
    
    Column Name : Speed_limit
    Unique values in this column : [30 40 60 70 50 20 10 15]
    30    941438
    60    234534
    40    120110
    70    107469
    50     47790
    20     16421
    10        14
    15        10
    Name: Speed_limit, dtype: int64
    ===================================
    
    
    Column Name : Light_Conditions
    Unique values in this column : ['Daylight: Street light present'
     'Darkness: Street lights present and lit' 'Darkeness: No street lighting'
     'Darkness: Street lights present but unlit'
     'Darkness: Street lighting unknown']
    Daylight: Street light present               1075716
    Darkness: Street lights present and lit       288309
    Darkeness: No street lighting                  81538
    Darkness: Street lighting unknown              15472
    Darkness: Street lights present but unlit       6751
    Name: Light_Conditions, dtype: int64
    ===================================
    
    
    Column Name : Weather_Conditions
    Unique values in this column : ['Raining with high winds' 'Raining without high winds'
     'Fine without high winds' 'Fine with high winds' 'Other' 'Unknown'
     'Snowing with high winds' 'Snowing without high winds' 'Fog or mist']
    Fine without high winds       1176211
    Raining without high winds     172917
    Other                           32848
    Unknown                         26241
    Raining with high winds         20356
    Fine with high winds            18091
    Snowing without high winds      11140
    Fog or mist                      8049
    Snowing with high winds          1933
    Name: Weather_Conditions, dtype: int64
    ===================================
    
    
    Column Name : Year
    Unique values in this column : [2005 2006 2007 2009 2010 2011 2012 2013 2014]
    2005    198453
    2006    188904
    2007    181879
    2009    163333
    2010    154185
    2011    151237
    2014    146098
    2012    145305
    2013    138392
    Name: Year, dtype: int64
    ===================================
    
    
    

## Exploring Traffic accidents

### Time Indicators

One of the possible indicator of traffic Accident is time.There might be more accidents in certain month , on certain time of day, or year.

We're going to look at a few line plots showing how the traffic volume changes according to the following:
- Year
- Month
- Day of the week
- Time of day


```python
#making copy of `dataset` and adding new column for Month
dataset_w_month = dataset.copy()
dataset_w_month['Month']= dataset['Date'].dt.month

sns.set_style('darkgrid')

fig,axes = plt.subplots(nrows=2, ncols=2,figsize=(12,8),sharey= True)

df_list = []
columns= ['Year','Month','Day_of_week','Hour_day']
for ax,col_name in zip(axes.flatten(),columns):
    title = col_name.replace('_',' ')
    
    df = dataset_w_month[col_name].value_counts(normalize = True)*100
    df=df.reset_index()
    df.rename(columns={'index':col_name,col_name:'frq (%)'},inplace= True)
    df_list.append(df)
    #line plot
    g = sns.lineplot(data = df , x= col_name, y = 'frq (%)', ax= ax)
    g.set_title('Distribution of accident by '+ title, y=0.9)
    g.set(xlabel='',ylabel='%')
plt.tight_layout()
```


    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_35_0.png)
    



```python
for df in df_list:
    display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>frq (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>13.520568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>12.869996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007</td>
      <td>12.391384</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009</td>
      <td>11.127848</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>10.504597</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011</td>
      <td>10.303750</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014</td>
      <td>9.953631</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012</td>
      <td>9.899604</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>9.428622</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>frq (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>9.222393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>9.052341</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8.793652</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>8.621148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>8.500217</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>8.492042</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>8.230355</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>8.191385</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>8.029440</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>7.919751</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>7.666921</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>7.280353</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Day_of_week</th>
      <th>frq (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Friday</td>
      <td>16.435162</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wednesday</td>
      <td>15.045654</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thursday</td>
      <td>15.039999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tuesday</td>
      <td>14.899038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Monday</td>
      <td>14.206839</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Saturday</td>
      <td>13.398275</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sunday</td>
      <td>10.975033</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hour_day</th>
      <th>frq (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.0</td>
      <td>8.881472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.0</td>
      <td>8.119099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>7.719381</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>7.284781</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.0</td>
      <td>6.915518</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13.0</td>
      <td>6.075886</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.0</td>
      <td>6.066552</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12.0</td>
      <td>5.930837</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11.0</td>
      <td>5.191561</td>
    </tr>
    <tr>
      <th>9</th>
      <td>19.0</td>
      <td>5.167988</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9.0</td>
      <td>4.980494</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0</td>
      <td>4.518642</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.0</td>
      <td>4.150809</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20.0</td>
      <td>3.805800</td>
    </tr>
    <tr>
      <th>14</th>
      <td>21.0</td>
      <td>3.050104</td>
    </tr>
    <tr>
      <th>15</th>
      <td>22.0</td>
      <td>2.658630</td>
    </tr>
    <tr>
      <th>16</th>
      <td>23.0</td>
      <td>2.125582</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6.0</td>
      <td>1.719120</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>1.534079</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>1.123938</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.0</td>
      <td>0.905309</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5.0</td>
      <td>0.801139</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.0</td>
      <td>0.721495</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4.0</td>
      <td>0.551783</td>
    </tr>
  </tbody>
</table>
</div>


We look further distribution of accidents by Hour and week days to see pattern during weekends, wekdays.


```python
dataset_heatmap_accident = dataset.copy()[['Day_of_week','Hour_day','Number_of_Casualties']]
dataset_heatmap_accident = dataset_heatmap_accident.groupby(['Day_of_week','Hour_day']).count()['Number_of_Casualties'].unstack()
g=sns.heatmap(dataset_heatmap_accident,cmap = 'rocket_r')
g.set_title('Accidents by Day & Hour')
g.set(ylabel = 'Day of Week',xlabel='Hour')
```




    [Text(34.0, 0.5, 'Day of Week'), Text(0.5, 16.0, 'Hour')]




    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_38_1.png)
    


Data Revealed
- Number of accidents gradually decreased each year,displaying downward trend. Except 2012.
- Coldest Month of the year February recorded lowest no. of accidents (7%).November was the peak month for accidents (9%). Generally warmer month recorded more accidents than colder months.
- Most accidents took place on Friday(17%).
- 5PM was the peak hour for accidents during weekdays(9%). With 4PM and 3PM taking 2nd , 3rd place in 'riskest driving hours'. While 12:00 noon was the peak hour for accidents during weekends.

### Categorical Indicators

#### Weather, Road Type, Road surface, Light condition
 


```python
sns.set()
sns.set_style('white')
sns.set_palette('hls')
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1,figsize=(8,14))
columns = ['Weather_Conditions','Road_Type' , 'Light_Conditions','Road_Surface_Conditions']
axes= [ax1,ax2,ax3,ax4]
for col,ax in zip(columns,axes):
    col_title=col.replace('_',' ')
    data = dataset[col].value_counts(normalize=True)*100
    df = data.to_frame().reset_index()
    g= sns.barplot(data= df,x = col , y = 'index' , ax=ax)
    g.set(xlabel='',ylabel='')
    g.set_title('Distribution of accidents by '+ col_title)
    print(df)
ax4.set_xlabel('Frequency (%)')
sns.despine(left= True)
plt.tight_layout()
```

                            index  Weather_Conditions
    0     Fine without high winds           80.135047
    1  Raining without high winds           11.780805
    2                       Other            2.237928
    3                     Unknown            1.787795
    4     Raining with high winds            1.386851
    5        Fine with high winds            1.232537
    6  Snowing without high winds            0.758966
    7                 Fog or mist            0.548377
    8     Snowing with high winds            0.131695
                    index  Road_Type
    0  Single carriageway  74.874743
    1    Dual carriageway  14.762983
    2          Roundabout   6.696208
    3      One way street   2.062222
    4           Slip road   1.045180
    5             Unknown   0.558665
                                           index  Light_Conditions
    0             Daylight: Street light present         73.288340
    1    Darkness: Street lights present and lit         19.642441
    2              Darkeness: No street lighting          5.555169
    3          Darkness: Street lighting unknown          1.054105
    4  Darkness: Street lights present but unlit          0.459944
                           index  Road_Surface_Conditions
    0                        Dry                68.901461
    1                   Wet/Damp                28.164732
    2                  Frost/Ice                 2.088179
    3                       Snow                 0.704667
    4  Flood (Over 3cm of water)                 0.140961
    


    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_41_1.png)
    


#### Speed limit


```python
sns.set_style('white')
sns.color_palette("hls")
speed= dataset['Speed_limit'].value_counts(normalize= True) *100
speed_df = speed.to_frame().reset_index()

g=sns.barplot(data = speed_df,x= 'index' ,y = 'Speed_limit')
g.set(xlabel='Speed Limit', ylabel='Frequency (%)')
sns.despine(left= True)

print(speed_df)
```

       index  Speed_limit
    0     30    64.140004
    1     60    15.978760
    2     40     8.183073
    3     70     7.321844
    4     50     3.255924
    5     20     1.118760
    6     10     0.000954
    7     15     0.000681
    


    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_43_1.png)
    


Data Also revealed
- A common misconception may be that most traffic accidents due to high speed limit.The majority (64%) reported accidents took place at low speeds of just 30mph.
- Accidents (75%) occured on single-carriageway roads.
- Surprising finding is that bad weather conditions are not necessaily cause of more traffic accidents. 80% of the time happened on when the weather was 'fine'(80%) and 12 % of accidents taking place on rainy days. Less than 2% happening in other conditions like fog/mist/snowy/windy.
- Majority (73%) accidents occured during day time , with 69 % times road conditions were dry.

### Exploring Casualties


```python
print('This dataset records {} number of traffic casualties between 2005 & 2014 (2008 not included).'.format(dataset_w_month['Number_of_Casualties'].sum()))
```

    This dataset records 1983085 number of traffic casualties between 2005 & 2014 (2008 not included).
    

Highest Casualty 94 was recorded on Oct 2014.


```python
dataset_w_month[dataset_w_month['Number_of_Casualties'] == dataset_w_month['Number_of_Casualties'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Police_Force</th>
      <th>Accident_Severity</th>
      <th>Number_of_Vehicles</th>
      <th>Number_of_Casualties</th>
      <th>Date</th>
      <th>Time</th>
      <th>Road_Type</th>
      <th>Speed_limit</th>
      <th>Pedestrian_Crossing-Physical_Facilities</th>
      <th>Light_Conditions</th>
      <th>Weather_Conditions</th>
      <th>Road_Surface_Conditions</th>
      <th>Year</th>
      <th>Hour_day</th>
      <th>Day_of_week</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>410419</th>
      <td>41</td>
      <td>2</td>
      <td>2</td>
      <td>93</td>
      <td>2014-10-20</td>
      <td>2022-12-24 08:22:00</td>
      <td>Single carriageway</td>
      <td>60</td>
      <td>No physical crossing within 50 meters</td>
      <td>Daylight: Street light present</td>
      <td>Fine without high winds</td>
      <td>Dry</td>
      <td>2014</td>
      <td>8.0</td>
      <td>Monday</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



Majority (93%) of accidents show 1-2 casualties. 


```python
count_casualties = dataset_w_month['Number_of_Casualties'].value_counts(normalize= True)*100
count_casualties = count_casualties.reset_index()
count_casualties.rename(columns={'index':'No of Casualties','Number_of_Casualties':'Frequency(%)'},inplace= True)
print(count_casualties[:5])
print(count_casualties[-5:])

```

       No of Casualties  Frequency(%)
    0                 1     76.692720
    1                 2     16.062900
    2                 3      4.580504
    3                 4      1.661618
    4                 5      0.612691
        No of Casualties  Frequency(%)
    42                33      0.000068
    43                70      0.000068
    44                46      0.000068
    45                54      0.000068
    46                93      0.000068
    


```python
fig, axes = plt.subplots(2,2,figsize=(12,8),sharey=True)
col_names = ['Year','Month','Day_of_week','Hour_day']
sns.set_style('darkgrid')
for ax,col_name in zip(axes.flatten(),col_names):
    title = col_name.replace('_',' ')
    df = dataset_w_month.groupby(col_name).mean()['Number_of_Casualties']
    g=sns.lineplot(data=df,x=df.index, y = df.values,ax=ax)
    g.set_ylim([1.25,1.5])
    g.set(xlabel='',ylabel='Avg Casualties')
    g.set_title('Avg Casualties by '+ title,y=0.90)
# sns.despine(left= True)
plt.tight_layout()
```


    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_51_0.png)
    



```python
dataset_heatmap_causualties = dataset_w_month.copy()[['Day_of_week','Hour_day','Number_of_Casualties']]
dataset_heatmap_causualties  = dataset_heatmap_causualties.groupby(['Day_of_week','Hour_day']).mean()['Number_of_Casualties'].unstack()
g= sns.heatmap(dataset_heatmap_causualties,cmap='rocket_r')
g.set_title('Average Casualties by Day & Hour',y=1.05)
g.set(xlabel =' Hour ', ylabel='Day')
```




    [Text(0.5, 12.5, ' Hour '), Text(30.453125, 0.5, 'Day')]




    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_52_1.png)
    


Interesting Findings

- Average Casualties gradually decreased each year,displaying downward trend.
- Warmer months August and July were months with higher average casualties. Colds Month of the year recorded low Average Casualties.
- In contrast to our Accidents findings,High Casualties were records on Sunday and Saturday regardless of day/night time. While during weekdays, High Casualties occured during night time between 8PM - 3AM.



### Exploring Accident severity & Avg Casualties


```python
severity_count = dataset['Accident_Severity'].value_counts(normalize=True)*100
severity_count = severity_count.reset_index()
severity_count.rename(columns={'index':'accident severity','Accident_Severity':'frequency (%)'},inplace =True)
severity_count= severity_count.sort_values('frequency (%)',ascending = True)
display(severity_count)
severity_casualties =  dataset.groupby('Accident_Severity')['Number_of_Casualties'].mean()
severity_casualties = severity_casualties.reset_index()
severity_casualties.rename(columns={'Accident_Severity':'accident severity','Number_of_Casualties':'avg casualties'},inplace= True)
display(severity_casualties)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident severity</th>
      <th>frequency (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.296102</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>13.536510</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>85.167388</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accident severity</th>
      <th>avg casualties</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.894449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.455848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.326150</td>
    </tr>
  </tbody>
</table>
</div>



```python
sns.set_style('white')
fig,ax = plt.subplots(figsize=(4,4))
ax.bar(x=severity_count['accident severity'] , height= severity_count['frequency (%)'],color='#0c0750')
ax.set_xlabel('Accident Severity')
ax.set_ylabel('Frequency (%)')
ax2 = ax.twinx()
ax2.plot(severity_casualties['accident severity'] ,severity_casualties['avg casualties'],color='#26df3d')
ax2.set_ylabel('Avg Casualties')
```




    Text(0, 0.5, 'Avg Casualties')




    
![png](Traffic%20Accident%20Analysis_files/Traffic%20Accident%20Analysis_56_1.png)
    



```python
dataset_w_month.corr()['Number_of_Casualties']
```




    Police_Force            0.007100
    Number_of_Vehicles      0.236699
    Number_of_Casualties    1.000000
    Speed_limit             0.139754
    Year                   -0.015479
    Hour_day                0.025223
    Month                   0.001318
    Name: Number_of_Casualties, dtype: float64



Findings:
- Though Accident severity(level 3) occured 85% of times, average casualties is lower than of (level 1)
- however there isnt strong correlation between accident severity and Casualties to conclude they are affect eachother negatively or positively.
