{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2a215672",
   "metadata": {
    "_cell_guid": "72c8a032-ad51-49f9-9113-771053c6729a",
    "_uuid": "4f8f99f3-b475-4711-a516-f2f2d015be5f",
    "papermill": {
     "duration": 0.014138,
     "end_time": "2022-10-10T12:32:29.630693",
     "exception": false,
     "start_time": "2022-10-10T12:32:29.616555",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Explanatory Data Analysis with Python and Pandas"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2729542",
   "metadata": {
    "_cell_guid": "6f458758-6da9-4fb1-9229-194a6305e3be",
    "_uuid": "0c488257-5bcc-4523-8c4d-257a79363508",
    "papermill": {
     "duration": 0.012257,
     "end_time": "2022-10-10T12:32:29.655639",
     "exception": false,
     "start_time": "2022-10-10T12:32:29.643382",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "896424d2",
   "metadata": {
    "_cell_guid": "5d18d33f-1fc6-4cbe-b202-76008563fff4",
    "_uuid": "e6e14686-0b34-429a-bebf-59daf21ece8f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:29.684680Z",
     "iopub.status.busy": "2022-10-10T12:32:29.683926Z",
     "iopub.status.idle": "2022-10-10T12:32:30.894104Z",
     "shell.execute_reply": "2022-10-10T12:32:30.892789Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.227388,
     "end_time": "2022-10-10T12:32:30.897208",
     "exception": false,
     "start_time": "2022-10-10T12:32:29.669820",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import matplotlib\n",
    "\n",
    "%matplotlib inline\n",
    "matplotlib.rcParams['figure.figsize'] = (12,8) # use to increase the size of plots"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "932f8b49",
   "metadata": {
    "_cell_guid": "055298ed-9125-45e3-88be-fe9218e793ab",
    "_uuid": "a012f81f-9e20-4440-b7a4-a23cf61889ad",
    "papermill": {
     "duration": 0.013246,
     "end_time": "2022-10-10T12:32:30.923308",
     "exception": false,
     "start_time": "2022-10-10T12:32:30.910062",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "link to data source: https://www.kaggle.com/datasets/pranavuikey/black-friday-sales-eda\n",
    "### context\n",
    " A retail company \"ABC Private Limited\" wants to understand the customer purchase behaviour (specifically, purchase amount)\n",
    "against various product from last month\n",
    " Now they want to bulid a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers againsts different products"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1861012",
   "metadata": {
    "_cell_guid": "04f76b05-9df5-4417-991d-9cdb4edee9bd",
    "_uuid": "390fd623-e89f-45be-b88b-3c4562aaaf96",
    "papermill": {
     "duration": 0.012218,
     "end_time": "2022-10-10T12:32:30.948130",
     "exception": false,
     "start_time": "2022-10-10T12:32:30.935912",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Intial Data Exploration\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c303d6b2",
   "metadata": {
    "_cell_guid": "bf328a01-dc07-4a12-b68b-f61e67a03e03",
    "_uuid": "ccc6794b-0327-4975-a835-3730abc62ad4",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:30.975058Z",
     "iopub.status.busy": "2022-10-10T12:32:30.974632Z",
     "iopub.status.idle": "2022-10-10T12:32:31.980020Z",
     "shell.execute_reply": "2022-10-10T12:32:31.978817Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.022244,
     "end_time": "2022-10-10T12:32:31.982797",
     "exception": false,
     "start_time": "2022-10-10T12:32:30.960553",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# import the dataset\n",
    "\n",
    "df = pd.read_csv(r'/kaggle/input/black-friday-sales-eda/train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "39a44a07",
   "metadata": {
    "_cell_guid": "38cb9d6f-57a6-4d48-a406-4424bf14b288",
    "_uuid": "04432f71-5948-4d76-89ea-e3b4df207843",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.010116Z",
     "iopub.status.busy": "2022-10-10T12:32:32.009668Z",
     "iopub.status.idle": "2022-10-10T12:32:32.036147Z",
     "shell.execute_reply": "2022-10-10T12:32:32.034838Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.044484,
     "end_time": "2022-10-10T12:32:32.039651",
     "exception": false,
     "start_time": "2022-10-10T12:32:31.995167",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN      8370  \n",
       "1                 6.0                14.0     15200  \n",
       "2                 NaN                 NaN      1422  \n",
       "3                14.0                 NaN      1057  \n",
       "4                 NaN                 NaN      7969  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9f1c8d6e",
   "metadata": {
    "_cell_guid": "a4d06767-b729-4070-af36-deb79184f43d",
    "_uuid": "73d17d05-ac87-46c7-b78e-c9d454c4d8d6",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.067932Z",
     "iopub.status.busy": "2022-10-10T12:32:32.067485Z",
     "iopub.status.idle": "2022-10-10T12:32:32.084357Z",
     "shell.execute_reply": "2022-10-10T12:32:32.083169Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.034292,
     "end_time": "2022-10-10T12:32:32.086841",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.052549",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>550063</th>\n",
       "      <td>1006033</td>\n",
       "      <td>P00372445</td>\n",
       "      <td>M</td>\n",
       "      <td>51-55</td>\n",
       "      <td>13</td>\n",
       "      <td>B</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>550064</th>\n",
       "      <td>1006035</td>\n",
       "      <td>P00375436</td>\n",
       "      <td>F</td>\n",
       "      <td>26-35</td>\n",
       "      <td>1</td>\n",
       "      <td>C</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>371</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>550065</th>\n",
       "      <td>1006036</td>\n",
       "      <td>P00375436</td>\n",
       "      <td>F</td>\n",
       "      <td>26-35</td>\n",
       "      <td>15</td>\n",
       "      <td>B</td>\n",
       "      <td>4+</td>\n",
       "      <td>1</td>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>550066</th>\n",
       "      <td>1006038</td>\n",
       "      <td>P00375436</td>\n",
       "      <td>F</td>\n",
       "      <td>55+</td>\n",
       "      <td>1</td>\n",
       "      <td>C</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>365</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>550067</th>\n",
       "      <td>1006039</td>\n",
       "      <td>P00371644</td>\n",
       "      <td>F</td>\n",
       "      <td>46-50</td>\n",
       "      <td>0</td>\n",
       "      <td>B</td>\n",
       "      <td>4+</td>\n",
       "      <td>1</td>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>490</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        User_ID Product_ID Gender    Age  Occupation City_Category  \\\n",
       "550063  1006033  P00372445      M  51-55          13             B   \n",
       "550064  1006035  P00375436      F  26-35           1             C   \n",
       "550065  1006036  P00375436      F  26-35          15             B   \n",
       "550066  1006038  P00375436      F    55+           1             C   \n",
       "550067  1006039  P00371644      F  46-50           0             B   \n",
       "\n",
       "       Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "550063                          1               1                  20   \n",
       "550064                          3               0                  20   \n",
       "550065                         4+               1                  20   \n",
       "550066                          2               0                  20   \n",
       "550067                         4+               1                  20   \n",
       "\n",
       "        Product_Category_2  Product_Category_3  Purchase  \n",
       "550063                 NaN                 NaN       368  \n",
       "550064                 NaN                 NaN       371  \n",
       "550065                 NaN                 NaN       137  \n",
       "550066                 NaN                 NaN       365  \n",
       "550067                 NaN                 NaN       490  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the last 5 rows \n",
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b48f3184",
   "metadata": {
    "_cell_guid": "39db7d6a-b5b3-4653-9292-e3845f8761e3",
    "_uuid": "1dc7336b-2460-4a98-bc5b-a68c8fdbff12",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.114718Z",
     "iopub.status.busy": "2022-10-10T12:32:32.114322Z",
     "iopub.status.idle": "2022-10-10T12:32:32.121406Z",
     "shell.execute_reply": "2022-10-10T12:32:32.120328Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.023682,
     "end_time": "2022-10-10T12:32:32.123710",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.100028",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',\n",
       "       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',\n",
       "       'Product_Category_2', 'Product_Category_3', 'Purchase'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the columns names\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "45937aba",
   "metadata": {
    "_cell_guid": "6abb2cee-e087-4414-846e-ac5292c9aee8",
    "_uuid": "d94b9af7-5a17-46eb-be79-550c9e56b137",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.151937Z",
     "iopub.status.busy": "2022-10-10T12:32:32.151446Z",
     "iopub.status.idle": "2022-10-10T12:32:32.160100Z",
     "shell.execute_reply": "2022-10-10T12:32:32.158890Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.025621,
     "end_time": "2022-10-10T12:32:32.162471",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.136850",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                         int64\n",
       "Product_ID                     object\n",
       "Gender                         object\n",
       "Age                            object\n",
       "Occupation                      int64\n",
       "City_Category                  object\n",
       "Stay_In_Current_City_Years     object\n",
       "Marital_Status                  int64\n",
       "Product_Category_1              int64\n",
       "Product_Category_2            float64\n",
       "Product_Category_3            float64\n",
       "Purchase                        int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the data types of the column in a dataset\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7224842",
   "metadata": {
    "_cell_guid": "52bec7d8-bb00-4141-85d8-39c5a605ec56",
    "_uuid": "a96eff43-65ad-4e9c-be7e-0df6f318108f",
    "papermill": {
     "duration": 0.012708,
     "end_time": "2022-10-10T12:32:32.188117",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.175409",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### summary of numeric columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cd183d01",
   "metadata": {
    "_cell_guid": "960c2703-11e1-4ccf-8c16-9eb08c0dc668",
    "_uuid": "da1d2ccd-2e09-4693-b08c-68edc247c2b5",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.217742Z",
     "iopub.status.busy": "2022-10-10T12:32:32.217348Z",
     "iopub.status.idle": "2022-10-10T12:32:32.421284Z",
     "shell.execute_reply": "2022-10-10T12:32:32.420323Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.22138,
     "end_time": "2022-10-10T12:32:32.423777",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.202397",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5.500680e+05</td>\n",
       "      <td>550068.000000</td>\n",
       "      <td>550068.000000</td>\n",
       "      <td>550068.000000</td>\n",
       "      <td>376430.000000</td>\n",
       "      <td>166821.000000</td>\n",
       "      <td>550068.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.003029e+06</td>\n",
       "      <td>8.076707</td>\n",
       "      <td>0.409653</td>\n",
       "      <td>5.404270</td>\n",
       "      <td>9.842329</td>\n",
       "      <td>12.668243</td>\n",
       "      <td>9263.968713</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.727592e+03</td>\n",
       "      <td>6.522660</td>\n",
       "      <td>0.491770</td>\n",
       "      <td>3.936211</td>\n",
       "      <td>5.086590</td>\n",
       "      <td>4.125338</td>\n",
       "      <td>5023.065394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000001e+06</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.001516e+06</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>5823.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.003077e+06</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>8047.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.004478e+06</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>12054.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.006040e+06</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>23961.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            User_ID     Occupation  Marital_Status  Product_Category_1  \\\n",
       "count  5.500680e+05  550068.000000   550068.000000       550068.000000   \n",
       "mean   1.003029e+06       8.076707        0.409653            5.404270   \n",
       "std    1.727592e+03       6.522660        0.491770            3.936211   \n",
       "min    1.000001e+06       0.000000        0.000000            1.000000   \n",
       "25%    1.001516e+06       2.000000        0.000000            1.000000   \n",
       "50%    1.003077e+06       7.000000        0.000000            5.000000   \n",
       "75%    1.004478e+06      14.000000        1.000000            8.000000   \n",
       "max    1.006040e+06      20.000000        1.000000           20.000000   \n",
       "\n",
       "       Product_Category_2  Product_Category_3       Purchase  \n",
       "count       376430.000000       166821.000000  550068.000000  \n",
       "mean             9.842329           12.668243    9263.968713  \n",
       "std              5.086590            4.125338    5023.065394  \n",
       "min              2.000000            3.000000      12.000000  \n",
       "25%              5.000000            9.000000    5823.000000  \n",
       "50%              9.000000           14.000000    8047.000000  \n",
       "75%             15.000000           16.000000   12054.000000  \n",
       "max             18.000000           18.000000   23961.000000  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7fd2b9c0",
   "metadata": {
    "_cell_guid": "de6dbf1c-add9-44af-b2d3-d89b0becb753",
    "_uuid": "2381ed34-c9ed-4e3f-bf55-c9514dfc91a3",
    "papermill": {
     "duration": 0.013321,
     "end_time": "2022-10-10T12:32:32.450641",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.437320",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Checking missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f42a730f",
   "metadata": {
    "_cell_guid": "0ed40b16-c8b1-4b13-89f6-dbd37cb747ff",
    "_uuid": "255c2d23-951b-4f36-9777-083bb14b30d2",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.479944Z",
     "iopub.status.busy": "2022-10-10T12:32:32.479237Z",
     "iopub.status.idle": "2022-10-10T12:32:32.606139Z",
     "shell.execute_reply": "2022-10-10T12:32:32.605245Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.144196,
     "end_time": "2022-10-10T12:32:32.608492",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.464296",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                            0\n",
       "Product_ID                         0\n",
       "Gender                             0\n",
       "Age                                0\n",
       "Occupation                         0\n",
       "City_Category                      0\n",
       "Stay_In_Current_City_Years         0\n",
       "Marital_Status                     0\n",
       "Product_Category_1                 0\n",
       "Product_Category_2            173638\n",
       "Product_Category_3            383247\n",
       "Purchase                           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea5ec975",
   "metadata": {
    "_cell_guid": "2143882b-63c3-4e4c-87ca-c905ba078d53",
    "_uuid": "6e181afe-81d7-40ff-80f9-bc7cfc21cf23",
    "papermill": {
     "duration": 0.01318,
     "end_time": "2022-10-10T12:32:32.635383",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.622203",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "In Product_Category_2 and Product_Category_3 column has missing values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a893ac55",
   "metadata": {
    "_cell_guid": "648e2717-9b43-40bf-96d0-8834df3f7138",
    "_uuid": "a38e7988-eabc-406f-b76d-bbb69bcd1cec",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.664597Z",
     "iopub.status.busy": "2022-10-10T12:32:32.663976Z",
     "iopub.status.idle": "2022-10-10T12:32:32.795339Z",
     "shell.execute_reply": "2022-10-10T12:32:32.794065Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.149341,
     "end_time": "2022-10-10T12:32:32.798333",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.648992",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                        0.00\n",
       "Product_ID                     0.00\n",
       "Gender                         0.00\n",
       "Age                            0.00\n",
       "Occupation                     0.00\n",
       "City_Category                  0.00\n",
       "Stay_In_Current_City_Years     0.00\n",
       "Marital_Status                 0.00\n",
       "Product_Category_1             0.00\n",
       "Product_Category_2            31.57\n",
       "Product_Category_3            69.67\n",
       "Purchase                       0.00\n",
       "dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "round((df.isna().sum()/len(df)) * 100, 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e720c30c",
   "metadata": {
    "_cell_guid": "b47b3eba-620e-40ce-ace7-5a715d2f95d4",
    "_uuid": "b78be311-ce13-4cf8-9751-4618016af483",
    "papermill": {
     "duration": 0.01317,
     "end_time": "2022-10-10T12:32:32.825416",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.812246",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Those two columns have 31 and 70 percent of the have null values respectively."
   ]
  },
  {
   "cell_type": "raw",
   "id": "ea7aebe0",
   "metadata": {
    "_cell_guid": "53c7cc2f-bf6a-4cf2-a531-315eaef8050c",
    "_uuid": "8a7d20da-0ef9-4702-9cce-f38cc69a3130",
    "papermill": {
     "duration": 0.01328,
     "end_time": "2022-10-10T12:32:32.852276",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.838996",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "we will replace the values with zero of Product Catergory column 2 and 3."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0b1000ba",
   "metadata": {
    "_cell_guid": "af2e2a5f-df31-43b8-a523-46c73e6d4616",
    "_uuid": "a1a5d3c9-a0ec-4c61-8fe5-491b0848e50e",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:32.882135Z",
     "iopub.status.busy": "2022-10-10T12:32:32.880981Z",
     "iopub.status.idle": "2022-10-10T12:32:33.008547Z",
     "shell.execute_reply": "2022-10-10T12:32:33.007671Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.145539,
     "end_time": "2022-10-10T12:32:33.011416",
     "exception": false,
     "start_time": "2022-10-10T12:32:32.865877",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# used fillna to replace null values with zero and inplace option to make changes permanent.\n",
    "df.fillna(0, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cf5739ce",
   "metadata": {
    "_cell_guid": "f3c74d7f-6a56-4de8-bed1-86fb360fecf4",
    "_uuid": "f887d24e-918c-4d8b-a1af-d8567a754e1a",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:33.040802Z",
     "iopub.status.busy": "2022-10-10T12:32:33.040345Z",
     "iopub.status.idle": "2022-10-10T12:32:33.172400Z",
     "shell.execute_reply": "2022-10-10T12:32:33.171262Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.149961,
     "end_time": "2022-10-10T12:32:33.175167",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.025206",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                       0.0\n",
       "Product_ID                    0.0\n",
       "Gender                        0.0\n",
       "Age                           0.0\n",
       "Occupation                    0.0\n",
       "City_Category                 0.0\n",
       "Stay_In_Current_City_Years    0.0\n",
       "Marital_Status                0.0\n",
       "Product_Category_1            0.0\n",
       "Product_Category_2            0.0\n",
       "Product_Category_3            0.0\n",
       "Purchase                      0.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to verify the datase has no null values.\n",
    "round((df.isna().sum()/len(df)) * 100, 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b5238ae",
   "metadata": {
    "_cell_guid": "f658fbb7-0993-493e-9733-366b17d9abf7",
    "_uuid": "e5a100aa-3b69-4f11-92d0-1bec5d2b5aae",
    "papermill": {
     "duration": 0.014073,
     "end_time": "2022-10-10T12:32:33.203048",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.188975",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## check for duplicates values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "54894bad",
   "metadata": {
    "_cell_guid": "acca4374-ffaa-421d-8bdf-42da21373668",
    "_uuid": "2f9f1405-8d95-4656-828a-7f614f7fd75a",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:33.233288Z",
     "iopub.status.busy": "2022-10-10T12:32:33.232866Z",
     "iopub.status.idle": "2022-10-10T12:32:33.625104Z",
     "shell.execute_reply": "2022-10-10T12:32:33.623895Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.410497,
     "end_time": "2022-10-10T12:32:33.628214",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.217717",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db3992ca",
   "metadata": {
    "_cell_guid": "94adaa87-b09b-4c40-aca1-2d92997b17f1",
    "_uuid": "573a36e3-b33e-4119-9eb6-b961ada8c554",
    "papermill": {
     "duration": 0.013896,
     "end_time": "2022-10-10T12:32:33.656307",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.642411",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "There is no duplicate rows in dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "26a14ffb",
   "metadata": {
    "_cell_guid": "bd876189-aa97-4972-88c4-fd67400e0a7e",
    "_uuid": "0f4c643f-65c6-4d13-b89e-c5960d64c727",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:33.687200Z",
     "iopub.status.busy": "2022-10-10T12:32:33.685680Z",
     "iopub.status.idle": "2022-10-10T12:32:33.703868Z",
     "shell.execute_reply": "2022-10-10T12:32:33.702685Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.036082,
     "end_time": "2022-10-10T12:32:33.706445",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.670363",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1422</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7969</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 0.0                 0.0      8370  \n",
       "1                 6.0                14.0     15200  \n",
       "2                 0.0                 0.0      1422  \n",
       "3                14.0                 0.0      1057  \n",
       "4                 0.0                 0.0      7969  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe687bb7",
   "metadata": {
    "_cell_guid": "36c1e892-be4e-4588-a0b7-114156ee4b32",
    "_uuid": "4e9940f0-e6cc-4393-9a18-b689c65d2962",
    "papermill": {
     "duration": 0.013759,
     "end_time": "2022-10-10T12:32:33.734405",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.720646",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Univariate Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7ace048",
   "metadata": {
    "_cell_guid": "f75c5c3d-00a7-46b4-af8f-2e02cc9e6783",
    "_uuid": "4e8cfb0a-6c81-4e7e-98e3-19f7e229368f",
    "papermill": {
     "duration": 0.013953,
     "end_time": "2022-10-10T12:32:33.763149",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.749196",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Question:\n",
    "    which age group customer is largest number of customer?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "353d2f5c",
   "metadata": {
    "_cell_guid": "d9c5fa93-2814-41a6-8825-bfc0923162b8",
    "_uuid": "a203adc9-a061-4837-b04a-6fe7fe298709",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:33.793682Z",
     "iopub.status.busy": "2022-10-10T12:32:33.792815Z",
     "iopub.status.idle": "2022-10-10T12:32:35.231621Z",
     "shell.execute_reply": "2022-10-10T12:32:35.230286Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 1.45732,
     "end_time": "2022-10-10T12:32:35.234541",
     "exception": false,
     "start_time": "2022-10-10T12:32:33.777221",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f65ba70edd0>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df['Age'].sort_values())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6b2ec6f",
   "metadata": {
    "_cell_guid": "bff65321-2d9f-4d5f-9318-d81a87afcf4c",
    "_uuid": "4fb0a6b6-25b8-47f7-9ec8-86cdabb68c3c",
    "papermill": {
     "duration": 0.014109,
     "end_time": "2022-10-10T12:32:35.263384",
     "exception": false,
     "start_time": "2022-10-10T12:32:35.249275",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The products are highly purchased by age group between 18 to 45. In that top is 26-35 spending the products."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1cfd3d41",
   "metadata": {
    "_cell_guid": "ded740e5-0cb8-4c70-b970-93d1561c89a1",
    "_uuid": "eb7e9a2e-665b-45cc-8590-8d2900809bee",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:35.295290Z",
     "iopub.status.busy": "2022-10-10T12:32:35.294520Z",
     "iopub.status.idle": "2022-10-10T12:32:35.767819Z",
     "shell.execute_reply": "2022-10-10T12:32:35.766940Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.492396,
     "end_time": "2022-10-10T12:32:35.770256",
     "exception": false,
     "start_time": "2022-10-10T12:32:35.277860",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = df['Gender'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6db9b8b1",
   "metadata": {
    "_cell_guid": "1ce45264-cb5c-4409-a7d3-1f57b01258c8",
    "_uuid": "1572e08d-e5bc-495a-8d87-bd584d6ef650",
    "papermill": {
     "duration": 0.014526,
     "end_time": "2022-10-10T12:32:35.799539",
     "exception": false,
     "start_time": "2022-10-10T12:32:35.785013",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The most customer are Male compare to Female."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "2047b449",
   "metadata": {
    "_cell_guid": "d5030355-321e-41dd-a39d-e64b425799ab",
    "_uuid": "334c9fb7-e297-4336-85b4-f7b62a132117",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:35.831533Z",
     "iopub.status.busy": "2022-10-10T12:32:35.830736Z",
     "iopub.status.idle": "2022-10-10T12:32:36.291588Z",
     "shell.execute_reply": "2022-10-10T12:32:36.289984Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.479888,
     "end_time": "2022-10-10T12:32:36.294321",
     "exception": false,
     "start_time": "2022-10-10T12:32:35.814433",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='City_Category', ylabel='count'>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x =df['City_Category'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "092b87f2",
   "metadata": {
    "_cell_guid": "3db84270-9f28-4e98-b558-557008d57451",
    "_uuid": "3b70bb63-1911-41fd-9f0f-36d718922a68",
    "papermill": {
     "duration": 0.014977,
     "end_time": "2022-10-10T12:32:36.324916",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.309939",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Above barchart shows City B is highest amount customer are purchasing the product followed by City C and A."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9b6baf14",
   "metadata": {
    "_cell_guid": "883db047-9be8-447a-8878-1ab53c6f7fe4",
    "_uuid": "4cb360db-e5f6-42b5-946f-ee125567a60b",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:36.358772Z",
     "iopub.status.busy": "2022-10-10T12:32:36.357950Z",
     "iopub.status.idle": "2022-10-10T12:32:36.382258Z",
     "shell.execute_reply": "2022-10-10T12:32:36.381109Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.044718,
     "end_time": "2022-10-10T12:32:36.384878",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.340160",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# change the Martial status as Numeric to Married for 1 and Unmarried for 0\n",
    "df['Marital_Status'] = df[\"Marital_Status\"].map({0: \"unmarried\", 1:\"married\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0a0ea066",
   "metadata": {
    "_cell_guid": "e0e36815-074c-4265-91e4-45938d390ba2",
    "_uuid": "ed7e868f-dbca-4f86-a2ec-37fda810f6d5",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:36.417319Z",
     "iopub.status.busy": "2022-10-10T12:32:36.416938Z",
     "iopub.status.idle": "2022-10-10T12:32:36.881799Z",
     "shell.execute_reply": "2022-10-10T12:32:36.880856Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.484015,
     "end_time": "2022-10-10T12:32:36.884176",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.400161",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Marital_Status', ylabel='count'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x = df[\"Marital_Status\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9aedc033",
   "metadata": {
    "_cell_guid": "bd3b5023-a507-467b-bdd2-929e3313231b",
    "_uuid": "2e7be549-cdfd-4ab5-8b87-8add3e46f67e",
    "papermill": {
     "duration": 0.015103,
     "end_time": "2022-10-10T12:32:36.915036",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.899933",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Above Bar char clear shows Unmarried customers are spending more than Married customers."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "796d5d83",
   "metadata": {
    "_cell_guid": "7201fd48-c03a-4f34-9d55-8f8631cd38d9",
    "_uuid": "73d50d29-0168-439a-b598-481b31fad7f8",
    "papermill": {
     "duration": 0.015128,
     "end_time": "2022-10-10T12:32:36.945603",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.930475",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Bivariate Analysis\n",
    "Is there any relationship between age group vs purchase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "84fed7b2",
   "metadata": {
    "_cell_guid": "62d76cc9-cfd0-4210-bc5b-94edaf777b56",
    "_uuid": "6328eb56-25e6-4311-ae24-edd2be2a3181",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:36.978838Z",
     "iopub.status.busy": "2022-10-10T12:32:36.978110Z",
     "iopub.status.idle": "2022-10-10T12:32:37.244745Z",
     "shell.execute_reply": "2022-10-10T12:32:37.243577Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.286461,
     "end_time": "2022-10-10T12:32:37.247485",
     "exception": false,
     "start_time": "2022-10-10T12:32:36.961024",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0-17</th>\n",
       "      <td>15143112813</td>\n",
       "      <td>132309</td>\n",
       "      <td>76775</td>\n",
       "      <td>96155.0</td>\n",
       "      <td>57725.0</td>\n",
       "      <td>134913183</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18-25</th>\n",
       "      <td>99939196632</td>\n",
       "      <td>671348</td>\n",
       "      <td>509371</td>\n",
       "      <td>654936.0</td>\n",
       "      <td>388041.0</td>\n",
       "      <td>913848675</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26-35</th>\n",
       "      <td>220270500414</td>\n",
       "      <td>1734073</td>\n",
       "      <td>1166945</td>\n",
       "      <td>1473278.0</td>\n",
       "      <td>846624.0</td>\n",
       "      <td>2031770578</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36-45</th>\n",
       "      <td>110350311441</td>\n",
       "      <td>972225</td>\n",
       "      <td>604438</td>\n",
       "      <td>750081.0</td>\n",
       "      <td>424412.0</td>\n",
       "      <td>1026569884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46-50</th>\n",
       "      <td>45846804203</td>\n",
       "      <td>389239</td>\n",
       "      <td>262424</td>\n",
       "      <td>315572.0</td>\n",
       "      <td>173059.0</td>\n",
       "      <td>420843403</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51-55</th>\n",
       "      <td>38615925320</td>\n",
       "      <td>339198</td>\n",
       "      <td>222313</td>\n",
       "      <td>267570.0</td>\n",
       "      <td>146334.0</td>\n",
       "      <td>367099644</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55+</th>\n",
       "      <td>21568218459</td>\n",
       "      <td>204346</td>\n",
       "      <td>130450</td>\n",
       "      <td>147356.0</td>\n",
       "      <td>77134.0</td>\n",
       "      <td>200767375</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            User_ID  Occupation  Product_Category_1  Product_Category_2  \\\n",
       "Age                                                                       \n",
       "0-17    15143112813      132309               76775             96155.0   \n",
       "18-25   99939196632      671348              509371            654936.0   \n",
       "26-35  220270500414     1734073             1166945           1473278.0   \n",
       "36-45  110350311441      972225              604438            750081.0   \n",
       "46-50   45846804203      389239              262424            315572.0   \n",
       "51-55   38615925320      339198              222313            267570.0   \n",
       "55+     21568218459      204346              130450            147356.0   \n",
       "\n",
       "       Product_Category_3    Purchase  \n",
       "Age                                    \n",
       "0-17              57725.0   134913183  \n",
       "18-25            388041.0   913848675  \n",
       "26-35            846624.0  2031770578  \n",
       "36-45            424412.0  1026569884  \n",
       "46-50            173059.0   420843403  \n",
       "51-55            146334.0   367099644  \n",
       "55+               77134.0   200767375  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "age_grp = df.groupby(df.Age).sum()\n",
    "age_grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "5ed6d7b6",
   "metadata": {
    "_cell_guid": "42cc2703-364b-428d-aa21-bf88880eebec",
    "_uuid": "5a264622-9006-4608-9cdc-0ae6ad3f462c",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:37.281441Z",
     "iopub.status.busy": "2022-10-10T12:32:37.281052Z",
     "iopub.status.idle": "2022-10-10T12:32:37.489633Z",
     "shell.execute_reply": "2022-10-10T12:32:37.488420Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.22848,
     "end_time": "2022-10-10T12:32:37.492161",
     "exception": false,
     "start_time": "2022-10-10T12:32:37.263681",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Age', ylabel='Purchase'>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x = age_grp.index,\n",
    "           y = age_grp.Purchase)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bec87ae",
   "metadata": {
    "_cell_guid": "2d399b1c-48c4-49f0-b7c8-070d6affb954",
    "_uuid": "aa8f0e04-994a-436e-ab51-a0138cc99533",
    "papermill": {
     "duration": 0.015761,
     "end_time": "2022-10-10T12:32:37.524021",
     "exception": false,
     "start_time": "2022-10-10T12:32:37.508260",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Highest number of purchase by the Age group 26-35 due to number of customer between age group 26-35 is also high."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "245c03e7",
   "metadata": {
    "_cell_guid": "a73afae9-94df-4308-8a4b-5a143b51662f",
    "_uuid": "2dbd4a0d-618a-4200-83e5-52df87e67510",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:37.559057Z",
     "iopub.status.busy": "2022-10-10T12:32:37.557617Z",
     "iopub.status.idle": "2022-10-10T12:32:37.672412Z",
     "shell.execute_reply": "2022-10-10T12:32:37.671212Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.134995,
     "end_time": "2022-10-10T12:32:37.675238",
     "exception": false,
     "start_time": "2022-10-10T12:32:37.540243",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Occupation</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>72521952247</td>\n",
       "      <td>375044</td>\n",
       "      <td>478959.0</td>\n",
       "      <td>279431.0</td>\n",
       "      <td>666244484</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>69847008662</td>\n",
       "      <td>378374</td>\n",
       "      <td>464793.0</td>\n",
       "      <td>263929.0</td>\n",
       "      <td>635406958</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>59315918984</td>\n",
       "      <td>320171</td>\n",
       "      <td>402950.0</td>\n",
       "      <td>228545.0</td>\n",
       "      <td>557371587</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>47584821210</td>\n",
       "      <td>269249</td>\n",
       "      <td>319095.0</td>\n",
       "      <td>172085.0</td>\n",
       "      <td>424614144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>40164327766</td>\n",
       "      <td>202462</td>\n",
       "      <td>278263.0</td>\n",
       "      <td>180304.0</td>\n",
       "      <td>393281453</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>31270073799</td>\n",
       "      <td>161372</td>\n",
       "      <td>217225.0</td>\n",
       "      <td>133868.0</td>\n",
       "      <td>305449446</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>33658243797</td>\n",
       "      <td>190307</td>\n",
       "      <td>219662.0</td>\n",
       "      <td>116538.0</td>\n",
       "      <td>296570442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>27396517676</td>\n",
       "      <td>144235</td>\n",
       "      <td>189332.0</td>\n",
       "      <td>107053.0</td>\n",
       "      <td>259454692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>25449031066</td>\n",
       "      <td>140035</td>\n",
       "      <td>173079.0</td>\n",
       "      <td>94139.0</td>\n",
       "      <td>238346955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>26672083675</td>\n",
       "      <td>150096</td>\n",
       "      <td>176408.0</td>\n",
       "      <td>95243.0</td>\n",
       "      <td>238028583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>20421422468</td>\n",
       "      <td>114201</td>\n",
       "      <td>135849.0</td>\n",
       "      <td>73840.0</td>\n",
       "      <td>188416784</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>17694824027</td>\n",
       "      <td>98740</td>\n",
       "      <td>119452.0</td>\n",
       "      <td>63335.0</td>\n",
       "      <td>162002168</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>12207012232</td>\n",
       "      <td>64217</td>\n",
       "      <td>83070.0</td>\n",
       "      <td>50851.0</td>\n",
       "      <td>118960211</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>12962332151</td>\n",
       "      <td>64308</td>\n",
       "      <td>82153.0</td>\n",
       "      <td>51357.0</td>\n",
       "      <td>115844465</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>12218466631</td>\n",
       "      <td>62643</td>\n",
       "      <td>80713.0</td>\n",
       "      <td>48775.0</td>\n",
       "      <td>113649759</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>11618591399</td>\n",
       "      <td>64812</td>\n",
       "      <td>76751.0</td>\n",
       "      <td>41776.0</td>\n",
       "      <td>106751618</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>8483171017</td>\n",
       "      <td>46682</td>\n",
       "      <td>56983.0</td>\n",
       "      <td>30559.0</td>\n",
       "      <td>73700617</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>7748386210</td>\n",
       "      <td>46930</td>\n",
       "      <td>53016.0</td>\n",
       "      <td>27019.0</td>\n",
       "      <td>71919481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>6642780533</td>\n",
       "      <td>37088</td>\n",
       "      <td>43285.0</td>\n",
       "      <td>25159.0</td>\n",
       "      <td>60721461</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>6307964046</td>\n",
       "      <td>34513</td>\n",
       "      <td>43595.0</td>\n",
       "      <td>22767.0</td>\n",
       "      <td>54340046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1549139686</td>\n",
       "      <td>7237</td>\n",
       "      <td>10315.0</td>\n",
       "      <td>6756.0</td>\n",
       "      <td>14737388</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                User_ID  Product_Category_1  Product_Category_2  \\\n",
       "Occupation                                                        \n",
       "4           72521952247              375044            478959.0   \n",
       "0           69847008662              378374            464793.0   \n",
       "7           59315918984              320171            402950.0   \n",
       "1           47584821210              269249            319095.0   \n",
       "17          40164327766              202462            278263.0   \n",
       "12          31270073799              161372            217225.0   \n",
       "20          33658243797              190307            219662.0   \n",
       "14          27396517676              144235            189332.0   \n",
       "16          25449031066              140035            173079.0   \n",
       "2           26672083675              150096            176408.0   \n",
       "6           20421422468              114201            135849.0   \n",
       "3           17694824027               98740            119452.0   \n",
       "15          12207012232               64217             83070.0   \n",
       "10          12962332151               64308             82153.0   \n",
       "5           12218466631               62643             80713.0   \n",
       "11          11618591399               64812             76751.0   \n",
       "19           8483171017               46682             56983.0   \n",
       "13           7748386210               46930             53016.0   \n",
       "18           6642780533               37088             43285.0   \n",
       "9            6307964046               34513             43595.0   \n",
       "8            1549139686                7237             10315.0   \n",
       "\n",
       "            Product_Category_3   Purchase  \n",
       "Occupation                                 \n",
       "4                     279431.0  666244484  \n",
       "0                     263929.0  635406958  \n",
       "7                     228545.0  557371587  \n",
       "1                     172085.0  424614144  \n",
       "17                    180304.0  393281453  \n",
       "12                    133868.0  305449446  \n",
       "20                    116538.0  296570442  \n",
       "14                    107053.0  259454692  \n",
       "16                     94139.0  238346955  \n",
       "2                      95243.0  238028583  \n",
       "6                      73840.0  188416784  \n",
       "3                      63335.0  162002168  \n",
       "15                     50851.0  118960211  \n",
       "10                     51357.0  115844465  \n",
       "5                      48775.0  113649759  \n",
       "11                     41776.0  106751618  \n",
       "19                     30559.0   73700617  \n",
       "13                     27019.0   71919481  \n",
       "18                     25159.0   60721461  \n",
       "9                      22767.0   54340046  \n",
       "8                       6756.0   14737388  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Occ_grp = df.groupby(df.Occupation).sum()\n",
    "Occ_grp = Occ_grp.sort_values(by=[\"Purchase\"], ascending = False)\n",
    "\n",
    "Occ_grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f1141f06",
   "metadata": {
    "_cell_guid": "0c4501be-9685-4a81-ab43-0f1fd868fcba",
    "_uuid": "c8066e29-a46c-4b5a-baf2-ba4d08850102",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:37.712190Z",
     "iopub.status.busy": "2022-10-10T12:32:37.711121Z",
     "iopub.status.idle": "2022-10-10T12:32:38.011040Z",
     "shell.execute_reply": "2022-10-10T12:32:38.009848Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.320809,
     "end_time": "2022-10-10T12:32:38.013508",
     "exception": false,
     "start_time": "2022-10-10T12:32:37.692699",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Occupation', ylabel='Purchase'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# comparison between the occupation and purchase\n",
    "# Occupation details are shown in number from 0 to 20\n",
    "\n",
    "sns.barplot(x=Occ_grp.index, y = Occ_grp.Purchase)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c38ab88",
   "metadata": {
    "_cell_guid": "efca656f-75c8-4cdd-bc16-1001c8a91c09",
    "_uuid": "0013554b-cbdb-4dcb-943a-8dfa6bb2480f",
    "papermill": {
     "duration": 0.017907,
     "end_time": "2022-10-10T12:32:38.048973",
     "exception": false,
     "start_time": "2022-10-10T12:32:38.031066",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Data visualization for the top five occupation purchased product in ABC company."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3143e163",
   "metadata": {
    "_cell_guid": "a7fde759-10aa-4866-9a2a-e76c2d7329e2",
    "_uuid": "a3f22945-304a-4e1f-9ce7-0145887868c8",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:38.085257Z",
     "iopub.status.busy": "2022-10-10T12:32:38.084834Z",
     "iopub.status.idle": "2022-10-10T12:32:46.044557Z",
     "shell.execute_reply": "2022-10-10T12:32:46.043766Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 7.981163,
     "end_time": "2022-10-10T12:32:46.047236",
     "exception": false,
     "start_time": "2022-10-10T12:32:38.066073",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Product_Category_1', ylabel='Purchase'>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x= df.Product_Category_1, y = df.Purchase)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "faf5d2e3",
   "metadata": {
    "_cell_guid": "c3f5f8aa-fed4-40b2-8489-a696be24d5ac",
    "_uuid": "337bcd18-3986-4918-acfb-d83698043d36",
    "papermill": {
     "duration": 0.017606,
     "end_time": "2022-10-10T12:32:46.082607",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.065001",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Comparison between product category 1 vs Purchase values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1aaac254",
   "metadata": {
    "_cell_guid": "a9e3ad4b-af67-4752-88ed-04342de24c56",
    "_uuid": "9551a7fc-85d0-4133-ae3a-0763b229f81f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:46.120908Z",
     "iopub.status.busy": "2022-10-10T12:32:46.120183Z",
     "iopub.status.idle": "2022-10-10T12:32:46.258876Z",
     "shell.execute_reply": "2022-10-10T12:32:46.258065Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.160099,
     "end_time": "2022-10-10T12:32:46.261328",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.101229",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Gender</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>136234060927</td>\n",
       "      <td>915426</td>\n",
       "      <td>776517</td>\n",
       "      <td>916139.0</td>\n",
       "      <td>468179.0</td>\n",
       "      <td>1186232642</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>M</th>\n",
       "      <td>415500008355</td>\n",
       "      <td>3527312</td>\n",
       "      <td>2196199</td>\n",
       "      <td>2788809.0</td>\n",
       "      <td>1645150.0</td>\n",
       "      <td>3909580100</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             User_ID  Occupation  Product_Category_1  Product_Category_2  \\\n",
       "Gender                                                                     \n",
       "F       136234060927      915426              776517            916139.0   \n",
       "M       415500008355     3527312             2196199           2788809.0   \n",
       "\n",
       "        Product_Category_3    Purchase  \n",
       "Gender                                  \n",
       "F                 468179.0  1186232642  \n",
       "M                1645150.0  3909580100  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gender_grp = df.groupby(df.Gender).sum()\n",
    "\n",
    "gender_grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "dd61e233",
   "metadata": {
    "_cell_guid": "d68979e9-b528-4ddd-9337-d0f28faec61e",
    "_uuid": "5b4c3e8f-7608-4ced-ac5d-97fd2e99e7ea",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:46.299617Z",
     "iopub.status.busy": "2022-10-10T12:32:46.298960Z",
     "iopub.status.idle": "2022-10-10T12:32:46.433902Z",
     "shell.execute_reply": "2022-10-10T12:32:46.432274Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.159219,
     "end_time": "2022-10-10T12:32:46.438512",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.279293",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "gender = ['Female', 'Male']\n",
    "mycolors = [\"hotpink\", \"Blue\"]\n",
    "plt.pie(gender_grp.Purchase, labels =gender ,colors = mycolors, autopct='%1.1f%%',pctdistance=1.5,\n",
    "       wedgeprops={'edgecolor':'k', 'linestyle': '-',    \n",
    "        'antialiased':True})\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8fec459",
   "metadata": {
    "_cell_guid": "c57b1f08-1d25-4ed0-8a0d-28205fc6009a",
    "_uuid": "9cab0f54-14c1-4470-be91-96d6963fc131",
    "papermill": {
     "duration": 0.02138,
     "end_time": "2022-10-10T12:32:46.506687",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.485307",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Percentage of total purchase by the Gender comparison. Males are purchasing more than female."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "f1bedfc2",
   "metadata": {
    "_cell_guid": "d79f2032-49b9-4939-bf8e-ccd0db984db4",
    "_uuid": "f9ed84d7-ade8-437d-85d5-923e8bdad3e9",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:46.544980Z",
     "iopub.status.busy": "2022-10-10T12:32:46.544252Z",
     "iopub.status.idle": "2022-10-10T12:32:46.684161Z",
     "shell.execute_reply": "2022-10-10T12:32:46.682850Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.161823,
     "end_time": "2022-10-10T12:32:46.686667",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.524844",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Marital_Status</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>married</th>\n",
       "      <td>226029063614</td>\n",
       "      <td>1862821</td>\n",
       "      <td>1238958</td>\n",
       "      <td>1519591.0</td>\n",
       "      <td>857904.0</td>\n",
       "      <td>2086885295</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unmarried</th>\n",
       "      <td>325705005668</td>\n",
       "      <td>2579917</td>\n",
       "      <td>1733758</td>\n",
       "      <td>2185357.0</td>\n",
       "      <td>1255425.0</td>\n",
       "      <td>3008927447</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     User_ID  Occupation  Product_Category_1  \\\n",
       "Marital_Status                                                 \n",
       "married         226029063614     1862821             1238958   \n",
       "unmarried       325705005668     2579917             1733758   \n",
       "\n",
       "                Product_Category_2  Product_Category_3    Purchase  \n",
       "Marital_Status                                                      \n",
       "married                  1519591.0            857904.0  2086885295  \n",
       "unmarried                2185357.0           1255425.0  3008927447  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "status_grp = df.groupby(['Marital_Status']).sum()\n",
    "status_grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6ac57ab2",
   "metadata": {
    "_cell_guid": "be183ae1-4867-413f-84ed-870d371d67c2",
    "_uuid": "f0976f17-0119-4cd6-997a-d2be6c6b4593",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:46.725488Z",
     "iopub.status.busy": "2022-10-10T12:32:46.725095Z",
     "iopub.status.idle": "2022-10-10T12:32:46.940603Z",
     "shell.execute_reply": "2022-10-10T12:32:46.939260Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.238168,
     "end_time": "2022-10-10T12:32:46.943296",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.705128",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Marital_Status', ylabel='Purchase'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=status_grp.index, y = status_grp.Purchase)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8578355e",
   "metadata": {
    "_cell_guid": "95efabf8-6a91-4d98-a399-b337689174c6",
    "_uuid": "3f65b5c4-7eba-452f-811a-356de01a344d",
    "papermill": {
     "duration": 0.018297,
     "end_time": "2022-10-10T12:32:46.980299",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.962002",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Unmarried peoples are purchasing more compare to Married people."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "6cb34cd5",
   "metadata": {
    "_cell_guid": "af76d152-b616-4aba-901b-82b800a409e6",
    "_uuid": "f5fcb325-5f59-4650-b4f3-baa734b54b51",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2022-10-10T12:32:47.020209Z",
     "iopub.status.busy": "2022-10-10T12:32:47.019752Z",
     "iopub.status.idle": "2022-10-10T12:32:47.168185Z",
     "shell.execute_reply": "2022-10-10T12:32:47.166342Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.173244,
     "end_time": "2022-10-10T12:32:47.172703",
     "exception": false,
     "start_time": "2022-10-10T12:32:46.999459",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "d = df['City_Category'].value_counts()\n",
    "keys = ['A','B','C']\n",
    "palette_color = sns.color_palette('bright')\n",
    "plt.pie(d,labels = keys, colors=palette_color, autopct='%.0f%%')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8004cbc0",
   "metadata": {
    "_cell_guid": "76f35c1f-2569-4aa2-bcd5-c65b19afc7b0",
    "_uuid": "8146beb0-6b73-4858-b36c-4d460e39788f",
    "papermill": {
     "duration": 0.020225,
     "end_time": "2022-10-10T12:32:47.243101",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.222876",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "City wise count of purchase for company"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eaeee807",
   "metadata": {
    "_cell_guid": "6bab5b63-a72f-4679-b49e-d9abebed34f6",
    "_uuid": "2a307306-3093-4ad4-85a0-b484c82e4cc8",
    "papermill": {
     "duration": 0.018252,
     "end_time": "2022-10-10T12:32:47.280112",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.261860",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "753dc1fe",
   "metadata": {
    "_cell_guid": "0881a759-cd76-420b-a131-69ede3c13acf",
    "_uuid": "d661a86d-8c57-4e8b-96bb-06fdb38021cc",
    "papermill": {
     "duration": 0.018526,
     "end_time": "2022-10-10T12:32:47.317258",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.298732",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The dataset contains one month data only. This not enough to predict the behaviour or trends in purchase of the product."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eefd0b66",
   "metadata": {
    "_cell_guid": "3caff1d2-58a8-4d93-8040-03cc2e1d791f",
    "_uuid": "e64f52e5-765b-4e46-b38d-560f80b8ba0e",
    "papermill": {
     "duration": 0.020183,
     "end_time": "2022-10-10T12:32:47.356025",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.335842",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### With the one month of data some observations:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ef9abdc",
   "metadata": {
    "_cell_guid": "ad0356e2-01fd-4313-8b5d-6075a7910199",
    "_uuid": "388c46da-a71d-45c1-813c-e63be984b0e8",
    "papermill": {
     "duration": 0.018742,
     "end_time": "2022-10-10T12:32:47.394218",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.375476",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "    Age group between 26 to 35 did more purchase. we need to check the what are category they are purchased most, for that we need  category of product required\n",
    "\t\n",
    "    Why the 26-35 age group having more purchase means there are more customer are in this age group\n",
    "\n",
    "\tUnmarried people are spending more than married peoples means with assumptions they have less responsiblity than the married peoples.\n",
    "\t\n",
    "    Males are more purchase than the females. we can attract female by increase the female product with offers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c4d775a",
   "metadata": {
    "_cell_guid": "e70bcaaa-2b6a-4a11-8aa9-86fe9ab69072",
    "_uuid": "1a17b972-12ce-4e57-a480-9730083e65c7",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.018516,
     "end_time": "2022-10-10T12:32:47.431411",
     "exception": false,
     "start_time": "2022-10-10T12:32:47.412895",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 28.274015,
   "end_time": "2022-10-10T12:32:48.272458",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-10-10T12:32:19.998443",
   "version": "2.3.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
