{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1786,
   "id": "a51fb650-f7ad-470c-b65c-f09cb127a5f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt \n",
    "import sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1787,
   "id": "d2850231-40bf-4b98-a82b-d0302a984f11",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required Python libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1788,
   "id": "365a7b7a-38b9-4705-a769-6122e3e38ce1",
   "metadata": {},
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 1788,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('titanic.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1789,
   "id": "8ee88551-fc08-478d-8bbb-cbb0c9b910ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reading the dataset using a method in the Pandas library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1790,
   "id": "8537e9d4-bb5e-4b11-8a36-cee7c57c9b3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1791,
   "id": "0ebaccdb-ee22-410f-a899-b34a3d1807ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get more information from the dataset with the help of a method from Pandas\n",
    "# We have a number of nulls in the Age and Cabin and Embarked columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1792,
   "id": "bef96f90-fa2a-4c66-8a96-62004350b43c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 1792,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjgAAAGwCAYAAACkfh/eAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAy60lEQVR4nO3de1xVZb7H8e9mI7CJGOSqqGneEBCRg2mWppTmJZ0Um9PUpJlN1kljTDsWepxMJU/SRVHUHLWxMDU1uzhNlp3GysZLOOAtGMEyzBtk2iGBjZt9/ui4Zxg0YQuszfLzfr32q/bzrLWe3wIWfF3r2WtZnE6nUwAAACbiZXQBAAAA9Y2AAwAATIeAAwAATIeAAwAATIeAAwAATIeAAwAATIeAAwAATMfb6AKMUlVVpfPnz8vLy0sWi8XocgAAQC04nU5VVVXJ29tbXl6XPk9z1Qac8+fPa9++fUaXAQAA3BAXFycfH59L9l+1AedC6ouLi5PVajW4GgAAUBsOh0P79u372bM30lUccC5clrJarQQcAACamMtNL2GSMQAAMB0CDgAAMB0CDgAAMJ2rdg4OAABNmcPhUGVlpdFl1LtmzZrVy9xYAg4AAE2I0+nUiRMndObMGaNLaTBBQUFq0aLFFd2njoADAEATciHchIeHy9/f31Q3q3U6nTp37pxOnTolSWrZsqXb2yLgAADQRDgcDle4CQkJMbqcBmGz2SRJp06dUnh4uNuXq5hkDABAE3Fhzo2/v7/BlTSsC/t3JXOMCDgAADQxZrosdTH1sX8EHAAAYDoEHAAAYDoEHAAATObWW29VVFSU6xUbG6vBgwfrj3/8Y63WffPNNxu+yAbGp6gAADChadOmaejQoZKk8+fPa8eOHZo+fbqCgoI0YsQIY4trBJzBAQDAhK699lqFhYUpLCxMLVu21MiRI9W7d2998MEHRpfWKAg4AABcJby9vdWsWTOdP39eL774ovr06aPExESlpKTo+++/r7F8aWmpUlNT1bt3b3Xt2lWDBw/W1q1bXf3vvfeeBg0apLi4OA0dOrRa36uvvqqkpCTFxcUpOTlZX3zxRaPs4wUEHKAROR0Oo0vwCHwdgMZVWVmpDz74QNu3b9dtt92mBQsWaNOmTXr22We1bt06fffdd3r66adrrJeWlqavvvpKK1eu1ObNm9WjRw9Nnz5ddrtd3333naZOnaqHH35Y77//vkaNGqXJkyfrzJkzOnjwoObNm6enn35af/7zn9WjRw9NmjRJVVVVjbbPzMEBGpHFalXxpEmqLCgwuhTDNOvYUWHz5xtdBmB6Tz/9tGbPni1JKi8vl5+fn+6//34NHz5cN954o5588kndcsstkqRnnnlGf/7zn2ts44YbbtADDzygzp07S5LGjRun9evX67vvvtP333+vyspKtWjRQq1atdK4ceMUFRUlX19fffvtt7JYLIqMjFTr1q01adIkJSUlqaqqSl5ejXNuhYADNLLKggLZDxwwugwAJpeSkqLbb79dkuTr66uwsDBZrVadPn1aZ86cUWxsrGvZjh076rHHHquxjREjRmjr1q164403dPjwYR34/99dDodD0dHR6t+/vx544AFdf/31uu222/SrX/1KNptNffr0UefOnTV8+HDFxMS4+ry9Gy92cIkKAAATCgkJUdu2bdW2bVu1aNHC9UynuoSMqVOn6rnnnlNgYKDuuecevfzyy64+i8Wil19+WevXr9egQYP08ccfa+TIkfryyy9ls9m0fv16rVq1Sj179tSbb76p5ORknTx5st7381IIOAAAXEUCAwPVvHlz5eXludq+/PJL3XLLLSovL3e1lZaWavPmzXrppZeUkpKigQMH6uzZs5J+eup3YWGhnnvuOXXr1k2PP/64/vSnP6lly5b69NNP9be//U0vv/yybrzxRqWmpur9999XRUWFsrOzG20/uUQFAMBVZvTo0VqwYIEiIiIUEhKitLQ0de/eXX5+fq5lfHx8ZLPZ9MEHHyg4OFhfffWVZs2aJUmy2+0KDAzUmjVrdO2112r48OEqKCjQt99+q5iYGPn5+SkzM1OhoaHq3bu3du/erXPnzikqKqrR9pGAAwDAVWb8+PH63//9X02aNEnnz59X//79NWPGjGrL+Pj4KD09Xc8995xee+01tW7dWv/xH/+h+fPn68svv9SwYcO0cOFCPf/881q6dKlCQkI0efJk9enTR9JPn8BavHixZs2apcjISKWnp6tDhw6Nto8Wp9PpbLTRPIjD4VBOTo66d+/uui4JNIZjw4Zd1ZOMfWJjFbl5s9FlAE1SeXm5vvrqK11//fXVzraYzc/tZ23/fjMHBwAAmA4BBwAAmA4BBwAAmA4BBwAAmA4BBwAAmA4BBwAAmA4BBwAAmA4BBwAA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6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6Xh0wPnwww8VFRVV7ZWSkiJJOnjwoH71q18pPj5eo0aN0v79+w2uFgAAeAqPDjgFBQVKSkrSZ5995nrNmTNH586d0/jx49WjRw+9+eabSkhI0MMPP6xz584ZXTIAAPAAHh1wCgsL1blzZ4WFhblegYGBeu+99+Tr66upU6eqQ4cOmj59uq655hq9//77RpcMAAA8gLfRBfycwsJC3XTTTTXac3NzlZiYKIvFIkmyWCz6t3/7N+Xk5Cg5OblOYzgcjnqp9WKsVmuDbRto6hry2GsMHN/ApTXk8V3bbXtswHE6nfrqq6/02Wef6eWXX5bD4dDgwYOVkpKi4uJidezYsdryISEhOnToUJ3H2bdvX32VXI3NZlNMTEyDbBswg/z8fJWVlRldhls4voGf5wnHt8cGnGPHjqmsrEw+Pj6aP3++jh49qjlz5qi8vNzV/s98fHxkt9vrPE5cXBz/EgMMEBUVZXQJABpIQx7fDoejVicnPDbgtGrVSjt37tQvfvELWSwWRUdHq6qqSv/5n/+pnj171ggzdrtdfn5+dR7HarUScAADcNwB5uUJx7fHBhxJCgoKqva+Q4cOqqioUFhYmEpKSqr1lZSUKDw8vBGrAwAAnspjP0X16aefqlevXtWu4X355ZcKCgpSYmKi/va3v8npdEr6ab7Onj17FB8fb1S5AADAg3hswElISJCvr6/+67/+S4cPH9a2bds0b948/fa3v9XgwYP1ww8/KC0tTQUFBUpLS1NZWZmGDBlidNkAAMADeGzACQgI0IoVK3T69GmNGjVK06dP1913363f/va3CggI0Msvv6zs7GwlJycrNzdXy5Ytk7+/v9FlAwAAD+DRc3A6deqkV1555aJ93bp106ZNmxq5IgAA0BR47BkcAAAAdxFwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6RBwAACA6TTpgFNRUaFp06apR48e6tOnj1auXGl0SQAAwAN4G13AlZg3b57279+vVatW6dixY3ryyScVGRmpwYMHG10aAAAwUJMNOOfOndP69ev1hz/8QbGxsYqNjdWhQ4e0evVqAg4AAFe5JnuJKi8vT+fPn1dCQoKrLTExUbm5uaqqqjKwMgAAYLQmewanuLhYzZs3l4+Pj6stNDRUFRUVOnPmjIKDg392fafTKUmy2+2yWq0NUqPValVYtFVevk32y3zFgtp5yeFwKMTp33TTdD35hdNPDodD1uhoefv6Gl2OYazt28vhcMjhcBhdyhXh+Ob4/mcc3z9pjOP7wrYv/B2/lCZ7ZJaVlVULN5Jc7+12+2XXv3CW5+DBg/Vf3D9p+2upbYOO4PlycqRQSaG6eg/6n5QqRznSr39tdCGGO5qTY3QJ9YLjm+P7Hzi+L2is4/tyV2uabMDx9fWtEWQuvPfz87vs+t7e3oqLi5OXl5csFkuD1AgAAOqX0+lUVVWVvL1/PsI02YATERGh77//XufPn3ftZHFxsfz8/BQYGHjZ9b28vGqcAQIAAObQZC+bRkdHy9vbWzn/dCosOzvbdVYGAABcvZpsErDZbBoxYoRmzpypvXv3auvWrVq5cqXGjBljdGkAAMBgFuflpiF7sLKyMs2cOVMffPCBAgIC9OCDD2rs2LFGlwUAAAzWpAMOAADAxTTZS1QAAACXQsABAACmQ8ABAACmQ8CBqVVUVGjatGnq0aOH+vTpo5UrVxpdEoB6ZrfbNWzYMO3cudPoUuBBmuyN/oDamDdvnvbv369Vq1bp2LFjevLJJxUZGckT5wGTqKio0JQpU3To0CGjS4GHIeDAtM6dO6f169frD3/4g2JjYxUbG6tDhw5p9erVBBzABAoKCjRlypTLPnQRVycuUcG08vLydP78eSUkJLjaEhMTlZube9mHtAHwfLt27VKvXr20bt06o0uBB+IMDkyruLhYzZs3r/bMsdDQUFVUVOjMmTMKDg42sDoAV+ree+81ugR4MM7gwLTKyspqPFD1wvt/fRI9AMBcCDgwLV9f3xpB5sJ7Pz8/I0oCADQSAg5MKyIiQt9//73Onz/vaisuLpafn58CAwMNrAwA0NAIODCt6OhoeXt7Kycnx9WWnZ2tuLg4eXnxow8AZsZveZiWzWbTiBEjNHPmTO3du1dbt27VypUrNWbMGKNLAwA0MD5FBVNLTU3VzJkzdf/99ysgIECPPfaYbr/9dqPLAgA0MIuTOyQBAACT4RIVAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOgEZVWVmphQsX6rbbblPXrl3Vv39/zZ07V6WlpfU+1sKFCzV69Oh6364kRUVFaefOnQ2ybQBXjkc1AGhUzz//vD7//HPNmTNHbdq0UVFRkdLS0nTkyBEtXbq0XscaN25cgwUcAJ6NgAOgUW3atEnPPvusevfuLUlq3bq1Zs6cqd/85jc6deqUwsPD622sa665pt62BaBp4RIVgEZlsVi0Y8cOVVVVudoSEhL0pz/9Sc2bN9ett96qN99809W3c+dORUVFSZKOHj2qqKgoZWZm6oYbblBqaqri4uK0Y8cO1/KlpaWKi4vTF1984bpEVVVVpb59+2rjxo2u5ZxOp2655Ra9/fbbkqQvvvhCycnJ6tatm4YPH64tW7ZUq3vRokXq3bu3evXqpfXr1zfI1wZA/eEMDoBGNWbMGGVkZGjr1q3q16+fbrrpJvXp00cdO3as9Tb27NmjjRs3qqqqSmfPntWHH36oG2+8UZL0l7/8RcHBwUpMTNRf//pXSZKXl5cGDx6sDz/8UKNGjZIk5eTk6MyZM7rttttUXFyshx9+WI8//rj69u2rnJwcPfXUUwoJCVGPHj20bt06vfrqq3ruuefUokULPfPMM/X/hQFQrziDA6BRTZgwQenp6WrRooXeeOMNpaSk1Di7cjn333+/rrvuOrVr10533HGHPvzwQzmdTknSli1bNGTIEFkslmrr3HHHHdq+fbtrMvOWLVvUr18/BQQEaPXq1brpppt03333qW3btrrzzjt19913a9WqVZKkN954Q/fff7+SkpIUHR2tOXPm1NNXA0BDIeAAaHS//OUvtXbtWn3++ed6/vnn1alTJ02fPl379++v1fqtWrVy/X9SUpJ++OEH5ebmqqysTJ9++qmGDh1aY53u3bsrLCxM27ZtkyR98MEHruUOHz6sjz/+WAkJCa5XVlaWvv76a0lSYWGhoqOjXdvq2LGj/P393d19AI2AS1QAGk1eXp7eeustPfXUU5Kk5s2ba/jw4Ro0aJBuv/32anNpLnA4HDXafH19Xf/v7++vpKQkbdmyRSdPnlRoaKi6det20fGHDh2qLVu2qG3btvr+++/Vv39/SdL58+c1fPhwPfLII9WW9/b+x6/IC2eILtYHwPNwBgdAo3E4HHrllVd08ODBau0+Pj7y8/NTcHCwmjVrph9//NHVV1RUdNnt3nHHHdq2bZu2bt160bM3/7zc9u3btWXLFt16662y2WySpOuvv15HjhxR27ZtXa+PPvpI7777riSpU6dO2rdvn2s7R48e1Q8//FCnfQfQuAg4ABpNbGys+vfvr0cffVTvvvuujh49qpycHD399NOy2+26/fbbFRcXpw0bNujvf/+7du7cqZUrV152u7fccotOnTp12YATHR2t8PBwZWVlaciQIa72e++9V/v379dLL72kr7/+Wu+++65efPFFRUZGSpLuu+8+vfrqq9qyZYv+/ve/a/r06fLy4tcn4Mk4QgE0qvnz5+vOO+/UokWLNGTIED388MMqLS1VVlaWAgICNGnSJAUGBio5OVlpaWn63e9+d9lt+vj4aMCAAWrRooW6dOnys8sOHTpUVqtVt9xyi6utVatWWrp0qT799FMNGzZM8+fP11NPPaVf/vKXkqQ777xTKSkpmj17tu69917dfPPNCgwMvLIvBIAGZXH+64VlAACAJo4zOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHQIOAAAwHT+D7HvBHTv4LR8AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Survived' ,hue='Pclass' ,data=data,palette='rainbow')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1793,
   "id": "d25b152c-2807-44db-9f21-b7b209b19007",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drawing a graph from the Seaborn library to discover a pattern between the Pclass variable and the survivors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1794,
   "id": "6d833cc3-9217-41d5-b154-87ac0a9c38a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 1794,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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BzDAgQMH9Mgjj1zyTRfROnGKCsbat2+fqqqqlJSU5FmWnJysgoKCS75JG4Dm77PPPlNqaqreeOMNq0dBM8QRHBirpKREHTt2rPGeY2FhYaqsrFRZWZlCQkIsnA7A5brnnnusHgHNGEdwYCyHw1HrDVXPf/6P70QPADALgQNj+fv71wqZ858HBARYMRIAoIkQODBWeHi4Tpw4oaqqKs+ykpISBQQEKDg42MLJAACNjcCBsXr16iVfX1/l5+d7luXl5Sk+Pl4+PvynDwAm41EexgoMDNRtt92mrKws7dq1S5s3b9bKlSs1duxYq0cDADQynkUFo2VmZiorK0v333+/2rVrpylTpuimm26yeiwAQCOzuXmFJAAAYBhOUQEAAOMQOAAAwDgEDgAAMA6BAwAAjEPgAAAA4xA4AADAOAQOAAAwDoEDAACMQ+AAaFLnzp3TokWLdOONNyouLk6DBg1Sdna2ysvLG/y+Fi1apDFjxjT4diUpJiZG27dvb5RtA7h8vFUDgCb1/PPPa+vWrXrqqacUGRmpoqIizZs3T0eOHNHy5csb9L7Gjx/faIEDoHkjcAA0qfXr1+vpp59Wv379JEkRERHKysrSvffeq+PHj6tz584Ndl9t27ZtsG0BaFk4RQWgSdlsNn366aeqrq72LEtKStI777yjjh07Kj09XW+99ZZn3fbt2xUTEyNJOnr0qGJiYrRkyRL17dtXmZmZio+P16effuq5fXl5ueLj47Vjxw7PKarq6moNGDBAubm5ntu53W4NHDhQ//3f/y1J2rFjhzIyMtSnTx/dcsst2rRpU425Fy9erH79+ik1NVVr165tlO8NgIbDERwATWrs2LFauHChNm/erBtuuEHXX3+90tLS1LNnzzpvY+fOncrNzVV1dbVOnjyp9957T7/85S8lSR988IFCQkKUnJysbdu2SZJ8fHz0q1/9Su+9955GjRolScrPz1dZWZluvPFGlZSU6KGHHtJvf/tbDRgwQPn5+Zo+fbpCQ0OVkpKiN954Q6tWrdKzzz6rLl266Mknn2z4bwyABsURHABNatKkSZo/f766dOmiN998U1OnTq11dOVS7r//fl155ZW66qqrNHz4cL333ntyu92SpE2bNmno0KGy2Ww1vmb48OH65JNPPBczb9q0STfccIPatWun119/Xddff73uu+8+RUVFaeTIkbrrrrv06quvSpLefPNN3X///Ro8eLB69eqlp556qoG+GwAaC4EDoMndeuutWrNmjbZu3arnn39e11xzjWbOnKk9e/bU6eu7d+/u+ffBgwfr1KlTKigokMPh0EcffaRhw4bV+prExER16tRJH374oSTpL3/5i+d2Bw8e1JYtW5SUlOT5WL16tQ4fPixJ+vrrr9WrVy/Ptnr27KmgoCBvdx9AE+AUFYAms2/fPm3YsEHTp0+XJHXs2FG33HKLbr75Zt100001rqU5z+Vy1Vrm7+/v+fegoCANHjxYmzZt0vfff6+wsDD16dPngvc/bNgwbdq0SVFRUTpx4oQGDRokSaqqqtItt9yiCRMm1Li9r+//PUSeP0J0oXUAmh+O4ABoMi6XS6+88or27t1bY7mfn58CAgIUEhKiNm3a6MyZM551RUVFl9zu8OHD9eGHH2rz5s0XPHrz09t98skn2rRpk9LT0xUYGChJio6O1pEjRxQVFeX5eP/997Vx40ZJ0jXXXKPdu3d7tnP06FGdOnWqXvsOoGkROACaTGxsrAYNGqSJEydq48aNOnr0qPLz8zV79mw5nU7ddNNNio+P17p16/TVV19p+/btWrly5SW3O3DgQB0/fvySgdOrVy917txZq1ev1tChQz3L77nnHu3Zs0cvvfSSDh8+rI0bN+rFF19Ut27dJEn33XefVq1apU2bNumrr77SzJkz5ePDwyfQnPEbCqBJLViwQCNHjtTixYs1dOhQPfTQQyovL9fq1avVrl07/du//ZuCg4OVkZGhefPm6eGHH77kNv38/DRkyBB16dJF11577UVvO2zYMNntdg0cONCzrHv37lq+fLk++ugjjRgxQgsWLND06dN16623SpJGjhypqVOnau7cubrnnnvUv39/BQcHX943AkCjsrn/8cQyAABAC8cRHAAAYBwCBwAAGIfAAQAAxiFwAACAcQgcAABgHAIHAAAYh8ABAADGIXAAAIBxCBwAAGAcAgcAABiHwAEAAMb5/6o4pGTzY9aVAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style('whitegrid')\n",
    "sns.countplot(x='Survived', hue='Sex', data=data,palette='RdBu_r')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1795,
   "id": "0e993e60-10ad-482a-9e34-0b728626ba76",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drawing a graph from the Seaborn library to discover a pattern between the Sex variable and the survivors\n",
    "# According to the graph, 2 times more women survived than men"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1796,
   "id": "2ea43073-6f76-409d-b8ce-1975d0de92de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: ylabel='Frequency'>"
      ]
     },
     "execution_count": 1796,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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iIgIn0VJBQUFSktLU3i4vZ4+l8ul/fv36+TJk96xgoICuVwuE6vyz7Jly5Sfn68lS5b4/IvILj1+8803mjhxoo4ePeod27dvn9q1a6eMjAzL9/j6669r48aN2rBhgzZs2KBBgwZp0KBB2rBhg1wul/bs2eO9lohhGNq9e7el+pN+/h9e3759fY6Wffnll2rbtq0yMjJs0aPL5VJdXZ2++uor79jhw4fVqVMn2/weG+zdu1e9e/f2GbNDjwkJCTpy5IjPkZbDhw/rl7/8pen92ev/wEHgdDp16623as6cOdq7d6+2bNmiVatW6b777jO7tIDLzMzU5ZdfrunTp6u0tFR5eXnau3evbr/9drNLa5ZDhw7p+eef129/+1tlZGSosrLS+2WXHtPS0tS9e3fNmDFDZWVl2rp1qxYtWqQHHnjAFj126tRJSUlJ3q82bdqoTZs2SkpKUlZWlk6cOKHc3FyVlZUpNzdXtbW1Gjp0qNllN0t6erqioqL0+OOP6/Dhw9q6dasWLlyocePG2abHq6++WgMHDtT06dN18OBBbdu2TXl5ebrzzjtt02OD0tJSn0/LSbJFj4MGDdIll1yixx9/XF999ZU+/PBDvfDCC7r33nvN7++ifFjb4mpqaoypU6cavXr1MgYMGGC88sorZpcUMP98HRjDMIyvv/7auPvuu40ePXoYw4cPN3bs2GFidf5ZuXKl0aVLl7N+GYY9ejQMw/juu++MCRMmGL179zb69+9vrFixwns9Brv02GDatGne68AYhmEUFRUZt956q5GWlmbcfvvtxv79+02szn9//etfjbFjxxq9evUy+vfvb/zhD3/w/g7t0uOJEyeMKVOmGL169TL69etnyx4NwzDS0tKMjz/+uNG4HXosLS01xo4da/Tu3dsYPHiw8corr4TE7zDMMP7/sR8AAACL4BQSAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwHAIMAACwnP8HO9lNs2rK/yYAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.dropna(subset=['Age'], axis=0)\n",
    "data['Age'].plot.hist(bins=30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1797,
   "id": "c0bb6ee4-cf4c-4dee-aba2-7a6d15c78e81",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Temporary removal of null data from the age column and visual inspection of the age column data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1798,
   "id": "fb846146-819a-428a-ab71-58703e08e054",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Age'>"
      ]
     },
     "execution_count": 1798,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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ANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANshwAAAANu5oABTX18vl8slt9t9seoBAAD4t9odYNxut4qLi3X11VcrNTVVR48e1S9/+UvNmzdPLS0tl6JGAACANtodYAoLC/Xmm2/q8ccfV0BAgCTptttu05YtW/TEE09c9AIBAAC+qd0BZsOGDfrVr36lkSNHys/PT5I0bNgwLVq0SO++++5FLxAAAOCb2h1gvvjiC0VGRp7RHhoaqsbGxotSFAAAwHdpd4C55pprtHLlyjZtDQ0NevLJJ3X11VdftMIAAAC+TbsDzPz587V7924NGzZMzc3NmjFjhn70ox/p6NGjmjt37qWoEQAAoI1O7X1Bz5499dprr6msrEwHDx7UqVOn1LdvX1177bXy9z//u7IzMjIUHh6uxx9/XJK0e/du5ebmau/evYqJiVFeXp4GDRp03u8PAAA6jnYnjqqqKlVVVSk6OlojR47UjTfeqJiYGFVXV6u2tlZOp7PdRbzzzjv66KOPPF83NjYqIyNDQ4YM0fr165WcnKypU6cyxwYAAEg6jxGYG2+8US6X69vfsFMn3XDDDVqwYIFCQkL+7fvV1dXpiSeeUHx8vKdt48aNCgwM1OzZs+Xn56ecnByVlpZq06ZNSk9Pb2/JAACgg2n3CExeXp6io6O1YsUKffrpp/r000/13HPPKSYmRg899JBKSkpUW1vruRT07yxatEjjxo1TTEyMp23Hjh1KSUnx3Kbt5+enwYMHq7y8vL3lAgCADqjdIzC/+93v9NRTT2nw4MGettTUVC1YsED/9V//pYyMDGVnZ2vKlCnKz8//zvcqKyvTX/7yF7311luaP3++p72mpqZNoJGkiIgI7du3r73lntclrQt5HTo+p9NJ/zCs9fzzbwGroE9ePOd6/todYP71r3+pU6czX+bv76+vvvpKkhQSEqKTJ09+5/s0NzcrNzdX8+bNU1BQUJttTU1Nnqf8tgoICDivpQp27tzZ7tdI0pEjR87rdej49u7dy3wsw1q/P/m3gFXQJ72v3QFm1KhRmjNnjubNm6dBgwbJ7XZr165dys/P1w033KCmpiYtX75cCQkJ3/k+zzzzjAYNGqThw4efsS0wMPCMsNLS0nJG0DkX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qwxVAwAArMySIzAAAADfhQADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABshwADAABsp5PpAgDYT1VVlRoaGowcu6WlRc8995wkaenSpZoyZYoCAgKM1BISEqLevXsbOTbg6wgwANqlrq5Od911l1wul+lStG3bNm3bts3Y8f39/bV+/Xp169bNWA2AryLAAGiXbt26adWqVV4fgSksLNSOHTvkcDh044036tprr9XHH3+s999/X06nU4mJicrMzPRqTSEhIYQXwBACDIB28/Zlk6amJu3YsUOdO3fWO++847lkdO211+rhhx/WTTfdpB07duh73/ueunTp4tXaAJjBJF4Alrds2TJJ0vjx4yVJa9eu1W9+8xutXbtWknTHHXe02Q9Ax8cIDADLO3LkiCSpvr5eY8aMkdPp9GxbunSpRo0a1WY/AB0fIzAALO+KK66QJL3zzjsKDQ3VrFmztG7dOs2aNUuhoaHauHFjm/0AdHwEGACWd88993j+XFJSoptvvlkRERG6+eabVVJSctb9AHRsXEICYHnvvfee58+33nqr4uPjFRERoS+++EI7d+5ss1/rPBkAHRsBBoDlVVVVSZJ69uypY8eOafv27W22R0VFqbq62rMfgI6PS0gALK/1tu1jx46ddXt1dXWb/QB0fAQYAJY3ZswYz5/9/PzabDv969P3A9CxEWAAWN5bb73l+bPD4dDEiRP18ssva+LEiXI4HGfdD0DHZuk5MM3NzcrLy9N7772noKAgTZkyRVOmTDFdlleZXDRPkiorK9v8bhIL5/muLVu2SJJCQ0PV0NCg1atXa/Xq1ZK+Xo+oa9eu+uqrr7RlyxZNnDjRZKkAvMTSAeaJJ57QZ599phdffFFVVVV65JFH1Lt3b40ePdp0aV5hpUXzCgoKTJfAwnk+7F//+pck6frrr9f06dP1xhtvqKqqSr1799a4ceNUWFioN954w7MfgI7PsgGmsbFRa9eu1YoVKxQXF6e4uDjt27dPJSUlPhNgTC2aZ1UsnOe7+vbtq0OHDmnTpk3KzMxsc6v0qVOnPLdZ9+3b11SJMIzR6q/50ki1ZQPMX//6V506dUrJycmetpSUFC1dulQul0v+/r4x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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='Pclass', y='Age', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1799,
   "id": "bff303cf-6f72-477a-81ee-20d10afb6891",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking the age of people in each cabin with the help of Seaborn Library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1800,
   "id": "235b261d-d0cd-4490-b6ed-4c0fc8010b1d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(38.233440860215055, 29.87763005780347, 25.14061971830986)"
      ]
     },
     "execution_count": 1800,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Pclass']==1]['Age'].mean(), data[data['Pclass']==2]['Age'].mean(), data[data['Pclass']==3]['Age'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1801,
   "id": "b476ceb1-3535-4e18-a904-adef987e61db",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the average age value of the people in each column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1802,
   "id": "afc50a19-9671-455a-9012-ebfcbfb58b07",
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_age (row):\n",
    "    Age = row['Age']\n",
    "    Pclass = row['Pclass']\n",
    "    if pd.isnull(Age):\n",
    "        if Pclass == 1:\n",
    "            return 38\n",
    "        elif Pclass == 2:\n",
    "            return 30\n",
    "        else:\n",
    "            return 25\n",
    "    else:\n",
    "        return Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1803,
   "id": "0b5f9289-01b2-40a1-be68-aafd1108ac47",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define a function to replace the average age in each Pclass instead of nulls "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1804,
   "id": "6ecd405a-b9e3-46be-8080-8ae5c650e08d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Age'] = data.apply(impute_age,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1805,
   "id": "4b5848fc-fb03-4c33-b981-6061e8dd72a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the apply method to apply changes to the dataframe."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1806,
   "id": "d1729816-0644-41ba-bc12-89f65ed67f58",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop('Cabin' ,axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1807,
   "id": "2b07df33-1271-4c0a-ac18-a9a026829034",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Removing the cabin column due to the small number of data and the insignificant effect of this column on the mortality rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1808,
   "id": "c514a556-950a-4d39-846a-1162c6f05684",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "S    644\n",
       "C    168\n",
       "Q     77\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 1808,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1809,
   "id": "6cbd7fb3-7f3c-4717-9c13-e94a891605f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking the number of variables in the Embarked column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1810,
   "id": "9ef3fc51-d19c-4cf3-a9f6-eb83e991fbb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Embarked'].replace(np.nan, 'S', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1811,
   "id": "89e04f39-d2b3-40dd-8179-4ba20203c914",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filling the blanks of the Embarked column with the mod method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1812,
   "id": "dcd3a054-e255-4b74-aa0b-0cb29df6b2d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          891 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Embarked     891 non-null    object \n",
      "dtypes: float64(2), int64(5), object(4)\n",
      "memory usage: 76.7+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1813,
   "id": "d171fa8d-f61e-4e88-bb1a-6b8737f5040d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Double check to make sure all null values ​​are inserted"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1814,
   "id": "ffa9944d-a93d-489f-bb6f-78b2ae532d9c",
   "metadata": {},
   "outputs": [],
   "source": [
    " sex= pd.get_dummies(data['Sex'],drop_first=True).astype(int) #sex\n",
    " embark= pd.get_dummies(data['Embarked'],drop_first=True).astype(int) #embark\n",
    " pclass= pd.get_dummies(data['Pclass'],drop_first=True).astype(int) #Pclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1815,
   "id": "5e32acdc-16c8-4bbd-8042-d5e5a1a872cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To understand our model, we binarize the categorical data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1816,
   "id": "00f1ae21-4150-42e7-9e00-21668c922b27",
   "metadata": {},
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
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>male</th>\n",
       "      <th>Q</th>\n",
       "      <th>S</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived   Age  SibSp  Parch     Fare  male  Q  S  2  3\n",
       "0         0  22.0      1      0   7.2500     1  0  1  0  1\n",
       "1         1  38.0      1      0  71.2833     0  0  0  0  0\n",
       "2         1  26.0      0      0   7.9250     0  0  1  0  1\n",
       "3         1  35.0      1      0  53.1000     0  0  1  0  0\n",
       "4         0  35.0      0      0   8.0500     1  0  1  0  1"
      ]
     },
     "execution_count": 1816,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.drop(['PassengerId','Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)\n",
    "train = pd.concat([data,sex,embark,pclass],axis=1)\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1817,
   "id": "83b19e08-668b-4061-bfb0-fe516203626a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping extra columns and pasting new columns into the dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1818,
   "id": "f25806cf-7ffe-4cc4-88f0-2b401ec88fe9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1819,
   "id": "aeba5286-78ca-4255-84e2-cf9a8599749a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate the Training and testing categories from the Sklearn library in such a way that we keep 30% of the dataset as testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1820,
   "id": "5e603ccc-782e-4da8-91b4-b11c03f26f14",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "235513bb-66ab-41ae-902d-fcb92c5347eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Choosing the logistic regression algorithm to create the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1821,
   "id": "6a0eff06-ae11-4a24-845c-3ba06a8c173c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.columns = X_train.columns.astype(str)\n",
    "X_test.columns = X_test.columns.astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc42fc83-db36-4228-9121-93e7d7117716",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Transforming name type of columns to strings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1822,
   "id": "cd3a408e-d305-4acf-9126-989cae4dfa37",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-14 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-14 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-14 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-14 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-14 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-14 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-14 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-14 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-14 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-14 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-14 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-14 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-14 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-14 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-14 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-14\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(max_iter=200)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" checked><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression(max_iter=200)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(max_iter=200)"
      ]
     },
     "execution_count": 1822,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logmodel = LogisticRegression(max_iter=200)\n",
    "logmodel.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1823,
   "id": "4927ab67-65ba-4c43-83ac-dfd75db2cb9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a logistic regression model with the number of iterantions increased to 200\n",
    "# Train the model using the training dat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1824,
   "id": "d5dc5dd8-ec5a-4399-b10d-d7c3bd89228c",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = logmodel.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b29cc048-39fa-4240-9f72-c9ff15ff6fde",
   "metadata": {},
   "outputs": [],
   "source": [
    "# defined the test model here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1825,
   "id": "22b28df3-6fb1-4fea-bc81-3239c10629b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[137,  17],\n",
       "       [ 38,  76]], dtype=int64)"
      ]
     },
     "execution_count": 1825,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(y_test,predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1826,
   "id": "ca30255c-f70b-47c1-91e0-b6c6df587c09",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Resualt of test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1827,
   "id": "95c38954-c5dc-41a9-83de-35758e1115fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.89      0.83       154\n",
      "           1       0.82      0.67      0.73       114\n",
      "\n",
      "    accuracy                           0.79       268\n",
      "   macro avg       0.80      0.78      0.78       268\n",
      "weighted avg       0.80      0.79      0.79       268\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5b5578e-dc53-4d82-8514-6044aa5d5c5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Displays precision , recall , f1-score and support for each class , along with marco and weighted avarages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c796935f-b4b0-4aef-a7d7-096160abcf1a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.12.0b4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
