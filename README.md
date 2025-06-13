# Titanic Data Preprocessing: Handling Mixed Variables

This project demonstrates a common data cleaning and feature engineering technique: handling **mixed variables**. Mixed variables are columns that contain both numerical and categorical information within the same field. In the Titanic dataset, the 'Ticket' and 'Cabin' columns are prime examples of this.

## Project Goal

The primary goal of this script is to separate the mixed information within the 'Ticket' and 'Cabin' columns into distinct numerical and categorical features. This allows for more effective analysis and model building, as machine learning algorithms typically require clean, well-defined numerical or categorical inputs.

## Code Explanation

Here's a breakdown of the steps taken in the provided Python script:

---

### 1. Import Libraries and Load Data

First, we import the necessary libraries: `numpy` for numerical operations and `pandas` for data manipulation. Then, the Titanic dataset is loaded into a pandas DataFrame.

```python
import numpy as np
import pandas as pd

df = pd.read_csv('titanic.csv')
```

---

### 2. Initial Data Inspection

We inspect the first few rows of the DataFrame and check the unique values in the 'number' column (which seems to be another mixed variable, though not explicitly mentioned for further processing in your provided code, it's good to note its mixed nature).

```python
df.head()
df['number'].unique()
```

The `df['number'].unique()` output clearly shows a mix of numerical-like strings ('5', '3', '6', '2', '1', '4') and a categorical string ('A'). This confirms the presence of mixed data.

---

### 3. Visualizing 'number' Column (Initial Exploration)

A bar plot is generated to visualize the frequency of each unique value in the 'number' column.

```python
fig = df['number'].value_counts().plot.bar()
fig.set_title('Passengers travelling with')
```

---

### 4. Separating 'number' Column into Numerical and Categorical Parts

To handle the mixed 'number' column, we create two new columns:
- **`number_numerical`**: Attempts to convert the 'number' column to numeric. If a value cannot be converted (e.g., 'A'), it's coerced to `NaN` (Not a Number). `downcast='integer'` optimizes memory by using the smallest possible integer type.
- **`number_categorical`**: Uses `np.where` to assign the original 'number' value if `number_numerical` is `NaN` (meaning it was originally non-numeric), otherwise it assigns `NaN`.

```python
# extract numerical part
df['number_numerical'] = pd.to_numeric(df["number"], errors='coerce', downcast='integer')

# extract categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(), df['number'], np.nan)
df.head()
```

The `df.head()` output after this step shows how 'A' from the original 'number' column is now correctly placed under 'number_categorical', while numerical values are in 'number_numerical'.

---

### 5. Inspecting 'Cabin' and 'Ticket' Columns

Before processing, we examine the unique values in the 'Cabin' and 'Ticket' columns to understand their structure and identify the mixed patterns.

```python
df['Cabin'].unique()
df['Ticket'].unique()
```

The output for both columns clearly shows a combination of letters, numbers, and sometimes special characters, confirming their mixed nature.

---

### 6. Separating 'Cabin' into Numerical and Categorical Parts

For the 'Cabin' column, we leverage string extraction methods:
- **`cabin_num`**: Extracts only the numerical part of the cabin string using a regular expression `(\d+)`. This captures one or more digits.
- **`cabin_cat`**: Extracts the first character of the cabin string, which typically represents the deck or cabin class.

```python
df['cabin_num'] = df['Cabin'].str.extract('(\d+)') # captures numerical part
df['cabin_cat'] = df['Cabin'].str[0] # captures the first letter
df.head()
```

---

### 7. Visualizing 'cabin_cat'

A bar plot is generated to show the distribution of the extracted cabin categories.

```python
df['cabin_cat'].value_counts().plot(kind='bar')
```

---

### 8. Separating 'Ticket' into Numerical and Categorical Parts

The 'Ticket' column is more complex, often containing prefixes followed by numbers. We handle it in two steps:
- **`ticket_num`**: Extracts the last space-separated part of the ticket string. This is then converted to a numeric type, with `errors='coerce'` handling any non-numeric last parts (turning them into `NaN`). `downcast='integer'` is applied for memory efficiency.
- **`ticket_cat`**: Extracts the first space-separated part of the ticket string. Then, `np.where` is used to set this to `NaN` if the extracted part is purely numeric (as those are likely just ticket numbers without a clear categorical prefix), otherwise, it keeps the extracted string.

```python
# extract the last bit of ticket as number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],
                                   errors='coerce',
                                   downcast='integer')

# extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan,
                              df['ticket_cat'])
df.head(20)
```

---

### 9. Inspecting Unique 'ticket_cat' Values

Finally, we look at the unique values in the newly created `ticket_cat` column to confirm that the categorical prefixes have been correctly extracted.

```python
df['ticket_cat'].unique()
```

This output shows a clean list of categorical prefixes like 'A/5', 'PC', 'STON/O2.', etc., and `NaN` where the ticket was purely numerical.

## Conclusion

By following these steps, we have successfully decomposed mixed variables ('number', 'Cabin', 'Ticket') into separate numerical and categorical features. This makes the data much more amenable to further analysis, visualization, and machine learning model training.
```
