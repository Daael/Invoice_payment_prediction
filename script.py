import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from itertools import combinations
import joblib

# Set the document path to the current directory
base_path = os.getcwd()

# Read the data file
df = pd.read_csv(os.path.join(base_path, "data.csv"))

# Translate the variable names from Spanish to English
df.rename(columns={
    'PROVEEDOR_BANCARROTA': 'SUPPLIER_BANKRUPTCY',
    'PROVEEDOR_BAD_DEBT': 'SUPPLIER_BAD_DEBT',
    'FECHA_FACTURA': 'DATE_INVOICE',
    'FECHA_ANIO_PRESUPESTARIO': 'DATE_BUDGET_YEAR',
    'FECHA_CREACION_FACTURA': 'DATE_INVOICE_CREATION',
    'CODIGO_COMPRAS': 'PURCHASE_CODE',
    'MONEDA_FACTURA': 'INVOICE_CURRENCY',
    'TIPO_GASTO': 'EXPENSE_TYPE',
    'OPERADORA': 'OPERATOR',
    'OPERADORA_PAIS': 'OPERATOR_COUNTRY',
    'AREA_ROSETA_NIVEL_2': 'ROSETTE_AREA_LEVEL_2',
    'PROVEEDOR': 'SUPPLIER',
    'PROVEEDOR_PAIS': 'SUPPLIER_COUNTRY',
    'PROVEEDOR_TIPO': 'SUPPLIER_TYPE',
    'PROVEEDOR_CONCILIADO': 'SUPPLIER_RECONCILED',
    'PROVEEDOR_TIENE_GRUPO': 'SUPPLIER_HAS_GROUP',
    'PROVEEDOR_GRUPO': 'SUPPLIER_GROUP',
    'PROVEEDOR_LOGISTICO': 'SUPPLIER_LOGISTICS',
    'PROVEEDOR_BLOQUEADO_ALGUNA_VEZ': 'SUPPLIER_EVER_BLOCKED',
    'CONDICION_DE_PAGO': 'PAYMENT_CONDITION',
    'FECHA_EMISION': 'ISSUE_DATE',
    'FECHA_CIERRE': 'CLOSING_DATE',
    'COBRADO': 'COLLECTED',
    'PORCENTAJE_COBRADO': 'PERCENT_COLLECTED',
    'FEE_FACTURAS_EUROS': 'INVOICE_FEE_EUROS',
    'FEE_FACTURAS': 'INVOICE_FEE'
}, inplace=True)


# Convert object type variable to numeric, replacing commas.
df['INVOICE_FEE'] = pd.to_numeric(df['INVOICE_FEE'].str.replace(',', ''), errors='coerce')
df['INVOICE_FEE_EUROS'] = pd.to_numeric(df['INVOICE_FEE_EUROS'].str.replace(',', ''), errors='coerce')

# Convert object type date variables to date format.
df['DATE_INVOICE'] = pd.to_datetime(df['DATE_INVOICE'], format='%d/%m/%Y')
df['DATE_BUDGET_YEAR'] = pd.to_datetime(df['DATE_BUDGET_YEAR'], format='%d/%m/%Y')
df['DATE_INVOICE_CREATION'] = pd.to_datetime(df['DATE_INVOICE_CREATION'], format='%d/%m/%Y')

# There are only two clear outliers; we should remove them.    
df = df.iloc[2:]

# Dropping useless columns
df = df.drop(columns=['DATE_BUDGET_YEAR', 'INVOICE_FEE', 'ISSUE_DATE', 'CLOSING_DATE', 'PERCENT_COLLECTED'])

# We use the mean of the corresponding "SUPPLIER" to do the imputation.
df['INVOICE_FEE_EUROS'] = df.groupby('SUPPLIER')['INVOICE_FEE_EUROS']\
                            .transform(lambda x: x.fillna(x.mean()))

# Dropping highly correlated features
df = df.drop(columns=['SUPPLIER', 'SUPPLIER_GROUP', 'ROSETTE_AREA_LEVEL_2', 'INVOICE_CURRENCY'])

# List of categorical variables to group rare categories
categorical_vars = ['OPERATOR', 'SUPPLIER_COUNTRY', 'PAYMENT_CONDITION', 'OPERATOR_COUNTRY']

# Apply grouping on the working DataFrame
for var in categorical_vars:
    counts = df[var].value_counts(normalize=True)
    rare = counts[counts < 0.0026].index
    df[var] = df[var].replace(rare, 'Other')

# We create a new column that will be the difference in days between the two dates
df['DAYS_DIFFERENCE'] = (df['DATE_INVOICE_CREATION'] - df['DATE_INVOICE']).dt.days

# We can drop the DATE_INVOICE_CREATION now, since its effect is represented in the variable created DAYS_DIFFERENCE
df = df.drop(columns=['DATE_INVOICE_CREATION'])

# Now we transform the remaining date column to use it in our model, we take a column for the year and one for the month.
df["YEAR_DATE_INVOICE"] = df["DATE_INVOICE"].dt.year
df["MONTH_DATE_INVOICE"] = df["DATE_INVOICE"].dt.month

# Map months to financial seasons / quarters
season_map = {
1: 'Q1', 2: 'Q1', 3: 'Q1',     # Jan, Feb, Mar
4: 'Q2', 5: 'Q2', 6: 'Q2',     # Apr, May, Jun
7: 'Q3', 8: 'Q3', 9: 'Q3',     # Jul, Aug, Sep
10: 'Q4', 11: 'Q4', 12: 'Q4'   # Oct, Nov, Dec
}
df['SEASON'] = df['MONTH_DATE_INVOICE'].map(season_map)

    
# Create the binary variable (seasonality variable). Define the coronavirus period
corona_start_date = pd.to_datetime('2020-01-01')
corona_end_date = pd.to_datetime('2021-12-31')
df['CORONAVIRUS_PERIOD'] = df['DATE_INVOICE'].apply(lambda x: 1 if corona_start_date <= x <= corona_end_date else 0)

# Create a copy of the data frame before adding the macroeconomic variables.
df_2 = df.copy()

# Grouping Venezuela with "Other".
df.loc[df['OPERATOR_COUNTRY'] == 'VE', 'OPERATOR_COUNTRY'] = 'Other'

# Load the dataset of the macroeconomic variables
ER = pd.read_excel(os.path.join(base_path, "Macro", "ExchangeRate", "ER.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
ER['Date'] = pd.to_datetime(ER['Date'], format='%Y-%m')
ER['YEAR_DATE_INVOICE'] = ER['Date'].dt.year
ER['MONTH_DATE_INVOICE'] = ER['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
ER.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, ER, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

IR = pd.read_excel(os.path.join(base_path, "Macro", "InterestRate", "IR.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
IR['Date'] = pd.to_datetime(IR['Date'], format='%Y-%m')
IR['YEAR_DATE_INVOICE'] = IR['Date'].dt.year
IR['MONTH_DATE_INVOICE'] = IR['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
IR.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, IR, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

UR = pd.read_excel(os.path.join(base_path, "Macro", "UnemploymentRate", "UR.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
UR['Date'] = pd.to_datetime(UR['Date'], format='%Y-%m')
UR['YEAR_DATE_INVOICE'] = UR['Date'].dt.year
UR['MONTH_DATE_INVOICE'] = UR['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
UR.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, UR, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

df.drop('Date_x', axis=1, inplace=True)
df.drop('Date_y', axis=1, inplace=True)

GDP = pd.read_excel(os.path.join(base_path, "Macro", "GDP", "GDP.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
GDP['Date'] = pd.to_datetime(GDP['Date'], format='%Y-%m')
GDP['YEAR_DATE_INVOICE'] = GDP['Date'].dt.year
GDP['MONTH_DATE_INVOICE'] = GDP['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
GDP.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, GDP, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

df.drop('Date_x', axis=1, inplace=True)
df.drop('Date_y', axis=1, inplace=True)

Inflation = pd.read_excel(os.path.join(base_path, "Macro", "Inflation", "Inflation.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
Inflation['Date'] = pd.to_datetime(Inflation['Date'], format='%Y-%m')
Inflation['YEAR_DATE_INVOICE'] = Inflation['Date'].dt.year
Inflation['MONTH_DATE_INVOICE'] = Inflation['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
Inflation.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, Inflation, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

Population = pd.read_excel(os.path.join(base_path, "Macro", "Population", "Population.xlsx"))

# Convert the "Date" column in the Excel data to datetime format and extract year and month:
Population['Date'] = pd.to_datetime(Population['Date'], format='%Y-%m')
Population['YEAR_DATE_INVOICE'] = Population['Date'].dt.year
Population['MONTH_DATE_INVOICE'] = Population['Date'].dt.month
#Changing the name of Excel column so it matches the name in English of our dataset
Population.rename(columns={'OPERADORA_PAIS': 'OPERATOR_COUNTRY'}, inplace=True)
Population.drop('Date', axis=1, inplace=True)

# Merging the dataframes on the common columns: 'Year', 'Month', and 'OPERADORA_PAIS'
df = pd.merge(df, Population, on=['YEAR_DATE_INVOICE', 'MONTH_DATE_INVOICE', 'OPERATOR_COUNTRY'], how='outer')

# Create a variable for the Human Development Index according to the World Bank
hdi_values = {
    'ES': 3,
    'BR': 1,
    'DE': 3,
    'AR': 2,
    'UK': 3,
    'PE': 1,
    'CL': 2,
    'CO': 1,
    'MX': 1,
    'EC': 1,
    'UY': 2,
    'SV': 0,
    'US': 3,
    'CR': 2,
    'PA': 1,
    'GT': 0,
    'NI': 0,
    'Other': 1
}
df['HDI'] = df['OPERATOR_COUNTRY'].map(hdi_values)

# Impute missing values using the mean within each OPERATOR_COUNTRY
macro_vars = ['ER_SDR', 'ER_USD', 'IR', 'UR', 'GDP_N', 'GDP_R', 
              'GDP_N_ChangeRate', 'GDP_R_ChangeRate', 'Population', 'Inflation']

for var in macro_vars:
    # Fill using group mean
    df[var] = df.groupby('OPERATOR_COUNTRY')[var].transform(lambda x: x.fillna(x.mean()))
    
    # Fill remaining NaNs (like 'Other') using overall mean
    df[var] = df[var].fillna(df[var].mean())


# During the outer merge, 59 rows were created for Operator_Country–month–year combinations with no corresponding invoices.
# These unmatched entries are removed to maintain data consistency.
df = df.dropna(subset=['DATE_INVOICE'])

# We drop the date-related columns for both datasets because we already used them for merging.
df = df.drop(columns=['DATE_INVOICE', 'Date', 'MONTH_DATE_INVOICE', 'YEAR_DATE_INVOICE'])
df_2 = df_2.drop(columns=['DATE_INVOICE', 'MONTH_DATE_INVOICE', 'YEAR_DATE_INVOICE'])

# We also drop the OPERATOR_COUNTRY variable due to its strong correlation with OPERATOR and because it has already been used for merging.
df = df.drop(columns=['OPERATOR_COUNTRY'])
df_2 = df_2.drop(columns=['OPERATOR_COUNTRY'])

# Drop the values with high VIF and check again
df = df.drop(columns=['ER_USD','ER_SDR', 'GDP_R', 'GDP_N_ChangeRate'])

### DATA SPLIT ###
# Doing one-hot encoding to the categorical variables (variables of type object)
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)

# Separate features and target variable
X = df.drop(columns=['COLLECTED'])
y = df['COLLECTED']

# Split the data into training (80%) and testing (20%) sets
# X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Split the training data into training (75%) and validation (25%) sets
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=4)

# Load the models
rf_model = joblib.load("final_model_random_forest.pkl")
gb_model = joblib.load("final_model_gradient_boosting.pkl")

### DATA SPLIT 2 ###
# Doing one-hot encoding to the categorical variables (variables of type object)
df_2 = pd.get_dummies(df_2, columns=df_2.select_dtypes(include=['object']).columns)

# Separate features and target variable
X_2 = df_2.drop(columns=['COLLECTED'])
y_2 = df_2['COLLECTED']

# Split the data into training (80%) and testing (20%) sets
# X_2_train_full, X_2_test, y_2_train_full, y_2_test = train_test_split(X_2, y_2, test_size=0.2, random_state=4)

# Split the training data into training (75%) and validation (25%) sets
# X_2_train, X_2_val, y_2_train, y_2_val = train_test_split(X_2_train_full, y_2_train_full, test_size=0.25, random_state=4)

# Load the models 2
rf_model_2 = joblib.load("final_model_2_random_forest.pkl")
gb_model_2 = joblib.load("final_model_2_gradient_boosting.pkl")

### Model ###

# Store models in dictionary
results = {
    "Gradient Boosting": gb_model,
    "Random Forest": rf_model
}

# Use the best model to predict on the test set
for model_name, best_model in results.items():
    
    # Predict on the test set
    y_test_pred = best_model.predict(X)
    
    # Evaluate the performance on the test set
    test_accuracy = accuracy_score(y, y_test_pred)
    test_report = classification_report(y, y_test_pred)
    conf_matrix = confusion_matrix(y, y_test_pred)
    
    # Calculate ROC curve
    y_test_prob = best_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    # Display the results
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_accuracy}")
    print("Test Classification Report:")
    print(test_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 80)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(ax=ax, cmap=plt.cm.Greens)

    # Customize the display
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout() 
    plt.show()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Store models in dictionary
results_2 = {
    "Gradient Boosting": gb_model_2,
    "Random Forest": rf_model_2
}

# Use the best model to predict on the test set
for model_name, best_model_2 in results_2.items():
    
    # Predict on the test set
    y_2_test_pred = best_model_2.predict(X_2)
    
    # Evaluate the performance on the test set
    test_accuracy_2 = accuracy_score(y_2, y_2_test_pred)
    test_report_2 = classification_report(y_2, y_2_test_pred)
    conf_matrix = confusion_matrix(y_2, y_2_test_pred)
    
    # Calculate ROC curve and AUC
    y_2_test_prob = best_model_2.predict_proba(X_2)[:, 1]
    fpr, tpr, _ = roc_curve(y_2, y_2_test_prob)
    roc_auc = auc(fpr, tpr)
    
    # Display the results
    print(f"Model: {model_name} on df_2")
    print(f"Test Accuracy: {test_accuracy_2}")
    print("Test Classification Report:")
    print(test_report_2)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 80)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



