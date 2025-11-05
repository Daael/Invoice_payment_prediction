# Invoice Payment Prediction

This project presents a **real-world machine learning application** developed for a multinational telecommunications company.  
The primary goal is to **predict whether an invoice issued to a sister company will be paid or not**.
**Skills used**: Statistical Modelling, EDA, Data Cleaning, Feature Engineering, Model Selection and Evaluation, Hyperparameter Tunning, Virtual Machines etc.
**Technologies**: Python, Azure, Matplotlib, Numpy, Pandas, Seaborn, Sklearn, Sheets, Jupyter, etc.

If you want to deep dive in the techniques used, you can explore the entire machine learning process in the next Jupyter notebooks.
[Project Notebook](https://github.com/Daael/Invoice_payment_prediction/blob/main/Invoice_Prediction.ipynb).

---

## Overview

A global telecom subsidiary issues invoices to operators worldwide for technology products and services. These invoices are not always paid, or they take too long to be paid. The company wants to predict which newly generated invoices were most likely to go unpaid. 
The goal: Put pressure in the high risk invoices operators, while adapting budget and resource planning based only on invoices with a high likelihood of payment. This enables informed business decisions regarding operators with elevated non-payment risk.

**Goal**: Build a machine-learning-based risk scoring system to flag invoices likely to become overdue at the moment they are issued.

The data was provided by the subsidiary company historical invoice operations. Macroeconomic data was integrated in the analysis obtained from different institutions like the World Bank, IMF, the Country's government pages, etc. 

---

## Project Overview

The project is structured around two predictive models:
1. **Base Model** – Uses only internal company data.  
2. **Extended Model** – Incorporates additional macroeconomic indicators from the countries where the sister companies operate.

The dataset provided in this repository is **anonymized** and represents **5% of the original data** for efficiency.


- **Full dataset download:** [Google Drive link](https://drive.google.com/file/d/1hlmHiU9xRZyPFD9c9qu5ZDSIZHB5Q_FX/view?usp=sharing)

---

## Summary ##

The model achieved an **overall accuracy of over 93%** in predicting whether newly issued invoices would be paid. When an invoice was likely to be paid, the model performed exceptionally well, reaching 99% accuracy. While predicting non-payment is inherently more challenging due to class imbalance and uncertainty, the model still reached 50% accuracy in identifying high-risk invoices.

With this capability, **the company can now anticipate which invoices are unlikely to be settled and take proactive actions** with those operators. Beyond prediction, the analysis discovered some interesting insights. For example, macroeconomic indicators proved significant in the model, revealing that **a country’s economic stability plays a direct role in an operator’s ability to fulfill its financial obligations**.  

![Confusion Matrix Preview](https://raw.githubusercontent.com/Daael/Invoice_payment_prediction/main/Images/ConfusionMatrix.png)

---

## Insights and Recommendations ##

Some interesting Insights found were:

![Variables Preview](https://raw.githubusercontent.com/Daael/Invoice_payment_prediction/main/Images/Variables1.png)
- The macroeconomic variables included in the model were found to have significant influence on invoice payment predictions. This means that an operator’s financial reliability is closely related to the economic conditions of its country. This is a key insight: the company can proactively flag invoices from operators in unstable economic environments, allowing them to either delay issuance, allocate resources more cautiously, or apply targeted pressure on these high-risk invoices. By doing so, the subsidiary can potentially save substantial amounts and reduce financial exposure.

![Fees Preview](https://raw.githubusercontent.com/Daael/Invoice_payment_prediction/main/Images/Fees.png)
- One might assume that invoices with higher fees would be harder to pay (the more expensive the invoice, the greater the risk.). And the analysis confirmed that, the amount of fees in euros is important for the invoice risk assesment. Also, the identity of the product or service provider proved a good predictor. Along with the macroeconomic variables, we recommend focusing in the invoice amount and supplier.

![Coronavirus Preview](https://raw.githubusercontent.com/Daael/Invoice_payment_prediction/main/Images/Coronavirus.png)
- It seems that the risk of paying an invoice generated in the coronavirus period shows diffent behavior. Given this shift, we recommend training a separate model using only post-2022 data. This ensures the prediction engine reflects the current economic environment and more accurately represents the payment patterns of newly issued invoices, leading to more reliable risk assessment moving forward.

## Variables

### Independent Variables

| Column | Description |
|--------|-------------|
| `PURCHASE_CODE` (`CODIGO_COMPRAS`) | Code of the type of product ordered for the invoice. |
| `INVOICE_FEE` (`FEE_FACTURAS`) | Numeric quantity of the fee to pay. |
| `INVOICE_FEE_EUROS` (`FEE_FACTURAS_EUROS`) | Numeric quantity of the fee in euros. |
| `INVOICE_CURRENCY` (`MONEDA_FACTURA`) | Currency type of the issued invoice. |
| `OPERATOR` (`OPERADORA`) | Name of the company that ordered the invoice. |
| `OPERATOR_COUNTRY` (`OPERADORA_PAIS`) | Country of operation of the company. |
| `SUPPLIER_BAD_DEBT` (`PROVEEDOR_BAD_DEBT`) | Supplier had bankruptcy history or creditor issues. |
| `SUPPLIER_EVER_BLOCKED` (`PROVEEDOR_BLOQUEADO_ALGUNA_VEZ`) | Whether the supplier has been blocked in the past. |
| `SUPPLIER_RECONCILED` (`PROVEEDOR_CONCILIADO`) | Indicates if a conciliation process is required before invoicing. |
| `SUPPLIER_LOGISTICS` (`PROVEEDOR_LOGISTICO`) | Supplier responsible for logistics. |
| `SUPPLIER_TYPE` (`PROVEEDOR_TIPO`) | Size of the supplier (small, medium, etc.). |
| `SUPPLIER_HAS_GROUP` (`PROVEEDOR_TIENE_GRUPO`) | Whether the supplier is part of a group. |
| `SUPPLIER_GROUP` (`PROVEEDOR_GRUPO`) | Name of the group the supplier belongs to. |
| `EXPENSE_TYPE` (`TIPO_GASTO`) | Type of expense (investment, mobile, etc.). |

### Dependent Variable

| Column | Description |
|--------|-------------|
| `COLLECTED` (`COBRADO`) | Indicates if the invoice has been paid (`1`) or not (`0`). |

