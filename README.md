# Invoice Payment Prediction

This project presents a **real-world machine learning application** developed for a multinational telecommunications company.  
The primary goal is to **predict whether an invoice issued to a sister company will be paid or not**.

You can explore the entire machine learning process in the included Jupyter notebooks.

---

## Project Overview

The project is structured around two predictive models:

1. **Base Model** – Uses only internal company data.  
2. **Extended Model** – Incorporates additional macroeconomic indicators from the countries where the sister companies operate.
The models are used by the company according to their needs and and preferences

The dataset provided in this repository is **anonymized** and represents **5% of the original data** for efficiency.  

- **Full dataset download:** [Google Drive link](https://drive.google.com/file/d/1hlmHiU9xRZyPFD9c9qu5ZDSIZHB5Q_FX/view?usp=sharing)

---

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

