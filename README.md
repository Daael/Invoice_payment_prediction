# Invoice_payment_prediction
The following project presents a real-world machine learning application developed for a multinational telecommunications company. Its primary goal is to predict whether an invoice issued to a sister company will be paid or not.
You can see the whole machine learning process in the next jupyter notebook:


The project is structured around two predictive models:
Base Model –> uses only internal company data.
Extended Model –> incorporates additional macroeconomic indicators from the countries where the sister companies operate.


The dataset provided is anonimized; The csv file in this repository represents 5% of the original data for efficiency reasons. If you wish to run the project on the full dataset, it can be downloaded from the following source:
https://docs.google.com/spreadsheets/d/1ZN4CqO7DnnjoYrlBbVqBUcjt8lK7KGRw/edit?usp=sharing&ouid=103397986873263038565&rtpof=true&sd=true

Independent variables:
- PURCHASE_CODE (CODIGO_COMPRAS): Code of the type of product that was ordered for the invoice.
- INVOICE_FEE (FEE_FACTURAS): Numeric quantity of the fee to pay.
- INVOICE_FEE_EUROS (FEE_FACTURAS_EUROS): Numeric quantity of the fee to pay in euros.
- INVOICE_CURRENCY (MONEDA_FACTURA): Currency type of the issued invoice.
- OPERATOR (OPERADORA): Name of the company that ordered the Invoice.
- OPERATOR_COUNTRY (OPERADORA_PAIS): Country of operation of the company that ordered the invoice.
- SUPPLIER_BAD_DEBT (PROVEEDOR_BAD_DEBT): Suppliers had declared bankruptcy in the past or had problems with creditors.
- SUPPLIER_EVER_BLOCKED (PROVEEDOR_BLOQUEADO_ALGUNA_VEZ): If the Supplier has not paid on time in the past and has been blocked.
- SUPPLIER_RECONCILED (PROVEEDOR_CONCILIADO): It defines if a conciliation process has to be done before emitting the invoice.
- SUPPLIER_LOGISTICS (PROVEEDOR_LOGISTICO): Supplier that does the logistics of the operation.
- SUPPLIER_TYPE (PROVEEDOR_TIPO): Size of the supplier (small, medium, etc).
- SUPPLIER_HAS_GROUP (PROVEEDOR_TIENE_GRUPO): Indicates if the supplier is part of a group.
- SUPPLIER_GROUP (PROVEEDOR_GRUPO): Name of the group that is part of.
- EXPENSE_TYPE (TIPO_GASTO): Type of expense (investment, movil, etc.).

Dependent variable:
- COLLECTED (COBRADO): Shows if the invoice has been paid or not.

