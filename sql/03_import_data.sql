BULK INSERT CustomerChurn
FROM 'D:\Desktop\Projects\customer-churn-analytics\data\processed\customer_churn_clean.csv'
WITH
(
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO