--Analytical queries
---Churn_distribution
SELECT
    Churned,
    COUNT(*) AS Customer_Count,
    CAST(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER()
        AS DECIMAL(5,2)
    ) AS Percentage
FROM CustomerChurn
GROUP BY Churned;

---Churn_by_country
SELECT
    Country,
    COUNT(*) AS Total_Customers,
    SUM(CAST(Churned AS INT)) AS Churned_Customers,
    CAST(
        SUM(CAST(Churned AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS Churn_Rate
FROM CustomerChurn
GROUP BY Country
ORDER BY Churn_Rate DESC;

---Churn_by_gender
SELECT
    Gender,
    COUNT(*) AS Total_Customers,
    SUM(CAST(Churned AS INT)) AS Churned_Customers,
    CAST(
        SUM(CAST(Churned AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS Churn_Rate
FROM CustomerChurn
GROUP BY Gender
ORDER BY Churn_Rate DESC;

---Churn_by_age_group
SELECT
    CASE
        WHEN Age < 25 THEN '<25'
        WHEN Age BETWEEN 25 AND 34 THEN '25-34'
        WHEN Age BETWEEN 35 AND 44 THEN '35-44'
        WHEN Age BETWEEN 45 AND 54 THEN '45-54'
        ELSE '55+'
    END AS Age_Group,
    COUNT(*) AS Total_Customers,
    SUM(CAST(Churned AS INT)) AS Churned_Customers,
    CAST(
        SUM(CAST(Churned AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS Churn_Rate
FROM CustomerChurn
GROUP BY
    CASE
        WHEN Age < 25 THEN '<25'
        WHEN Age BETWEEN 25 AND 34 THEN '25-34'
        WHEN Age BETWEEN 35 AND 44 THEN '35-44'
        WHEN Age BETWEEN 45 AND 54 THEN '45-54'
        ELSE '55+'
    END
ORDER BY Churn_Rate DESC;

---Churn_by_membership
SELECT
    Membership_Years,
    COUNT(*) AS Total_Customers,
    SUM(CAST(Churned AS INT)) AS Churned_Customers,
    CAST(
        SUM(CAST(Churned AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS Churn_Rate
FROM CustomerChurn
GROUP BY Membership_Years
ORDER BY Membership_Years;

---Behavior_by_churn 
SELECT
    Churned,
    AVG(Login_Frequency) AS Avg_Login_Frequency,
    AVG(Session_Duration_Avg) AS Avg_Session_Duration,
    AVG(Cart_Abandonment_Rate) AS Avg_Cart_Abandonment,
    AVG(Days_Since_Last_Purchase) AS Avg_Days_Last_Purchase,
    AVG(Returns_Rate) AS Avg_Return_Rate,
    AVG(Email_Open_Rate) AS Avg_Email_Open_Rate
FROM CustomerChurn
GROUP BY Churned;