-- 1. Kiểm tra tổng số record
SELECT COUNT(*) AS Total_Customers
FROM CustomerChurn;

-- 2. Kiểm tra giá trị NULL ở các cột quan trọng
SELECT
    COUNT(*) - COUNT(Age) AS Age_Null,
    COUNT(*) - COUNT(Gender) AS Gender_Null,
    COUNT(*) - COUNT(Country) AS Country_Null,
    COUNT(*) - COUNT(Membership_Years) AS Membership_Null,
    COUNT(*) - COUNT(Login_Frequency) AS Login_Null,
    COUNT(*) - COUNT(Lifetime_Value) AS LTV_Null,
    COUNT(*) - COUNT(Churned) AS Churn_Null
FROM CustomerChurn;

-- 3. Kiểm tra các giá trị bất thường (ví dụ: Tuổi)
SELECT *
FROM CustomerChurn
WHERE Age < 0 OR Age > 100;