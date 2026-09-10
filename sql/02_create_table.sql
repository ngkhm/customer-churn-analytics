CREATE TABLE CustomerChurn
(
    Age FLOAT,
    Gender VARCHAR(20),
    Country VARCHAR(50),
    City VARCHAR(100),
    Membership_Years FLOAT,
    Login_Frequency FLOAT,
    Session_Duration_Avg FLOAT,
    Pages_Per_Session FLOAT,
    Cart_Abandonment_Rate FLOAT,
    Wishlist_Items FLOAT,
    
    Total_Purchases FLOAT,
    Average_Order_Value FLOAT,
    Days_Since_Last_Purchase FLOAT,
    Discount_Usage_Rate FLOAT,
    Returns_Rate FLOAT, 
    
    Email_Open_Rate FLOAT,
    Customer_Service_Calls FLOAT,
    Product_Reviews_Written FLOAT,
    Social_Media_Engagement_Score FLOAT,
    Mobile_App_Usage FLOAT,
    Payment_Method_Diversity FLOAT,
    
    Lifetime_Value FLOAT,
    Credit_Balance FLOAT,
    Churned BIT,
    Signup_Quarter VARCHAR(10)
);
GO