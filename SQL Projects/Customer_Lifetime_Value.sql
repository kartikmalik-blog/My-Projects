-- Project: Customer Lifetime Value (CLV) Analysis

/*Step 1: Create the Customers Table
This table stores basic demographic information.

SQL*/

CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    City VARCHAR(50),
    AcquisitionChannel VARCHAR(50) -- e.g., 'Web', 'Referral', 'Social'
);

/*Step 2: Insert Customer Data
SQL*/

INSERT INTO Customers (CustomerID, FirstName, LastName, City, AcquisitionChannel) VALUES
(101, 'Liam', 'Chen', 'New York', 'Web'),
(102, 'Olivia', 'Garcia', 'Los Angeles', 'Social'),
(103, 'Noah', 'Singh', 'Chicago', 'Referral'),
(104, 'Emma', 'Brown', 'New York', 'Web'),
(105, 'Ava', 'Jones', 'Chicago', 'Social');


/*Step 3: Create the Orders Table
This table stores transactional data, linked by CustomerID.

SQL*/

CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    CustomerID INT NOT NULL,
    OrderDate DATE NOT NULL,
    TotalAmount DECIMAL(10, 2) NOT NULL,
    ShippedStatus VARCHAR(20)
);

/*Step 4: Insert Order Data
SQL*/

INSERT INTO Orders (OrderID, CustomerID, OrderDate, TotalAmount, ShippedStatus) VALUES
(1001, 101, '2025-11-01', 50.00, 'Shipped'),
(1002, 103, '2025-11-01', 120.00, 'Shipped'),
(1003, 101, '2025-11-05', 75.50, 'Pending'),
(1004, 104, '2025-11-06', 200.00, 'Shipped'),
(1005, 102, '2025-11-08', 30.00, 'Shipped'),
(1006, 103, '2025-11-10', 45.00, 'Pending'),
(1007, 105, '2025-11-12', 150.00, 'Shipped');

/* 📊 Final SQL Project Challenge
Now, let's execute a complex analytical query that combines every concept we've mastered.

Goal: Calculate the Customer Lifetime Value (CLV) for each city and acquisition channel, and then rank the cities by their average CLV.

This single query demonstrates: Filtering, Joining, Aggregation, and Window Functions.

The Combined CLV Analysis Query */

SELECT
    T1.City,
    T1.AcquisitionChannel,
    T1.Total_Customer_Value,
    -- Step 3: Rank the City based on Total Value (Window Function)
    RANK() OVER (
        ORDER BY T1.Total_Customer_Value DESC
    ) AS City_CLV_Rank
FROM (
    -- Step 1 & 2: Join Orders and Customers, then Aggregate to find CLV per City/Channel
    SELECT
        C.City,
        C.AcquisitionChannel,
        SUM(O.TotalAmount) AS Total_Customer_Value
    FROM
        Orders AS O
    INNER JOIN
        Customers AS C
    ON
        O.CustomerID = C.CustomerID -- Linking the two tables
    WHERE
        O.ShippedStatus = 'Shipped' -- Filtering rows before aggregation
    GROUP BY
        C.City,
        C.AcquisitionChannel -- Grouping for CLV calculation
) AS T1 -- Alias for the aggregated results
ORDER BY
    City_CLV_Rank ASC;

    