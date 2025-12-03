/* Project: Inventory and Supply Chain Analysis
This project simulates a business need to track inventory levels and analyze supplier performance.

Setup: Three New Tables
We need three new tables to practice complex multi-table joins and aggregations. */

-- 1. Products Table (Product master list)
CREATE TABLE Products (
    ProductID INT PRIMARY KEY,
    ProductName VARCHAR(100) NOT NULL,
    Cost DECIMAL(10, 2) NOT NULL,
    ReorderLevel INT
);

INSERT INTO Products (ProductID, ProductName, Cost, ReorderLevel) VALUES
(1, 'Graphene Sheet', 15.00, 50),
(2, 'Quantum Chip', 45.00, 20),
(3, 'Nano Wire', 5.00, 100),
(4, 'Fusion Core', 200.00, 10);

-- 2. Inventory Table (Current stock levels)
CREATE TABLE Inventory (
    LocationID INT PRIMARY KEY,
    ProductID INT NOT NULL,
    StockQuantity INT NOT NULL,
    Warehouse VARCHAR(50)
);

INSERT INTO Inventory (LocationID, ProductID, StockQuantity, Warehouse) VALUES
(10, 1, 45, 'A_Main'),
(11, 2, 15, 'B_East'),
(12, 1, 60, 'B_East'),
(13, 3, 110, 'A_Main'),
(14, 4, 8, 'A_Main'),
(15, 4, 3, 'C_West');

-- 3. Suppliers Table (Supplier information)
CREATE TABLE Suppliers (
    SupplierID INT PRIMARY KEY,
    SupplierName VARCHAR(100) NOT NULL,
    LeadTimeDays INT -- Days it takes to fulfill an order
);

INSERT INTO Suppliers (SupplierID, SupplierName, LeadTimeDays) VALUES
(501, 'Alpha Tech', 10),
(502, 'Beta Supplies', 5),
(503, 'Gamma Parts', 20);

/*Inventory Optimization
This single query combines four tables (conceptually, by joining the Inventory and Products tables multiple times in different ways) and demonstrates Window Functions, Complex Joins, and Aggregation for a critical business decision.

Goal: Identify low-stock products and their highest stock location, while calculating the average cost across all locations.

The Consolidated Inventory Report Query */

SELECT
    P.ProductName,
    I_Main.Warehouse AS Main_Warehouse_Location,
    I_Main.StockQuantity AS Highest_Stock,
    AVG_Cost.Average_Product_Cost,
    -- Determine if the stock is below the reorder level
    CASE
        WHEN I_Main.StockQuantity < P.ReorderLevel THEN 'ACTION REQUIRED'
        ELSE 'Stock OK'
    END AS Reorder_Status
FROM
    Products AS P

-- 1. SELF-JOIN: Link Products to the Inventory table (I_Main) to find the absolute maximum stock quantity across all locations for each product.
INNER JOIN (
    SELECT
        ProductID,
        MAX(StockQuantity) AS Max_Quantity
    FROM
        Inventory
    GROUP BY
        ProductID
) AS Max_Stock_Finder
ON P.ProductID = Max_Stock_Finder.ProductID

-- 2. INNER JOIN: Link the result back to Inventory (I_Main) to find the actual Warehouse name associated with that Max_Quantity.
INNER JOIN
    Inventory AS I_Main
ON
    P.ProductID = I_Main.ProductID
    AND I_Main.StockQuantity = Max_Stock_Finder.Max_Quantity

-- 3. INNER JOIN: Link Products to an Aggregation (AVG_Cost) to get the average cost (if Cost were stored per-location). Here we use the Cost from the Product table as a stand-in for complexity.
INNER JOIN (
    SELECT
        ProductID,
        AVG(Cost) AS Average_Product_Cost
    FROM
        Products -- In a real scenario, this would be a cost history table
    GROUP BY
        ProductID
) AS AVG_Cost
ON P.ProductID = AVG_Cost.ProductID

ORDER BY
    Reorder_Status DESC,
    P.Cost DESC;