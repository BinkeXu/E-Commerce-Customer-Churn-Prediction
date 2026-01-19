
with stg_transactions as (
    select * from {{ ref('stg_transactions') }}
),

max_date as (
    select max(InvoiceDate) as max_invoice_date from stg_transactions
),

customer_group as (
    select
        CustomerID,
        max(InvoiceDate) as LastPurchase,
        min(InvoiceDate) as FirstPurchase,
        count(distinct InvoiceNo) as TotalInvoices,
        count(distinct StockCode) as UniqueProducts,
        sum(TotalAmount) as TotalSpent
    from stg_transactions
    group by CustomerID
)

select
    t.CustomerID,
    -- RFM Metrics
    date_diff(m.max_invoice_date, t.LastPurchase, DAY) as Recency,
    t.TotalInvoices as Frequency,
    t.TotalSpent as Monetary,
    
    -- Time Features
    t.FirstPurchase,
    t.LastPurchase,
    date_diff(t.LastPurchase, t.FirstPurchase, DAY) as CustomerLifetime,
    date_diff(m.max_invoice_date, t.FirstPurchase, DAY) as DaysSinceFirstPurchase,
    
    -- AvgInterPurchaseTime (avoid divide by zero)
    case 
        when t.TotalInvoices > 1 then date_diff(t.LastPurchase, t.FirstPurchase, DAY) / (t.TotalInvoices - 1)
        else 0 
    end as AvgInterPurchaseTime,
    
    -- Behavioral Features
    safe_divide(t.TotalSpent, t.TotalInvoices) as AvgOrderValue,
    safe_divide(t.UniqueProducts, t.TotalInvoices) as ProductsPerOrder,
    safe_divide(t.TotalSpent, date_diff(m.max_invoice_date, t.FirstPurchase, DAY) + 1) as SpendingVelocity

from customer_group t
cross join max_date m
