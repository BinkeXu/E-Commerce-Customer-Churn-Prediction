
with raw_data as (
    select * from {{ source('ecommerce', 'OnlineRetail') }}
)

select
    cast(CustomerID as int64) as CustomerID,
    parse_timestamp('%m/%d/%Y %H:%M', InvoiceDate) as InvoiceDate,
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    UnitPrice,
    Country,
    (Quantity * UnitPrice) as TotalAmount
from raw_data
where CustomerID is not null
  and Quantity > 0
  and UnitPrice > 0
  and not starts_with(InvoiceNo, 'C')
