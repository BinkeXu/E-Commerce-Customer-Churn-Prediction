
with rfm_data as (
    select * from {{ ref('int_customer_rfm') }}
),

scored_data as (
    select
        *,
        -- Churn Label (90 days threshold)
        case when Recency > 90 then 1 else 0 end as Churned,
        
        -- RFM Scores (1-5)
        -- Recency: Lower is better (Oldest/Highest Recency -> Score 1, Newest/Lowest Recency -> Score 5)
        -- NTILE(5) over DESC: 1=Highest(Oldest), 5=Lowest(Newest). Matches Python logic "1(old) to 5(recent)".
        ntile(5) over (order by Recency desc) as RecencyScore,
        
        -- Frequency: Higher is better (1=Low, 5=High)
        -- NTILE(5) over ASC: 1=Lowest, 5=Highest. Matches Python logic.
        ntile(5) over (order by Frequency asc) as FrequencyScore,
        
        -- Monetary: Higher is better (1=Low, 5=High)
        ntile(5) over (order by Monetary asc) as MonetaryScore

    from rfm_data
)

select
    *,
    (RecencyScore + FrequencyScore + MonetaryScore) as RFMScore
from scored_data
where Recency is not null
  and Frequency is not null
  and Monetary is not null
