# Write your MySQL query statement below
select w1.id
from weather as w1, weather as w2
where w1.temperature> w2.temperature 
and datediff(w1.recordDate, w2.recordDate)=1;

# Write an SQL query to find all dates' Id with higher temperatures compared to its previous dates (yesterday).
#Return the result table in any order.
#The query result format is in the following example.
