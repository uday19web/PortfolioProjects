/*

Cleaning data in SQL Queries

*/

select * from `my-data-project12345-360606.Nashousing.housing`
---------------------------------------------------------------------------
-- Standardize Date Format
select SaleDate,cast(SaleDate AS date)
from `my-data-project12345-360606.Nashousing.housing` limit 10

-- Updating the standard date format
update `my-data-project12345-360606.Nashousing.housing`
set SaleDate = cast(SaleDate as date)

-------------------------------------------------------------------------------------

-- Populate property Address Data

-- Doing self join to check the null values in the PropertyAddress column
select
  T1.ParcelID, T1.PropertyAddress, T2.ParcelID, T2.PropertyAddress,
  ifnull(T1.PropertyAddress, T2.PropertyAddress) -- "ifnull" is used to replace the null value with given values
from `my-data-project12345-360606.Nashousing.housing` T1  
join `my-data-project12345-360606.Nashousing.housing` T2
on T1.ParcelID = T2.ParcelID
and T1.UniqueID <> T2.UniqueID
where T1.PropertyAddress is null 
limit 10

select  
  a.ParcelID, b.PropertyAddress, b.ParcelID, b.PropertyAddress
from `my-data-project12345-360606.Nashousing.housing` a
join `my-data-project12345-360606.Nashousing.housing` b
on a.ParcelID = b.ParcelID and a.UniqueID <> b.UniqueID
where a.PropertyAddress is null

select * from `my-data-project12345-360606.Nashousing.housing`
-- we are going to update the values in address with parcelid as reference to replace the null 
-- this human error to missed to update address  we came to know by using self join in above query

update T1
set PropertyAddress = ifnull(T1.PropertyAddress, T2.PropertyAddress)
from `my-data-project12345-360606.Nashousing.housing` T1  
join `my-data-project12345-360606.Nashousing.housing` T2
on T1.ParcelID = T2.ParcelID
and T1.UniqueID <> T2.UniqueID
where T1.PropertyAddress is null 

-----------------------------------------------------------------
-- Separating the propertyaddress column into Individual Column(address, City, State)
select 
  substring(PropertyAddress, 1, instr(PropertyAddress, ',',1, 1)-1) as address,
  substring(PropertyAddress, instr(PropertyAddress, ',',1, 1) + 1, length(PropertyAddress)) as city
from `my-data-project12345-360606.Nashousing.housing`

-- then created new column in the table by using Alter command

-- separating the owner address into individual column
select
 substring(OwnerAddress, 1, instr(OwnerAddress, ',',1, 1)-1) as address,
  substring(OwnerAddress, instr(OwnerAddress, ',',1, 1) + 1, (instr(OwnerAddress, ',',1, 2)- instr(OwnerAddress, ',',1, 1))-1) as city,
  substring(OwnerAddress, instr(OwnerAddress, ',',1, 2) + 1, length(OwnerAddress)) as state
from `my-data-project12345-360606.Nashousing.housing`

-- then created new column in the table by using Alter command

---------------------------------------------------------------------------

-- Change Y and N   to Yes and No in "Sold as Vacant" field
-- using case statement

select 
  case 
       when SoldAsVacant = 'Y' then 'Yes'
       when SoldAsVacant = 'N' then 'No'
       ELSE SoldAsVacant 
  end
from `my-data-project12345-360606.Nashousing.housing`

-- then update the column in table

-------------------------------------------------------------------------
-- Remove Duplicates

with rownumtemp as ( -- creating temp table to use the alias name "rownum"
select *,
  row_number() over(
    partition by ParcelID, PropertyAddress, SalePrice, SaleDate, LegalReference
    order by UniqueID
  ) as row_num
from `my-data-project12345-360606.Nashousing.housing`
-- order by ParcelID
)
delete -- deleting the duplicate row
from rownumtemp 
where row_num > 2
order by PropertyAddress

--------------------------------------------------------------------------
-- deleting unused columns
alter table `my-data-project12345-360606.Nashousing.housing`
drop column OwnerAddress, TaxDistrict, PropertyAddress, SaleDate
