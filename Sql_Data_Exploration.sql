SELECT
  *
FROM
  `my-data-project12345-360606.covid.Death`
ORDER BY
  location,
  date
LIMIT
  10 --
SELECT
  -- * --
FROM
  `my-data-project12345-360606.covid.vaccination` --
ORDER BY
  location,
  date --
LIMIT
  10 --
SELECT
  DATA that we are going TO be
USING
SELECT
  location,
  date,
  total_cases,
  new_cases,
  total_deaths,
  population
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  continent IS NOT NULL
ORDER BY
  1,
  2 -- Looking AT total cases vs total deaths
SELECT
  location,
  date,
  total_cases,
  total_deaths,
  (total_deaths/ total_cases) * 100 AS DeathPercentage -- calculating total percentage
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  location LIKE 'India%'
  AND continent IS NOT NULL
ORDER BY
  1,
  2 -- Looking AT Total Cases vs Population -- Shows What percentage OF population got Covid
SELECT
  location,
  date,
  total_cases,
  population,
  (total_cases/ population) * 100 AS Affected_Population_Percentage -- calculating total percentage
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  location LIKE 'India%'
  AND continent IS NOT NULL
ORDER BY
  1,
  2 -- Looking AT countries
WITH
  highest infection rate compared TO population
SELECT
  location,
  population,
  MAX(total_cases) AS HighestInfectionCount,
  MAX((total_cases/population)) * 100 AS InfectedPopulationPercentage
FROM
  `my-data-project12345-360606.covid.Death` --
WHERE
  location = 'India'
WHERE
  continent IS NOT NULL
GROUP BY
  location,
  population
ORDER BY
  InfectedPopulationPercentage DESC -- Showing Countries
WITH
  Highest Death Count per Population
SELECT
  location,
  MAX(total_deaths) AS TotalDeathCount
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  continent IS NOT NULL
GROUP BY
  location
ORDER BY
  2 DESC -- looking highest death count BY Continents
SELECT
  continent,
  MAX(total_deaths) AS TotalDeathCount
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  continent IS NOT NULL
GROUP BY
  continent
ORDER BY
  2 DESC --Global Numbers
SELECT
  date,
  SUM(new_cases) AS SumOfNewCases,
  SUM(new_deaths) AS SumOfNewDeath,
  SUM(new_deaths) / SUM(new_cases) * 100 AS DeathPercentage
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  continent IS NOT NULL
GROUP BY
  date
ORDER BY
  1,
  2 -- Total global numbers
SELECT
  SUM(new_cases) AS TotalCases,
  SUM(new_deaths) AS TotalDeaths,
  SUM(new_deaths) / SUM(new_cases) * 100 AS DeathPercentage
FROM
  `my-data-project12345-360606.covid.Death`
WHERE
  continent IS NOT NULL
ORDER BY
  1,
  2 -- joining the vaccination
  AND death dataset BY date
  AND location -- looking AT total population vs vaccination
SELECT
  dea.continent,
  dea.location,
  dea.date,
  dea.population,
  vac.new_vaccinations,
  SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS PeopleVaccination
FROM
  `my-data-project12345-360606.covid.Death` AS dea
JOIN
  `my-data-project12345-360606.covid.vaccination` AS vac
ON
  dea.location = vac.location
  AND dea.date = vac.date
WHERE
  dea.continent IS NOT NULL
ORDER BY
  2,
  3
LIMIT
  10 -- creating TEMP TABLE
WITH
  PopVsVac AS (
  SELECT
    dea.continent,
    dea.location,
    dea.date,
    dea.population,
    vac.new_vaccinations,
    SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS PeopleVaccination
  FROM
    `my-data-project12345-360606.covid.Death` AS dea
  JOIN
    `my-data-project12345-360606.covid.vaccination` AS vac
  ON
    dea.location = vac.location
    AND dea.date = vac.date
  WHERE
    dea.continent IS NOT NULL --
  ORDER BY
    2,
    3 )
SELECT
  *,
FROM
  PopVsVac
SELECT
  dea.continent,
  dea.location,
  dea.date,
  dea.population,
  vac.new_vaccinations,
  SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date) AS PeopleVaccination,
  (SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date)/dea.population) * 100
FROM
  `my-data-project12345-360606.covid.Death` AS dea
JOIN
  `my-data-project12345-360606.covid.vaccination` AS vac
ON
  dea.location = vac.location
  AND dea.date = vac.date
WHERE
  dea.continent IS NOT NULL
ORDER BY
  2,
  3

--------------------------
select 
  dea.continent as Continent, dea.location as Location, dea.date as Date, dea.population as Population, vac.new_vaccinations as New_vaccinations,
  sum(vac.new_vaccinations) over (partition by dea.location order by dea.location, dea.date) as PeopleVaccination
from `my-data-project12345-360606.covid.Death` as dea
join `my-data-project12345-360606.covid.vaccination` as vac
on
  dea.location = vac.location and dea.date = vac.date


