use zillow;

select * from predictions_2017
where transactiondate between 
	date('2017-01-01') and
    date('2017-12-31');
    
select distinct assessmentyear from properties_2017
	left join predictions_2017
		using(parcelid)
	left join propertylandusetype 
		using(propertylandusetypeid)
	where propertylandusedesc IN (
		'Single Family Residential',
		'Inferred Single Family Residential')
	and 
		transactiondate between 
			date('2017-01-01') and
			date('2017-12-31')
;


select 
p6.taxvaluedollarcnt as p6_value, p7.taxvaluedollarcnt as p7_value, p6.yearbuilt,
p6.taxamount as p6_tax, p7.taxamount as p7_tax,
p6.assessmentyear as ass_year_6, p7.assessmentyear as ass_year_7
from properties_2017 as p7
right join properties_2016 as p6
using (parcelid)
left join propertylandusetype as plut
on p6.propertylandusetypeid = plut.propertylandusetypeid
left join predictions_2017 as pr7
using(parcelid)
left join predictions_2016 as pr6
using(parcelid)
where propertylandusedesc IN (
'Single Family Residential',
'Inferred Single Family Residential')
and 
pr6.transactiondate between 
	date('2016-01-01') and
    date('2016-12-31')
;


select  p7.bedroomcnt, 
		p7.bathroomcnt, 
		p7.calculatedfinishedsquarefeet, 
		p7.taxvaluedollarcnt, 
		p7.yearbuilt,
		p7.fips, 
		p7.regionidzip, 
		p6.taxvaluedollarcnt as tax_value_2016, 
        p7.taxvaluedollarcnt as tax_value_2017, 
        p6.taxamount as tax_2016, 
        p7.taxamount as tax_2017
from properties_2017 as p7
left join properties_2016 as p6
using (parcelid)
left join propertylandusetype as plut
on p7.propertylandusetypeid = plut.propertylandusetypeid
left join predictions_2017 as pr7
using(parcelid)
where propertylandusedesc IN (
'Single Family Residential',
'Inferred Single Family Residential')
and 
transactiondate between 
	date('2017-01-01') and
    date('2017-12-31')
;

