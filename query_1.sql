select bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet, 
                    taxvaluedollarcnt, 
                    yearbuilt,
                    taxamount, 
                    fips, 
                    regionidzip
                from properties_2017
                left join propertylandusetype 
                    using(propertylandusetypeid)
                left join predictions_2017
                    using(parcelid)
                where propertylandusedesc IN (
                        'Single Family Residential',
                        'Inferred Single Family Residential')
                    and 
                        transactiondate between 
                            date('2017-01-01') and
                            date('2017-12-31')
                ;                
                

