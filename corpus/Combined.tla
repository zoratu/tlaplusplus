---- MODULE Combined ----
EXTENDS Naturals, FiniteSets

CONSTANTS MaxPrice, MinPrice

VARIABLES listings, sold, revenue

ValidListing(l) ==
    /\ l.price >= MinPrice
    /\ l.price <= MaxPrice
    /\ l.id >= 0

AllListingsValid == \A l \in listings : ValidListing(l)

TypeOK ==
    AllListingsValid /\ sold \subseteq listings /\ revenue >= 0

Init ==
    /\ listings = {
        [id |-> 1, price |-> 100],
        [id |-> 2, price |-> 200],
        [id |-> 3, price |-> 150]
       }
    /\ sold = {}
    /\ revenue = 0

Sell(listing) ==
    /\ listing \in listings
    /\ ~(listing \in sold)
    /\ sold' = sold \union {listing}
    /\ revenue' = revenue + listing.price
    /\ UNCHANGED listings

AddListing(id, price) ==
    /\ price >= MinPrice
    /\ price <= MaxPrice
    /\ \A l \in listings : l.id # id
    /\ listings' = listings \union {[id |-> id, price |-> price]}
    /\ UNCHANGED <<sold, revenue>>

ValidIDs == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
ValidPrices == {50, 100, 150, 200, 250, 300}

Next ==
    \/ \E l \in listings : Sell(l)
    \/ \E id \in ValidIDs, price \in ValidPrices : AddListing(id, price)

Spec == Init /\ [][Next]_<<listings, sold, revenue>>

RevenueNonNegative == revenue >= 0

AllSoldItemsHaveValidPrices ==
    \A l \in sold : l.price >= MinPrice /\ l.price <= MaxPrice

====
