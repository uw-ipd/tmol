def test_rama(default_database):
    db = default_database.scoring.rama

    alltables = [x.table_id for x in db.rama_tables]
    allrules = [x.table_id for x in db.rama_lookup]

    # ensure each table is defined
    for rrule in allrules:
        assert rrule in alltables

    # ensure there is a rule for each table, in the default database or in the
    # one that selects the symmetric glycine tables
    symm = default_database.with_symmetric_gly().scoring.rama
    reachable = set(allrules) | {x.table_id for x in symm.rama_lookup}
    for rtbl in alltables:
        assert rtbl in reachable
