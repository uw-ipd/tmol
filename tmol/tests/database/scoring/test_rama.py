def test_rama(default_database):
    db = default_database.scoring.rama

    # assert len(db.rama_tables) == 40
    # assert len(db.rama_lookup) == 42

    alltables = [x.name for x in db.rama_tables]
    allrules = [x.name for x in db.rama_lookup]

    # ensure each table is defined
    for rrule in allrules:
        assert rrule in alltables

    # ensure there is a rule for each table
    for rtbl in alltables:
        assert rtbl in allrules
