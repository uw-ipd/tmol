def test_multiline_error():
    assert False, """
A multiline.
"quoted"
Error.
\tMessage.
"""
