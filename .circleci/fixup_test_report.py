#!/usr/bin/env python3
"""Fixup pytest junit reports to propertly escape multiline error messages.

Workaround for pytest issue 1218:

https://github.com/pytest-dev/pytest/issues/1218

Could be resolved via update to peform escaping:
https://github.com/pytest-dev/pytest/blob/master/_pytest/junitxml.py#L193
"""

import re
import argparse

failure_message = r"failure message=\".*?\""
escapes = {"\n": '&#10;', "\r": '&#13;', "\t": '&#9;'}


def escape_failure(match):
    text = match.group()
    for f, t in escapes.items():
        text = text.replace(f, t)
    return text


def main():
    parser = argparse.ArgumentParser(
        description='Fixup whitespace escaping in junit test reports.'
    )
    parser.add_argument(
        'filename', type=str, nargs='+', help='Input junit test report.'
    )
    args = parser.parse_args()

    for f in args.filename:
        with open(f) as inf:
            contents = inf.read()

        contents = re.sub(
            failure_message,
            escape_failure,
            contents,
            flags=re.MULTILINE | re.DOTALL
        )

        with open(f, "w") as outf:
            outf.write(contents)


if __name__ == "__main__":
    main()
