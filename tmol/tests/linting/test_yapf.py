import pytest

from yapf.yapflib import file_resources
from yapf.yapflib.yapf_api import FormatFile

from .targets import lint_files, basedir

style = file_resources.GetDefaultStyleForDir(basedir)


@pytest.mark.parametrize("filename", lint_files)
def test_reformat_file(filename):

    assert style
    diff, encoding, is_changed = FormatFile(
        filename, style_config=style, print_diff=True
    )
    assert not is_changed, f"File formatting error:\n{diff}"
