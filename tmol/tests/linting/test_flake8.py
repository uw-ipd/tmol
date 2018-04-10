import pytest

from flake8.main import application

from .targets import lint_files


def flake8_file(filename):
    """Run flake8 over a single file, and return the number of failures."""
    args = []
    app = application.Application()
    app.parse_preliminary_options_and_args(args)
    app.make_config_finder()
    app.find_plugins()
    app.register_plugin_options()
    app.parse_configuration_and_cli(args)
    app.make_formatter()  # fix this
    app.make_notifier()
    app.make_guide()
    app.make_file_checker_manager()
    app.run_checks([str(filename)])
    app.formatter.start()
    app.report_errors()
    # app.report_statistics()
    # app.report_benchmarks()
    app.formatter.stop()

    return app.result_count


@pytest.mark.parametrize("filename", lint_files)
def test_flake8_file(filename):
    assert not flake8_file(filename), f"flake8 errors in file: {filename}"
