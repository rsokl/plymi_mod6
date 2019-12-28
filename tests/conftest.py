import os
import tempfile

import pytest


@pytest.fixture()
def cleandir() -> str:
    """ This fixture will use the stdlib `tempfile` module to
    change the current working directory to a tmp-dir for the
    duration of the test.

    Afterwards, the test session returns to its previous working
    directory, and the temporary directory and its contents
    will be automatically deleted.

    Yields
    ------
    str
        The name of the temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()
        os.chdir(tmpdirname)
        yield tmpdirname
        os.chdir(old_dir)


@pytest.fixture()
def dummy_email() -> str:
    """ This fixture will simply have pytest pass the string:
                   'dummy.email@plymi.com'
    to any test-function that has the parameter name `dummy_email` in
    its signature.
    """
    return "dummy.email@plymi.com"
