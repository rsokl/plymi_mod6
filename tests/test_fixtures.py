import pytest


# When run, this test will be executed within a
# temporary directory that will automatically be
# deleted - along with all of its contents - once
# the test ends.
#
# Thus we can have this test write a file, and we
# need not worry about having it clean up after itself.
@pytest.mark.usefixtures("cleandir")
def test_writing_a_file():
    with open("a_text_file.txt", mode="w") as f:
        f.write("hello world")

    with open("a_text_file.txt", mode="r") as f:
        file_content = f.read()

    assert file_content == "hello world"


# We can use the `dummy_email` fixture to provide
# the same email address to many tests. In this
# way, if we need to change the email address, we
# can simply update the fixture and all of the tests
# will be affected by the update.
#
# Note that we don't need to use a decorator here.
# pytest is smart, and will see that the parameter-name
# `dummy_email` matches the name of our fixture. It will
# thus call these tests using the value returned by our
# fixture

def test_email1(dummy_email: str):
    assert "dummy" in dummy_email


def test_email2(dummy_email: str):
    assert "plymi" in dummy_email


def test_email3(dummy_email: str):
    assert ".com" in dummy_email