import re

import embodiverse


def test_version_is_three_numbers():
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", embodiverse.__version__)
