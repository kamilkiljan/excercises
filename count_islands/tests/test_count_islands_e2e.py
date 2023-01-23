import subprocess
import sys


def test_count_islands_e2e():
    result = subprocess.run(
        [sys.executable, "../count_islands.py", "./files/test_input.txt"], capture_output=True, text=True
    )
    assert not result.stderr
    assert result.stdout == "4\n"
