from pathlib import Path

def test_ui_has_no_emoji():
    text=(Path(__file__).parents[1]/"app.py").read_text()
    ranges=[(0x1F300,0x1FAFF),(0x2600,0x27BF),(0x1F1E6,0x1F1FF)]
    assert not any(any(lo<=ord(ch)<=hi for lo,hi in ranges) for ch in text)
