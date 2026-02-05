from tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "hello\nworld"
    tok = CharTokenizer()
    tok.fit(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
