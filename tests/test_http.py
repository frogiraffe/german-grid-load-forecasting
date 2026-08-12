from loadfc.data.http import resilient_session


def test_resilient_session_mounts_retrying_adapters():
    session = resilient_session()
    for scheme in ("http://", "https://"):
        retry = session.adapters[scheme].max_retries
        assert retry.total == 4
        assert retry.connect == 4
        assert retry.read == 4
        assert 429 in retry.status_forcelist
        assert retry.allowed_methods == frozenset({"GET"})
