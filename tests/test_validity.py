import pytest
from scripts.routing.validity import is_query_valid, QueryValidity


def test_empty_query():
    res = is_query_valid("")
    assert res.validity == QueryValidity.INVALID


def test_symbols_only():
    res = is_query_valid("!!!@@@")
    assert res.validity == QueryValidity.INVALID


def test_too_short():
    res = is_query_valid("a")
    assert res.validity == QueryValidity.INVALID


def test_repeated_char():
    res = is_query_valid("aaaaaa")
    assert res.validity == QueryValidity.INVALID


def test_valid_simple_query():
    res = is_query_valid("iphone 15 price")
    assert res.validity == QueryValidity.VALID
    assert 0 <= res.validity_confidence <= 1


def test_valid_comparison_query():
    res = is_query_valid("iphone vs samsung")
    assert res.validity == QueryValidity.VALID


def test_noise_low_quality():
    res = is_query_valid("xqzzzzzz")
    assert res.validity == QueryValidity.INVALID

def test_small_electronics():
    res = is_query_valid("ps5 price")
    resw = is_query_valid("ps5")
    assert res.validity == QueryValidity.VALID
    assert resw.validity == QueryValidity.VALID


def test_normal():
    res = is_query_valid("best laptop under 50k")
    assert res.validity == QueryValidity.VALID
