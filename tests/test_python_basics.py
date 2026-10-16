import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number, mse,
    multiplicative_persistence, prime_factorization, pyramid,
)


def P(n):
    return PositiveIntegerInput(n)


# Задача 1
@pytest.mark.parametrize("text, expected", [
    ("hello", 2), ("AEIOU", 5), ("aEiOu", 5), ("bcdfg", 0), ("", 0), ("y", 0),
    ("Programming", 3),
])
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


# Задача 2
@pytest.mark.parametrize("text, expected", [
    ("abc", True), ("abca", False), ("", True), ("a", True), ("aA", True),
    ("  ", False), ("12 3", True),
])
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


# Задача 3
@pytest.mark.parametrize("number, expected", [
    (1, 1), (2, 1), (7, 3), (8, 1), (255, 8), (1023, 10), (2**64, 1), (2**64 - 1, 64),
])
def test_count_one_bits(number, expected):
    assert count_one_bits(P(number)) == expected


# Задача 4
@pytest.mark.parametrize("number, expected", [
    (39, 3), (4, 0), (999, 4), (1, 0), (9, 0), (10, 1), (25, 2), (277777788888899, 11),
])
def test_multiplicative_persistence(number, expected):
    assert multiplicative_persistence(P(number)) == expected


# Задача 5
@pytest.mark.parametrize("predicted, expected_vec, expected", [
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([1, 2, 3], [2, 3, 4], 1.0),
    ([0, 0], [3, 4], 12.5),
    ([5], [-5], 100.0),
    ((1, 2), (3, 2), 2.0),
])
def test_mse(predicted, expected_vec, expected):
    assert mse(VectorPairInput(predicted, expected_vec)) == pytest.approx(expected)


def test_mse_errors():
    with pytest.raises(ValueError):
        mse(VectorPairInput([1, 2], [1]))
    with pytest.raises(ValueError):
        mse(VectorPairInput([], []))


# Задача 6
@pytest.mark.parametrize("number, expected", [
    (86240, "(2**5)(5)(7**2)(11)"),
    (2, "(2)"), (13, "(13)"), (12, "(2**2)(3)"), (1024, "(2**10)"),
    (7919, "(7919)"), (2 * 7919, "(2)(7919)"), (1, ""),
    (9_999_991 ** 2, "(9999991**2)"),
])
def test_prime_factorization(number, expected):
    assert prime_factorization(P(number)) == expected


# Задача 7
@pytest.mark.parametrize("number, expected", [
    (1, 1), (5, 2), (14, 3), (30, 4), (385, 10), (338350, 100),
    (2, "It is impossible"), (4, "It is impossible"), (15, "It is impossible"),
    (338351, "It is impossible"),
])
def test_pyramid(number, expected):
    assert pyramid(P(number)) == expected


# Задача 8
@pytest.mark.parametrize("number, expected", [
    (1234006, True), (123456, False), (7, True), (12, True), (959, True),
    (1230987, False), (56239814, True), (1001, True), (1203, False),
])
def test_is_balanced_number(number, expected):
    assert is_balanced_number(P(number)) is expected
