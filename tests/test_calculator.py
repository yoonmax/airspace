from decimal import Decimal

import pytest

from airspace.miniapps.calculator.calculator import CalculatorModel
from airspace.miniapps.calculator.engine import CalcError, evaluate, format_number


@pytest.mark.parametrize("expr,expected", [
    ("1+2", "3"), ("2+3×4", "14"), ("(2+3)×4", "20"), ("10÷4", "2.5"), ("−5+2", "-3"), ("2×−3", "-6"),
    ("0.1+0.2", "0.3"), ("50%", "0.5"), ("200×10%", "20"), ("--4", "4"), ("1−2−3", "-4"), ("8÷2÷2", "2"),
    ("3.5×2", "7"), ("1/3×3", "1"),
])
def test_arithmetic(expr, expected):
    assert format_number(evaluate(expr)) == expected


@pytest.mark.parametrize("expr", ["1÷0", "1+", "(1+2", "1..2", "abc", "__import__('os')", "2**3", "", ")("])
def test_invalid_expressions_raise_calc_error(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_no_eval_or_names():
    with pytest.raises(CalcError):
        evaluate("open('x')")


def test_large_and_small_formatting():
    assert format_number(Decimal("1234567")) == "1,234,567"
    assert "e" in format_number(Decimal("1e30"))
    assert format_number(Decimal("-0.5")) == "-0.5"


def test_model_flow():
    m = CalculatorModel()
    for k in "12":
        m.press(k)
    m.press("+")
    m.press("3")
    assert m.result == "15"  # live preview
    m.press("=")
    assert m.just_evaluated and m.result == "15"
    m.press("×")  # continue from result
    m.press("2")
    m.press("=")
    assert m.result == "30"
    m.press("5")  # new number starts fresh
    assert m.expression == "5"


def test_model_decimal_and_operator_replacement():
    m = CalculatorModel()
    for k in [".", "5", ".", "5"]:
        m.press(k)
    assert m.expression == "0.55"
    m.press("+")
    m.press("×")  # operator replaced
    assert m.expression == "0.55×"


def test_model_errors_are_reported_not_raised():
    m = CalculatorModel()
    for k in ["1", "÷", "0", "="]:
        m.press(k)
    assert m.error == "Can't divide by zero"
    m.press("C")
    assert m.expression == "" and m.error is None


def test_backspace():
    m = CalculatorModel()
    for k in "123":
        m.press(k)
    m.press("⌫")
    assert m.expression == "12"
