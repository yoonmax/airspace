"""Safe arithmetic evaluation.

A tokenizer and recursive-descent parser over :class:`decimal.Decimal`.
Only numbers, + − × ÷ %, parentheses and unary signs are accepted — there is
no use of ``eval`` and no way to reach Python objects.

Grammar::

    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := ('+' | '-') factor | postfix
    postfix:= primary '%'*
    primary:= NUMBER | '(' expr ')'
"""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext

getcontext().prec = 28

OPERATORS = {"+", "-", "*", "/", "%", "(", ")"}
ALIASES = {"×": "*", "x": "*", "÷": "/", "−": "-", "–": "-", ",": ""}
MAX_LENGTH = 200


class CalcError(ValueError):
    pass


def tokenize(text: str) -> list[str]:
    if len(text) > MAX_LENGTH:
        raise CalcError("Expression too long")
    tokens: list[str] = []
    i = 0
    s = "".join(ALIASES.get(ch, ch) for ch in text)
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            dots = 0
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                dots += s[j] == "."
                j += 1
            if dots > 1:
                raise CalcError("Invalid number")
            num = s[i:j]
            if num == ".":
                raise CalcError("Invalid number")
            tokens.append(num)
            i = j
        elif ch in OPERATORS:
            tokens.append(ch)
            i += 1
        else:
            raise CalcError(f"Unexpected '{ch}'")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise CalcError("Incomplete expression")
        self.pos += 1
        return tok

    def parse(self) -> Decimal:
        if not self.tokens:
            raise CalcError("Empty expression")
        value = self.expr()
        if self.peek() is not None:
            raise CalcError("Invalid expression")
        return value

    def expr(self) -> Decimal:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self) -> Decimal:
        value = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            rhs = self.factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise CalcError("Can't divide by zero")
                value = value / rhs
        return value

    def factor(self) -> Decimal:
        if self.peek() in ("+", "-"):
            op = self.take()
            self.depth += 1
            if self.depth > 50:
                raise CalcError("Too many signs")
            v = self.factor()
            self.depth -= 1
            return v if op == "+" else -v
        return self.postfix()

    def postfix(self) -> Decimal:
        value = self.primary()
        while self.peek() == "%":
            self.take()
            value = value / Decimal(100)
        return value

    def primary(self) -> Decimal:
        tok = self.take()
        if tok == "(":
            self.depth += 1
            if self.depth > 50:
                raise CalcError("Too deeply nested")
            value = self.expr()
            if self.take() != ")":
                raise CalcError("Missing ')'")
            self.depth -= 1
            return value
        if tok in OPERATORS:
            raise CalcError("Invalid expression")
        try:
            return Decimal(tok)
        except InvalidOperation as exc:
            raise CalcError("Invalid number") from exc


def evaluate(text: str) -> Decimal:
    try:
        result = _Parser(tokenize(text)).parse()
    except (InvalidOperation, DivisionByZero) as exc:
        raise CalcError("Invalid operation") from exc
    if not result.is_finite():
        raise CalcError("Result out of range")
    return result


def format_number(value: Decimal, max_digits: int = 14) -> str:
    if value == 0:
        return "0"
    # Round to 12 significant digits so 1÷3×3 shows 1, not 0.9999999999999.
    q = Decimal(format(value, ".12g")).normalize()
    if q == 0:
        return "0"
    adjusted = q.adjusted()
    if -7 < adjusted < max_digits:
        text = format(q, "f")
        if "." in text:
            int_part, frac = text.split(".")
            room = max(0, max_digits - len(int_part.lstrip("-")))
            frac = frac[:room].rstrip("0")
            text = int_part + ("." + frac if frac else "")
        int_part, _, frac = text.partition(".")
        neg = int_part.startswith("-")
        int_part = f"{int(int_part.lstrip('-') or 0):,}"
        return ("-" if neg else "") + int_part + ("." + frac if frac else "")
    return f"{q:.{max_digits - 6}E}".replace("E+", "e").replace("E-", "e-")
