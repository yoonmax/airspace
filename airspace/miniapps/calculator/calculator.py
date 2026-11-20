"""calculator — large, gesture-friendly keys with a safe evaluator."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QKeyEvent, QPainter

from airspace.miniapps.base_app import MiniApp
from airspace.miniapps.calculator.engine import CalcError, evaluate, format_number
from airspace.ui.controls import Button
from airspace.ui.styles import Theme, with_alpha

KEYS = [
    ["C", "(", ")", "÷"],
    ["7", "8", "9", "×"],
    ["4", "5", "6", "−"],
    ["1", "2", "3", "+"],
    ["⌫", "0", ".", "="],
]
OPS = {"÷", "×", "−", "+"}


class CalculatorModel:
    """Input state machine separate from the UI (unit tested)."""

    def __init__(self) -> None:
        self.expression = ""
        self.result: str | None = None
        self.error: str | None = None
        self.just_evaluated = False
        self.history: list[tuple[str, str]] = []

    def press(self, key: str) -> None:
        self.error = None
        if key == "C":
            self.expression, self.result, self.just_evaluated = "", None, False
        elif key == "⌫":
            if self.just_evaluated:
                self.just_evaluated = False
            self.expression = self.expression[:-1]
        elif key == "=":
            self.evaluate()
        elif key in OPS or key == "%":
            if self.just_evaluated and self.result is not None:
                self.expression = self.result.replace(",", "")
            elif not self.expression and key != "−":
                self.expression = "0"
            if self.expression and self.expression[-1] in OPS and key != "−":
                self.expression = self.expression[:-1]
            self.expression += key
            self.just_evaluated = False
        else:  # digits, '.', parentheses
            if self.just_evaluated:
                self.expression = ""
                self.just_evaluated = False
            if key == ".":
                tail = self.expression
                for op in OPS | {"(", ")"}:
                    tail = tail.split(op)[-1]
                if "." in tail:
                    return
                if not tail:
                    key = "0."
            self.expression += key
        if len(self.expression) > 120:
            self.expression = self.expression[:120]
        self._preview()

    def _preview(self) -> None:
        if self.just_evaluated:
            return
        try:
            self.result = format_number(evaluate(self.expression)) if self.expression else None
        except CalcError:
            self.result = None

    def evaluate(self) -> None:
        if not self.expression:
            return
        try:
            value = format_number(evaluate(self.expression))
        except CalcError as exc:
            self.error = str(exc)
            return
        self.history = (self.history + [(self.expression, value)])[-4:]
        self.result = value
        self.just_evaluated = True


class CalculatorApp(MiniApp):
    app_type = "calculator"
    title = "calculator"
    icon = "calculator"
    default_size = (330.0, 470.0)
    min_size = (260.0, 380.0)
    max_size = (520.0, 720.0)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.model = CalculatorModel()
        self.keys: dict[str, Button] = {}
        for row in KEYS:
            for k in row:
                variant = "primary" if k == "=" else "key"
                b = Button(k, on_click=lambda k=k: self.model.press(k), variant=variant, font_size=16,
                           tooltip="")
                if k in OPS:
                    b.color = self.theme.accent
                self.keys[k] = self.controls.add(b)

    def layout(self, w: float, h: float) -> None:
        pad = 14 * self.theme.scale
        display_h = h * 0.27
        grid_top = display_h + pad * 0.5
        rows, cols = len(KEYS), 4
        gap = 8 * self.theme.scale
        cw = (w - 2 * pad - gap * (cols - 1)) / cols
        ch = (h - grid_top - pad - gap * (rows - 1)) / rows
        for r, row in enumerate(KEYS):
            for c, k in enumerate(row):
                self.keys[k].rect = QRectF(pad + c * (cw + gap), grid_top + r * (ch + gap), cw, ch)
                self.keys[k].radius = min(ch, cw) * 0.32

    def paint_content(self, p: QPainter, theme: Theme) -> None:
        m = self.model
        pad = 18 * theme.scale
        display_h = self.height * 0.27
        r = QRectF(pad, 6, self.width - 2 * pad, display_h - 12)
        if m.history:
            p.setFont(theme.font(9.5))
            p.setPen(theme.text_tertiary)
            expr, val = m.history[-1]
            p.drawText(QRectF(r.left(), r.top(), r.width(), 18), Qt.AlignmentFlag.AlignRight, f"{expr} = {val}")
        p.setFont(theme.font(13))
        p.setPen(theme.text_secondary)
        expr = m.expression or "0"
        fm = p.fontMetrics()
        expr = fm.elidedText(expr, Qt.TextElideMode.ElideLeft, int(r.width()))
        p.drawText(QRectF(r.left(), r.top() + 18, r.width(), r.height() * 0.4), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, expr)
        if m.error:
            p.setPen(theme.danger)
            p.setFont(theme.font(14, QFont.Weight.Medium))
            text = m.error
        else:
            text = m.result if m.result is not None else (m.expression and "…") or "0"
            p.setPen(theme.text if m.just_evaluated or not m.expression else with_alpha(theme.text, 0.7))
            size = 30.0
            p.setFont(theme.font(size, QFont.Weight.Light))
            while p.fontMetrics().horizontalAdvance(text) > r.width() and size > 14:
                size -= 2
                p.setFont(theme.font(size, QFont.Weight.Light))
        p.drawText(QRectF(r.left(), r.top() + r.height() * 0.45, r.width(), r.height() * 0.55),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)

    def key(self, event: QKeyEvent) -> bool:
        k = event.key()
        text = event.text()
        mapping = {
            Qt.Key.Key_Return: "=", Qt.Key.Key_Enter: "=", Qt.Key.Key_Equal: "=",
            Qt.Key.Key_Backspace: "⌫", Qt.Key.Key_Delete: "C", Qt.Key.Key_Escape: "C",
            Qt.Key.Key_Plus: "+", Qt.Key.Key_Minus: "−", Qt.Key.Key_Asterisk: "×", Qt.Key.Key_Slash: "÷",
            Qt.Key.Key_ParenLeft: "(", Qt.Key.Key_ParenRight: ")", Qt.Key.Key_Period: ".", Qt.Key.Key_Comma: ".",
            Qt.Key.Key_Percent: "%",
        }
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        key = mapping.get(k)
        if key is None and text and text in "0123456789":
            key = text
        if key is None and text in ("x", "X"):
            key = "×"
        if key is None:
            return False
        self.model.press(key)
        if key in self.keys:
            btn = self.keys[key]
            btn._press.value = 1.0  # brief visual key-press feedback
        return True

    def save_state(self) -> dict:
        return {"history": self.model.history}

    def restore_state(self, state: dict) -> None:
        hist = state.get("history", [])
        if isinstance(hist, list):
            self.model.history = [tuple(map(str, h))[:2] for h in hist if isinstance(h, (list, tuple)) and len(h) == 2][-4:]
