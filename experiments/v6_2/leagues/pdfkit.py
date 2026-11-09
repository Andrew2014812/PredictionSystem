"""Small formatting helpers shared by the league and temporal report builders (numbers come from result dicts)."""
from __future__ import annotations

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from scripts.report_kit import table


def T(rows, widths=None):
    return [table(rows, widths=[w * cm for w in widths] if widths else None), Spacer(1, 6)]


def f(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def pc(v, d=1, signed=True):
    if v is None:
        return "—"
    return f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"


def pp(v, d=1):
    """A fraction shown in percentage points with a sign."""
    return "—" if v is None else f"{v * 100:+.{d}f}"


def ci(c, mult=100, d=1):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0] * mult:+.{d}f}; {c[1] * mult:+.{d}f}]"


def dk(r, d=1):
    """paired-difference dict -> '+x.x [lo; hi]' in units of 10^-3."""
    if not r or r.get("diff") is None:
        return "—"
    return f"{r['diff'] * 1e3:+.{d}f} {ci(r['ci95'], 1e3, d)}"


def mk(r, d=1):
    """mean_ci dict -> '+x.x [lo; hi]' (×10^-3)."""
    if not r or r.get("mean") is None:
        return "—"
    return f"{r['mean'] * 1e3:+.{d}f} {ci(r['ci95'], 1e3, d)}"


def roi(r, d=1):
    """mean_ci of profits -> '-3.1% [-5.0; -1.2]'."""
    if not r or r.get("mean") is None:
        return "—"
    return f"{r['mean'] * 100:+.{d}f}% {ci(r['ci95'], 100, d)}"


def pv(p):
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
