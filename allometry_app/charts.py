"""The Model Lab's charts (Plotly, on the app's light theme). Each form keeps the same color everywhere (its place in
the catalogue of its target, never its rank), from a categorical palette checked for color-blind separation on the
app's background; a ninth form gets a gray dashed line instead of a made-up hue."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cl_allometry.forms import form as form_by_key, forms_for
from cl_allometry.registry import FitRecord

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OBSERVED = "#1D250E"  # the observed trees, as in the other pages of the app
INK, MUTED, GRID = "#35401E", "#6b6f5e", "#e6e3dc"
LABELS = {"dap_cm": "DAP (cm)", "ht_m": "Altura (m)", "vol_m3": "Volume (m³)", "agb_kg": "Biomassa (kg)",
          "crown_m2": "Área da copa (m²)", "rho_gcm3": "Densidade (g/cm³)"}


def form_style(key: str) -> dict:
    f = form_by_key(key)
    order = [x.key for x in forms_for(f.target)]
    i = order.index(key)
    return {"color": PALETTE[i], "dash": "solid"} if i < len(PALETTE) else {"color": MUTED, "dash": "dash"}


def _layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(color=INK), legend=dict(orientation="h", y=1.08, x=0),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def usable(rows: pd.DataFrame, rec: FitRecord) -> pd.DataFrame:
    """The rows the fit used: the filter's rows with every variable and the target present and positive."""
    f = form_by_key(rec.form)
    cols = [*f.variables, f.target]
    d = rows[cols].apply(pd.to_numeric, errors="coerce")
    return d[(d > 0).all(axis=1)]


def curves(rows: pd.DataFrame, records: Sequence[FitRecord]) -> go.Figure:
    """Observed trees and each model's curve over the range it was fitted on (one-variable forms), or observed
    against estimated (forms of several variables)."""
    f0 = form_by_key(records[0].form)
    fig = go.Figure()
    if len(f0.variables) == 1:
        x, y = f0.variables[0], f0.target
        d = usable(rows, records[0])
        fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", name="observado",
                                 marker=dict(color=OBSERVED, size=8, opacity=0.35, line=dict(width=0)),
                                 hovertemplate=f"{LABELS.get(x, x)} %{{x:.1f}}<br>{LABELS.get(y, y)} %{{y:.2f}}"
                                               "<extra>observado</extra>"))
        for rec in records:
            if not rec.fit.converged:
                continue
            lo, hi = rec.fit.ranges[x]
            grid = np.linspace(lo, hi, 120)
            style = form_style(rec.form)
            name = form_by_key(rec.form).name
            fig.add_trace(go.Scatter(x=grid, y=rec.fit.predict(**{x: grid}), mode="lines", name=name,
                                     line=dict(color=style["color"], width=2, dash=style["dash"]),
                                     hovertemplate=f"{LABELS.get(x, x)} %{{x:.1f}}<br>estimado %{{y:.2f}}"
                                                   f"<extra>{name}</extra>"))
            if len(records) <= 4:  # direct label at the end of the curve, in ink
                fig.add_annotation(x=grid[-1], y=float(rec.fit.predict(**{x: grid[-1:]})[0]), text=name,
                                   showarrow=False, xanchor="left", xshift=6, font=dict(color=INK, size=11))
        fig.update_xaxes(title=LABELS.get(x, x))
        fig.update_yaxes(title=LABELS.get(y, y))
    else:
        y = f0.target
        top = 0.0
        for rec in records:
            if not rec.fit.converged:
                continue
            d = usable(rows, rec)
            pred = rec.fit.predict(**{v: d[v].to_numpy(float) for v in form_by_key(rec.form).variables})
            style = form_style(rec.form)
            fig.add_trace(go.Scatter(x=pred, y=d[y], mode="markers", name=form_by_key(rec.form).name,
                                     marker=dict(color=style["color"], size=8, opacity=0.6)))
            top = max(top, float(np.nanmax(pred)), float(d[y].max()))
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="1:1",
                                 line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
        fig.update_xaxes(title=f"{LABELS.get(y, y)} estimado")
        fig.update_yaxes(title=f"{LABELS.get(y, y)} observado")
    return _layout(fig)


def residuals(rows: pd.DataFrame, rec: FitRecord) -> go.Figure:
    """Residuals (observed − estimated) against the estimate and against the first variable: a funnel says the
    spread grows with the tree (fit with weights); a curve says the form misses the shape."""
    f = form_by_key(rec.form)
    d = usable(rows, rec)
    pred = rec.fit.predict(**{v: d[v].to_numpy(float) for v in f.variables})
    resid = d[f.target].to_numpy(float) - pred
    x = f.variables[0]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("contra o estimado", f"contra {LABELS.get(x, x)}"))
    color = form_style(rec.form)["color"]
    for col, xs, title in ((1, pred, "estimado"), (2, d[x].to_numpy(float), LABELS.get(x, x))):
        fig.add_trace(go.Scatter(x=xs, y=resid, mode="markers", showlegend=False,
                                 marker=dict(color=color, size=8, opacity=0.45, line=dict(width=0)),
                                 hovertemplate=f"{title} %{{x:.2f}}<br>resíduo %{{y:.2f}}<extra></extra>"),
                      row=1, col=col)
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=1, col=col)
    fig.update_yaxes(title_text=f"resíduo ({LABELS.get(f.target, f.target)})", row=1, col=1)
    fig.update_annotations(font=dict(color=MUTED, size=12))
    return _layout(fig, height=340)


def sources(counts: pd.Series) -> go.Figure:
    """How many trees got their value from each source (one series: no legend)."""
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker=dict(color=PALETTE[0]),
                           hovertemplate="%{y}: %{x:,} árvores<extra></extra>", text=counts.values,
                           textposition="outside", textfont=dict(color=INK)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=60 + 40 * len(counts))
