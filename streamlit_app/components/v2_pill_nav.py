"""Lightweight Streamlit Components v2 pill navigation (inline JS, no iframe)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

_HTML = '<div data-pill-nav-root></div>'

_CSS = """
.lc-pill-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.125rem 0 0.25rem 0;
}
.lc-pill-nav button {
  border: 1px solid var(--st-border-color, rgba(128,128,128,.35));
  background: var(--st-secondary-background-color, #f5f5f5);
  color: var(--st-text-color, inherit);
  border-radius: 999px;
  padding: 0.35rem 0.7rem;
  cursor: pointer;
  font-size: 0.88rem;
  line-height: 1.2;
}
.lc-pill-nav button:hover {
  border-color: var(--st-primary-color, #ff4b4b);
}
.lc-pill-nav button[data-selected="true"] {
  background: color-mix(in oklab, var(--st-primary-color, #ff4b4b) 12%, white);
  border-color: var(--st-primary-color, #ff4b4b);
  font-weight: 600;
}
.lc-pill-nav-help {
  font-size: 0.82rem;
  color: var(--st-secondary-color, #6b7280);
  margin-bottom: 0.35rem;
}
"""

_JS = """
function resolveMountRoot(component) {
  if (!component || typeof component !== "object") return null;
  const candidates = [
    component.parentElement,
    component.element,
    component.parent,
    component.rootElement,
    component.hostElement,
  ];
  for (const node of candidates) {
    if (
      node &&
      typeof node === "object" &&
      typeof node.querySelector === "function" &&
      typeof node.appendChild === "function"
    ) {
      return node;
    }
  }
  return null;
}

export default function(component) {
  const safeSetTriggerValue =
    typeof component?.setTriggerValue === "function" ? component.setTriggerValue : null;

  try {
    const payload = component?.data || {};
    const items = Array.isArray(payload.items) ? payload.items : [];
    const selected = payload.selected || null;
    const helperText = payload.helperText || "";
    const mountRoot = resolveMountRoot(component);
    if (!mountRoot) return;

    let root = mountRoot.querySelector('[data-pill-nav-root]');
    if (!root) {
      root = document.createElement('div');
      root.setAttribute('data-pill-nav-root', '');
      mountRoot.appendChild(root);
    }

    root.innerHTML = '';

    if (helperText) {
      const helper = document.createElement('div');
      helper.className = 'lc-pill-nav-help';
      helper.textContent = helperText;
      root.appendChild(helper);
    }

    const wrap = document.createElement('div');
    wrap.className = 'lc-pill-nav';
    root.appendChild(wrap);

    for (const item of items) {
      if (!item || !item.id || !item.label) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = item.label;
      btn.title = item.description || item.label;
      btn.dataset.selected = String(item.id === selected);
      btn.onclick = (e) => {
        e.preventDefault();
        for (const sibling of wrap.querySelectorAll('button')) {
          sibling.dataset.selected = "false";
        }
        btn.dataset.selected = "true";
        if (safeSetTriggerValue) {
          safeSetTriggerValue('clicked', {
            id: item.id,
            label: item.label,
          });
        }
      };
      wrap.appendChild(btn);
    }
  } catch (err) {
    console.error('Pill nav v2 error', err);
  }
}
"""

_component = st.components.v2.component(
    "lc_v2_pill_nav",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def render_v2_pill_nav(
    items: Sequence[dict[str, str]],
    *,
    key: str,
    selected: str | None = None,
    helper_text: str | None = None,
) -> str | None:
    """Render pill navigation and return clicked item id (if any).

    Each item expects keys:
    - ``id``: stable identifier
    - ``label``: user-facing text
    - ``description`` (optional): tooltip
    """
    payload: dict[str, Any] = {
        "items": [dict(item) for item in items],
        "selected": selected,
        "helperText": helper_text or "",
    }
    try:
        result = _component(
            key=key,
            data=payload,
            height="content",
            width="stretch",
            default={"clicked": None},
            on_clicked_change=lambda: None,
        )
    except Exception:
        # Fail safe: don't break pages if Components v2 is unavailable/restricted.
        return None

    clicked = getattr(result, "clicked", None)
    if isinstance(clicked, dict):
        item_id = clicked.get("id")
        return str(item_id) if item_id is not None else None
    if isinstance(clicked, str):
        return clicked
    return None
