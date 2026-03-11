"""Minimal ECharts 5 wrapper built on Streamlit Components v2 (pilot use)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

_HTML = '<div data-lc-echarts-root style="width:100%;height:100%;"></div>'

_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "echarts-5.6.0.min.js"


def _load_echarts_source() -> str | None:
    try:
        return _ASSET_PATH.read_text(encoding="utf-8")
    except OSError:
        return None


_JS_TEMPLATE = r"""
const ECHARTS_INLINE_SOURCE = __ECHARTS_INLINE_SOURCE_JSON__;
const ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js";

function ensureEchartsLoaded() {
  if (globalThis.echarts) {
    return Promise.resolve(globalThis.echarts);
  }
  if (globalThis.__lcEchartsLoadingPromise) {
    return globalThis.__lcEchartsLoadingPromise;
  }
  globalThis.__lcEchartsLoadingPromise = new Promise((resolve, reject) => {
    const loadFromCdn = () => {
      const script = document.createElement("script");
      script.src = ECHARTS_CDN;
      script.async = true;
      script.onload = () => resolve(globalThis.echarts);
      script.onerror = () => reject(new Error("Failed to load ECharts 5 (local asset + CDN fallback)"));
      document.head.appendChild(script);
    };

    if (typeof ECHARTS_INLINE_SOURCE === "string" && ECHARTS_INLINE_SOURCE.length > 1024) {
      try {
        const inlineScript = document.createElement("script");
        inlineScript.type = "text/javascript";
        inlineScript.text = ECHARTS_INLINE_SOURCE + "\n//# sourceURL=echarts-5.6.0.min.js";
        document.head.appendChild(inlineScript);
        if (globalThis.echarts) {
          resolve(globalThis.echarts);
          return;
        }
      } catch (_err) {
        // Fall back to CDN below.
      }
    }

    loadFromCdn();
  });
  return globalThis.__lcEchartsLoadingPromise;
}

function resolveMountContext(component) {
  if (!component || typeof component !== "object") {
    return { mountRoot: null, styleHost: null };
  }
  const candidates = [
    component.parentElement,
    component.element,
    component.parent,
    component.rootElement,
    component.hostElement,
  ];

  let mountRoot = null;
  for (const node of candidates) {
    if (
      node &&
      typeof node === "object" &&
      typeof node.querySelector === "function" &&
      typeof node.appendChild === "function"
    ) {
      mountRoot = node;
      break;
    }
  }

  if (!mountRoot) {
    return { mountRoot: null, styleHost: null };
  }

  // HTMLElement host where we apply card-like styles.
  let styleHost = null;
  if (mountRoot.style && typeof mountRoot.style === "object") {
    styleHost = mountRoot;
  } else if (mountRoot.host && mountRoot.host.style) {
    styleHost = mountRoot.host; // ShadowRoot -> host element
  } else if (mountRoot.parentElement && mountRoot.parentElement.style) {
    styleHost = mountRoot.parentElement;
  }

  return { mountRoot, styleHost };
}

export default function(component) {
  const safeSetTriggerValue =
    typeof component?.setTriggerValue === "function" ? component.setTriggerValue : null;
  let disposed = false;
  let resizeObserver = null;
  let clickHandler = null;
  let root = null;

  try {
    const payload = component?.data || {};
    const option = payload.option || {};
    const chartHeight = Number.isFinite(payload.heightPx) ? payload.heightPx : 360;
    const { mountRoot, styleHost } = resolveMountContext(component);

    if (!mountRoot) {
      if (safeSetTriggerValue) safeSetTriggerValue("error", "Component mount root unavailable.");
      return () => {};
    }

    if (styleHost && styleHost.style) {
      styleHost.style.width = "100%";
      styleHost.style.background = "#ffffff";
      styleHost.style.border = "1px solid #E5EAF0";
      styleHost.style.borderRadius = "12px";
      styleHost.style.boxShadow = "0 1px 2px rgba(15,23,42,0.04), 0 6px 18px rgba(15,23,42,0.03)";
      styleHost.style.padding = "4px";
      styleHost.style.boxSizing = "border-box";
      styleHost.style.overflow = "hidden";
    }

    root = mountRoot.querySelector("[data-lc-echarts-root]");
    if (!root) {
      root = document.createElement("div");
      root.setAttribute("data-lc-echarts-root", "");
      mountRoot.appendChild(root);
    }
    root.style.width = "100%";
    root.style.height = `${chartHeight}px`;
    root.style.borderRadius = "10px";
    root.style.background = "#ffffff";

    ensureEchartsLoaded()
      .then((echarts) => {
        if (disposed || !echarts || !root) return;

        let chart = echarts.getInstanceByDom(root);
        if (!chart) {
          chart = echarts.init(root, null, { renderer: "canvas" });
        }

        chart.setOption(option, true);
        chart.resize();

        if (clickHandler) {
          chart.off("click", clickHandler);
        }
        clickHandler = (params) => {
          const dataObj =
            params && typeof params === "object" ? (params.data ?? null) : null;
          if (safeSetTriggerValue) {
            safeSetTriggerValue("clicked", {
              seriesName: params?.seriesName ?? null,
              name: params?.name ?? null,
              value: params?.value ?? null,
              data: dataObj,
            });
          }
        };
        chart.on("click", clickHandler);

        if (!resizeObserver && "ResizeObserver" in globalThis) {
          resizeObserver = new ResizeObserver(() => {
            try {
              chart.resize();
            } catch (_err) {}
          });
          resizeObserver.observe(root);
        }
      })
      .catch((err) => {
        console.error("ECharts pilot error", err);
        if (safeSetTriggerValue) {
          safeSetTriggerValue("error", String(err?.message || err));
        }
      });
  } catch (err) {
    console.error("ECharts pilot fatal error", err);
    if (safeSetTriggerValue) {
      safeSetTriggerValue("error", String(err?.message || err));
    }
    return () => {};
  }

  return () => {
    disposed = true;
    try {
      if (resizeObserver) resizeObserver.disconnect();
    } catch (_err) {}
    try {
      if (root && globalThis.echarts) {
        const chart = globalThis.echarts.getInstanceByDom(root);
        if (chart && clickHandler) chart.off("click", clickHandler);
        if (chart) chart.dispose();
      }
    } catch (_err) {}
  };
}
"""


def _build_js() -> str:
    source = _load_echarts_source()
    source_json = json.dumps(source) if source else "null"
    return _JS_TEMPLATE.replace("__ECHARTS_INLINE_SOURCE_JSON__", source_json)


_JS = _build_js()

_component = st.components.v2.component(
    "lc_v2_echarts",
    html=_HTML,
    js=_JS,
)


def render_v2_echarts(
    option: dict[str, Any],
    *,
    key: str,
    height_px: int = 360,
) -> dict[str, Any] | None:
    """Render an ECharts 5 chart and return click payload if available.

    The component returns a bidirectional state object. We expose the latest click payload
    and ignore errors by returning ``None`` (fail-safe pilot behavior).
    """
    try:
        result = _component(
            key=key,
            data={"option": option, "heightPx": int(height_px)},
            height=max(120, int(height_px)),
            width="stretch",
            default={"clicked": None, "error": None},
            on_clicked_change=lambda: None,
            on_error_change=lambda: None,
        )
    except Exception:
        return None

    if result is None:
        return None

    error = getattr(result, "error", None)
    if isinstance(error, str) and error.strip():
        st.caption(f"ECharts v2 pilot fallback: {error}")
        return None

    clicked = getattr(result, "clicked", None)
    if isinstance(clicked, dict):
        return clicked
    return None
