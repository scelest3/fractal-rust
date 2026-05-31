export const PANEL_STYLE = {
  background: "rgba(0,0,0,0.55)",
  color: "white",
  fontFamily: "monospace",
  fontSize: "12px",
  padding: "6px 10px",
  borderRadius: "4px",
  lineHeight: "1.5",
  zIndex: "998",
} as const;

export const PANEL_GAP = 4; // px — gap between stats overlay and palette panel
export const NARROW_PX = 500; // viewport width threshold for compact/mobile layout

export const INPUT_CSS =
  "background:#222;color:#eee;border:1px solid #555;border-radius:3px;" +
  "padding:2px 4px;font-family:monospace;font-size:12px;box-sizing:border-box;";

export const SELECT_CSS =
  "background:#222;color:#eee;border:1px solid #555;border-radius:3px;" +
  "padding:2px 6px;font-family:monospace;font-size:12px;cursor:pointer;";

export const SECTION_HEADER_CSS =
  "font-weight:bold;font-size:11px;opacity:0.7;cursor:pointer;" +
  "display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;";
