#!/usr/bin/env python3
"""
scripts/app.py
==============
FastAPI web interface for Apiro — with live SSE streaming.

Run with:
  uvicorn scripts.app:app --host 127.0.0.1 --port 8000

Features:
  - /run/stream  : Server-Sent Events endpoint — streams each traversal step
                   (seed_added, expanding, node_expanded, contradiction, etc.)
                   to the browser in real time via a thread-pool + asyncio queue.
  - /run         : Legacy sync endpoint (kept for backward compatibility).
  - New 3-column UI:
      Left   → Clinical input form + run statistics
      Center → Live "Thought Log" — stage cards slide in as the model reasons
      Right  → Incrementally-built D3 force graph + node inspector / differential tabs
"""

import sys
import logging
import time
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.investigate import build_components
from apiro.graph.belief_graph import BeliefGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apiro_app")

app = FastAPI(title="Apiro AI Detective")

# Shared components — initialised once at startup
logger.info("Initialising Apiro components...")
try:
    (traversal, axiom_extractor), doc_count = build_components()
    logger.info(f"Apiro ready. ChromaDB contains {doc_count:,} documents.")
except Exception as e:
    logger.error(f"Failed to initialise Apiro components: {e}")
    traversal, axiom_extractor, doc_count = None, None, 0

# Thread pool for running CPU-bound traversal without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=2)

DOMAIN_COLORS = {
    "pathophysiology": "#6366f1",
    "pharmacology":    "#f59e0b",
    "genetics":        "#10b981",
    "imaging":         "#3b82f6",
    "lab":             "#8b5cf6",
    "treatment":       "#06b6d4",
    "comorbidity":     "#f97316",
    "symptom":         "#ec4899",
    "unknown":         "#6b7280",
}


class InvestigationRequest(BaseModel):
    findings: str
    max_depth: int = 5
    real_entropy: bool = False


# ---------------------------------------------------------------------------
# HTML — complete frontend
# Placeholders replaced at request time: {doc_count}, {domain_colors}
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Apiro — an entropy-first AI clinical detective that reasons through biomedical knowledge graphs to generate differential diagnoses.">
<title>Apiro · AI Clinical Detective</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ── Design tokens ─────────────────────────────────────────────────────── */
  :root {
    --bg:        #060810;
    --surface:   #0d1020;
    --surface2:  #111828;
    --border:    #1a2035;
    --border2:   #232a40;
    --text:      #e2e8f0;
    --muted:     #3f4a60;
    --muted2:    #5a6880;
    --accent:    #6366f1;
    --accent-g:  rgba(99,102,241,0.25);
    --success:   #10b981;
    --danger:    #ef4444;
    --warning:   #f59e0b;
    --teal:      #14b8a6;
    --purple:    #a855f7;

    /* stage-card accent colours */
    --c-seed:   #3b82f6;
    --c-expand: #6366f1;
    --c-hypo:   #10b981;
    --c-rabbit: #f59e0b;
    --c-contra: #ef4444;
    --c-sat:    #14b8a6;
    --c-done:   #a855f7;
  }

  /* ── Reset ─────────────────────────────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ────────────────────────────────────────────────────────────── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    gap: 16px;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .logo-icon {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .logo-text h1 {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
  }
  .logo-text .sub {
    font-size: 10.5px;
    color: var(--muted2);
    margin-top: 1px;
  }
  #status-pill {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    background: var(--border);
    color: var(--muted2);
    border: 1px solid transparent;
    transition: all 0.25s ease;
    white-space: nowrap;
  }
  #status-pill.running {
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border-color: rgba(99,102,241,0.3);
    animation: pillPulse 2s ease-in-out infinite;
  }
  #status-pill.done {
    background: rgba(16,185,129,0.1);
    color: var(--success);
    border-color: rgba(16,185,129,0.25);
  }
  @keyframes pillPulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
  }

  /* ── Main layout ───────────────────────────────────────────────────────── */
  #main-layout {
    flex: 1;
    display: flex;
    overflow: hidden;
    min-height: 0;
  }

  /* ── LEFT — Input panel ────────────────────────────────────────────────── */
  #input-panel {
    width: 300px;
    min-width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 18px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow-y: auto;
    flex-shrink: 0;
  }
  #input-panel::-webkit-scrollbar { width: 3px; }
  #input-panel::-webkit-scrollbar-thumb { background: var(--border2); }

  .section-label {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted2);
    margin-bottom: 2px;
  }
  .form-group { display: flex; flex-direction: column; gap: 5px; }
  textarea, input[type="number"] {
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 7px;
    color: var(--text);
    padding: 9px 11px;
    font-size: 12px;
    font-family: 'Inter', sans-serif;
    line-height: 1.55;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  textarea { resize: none; height: 115px; }
  textarea:focus, input[type="number"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-g);
  }
  .hint { font-size: 10px; color: var(--muted); line-height: 1.5; }

  .toggle-row {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11.5px;
    color: var(--muted2);
    cursor: pointer;
    user-select: none;
  }
  input[type="checkbox"] { width: 13px; height: 13px; accent-color: var(--accent); cursor: pointer; }

  #run-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 11px 12px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    box-shadow: 0 2px 14px rgba(99,102,241,0.4);
    letter-spacing: -0.01em;
  }
  #run-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 5px 20px rgba(99,102,241,0.55); }
  #run-btn:active:not(:disabled) { transform: translateY(0); }
  #run-btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }

  .divider { height: 1px; background: var(--border); }

  /* Stats grid */
  .stats-grid { display: flex; flex-direction: column; gap: 5px; }
  .stat-row { display: flex; justify-content: space-between; align-items: center; }
  .stat-k  { font-size: 11px; color: var(--muted2); }
  .stat-v  { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--text); }
  .stat-v.accent { color: var(--accent); }

  /* ── CENTER — Thought Log ──────────────────────────────────────────────── */
  #thought-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border-right: 1px solid var(--border);
    overflow: hidden;
    min-width: 0;
  }
  #thought-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .panel-title-lg {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--muted2);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #thought-count {
    font-size: 10.5px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    background: var(--border);
    padding: 2px 8px;
    border-radius: 10px;
  }

  #thought-log {
    flex: 1;
    overflow-y: auto;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    scroll-behavior: smooth;
  }
  #thought-log::-webkit-scrollbar { width: 3px; }
  #thought-log::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* Empty state */
  #log-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--muted2);
    text-align: center;
    pointer-events: none;
    padding: 40px 20px;
  }
  #log-empty .e-icon { font-size: 44px; opacity: 0.2; }
  #log-empty p { font-size: 12.5px; max-width: 240px; line-height: 1.65; opacity: 0.5; }

  /* ── Stage Cards ───────────────────────────────────────────────────────── */
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 11px 13px;
    animation: slideUp 0.28s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    flex-shrink: 0;
  }
  .card.seed   { border-left-color: var(--c-seed); }
  .card.expand { border-left-color: var(--c-expand); background: rgba(99,102,241,0.04); }
  .card.hypo   { border-left-color: var(--c-hypo); }
  .card.rabbit { border-left-color: var(--c-rabbit); background: rgba(245,158,11,0.04); }
  .card.contra { border-left-color: var(--c-contra); background: rgba(239,68,68,0.04); }
  .card.sat    { border-left-color: var(--c-sat); background: rgba(20,184,166,0.04); }

  .card-head {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 7px;
  }
  .card-icon { font-size: 13px; line-height: 1; }
  .card-type {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .card-type.seed   { color: var(--c-seed); }
  .card-type.expand { color: var(--c-expand); }
  .card-type.hypo   { color: var(--c-hypo); }
  .card-type.rabbit { color: var(--c-rabbit); }
  .card-type.contra { color: var(--c-contra); }
  .card-type.sat    { color: var(--c-sat); }
  .card-badge {
    margin-left: auto;
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted2);
    background: var(--border);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .card-body {
    font-size: 12.5px;
    color: var(--text);
    line-height: 1.55;
    font-weight: 450;
    margin-bottom: 8px;
    word-break: break-word;
  }
  .card-body.dim { color: var(--muted2); font-size: 11.5px; font-style: italic; }

  .card-foot {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
  }
  .dpill {
    font-size: 8.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 10px;
    color: #fff;
  }
  .hpill {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted2);
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .mini-bar { width: 44px; height: 3px; background: var(--border2); border-radius: 2px; overflow: hidden; }
  .mini-fill { height: 100%; border-radius: 2px; }
  .foot-right { margin-left: auto; font-size: 9.5px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

  /* Final diagnosis banner */
  .dx-banner {
    background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(99,102,241,0.06));
    border: 1px solid rgba(168,85,247,0.22);
    border-left: 3px solid var(--c-done);
    border-radius: 10px;
    padding: 14px 16px;
    animation: slideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    flex-shrink: 0;
  }
  .dx-banner-title {
    font-size: 9.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--c-done);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .dx-list { list-style: none; display: flex; flex-direction: column; gap: 7px; }
  .dx-item { display: flex; align-items: flex-start; gap: 9px; font-size: 12.5px; color: var(--text); line-height: 1.45; }
  .dx-rank {
    font-size: 10px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    min-width: 24px;
    padding: 2px 5px;
    border-radius: 4px;
    text-align: center;
    flex-shrink: 0;
  }
  .dx-rank.r1 { background: rgba(251,191,36,0.18); color: #fbbf24; }
  .dx-rank.r2 { background: rgba(148,163,184,0.14); color: #94a3b8; }
  .dx-rank.r3 { background: rgba(180,83,9,0.14); color: #d97706; }

  /* ── RIGHT — Graph panel ────────────────────────────────────────────────── */
  #graph-panel {
    width: 350px;
    min-width: 300px;
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
  }
  #graph-header {
    border-bottom: 1px solid var(--border);
    padding: 10px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .graph-title { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted2); }
  #node-badge {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    background: rgba(99,102,241,0.1);
    padding: 2px 7px;
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.2);
  }
  #graph-viewport {
    flex: 0 0 52%;
    position: relative;
    overflow: hidden;
  }
  #g-svg {
    width: 100%;
    height: 100%;
    cursor: grab;
  }
  #g-svg:active { cursor: grabbing; }

  /* Inspector */
  #insp-panel {
    flex: 1;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-height: 0;
  }
  #insp-tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .tab {
    flex: 1;
    padding: 8px 6px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted2);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  #insp-body { flex: 1; overflow-y: auto; padding: 12px; }
  #insp-body::-webkit-scrollbar { width: 3px; }
  #insp-body::-webkit-scrollbar-thumb { background: var(--border2); }

  .ifield { margin-bottom: 9px; }
  .ilabel { font-size: 8.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted2); margin-bottom: 3px; }
  .ivalue { font-size: 11.5px; color: var(--text); line-height: 1.5; word-break: break-word; }
  .ivalue.mono { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--muted2); }
  .iebar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 5px; }
  .iebar-fill { height: 100%; border-radius: 2px; }
  .insp-empty { color: var(--muted2); font-size: 11.5px; text-align: center; padding: 20px 10px; line-height: 1.7; }

  /* D3 node/link styles */
  .g-link { stroke: #1a2540; stroke-width: 1.5; stroke-opacity: 0.85; fill: none; }
  .g-link.contra { stroke: var(--danger); stroke-dasharray: 5,3; stroke-width: 2; }
  .g-node circle { stroke-width: 2; cursor: pointer; transition: stroke-width 0.12s, filter 0.12s; }
  .g-node circle:hover { stroke-width: 3; filter: brightness(1.35); }
  .g-node.selected circle { stroke: #fff !important; stroke-width: 3; }
  .g-node.rhole circle { stroke-dasharray: 4,3; opacity: 0.35; }
  .g-node text { font-size: 6.5px; fill: rgba(148,163,184,0.7); pointer-events: none; text-anchor: middle; }

  /* Tooltip */
  #tooltip {
    position: fixed;
    background: rgba(10,13,28,0.97);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 11px;
    pointer-events: none;
    max-width: 240px;
    line-height: 1.45;
    display: none;
    z-index: 9999;
    box-shadow: 0 10px 28px rgba(0,0,0,0.6);
  }

  @keyframes nodeIn {
    from { transform: scale(0); opacity: 0; }
    to   { transform: scale(1); opacity: 1; }
  }
</style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-icon">⬡</div>
    <div class="logo-text">
      <h1>Apiro · Clinical Detective</h1>
      <div class="sub">Entropy-First Belief Graph &bull; {doc_count} corpus docs</div>
    </div>
  </div>
  <div id="status-pill">Idle</div>
</header>

<!-- ── Main layout ─────────────────────────────────────────────────────────── -->
<div id="main-layout">

  <!-- LEFT: Input -->
  <div id="input-panel">
    <div class="form-group">
      <div class="section-label">Clinical Findings</div>
      <textarea id="findings-input" placeholder="e.g. 45yo male, substernal chest pain, troponin 2.1, ST elevation V3-V5, diaphoresis, dyspnoea"></textarea>
      <div class="hint">Patient history, symptoms, labs, imaging — separated by commas.</div>
    </div>
    <div class="form-group">
      <div class="section-label">Max Graph Depth</div>
      <input type="number" id="depth-input" value="5" min="2" max="8">
    </div>
    <label class="toggle-row">
      <input type="checkbox" id="real-entropy-input">
      Real-time seed entropy (slower)
    </label>
    <button id="run-btn" onclick="startInvestigation()">▶ Run Detective</button>
    <div class="divider"></div>
    <div class="section-label">Run Statistics</div>
    <div class="stats-grid">
      <div class="stat-row"><span class="stat-k">Nodes</span><span class="stat-v" id="sv-nodes">—</span></div>
      <div class="stat-row"><span class="stat-k">Edges</span><span class="stat-v" id="sv-edges">—</span></div>
      <div class="stat-row"><span class="stat-k">Rabbit Holes</span><span class="stat-v" id="sv-rabbits">—</span></div>
      <div class="stat-row"><span class="stat-k">Contradictions</span><span class="stat-v" id="sv-contras">—</span></div>
      <div class="stat-row"><span class="stat-k">Stop Reason</span><span class="stat-v" id="sv-stop">—</span></div>
      <div class="stat-row"><span class="stat-k">Duration</span><span class="stat-v accent" id="sv-dur">—</span></div>
    </div>
  </div>

  <!-- CENTER: Thought Log -->
  <div id="thought-panel">
    <div id="thought-header">
      <span class="panel-title-lg">🧠 Detective's Reasoning</span>
      <span id="thought-count">0 thoughts</span>
    </div>
    <div id="thought-log">
      <div id="log-empty">
        <div class="e-icon">🔬</div>
        <p>Enter clinical findings and click <strong>Run Detective</strong> to watch the AI reason through the case in real time.</p>
      </div>
    </div>
  </div>

  <!-- RIGHT: Graph + Inspector -->
  <div id="graph-panel">
    <div id="graph-header">
      <span class="graph-title">Belief Graph</span>
      <span id="node-badge">0 nodes</span>
    </div>
    <div id="graph-viewport">
      <svg id="g-svg"></svg>
    </div>
    <div id="insp-panel">
      <div id="insp-tabs">
        <button class="tab active" id="tab-insp" onclick="switchTab('insp')">Node Inspector</button>
        <button class="tab" id="tab-dx" onclick="switchTab('dx')">Differential Dx</button>
      </div>
      <div id="insp-body">
        <div class="insp-empty">Click a graph node<br>to inspect it.</div>
      </div>
    </div>
  </div>

</div><!-- /#main-layout -->

<div id="tooltip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
/* ─── Config ───────────────────────────────────────────────────────────────── */
const DOMAIN_COLORS = {domain_colors};

/* ─── State ────────────────────────────────────────────────────────────────── */
let nodesData = [], linksData = [];
let linkGroup, nodeGroup, simulation;
let selectedId  = null;
let activeTab   = 'insp';
let savedDx     = [];
let tCount = 0, nCount = 0, eCount = 0, rCount = 0, cCount = 0;

/* ─── Colour helpers ───────────────────────────────────────────────────────── */
function entropyColor(h) {
  h = Math.max(0, Math.min(1, h || 0));
  if (h < 0.5) {
    const t = h * 2;
    return `rgb(${Math.round(t * 70)},${Math.round(130 + t * 70)},230)`;
  }
  const t = (h - 0.5) * 2;
  return `rgb(${Math.round(190 + t * 65)},${Math.round(90 * (1 - t))},${Math.round(210 * (1 - t))})`;
}

/* ─── D3 graph ─────────────────────────────────────────────────────────────── */
function initGraph() {
  const svg = d3.select('#g-svg');
  svg.selectAll('*').remove();
  const g = svg.append('g').attr('id', 'root-g');
  svg.call(d3.zoom().scaleExtent([0.15, 5]).on('zoom', e => g.attr('transform', e.transform)));
  linkGroup = g.append('g');
  nodeGroup = g.append('g');
  const vp = document.getElementById('graph-viewport');
  simulation = d3.forceSimulation([])
    .force('link', d3.forceLink([]).id(d => d.id).distance(55))
    .force('charge', d3.forceManyBody().strength(-110))
    .force('center', d3.forceCenter((vp.clientWidth || 350) / 2, (vp.clientHeight || 200) / 2))
    .force('collision', d3.forceCollide().radius(d => (d.r || 10) + 4))
    .on('tick', () => {
      linkGroup.selectAll('line')
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeGroup.selectAll('.g-node').attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function refreshGraph() {
  /* Links */
  const lSel = linkGroup.selectAll('line').data(linksData, d => d.key);
  lSel.enter().append('line').attr('class', d => 'g-link' + (d.contra ? ' contra' : ''));
  lSel.exit().remove();

  /* Nodes */
  const nSel = nodeGroup.selectAll('.g-node').data(nodesData, d => d.id);
  const enter = nSel.enter().append('g')
    .attr('class', d => 'g-node' + (d.rhole ? ' rhole' : ''))
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end',   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  enter.append('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.color)
    .attr('stroke', d => { try { return d3.color(d.color).darker(0.5).toString(); } catch { return '#333'; } });
  enter.append('text').attr('dy', 4).text(d => `H=${(d.entropy||0).toFixed(2)}`);
  nSel.exit().remove();

  /* Re-attach events to all nodes (enter + existing) */
  nodeGroup.selectAll('.g-node')
    .on('mouseover', (e, d) => {
      const tip = document.getElementById('tooltip');
      tip.style.display = 'block';
      tip.innerHTML = `<b style="color:#94a3b8">${d.domain||'unknown'}</b> · depth ${d.depth} · H=${(d.entropy||0).toFixed(3)}<br><span>${d.claim}</span>`;
    })
    .on('mousemove', e => {
      const tip = document.getElementById('tooltip');
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY - 30) + 'px';
    })
    .on('mouseout', () => { document.getElementById('tooltip').style.display = 'none'; })
    .on('click', (e, d) => { e.stopPropagation(); pickNode(d); });

  simulation.nodes(nodesData);
  simulation.force('link').links(linksData);
  simulation.alpha(0.4).restart();
  document.getElementById('node-badge').textContent = `${nodesData.length} nodes`;
}

function addNodeToGraph(ev) {
  if (nodesData.find(n => n.id === ev.node_id)) return;
  const depth  = ev.depth || 0;
  const r      = depth === 0 ? 15 : Math.max(7, 13 - depth * 1.5);
  nodesData.push({
    id: ev.node_id, claim: ev.claim, domain: ev.domain,
    entropy: ev.entropy || 0, depth, parent_id: ev.parent_id || null,
    rhole: false, r, color: entropyColor(ev.entropy || 0),
  });
  if (ev.parent_id && nodesData.find(n => n.id === ev.parent_id)) {
    const key = `${ev.parent_id}->${ev.node_id}`;
    if (!linksData.find(l => l.key === key))
      linksData.push({ key, source: ev.parent_id, target: ev.node_id, contra: false });
  }
  refreshGraph();
}

function markRabbitHole(nodeId) {
  const n = nodesData.find(n => n.id === nodeId);
  if (n) {
    n.rhole = true;
    nodeGroup.selectAll('.g-node').filter(d => d.id === nodeId).classed('rhole', true);
  }
}

/* ─── Inspector ────────────────────────────────────────────────────────────── */
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'insp') {
    const n = selectedId ? nodesData.find(n => n.id === selectedId) : null;
    n ? renderInspector(n) : setInspEmpty('Click a graph node<br>to inspect it.');
  } else {
    renderDx(savedDx);
  }
}

function pickNode(d) {
  selectedId = d.id;
  nodeGroup.selectAll('.g-node').classed('selected', n => n.id === d.id);
  switchTab('insp');
}

function renderInspector(d) {
  const dc = DOMAIN_COLORS[d.domain] || '#6b7280';
  const pct = Math.round((d.entropy || 0) * 100);
  const bc  = entropyColor(d.entropy || 0);
  document.getElementById('insp-body').innerHTML = `
    <div class="ifield"><div class="ilabel">Node ID</div><div class="ivalue mono">${d.id}</div></div>
    <div class="ifield"><div class="ilabel">Claim</div><div class="ivalue">${d.claim}</div></div>
    <div class="ifield"><div class="ilabel">Domain</div><span class="dpill" style="background:${dc}">${d.domain||'unknown'}</span></div>
    <div class="ifield">
      <div class="ilabel">Entropy</div>
      <div class="ivalue">${(d.entropy||0).toFixed(4)}</div>
      <div class="iebar"><div class="iebar-fill" style="width:${pct}%;background:${bc}"></div></div>
    </div>
    <div class="ifield"><div class="ilabel">Depth</div><div class="ivalue">${d.depth}</div></div>
    ${d.parent_id ? `<div class="ifield"><div class="ilabel">Parent</div><div class="ivalue mono">${d.parent_id}</div></div>` : ''}
    ${d.rhole ? `<div class="ifield"><div class="ilabel">Flag</div><span class="dpill" style="background:#7f1d1d">⚠ Rabbit Hole</span></div>` : ''}
  `;
}

function renderDx(dx) {
  if (!dx || dx.length === 0) { setInspEmpty('Run investigation<br>to generate differential.'); return; }
  const ranks = ['r1','r2','r3'];
  document.getElementById('insp-body').innerHTML = `
    <ul class="dx-list">
      ${dx.map((d, i) => `<li class="dx-item"><span class="dx-rank ${ranks[i]||''}">#${i+1}</span><span>${d}</span></li>`).join('')}
    </ul>`;
}

function setInspEmpty(html) {
  document.getElementById('insp-body').innerHTML = `<div class="insp-empty">${html}</div>`;
}

/* ─── Stats ────────────────────────────────────────────────────────────────── */
function updateStats() {
  document.getElementById('sv-nodes').textContent   = nCount || '—';
  document.getElementById('sv-edges').textContent   = eCount || '—';
  document.getElementById('sv-rabbits').textContent = rCount || '—';
  document.getElementById('sv-contras').textContent = cCount || '—';
}

/* ─── Thought log ──────────────────────────────────────────────────────────── */
function clearEmpty() {
  const e = document.getElementById('log-empty');
  if (e) e.remove();
}

function addCard(cls, icon, typeLabel, badge, bodyHtml, domain, entropy, footRight) {
  clearEmpty();
  const dc  = domain ? (DOMAIN_COLORS[domain] || '#6b7280') : null;
  const pct = entropy != null ? Math.round(entropy * 100) : 0;
  const bc  = entropy != null ? entropyColor(entropy) : '';
  const card = document.createElement('div');
  card.className = `card ${cls}`;
  card.innerHTML = `
    <div class="card-head">
      <span class="card-icon">${icon}</span>
      <span class="card-type ${cls}">${typeLabel}</span>
      ${badge ? `<span class="card-badge">${badge}</span>` : ''}
    </div>
    <div class="card-body${bodyHtml.length > 120 ? '' : ''}">${bodyHtml}</div>
    <div class="card-foot">
      ${dc ? `<span class="dpill" style="background:${dc}">${domain}</span>` : ''}
      ${entropy != null ? `
        <span class="hpill">H=${entropy.toFixed(3)}
          <span class="mini-bar"><span class="mini-fill" style="width:${pct}%;background:${bc}"></span></span>
        </span>` : ''}
      ${footRight ? `<span class="foot-right">${footRight}</span>` : ''}
    </div>`;
  const log = document.getElementById('thought-log');
  log.appendChild(card);
  log.scrollTop = log.scrollHeight;
  tCount++;
  document.getElementById('thought-count').textContent = `${tCount} thought${tCount !== 1 ? 's' : ''}`;
}

function addDxBanner(dx) {
  clearEmpty();
  const ranks = ['r1','r2','r3'];
  const el = document.createElement('div');
  el.className = 'dx-banner';
  el.innerHTML = `
    <div class="dx-banner-title">🏁 Final Differential Diagnosis</div>
    <ul class="dx-list">
      ${dx.map((d, i) => `<li class="dx-item"><span class="dx-rank ${ranks[i]||''}">#${i+1}</span><span>${d}</span></li>`).join('')}
    </ul>`;
  const log = document.getElementById('thought-log');
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

/* ─── Event dispatcher ─────────────────────────────────────────────────────── */
function handleEvent(ev) {
  const t = ev.event;

  if (t === 'patient_context_extracted') {
    addCard('seed', '🧑‍⚕️', 'Patient Profile', '', `Synthesized presentation:<br><b style="color:var(--text)">${ev.summary}</b>`, null, null, '');
  }
  else if (t === 'hypotheses_generated') {
    const list = ev.hypotheses || [];
    const html = `Based on the profile, I am considering ${list.length} initial hypotheses:<br><ul style="margin:8px 0 0 20px;padding:0;color:var(--text)">${list.slice(0,3).map(h=>`<li>${h}</li>`).join('')}${list.length>3?'<li>...</li>':''}</ul>`;
    addCard('hypo', '💡', 'Initial Differential', '', html, null, null, '');
  }
  else if (t === 'hypotheses_scored') {
    const ranked = ev.ranked || [];
    if (ranked.length > 0) {
      const top = ranked[0];
      const html = `Cross-referencing symptoms with literature.<br>Top candidate: <b style="color:var(--text)">${top.hypothesis}</b><br>(Score: ${(top.final_score||0).toFixed(2)})`;
      addCard('sat', '📊', 'Evidence Scoring', '', html, null, null, '');
    }
  }
  else if (t === 'enrichment_seed') {
    addCard('expand', '🔍', 'Deep Dive', '', `Investigating literature for:<br><b style="color:var(--text)">${ev.hypothesis}</b>`, null, null, '');
  }
  else if (t === 'traversal_complete') {
    const dx = ev.synthesis || [];
    savedDx = dx;
    if (dx.length) { addDxBanner(dx); if (activeTab === 'dx') renderDx(dx); }
    document.getElementById('sv-stop').textContent = ev.stop_reason || '—';
    document.getElementById('sv-dur').textContent  = ev.duration_seconds ? `${(+ev.duration_seconds).toFixed(1)}s` : '—';
    setStatus('done', `✓ Done · ${(ev.duration_seconds||0).toFixed(1)}s`);
  }
  else if (t === 'error') {
    addCard('contra', '❌', 'Error', '', ev.message || 'Unknown error', null, null, '');
    setStatus('idle', 'Error');
  }
}

/* ─── Status pill ──────────────────────────────────────────────────────────── */
function setStatus(state, text) {
  const el = document.getElementById('status-pill');
  el.textContent = text;
  el.className = state === 'running' ? 'running' : (state === 'done' ? 'done' : '');
}

/* ─── Main investigation ───────────────────────────────────────────────────── */
async function startInvestigation() {
  const findings = document.getElementById('findings-input').value.trim();
  const maxDepth = parseInt(document.getElementById('depth-input').value) || 5;
  const realEnt  = document.getElementById('real-entropy-input').checked;
  if (!findings) { alert('Please enter clinical findings.'); return; }

  // Reset all state
  nodesData = []; linksData = [];
  tCount = 0; nCount = 0; eCount = 0; rCount = 0; cCount = 0;
  selectedId = null; savedDx = [];

  // Reset UI
  document.getElementById('thought-log').innerHTML =
    `<div id="log-empty"><div class="e-icon" style="opacity:0.2;font-size:36px">⏳</div><p style="opacity:0.4;margin-top:8px">Starting analysis…</p></div>`;
  document.getElementById('thought-count').textContent = '0 thoughts';
  setInspEmpty('Click a graph node<br>to inspect it.');
  ['sv-nodes','sv-edges','sv-rabbits','sv-contras','sv-stop','sv-dur'].forEach(id => {
    document.getElementById(id).textContent = '—';
    document.getElementById(id).classList.remove('accent');
  });

  initGraph();
  document.getElementById('run-btn').disabled = true;
  setStatus('running', '⟳ Reasoning…');

  try {
    const resp = await fetch('/run/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ findings, max_depth: maxDepth, real_entropy: realEnt }),
    });
    if (!resp.ok) throw new Error(`Server error ${resp.status}`);

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (t.startsWith('data: ')) {
          try { handleEvent(JSON.parse(t.slice(6))); } catch(e) { console.warn('parse', t, e); }
        }
      }
    }
    // flush remainder
    if (buf.trim().startsWith('data: ')) {
      try { handleEvent(JSON.parse(buf.trim().slice(6))); } catch {}
    }
  } catch(err) {
    addCard('contra', '❌', 'Error', '', err.message, null, null, '');
    setStatus('idle', 'Error');
  } finally {
    document.getElementById('run-btn').disabled = false;
    if (document.getElementById('status-pill').classList.contains('running'))
      setStatus('done', 'Complete');
  }
}

// Init D3 graph on page load
initGraph();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def get_index():
    if not traversal:
        return HTMLResponse(
            "<h3 style='font-family:sans-serif;padding:40px;color:#ef4444'>"
            "Apiro engine not initialised. Ensure Ollama is running and run build_corpus first.</h3>",
            status_code=500
        )
    return INDEX_HTML \
        .replace("{doc_count}", f"{doc_count:,}") \
        .replace("{domain_colors}", json.dumps(DOMAIN_COLORS))


@app.post("/run/stream")
async def run_investigation_stream(req: InvestigationRequest):
    """
    Server-Sent Events endpoint.

    Runs the Apiro traversal in a ThreadPoolExecutor thread and emits each
    traversal event to the browser the moment it fires, so the UI can update
    the thought log and D3 graph in real time without waiting for completion.
    """
    if not traversal:
        raise HTTPException(status_code=500, detail="Apiro engine not initialised")

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def on_event(event: dict) -> None:
        """Called from the traversal thread — schedules a put on the async queue."""
        asyncio.run_coroutine_threadsafe(q.put(event), loop)

    def run_traversal() -> None:
        try:
            if not req.findings.strip():
                asyncio.run_coroutine_threadsafe(
                    q.put({"event": "error", "message": "No findings provided"}), loop
                )
                return
            graph = BeliefGraph()
            from apiro.axioms.seeding import build_seeds
            seeds, axioms, enriched_vignette = build_seeds(req.findings, axiom_extractor)
            traversal.run(
                seed_nodes=seeds,
                vignette=enriched_vignette,
                graph=graph,
                max_depth=req.max_depth,
                case_name="api_stream",
                on_event=on_event,
            )
        except Exception as exc:
            logger.error(f"Streaming traversal error: {exc}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                q.put({"event": "error", "message": str(exc)}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)   # sentinel → close stream

    # Launch traversal in background thread
    loop.run_in_executor(_executor, run_traversal)

    async def event_stream():
        while True:
            event = await q.get()
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.post("/run")
def run_investigation(req: InvestigationRequest):
    """Legacy synchronous endpoint — kept for backward compatibility."""
    if not traversal:
        raise HTTPException(status_code=500, detail="Apiro engine not initialised")

    t0 = time.time()
    try:
        from apiro.axioms.seeding import build_seeds
        seeds, axioms, enriched_vignette = build_seeds(req.findings, axiom_extractor)
        if not seeds:
            raise HTTPException(status_code=400, detail="Could not parse any valid findings")

        graph = BeliefGraph()
        result = traversal.run(
            seed_nodes=seeds,
            vignette=enriched_vignette,
            graph=graph,
            max_depth=req.max_depth,
            case_name="api_run",
        )
        elapsed = time.time() - t0

        nodes_list = [
            {
                "id":            n.id,
                "claim":         n.claim,
                "domain":        n.domain,
                "entropy_score": n.entropy_score,
                "resolved":      n.resolved,
                "is_rabbit_hole": n.is_rabbit_hole,
                "depth":         n.depth,
                "parent_id":     n.parent_id,
            }
            for n in graph.nodes.values()
        ]
        edges_list = [
            {"parent_id": e.parent_id, "child_id": e.child_id, "contradiction_flag": e.contradiction_flag}
            for e in graph.edges
        ]

        return {
            "synthesis":   result.synthesis or [],
            "nodes":       nodes_list,
            "edges":       edges_list,
            "duration":    elapsed,
            "stop_reason": result.stop_reason,
        }

    except Exception as exc:
        logger.error(f"Error during API investigation: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
