"""The single-page HTML served by the web interface."""

from __future__ import annotations

from .i18n import ui_strings

#: The page, with ``{{key}}`` markers filled in by :func:`index_html`.
INDEX_TEMPLATE = """<!doctype html>
<html lang="{{html_lang}}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
  :root { color-scheme: light dark; --bg:#faf9f7; --fg:#1a1a1a; --muted:#6b6b6b;
          --line:#e2e0dc; --card:#fff; --accent:#2f6f4f; --warn:#9a4a1e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16181a; --fg:#eceae6; --muted:#9a9a95; --line:#2c2f33; --card:#1e2124; --accent:#7fbf9a; --warn:#e0a06a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }
  header { border-bottom:1px solid var(--line); padding:20px 24px; }
  h1 { margin:0; font-size:20px; letter-spacing:-.01em; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:900px; margin:0 auto; padding:24px; display:grid; gap:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px; }
  label { display:block; font-weight:600; margin-bottom:6px; font-size:13px; }
  textarea { width:100%; padding:9px; border:1px solid var(--line); border-radius:7px;
        background:transparent; color:inherit; font:inherit; }
  /* The native file input renders its button in the browser's own language;
     a custom control keeps the page in the language it was rendered for. */
  .file { display:flex; align-items:center; gap:12px; width:100%; padding:8px 9px;
          border:1px solid var(--line); border-radius:7px; }
  .file input[type=file] { position:absolute; width:1px; height:1px; opacity:0; }
  .file span[role=button] { background:var(--line); color:inherit; border-radius:6px;
          padding:5px 12px; font-size:13px; white-space:nowrap; cursor:pointer; }
  .file input:focus-visible + span[role=button] { outline:2px solid var(--accent); outline-offset:2px; }
  .file .name { color:var(--muted); font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  textarea { min-height:64px; resize:vertical; }
  .field { margin-bottom:16px; }
  .hint { color:var(--muted); font-size:12px; margin-top:4px; }
  button { background:var(--accent); color:#fff; border:0; border-radius:7px; padding:10px 18px;
           font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:10px 0; }
  .bar > div { height:100%; background:var(--accent); transition:width .4s; }
  pre { white-space:pre-wrap; word-break:break-word; background:transparent; font-size:13px;
        max-height:65vh; overflow:auto; margin:0; }
  .status { font-size:13px; color:var(--muted); }
  .err { color:var(--warn); white-space:pre-wrap; font-size:13px; }
  .pill { display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px;
          border:1px solid var(--line); color:var(--muted); }
  .stats { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; margin-top:10px; }
  .stats b { font-size:18px; display:block; }
  [hidden] { display:none !important; }
</style>
</head>
<body>
<header>
  <h1>{{title}}</h1>
  <p>{{tagline}}</p>
</header>
<main>
  <section class="card">
    <div class="row" style="margin-bottom:14px"><span id="health" class="pill">{{checking_backends}}</span></div>
    <form id="form">
      <div class="field">
        <label for="video">{{video_label}}</label>
        <label class="file" for="video">
          <input type="file" id="video" name="video" accept="video/*,audio/*" required>
          <span role="button">{{choose_file}}</span>
          <span class="name" id="videoName">{{no_file}}</span>
        </label>
        <div class="hint">{{video_hint}}</div>
      </div>
      <div class="field">
        <label for="references">{{references_label}}</label>
        <label class="file" for="references">
          <input type="file" id="references" name="references" multiple accept=".pdf,.txt,.md,.docx,.html">
          <span role="button">{{choose_files}}</span>
          <span class="name" id="referencesName">{{no_file}}</span>
        </label>
        <div class="hint">{{references_hint}}</div>
      </div>
      <div class="field">
        <label for="urls">{{urls_label}}</label>
        <textarea id="urls" name="reference_urls" placeholder="https://…"></textarea>
      </div>
      <div class="row">
        <button type="submit" id="submit">{{submit}}</button>
        <label style="font-weight:400;font-size:13px;display:flex;gap:6px;align-items:center;margin:0">
          <input type="checkbox" id="offline" name="offline_mode"> {{offline_label}}
        </label>
      </div>
    </form>
  </section>

  <section class="card" id="progress" hidden>
    <div class="row"><strong id="stage">{{queued}}</strong><span class="status" id="elapsed"></span></div>
    <div class="bar"><div id="fill" style="width:0%"></div></div>
    <div class="status" id="loglines"></div>
    <div class="err" id="error" hidden></div>
    <div class="stats" id="stats" hidden></div>
  </section>

  <section class="card" id="reportCard" hidden>
    <div class="row" style="justify-content:space-between;margin-bottom:12px">
      <strong>{{report_heading}}</strong><a id="download" download="{{download_filename}}">{{download}}</a>
    </div>
    <pre id="report"></pre>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let timer = null;
// A poll started before the job finished can resolve after it: without this
// guard, its stale stage index overwrites the completed state.
let settled = false;

fetch('/api/health').then(r => r.json()).then(h => {
  const on = Object.entries(h.backends).filter(([,v]) => v).map(([k]) => k);
  const llm = h.llm.reachable ? `LLM: ${h.llm.model}` : `{{llm_unavailable}} (${h.llm.provider})`;
  $('health').textContent = `${llm} · backends: ${on.join(', ') || '{{no_optional_backends}}'}`;
}).catch(() => { $('health').textContent = '{{server_silent}}'; });

for (const [input, target] of [['video','videoName'], ['references','referencesName']]) {
  $(input).addEventListener('change', () => {
    const files = $(input).files;
    $(target).textContent = files.length === 0 ? '{{no_file}}'
      : files.length === 1 ? files[0].name
      : `{{n_files}}`.replace('{count}', files.length);
  });
}

$('form').addEventListener('submit', async e => {
  e.preventDefault();
  const data = new FormData();
  data.append('video', $('video').files[0]);
  for (const f of $('references').files) data.append('references', f);
  data.append('reference_urls', $('urls').value);
  data.append('offline_mode', $('offline').checked ? 'true' : 'false');

  settled = false;
  $('submit').disabled = true;
  $('progress').hidden = false; $('error').hidden = true; $('reportCard').hidden = true;
  $('stats').hidden = true; $('fill').style.width = '0%';

  const res = await fetch('/api/reviews', { method: 'POST', body: data });
  const job = await res.json();
  if (!res.ok) { fail(job.detail || '{{submit_failed}}'); return; }
  timer = setInterval(() => poll(job.job_id), 1500);
  poll(job.job_id);
});

async function poll(id) {
  const job = await (await fetch('/api/reviews/' + id)).json();
  if (settled) return;
  $('stage').textContent = job.status === 'done' ? '{{done}}' : job.stage;
  $('elapsed').textContent = job.elapsed + 's';
  $('fill').style.width = Math.round(100 * job.stage_index / job.stage_count) + '%';
  $('loglines').textContent = (job.log || []).slice(-3).join(' · ');

  if (job.status === 'failed') { settled = true; fail(job.error); return; }
  if (job.status !== 'done') return;

  settled = true;
  clearInterval(timer); timer = null; $('submit').disabled = false;
  $('fill').style.width = '100%';
  const s = job.summary || {};
  $('stats').hidden = false;
  $('stats').innerHTML = `
    <div><b>${s.analysed ?? 0}</b>{{stat_analysed}}</div>
    <div><b>${s.problems ?? 0}</b>{{stat_problems}}</div>
    <div><b>${Math.round(100 * (s.evidence_coverage ?? 0))}%</b>{{stat_coverage}}</div>
    <div><b>${s.dropped ?? 0}</b>{{stat_dropped}}</div>`;

  const md = await (await fetch(`/api/reviews/${id}/report`)).text();
  $('report').textContent = md;
  $('download').href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }));
  $('reportCard').hidden = false;
}

function fail(message) {
  settled = true;
  clearInterval(timer); timer = null;
  $('submit').disabled = false;
  $('error').hidden = false;
  $('error').textContent = message;
}
</script>
</body>
</html>"""


def index_html(language: str = "pt") -> str:
    """Render the interface in ``language``."""
    page = INDEX_TEMPLATE
    for key, value in ui_strings(language).items():
        page = page.replace("{{" + key + "}}", value)
    return page
