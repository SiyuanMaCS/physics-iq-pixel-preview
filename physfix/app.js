/* PhysFixBench sample50 — model repair judgment viewer (server-side verdicts). */
const $ = s => document.querySelector(s);
const VARIANTS = ['self_fixing', 'problem_only', 'problem_expected'];
const VARIANT_LABEL = {self_fixing: '自主修复', problem_only: '仅问题', problem_expected: '问题 + 预期'};
const STATUS_LABEL = {fixed: '已修复', partially_fixed: '部分修复', not_fixed: '未修复', pending: '待判定'};
const SOURCE_NAMES = {physion_eval: 'Physion', phyground: 'PhyGround', robojudge: 'RoboJudge', videophy2: 'VideoPhy2'};
const STATUS_COLOR = {fixed: 'var(--fixed)', partially_fixed: 'var(--partial)', not_fixed: 'var(--notfixed)', pending: 'var(--pending)'};

let results, judgments, selected = 0, model = '', variant = 'self_fixing', visible = [];
const videos = [$('#original'), $('#generated')];

const resultFor = (sample, m, v) => sample.results.find(r => r.model === m && r.variant === v);
const judgmentFor = (itemIdx, m, v) =>
  judgments.items.find(x => x.item === itemIdx)?.models?.[m]?.[v] || null;

function pauseAll() { videos.forEach(v => v.pause()); $('#play-both').textContent = '同时播放'; }

function renderProgress() {
  const box = $('#model-progress');
  box.replaceChildren();
  for (const m of results.models) {
    const jm = judgments.models[m.id];
    const card = document.createElement('div');
    card.className = 'mp-card';
    const total = jm.total_items, done = jm.judged_items;
    const seg = VARIANTS.map(v => {
      const t = jm.video_tally[v];
      return `<div style="display:contents">
        ${['fixed', 'partially_fixed', 'not_fixed'].map(s =>
          t[s] ? `<i title="${VARIANT_LABEL[v]} · ${STATUS_LABEL[s]} ${t[s]}" style="width:${t[s] / total * 100}%;background:${STATUS_COLOR[s]}"></i>` : '').join('')}
      </div>`;
    }).join('');
    const legend = VARIANTS.map(v => {
      const t = jm.video_tally[v];
      return `<span>${VARIANT_LABEL[v]}：
        <span class="chip fixed">${t.fixed}</span>
        <span class="chip partially_fixed">${t.partially_fixed}</span>
        <span class="chip not_fixed">${t.not_fixed}</span>
        <b style="margin-left:4px">${t.fixed + t.partially_fixed + t.not_fixed}/${total}</b></span>`;
    }).join('');
    card.innerHTML = `<h4>${m.name}</h4>
      <div class="mp-sub">判定进度 ${done} / ${total} 条样本</div>
      <div class="mp-bar">${seg}</div>
      <div class="mp-legend">${legend}</div>`;
    box.appendChild(card);
  }
}

function verdictOf(itemIdx) {
  const j = judgmentFor(itemIdx, model, variant);
  return j ? j.video_status : 'pending';
}

function renderQueue() {
  const q = $('#search').value.trim().toLowerCase();
  const src = $('#source-filter').value;
  const vf = $('#verdict-filter').value;
  visible = results.samples.filter(s => {
    if (src && s.source !== src) return false;
    if (q && !`${s.id} ${s.generator} ${s.dir_name}`.toLowerCase().includes(q)) return false;
    if (vf && verdictOf(s.item) !== vf) return false;
    return true;
  });
  $('#list-count').textContent = `${visible.length} / ${results.samples.length}`;
  const queue = $('#queue');
  queue.replaceChildren();
  for (const s of visible) {
    const st = verdictOf(s.item);
    const b = document.createElement('button');
    b.setAttribute('aria-current', String(s.item === selected));
    b.innerHTML = `<strong>${String(s.item).padStart(2, '0')} · ${s.id.split('::').at(-1)}</strong>
      <small><span class="dotmark ${st}"></span>${SOURCE_NAMES[s.source] || s.source} · ${STATUS_LABEL[st]}</small>`;
    b.onclick = () => select(s.item);
    queue.appendChild(b);
  }
  const pos = visible.findIndex(s => s.item === selected);
  $('#position').textContent = pos >= 0 ? `${pos + 1} / ${visible.length}` : '筛选外';
  $('#prev').disabled = pos <= 0;
  $('#next').disabled = pos < 0 || pos >= visible.length - 1;
}

function loadVideo(el, url, statusEl) {
  el.pause(); el.removeAttribute('src'); el.load();
  if (!url) { statusEl.textContent = '此结果尚未接入。'; return; }
  el.src = url; el.playbackRate = Number($('#speed').value); el.load();
  statusEl.textContent = '正在读取视频…';
}

function renderVerdict(sample) {
  const panel = $('#verdict-panel');
  const j = judgmentFor(sample.item, model, variant);
  if (!j) {
    panel.innerHTML = `<div class="vp-head"><span class="chip pending">待判定</span>
      <span class="big">该模型此提示词的视频尚未完成人工标准判定</span></div>
      <div class="pending-box">评审进行中，判定结果将在下次数据快照更新后出现在这里。</div>`;
    return;
  }
  const head = `<div class="vp-head"><span class="chip ${j.video_status}">视频级结论：${STATUS_LABEL[j.video_status]}</span>
    <span class="big muted">${results.models.find(m => m.id === model)?.short || model} · ${VARIANT_LABEL[variant]}</span></div>`;
  const cards = j.judgments.map(c => `
    <div class="issue-card">
      <div class="ic-head">
        <strong>问题 ${c.issue_index + 1} <span class="ic-ts">（${c.timestamp}s）</span></strong>
        <span class="chip ${c.status}">${STATUS_LABEL[c.status]}</span>
      </div>
      <p><span class="lab">标注问题：</span><span class="ev">${c.problem}</span></p>
      <p><span class="lab">期望行为：</span><span class="ev">${c.expected}</span></p>
      <p><span class="lab">画面证据：</span><span class="ev">${c.visual_evidence_zh}</span></p>
      <p><span class="lab">局限/不确定性：</span><span class="lim">${c.limitation_zh}</span></p>
    </div>`).join('');
  panel.innerHTML = `${head}${cards}
    <div class="montage"><a href="${j.montage}" target="_blank" rel="noopener">
      <img src="${j.montage}" alt="判定证据帧拼图" loading="lazy"></a>
      <p class="muted" style="font-size:12px;margin:6px 0 0">点击查看证据帧拼图原图（时间戳为视频内秒数）。</p></div>`;
}

function renderAnnotation(sample) {
  const j = judgmentFor(sample.item, model, variant);
  const problems = $('#problems'), expected = $('#expected');
  problems.replaceChildren(); expected.replaceChildren();
  if (j && j.judgments.length) {
    for (const c of j.judgments) {
      problems.appendChild(Object.assign(document.createElement('p'),
        {textContent: `第${c.timestamp}s：${c.problem}`}));
      expected.appendChild(Object.assign(document.createElement('p'), {textContent: c.expected}));
    }
  } else {
    sample.problems.forEach(t => problems.appendChild(Object.assign(document.createElement('p'), {textContent: t})));
    sample.expected.forEach(t => expected.appendChild(Object.assign(document.createElement('p'), {textContent: t})));
  }
}

function select(itemIdx) {
  pauseAll();
  selected = itemIdx;
  const sample = results.samples.find(s => s.item === itemIdx);
  const r = resultFor(sample, model, variant);
  renderQueue();
  $('#sample-source').textContent = `${SOURCE_NAMES[sample.source] || sample.source} / ${sample.generator}`;
  $('#sample-title').textContent = `${String(itemIdx).padStart(2, '0')} · ${sample.id.split('::').at(-1)}`;
  $('#record-id').textContent = sample.id;
  loadVideo(videos[0], sample.input_url, $('#original-status'));
  $('#original-link').href = sample.input_url;
  loadVideo(videos[1], r?.url, $('#generated-status'));
  $('#output-title').textContent = results.models.find(m => m.id === model)?.name || model;
  $('#output-badge').textContent = r?.status === 'success' ? '已生成' : '未接入';
  if (r?.url) $('#output-link').href = r.url; else $('#output-link').removeAttribute('href');
  $('#play-both').disabled = !r?.url;
  $('#prompt').textContent = r?.prompt || '暂无提示词';
  $('#params').textContent = JSON.stringify(r?.params || {}, null, 2);
  document.querySelectorAll('[data-variant]').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.variant === variant)));
  renderAnnotation(sample);
  renderVerdict(sample);
  location.hash = new URLSearchParams({item: itemIdx, model, variant}).toString();
}

videos.forEach((v, i) => {
  const st = i ? $('#generated-status') : $('#original-status');
  v.onloadedmetadata = () => { st.textContent = `${v.videoWidth} × ${v.videoHeight} · ${v.duration.toFixed(2)} s`; };
  v.onerror = () => { st.textContent = '视频加载失败，可单独打开或重试。'; };
  v.onended = () => { if (videos.every(x => x.paused)) $('#play-both').textContent = '同时播放'; };
});

$('#play-both').onclick = async () => {
  if (videos.some(v => !v.paused)) { pauseAll(); return; }
  $('#play-both').disabled = true;
  const outcomes = await Promise.allSettled(videos.map(v => { if (v.ended) v.currentTime = 0; return v.play(); }));
  $('#play-both').disabled = false;
  if (outcomes.some(x => x.status === 'rejected')) pauseAll();
  else $('#play-both').textContent = '全部暂停';
};
$('#reset-both').onclick = () => { pauseAll(); videos.forEach(v => { if (v.readyState) v.currentTime = 0; }); };
$('#speed').onchange = () => videos.forEach(v => v.playbackRate = Number($('#speed').value));
$('#search').oninput = renderQueue;
$('#source-filter').onchange = renderQueue;
$('#verdict-filter').onchange = renderQueue;
$('#model').onchange = () => { model = $('#model').value; renderQueue(); select(selected); };
document.querySelectorAll('[data-variant]').forEach(b =>
  b.onclick = () => { variant = b.dataset.variant; select(selected); });
$('#prev').onclick = () => { const i = visible.findIndex(s => s.item === selected); select(visible[i - 1].item); };
$('#next').onclick = () => { const i = visible.findIndex(s => s.item === selected); select(visible[i + 1].item); };
document.addEventListener('visibilitychange', () => { if (document.hidden) pauseAll(); });

async function init() {
  try {
    [results, judgments] = await Promise.all([
      fetch('./data/model-results.json').then(r => { if (!r.ok) throw Error(`HTTP ${r.status}`); return r.json(); }),
      fetch('./data/judgments.json').then(r => { if (!r.ok) throw Error(`HTTP ${r.status}`); return r.json(); }),
    ]);
    $('#snapshot-date').textContent = results.snapshot_date;
    $('#judge-updated').textContent = `判定更新于 ${judgments.updated_at.replace('T', ' ')} UTC`;
    $('#hf-link').href = results.repo_url;
    for (const m of results.models) $('#model').add(new Option(m.short, m.id));
    for (const s of [...new Set(results.samples.map(x => x.source))])
      $('#source-filter').add(new Option(SOURCE_NAMES[s] || s, s));
    const hash = new URLSearchParams(location.hash.slice(1));
    model = results.models.some(m => m.id === hash.get('model')) ? hash.get('model') : results.models[0].id;
    if (VARIANTS.includes(hash.get('variant'))) variant = hash.get('variant');
    $('#model').value = model;
    renderProgress();
    $('#workspace').hidden = false;
    $('#load-status').hidden = true;
    const hItem = Number(hash.get('item'));
    select(Number.isInteger(hItem) && results.samples.some(s => s.item === hItem) ? hItem : 0);
  } catch (e) {
    $('#load-status').textContent = `结果清单加载失败，请刷新重试：${e.message}`;
  }
}
init();
