"""
TLabel面板模板生成器 — 生成Jupyter嵌入的HTML+JS+CSS

核心设计：
- Jupyter _repr_html_ 渲染
- Canvas雷达图+彩色时间轴
- 中英文切换
- 帧详情编辑器+批量修正
"""

import json
from typing import Optional


def generate_panel_html(data_dict: dict, lang: str = "auto", instance_id: str = "tlabel") -> str:
    """生成完整面板HTML"""
    data_json = json.dumps(data_dict, ensure_ascii=False, default=str)

    return f"""<!DOCTYPE html>
<div id="{instance_id}-root" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
     max-width: 960px; margin: 0 auto; background: #f8f9fa; color: #343a40;
     border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">

<!-- Header -->
<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;
     background:linear-gradient(135deg,#e9ecef,#f1f3f5);border-bottom:1px solid #dee2e6;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">🦞</span>
    <span style="font-size:16px;font-weight:700;color:#e85d75;" data-i18n="app.title">TLabel 触觉标注工具</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:12px;color:#868e96;" id="{instance_id}-sensor-info"></span>
    <button style="padding:4px 10px;border-radius:6px;border:1px solid #e85d75;background:transparent;
                   color:#e85d75;cursor:pointer;font-size:12px;" id="{instance_id}-lang-btn">EN</button>
  </div>
</div>

<!-- Stats Bar -->
<div style="display:flex;gap:16px;padding:10px 20px;background:#f1f3f5;font-size:12px;">
  <span>📊 <span data-i18n="stats.frames">帧数</span>: <b id="{instance_id}-stat-frames">0</b></span>
  <span>⏱ <span data-i18n="stats.duration">时长</span>: <b id="{instance_id}-stat-duration">0s</b></span>
  <span>🟢 <span data-i18n="stats.contact">接触率</span>: <b id="{instance_id}-stat-contact">0%</b></span>
  <span>🔴 <span data-i18n="stats.slip">滑移率</span>: <b id="{instance_id}-stat-slip">0%</b></span>
  <span>✏️ <span data-i18n="stats.modified">已修正</span>: <b id="{instance_id}-stat-modified">0</b></span>
</div>

<!-- Timeline -->
<div style="padding:12px 20px;background:#e9ecef;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <span style="font-size:12px;color:#868e96;" data-i18n="timeline.title">时间轴</span>
    <span style="font-size:12px;color:#e85d75;" id="{instance_id}-frame-label">-</span>
  </div>
  <canvas id="{instance_id}-timeline" width="920" height="60" 
          style="width:100%;height:60px;border-radius:8px;cursor:pointer;background:#fff;"></canvas>
  <div style="display:flex;justify-content:center;gap:8px;margin-top:6px;">
    <button id="{instance_id}-btn-prev"
            style="padding:4px 12px;border-radius:6px;border:1px solid #ced4da;background:#fff;color:#495057;cursor:pointer;">◀</button>
    <input type="number" id="{instance_id}-frame-input" min="0" style="width:60px;text-align:center;
           border-radius:6px;border:1px solid #ced4da;background:#fff;color:#343a40;padding:2px 4px;">
    <button id="{instance_id}-btn-next"
            style="padding:4px 12px;border-radius:6px;border:1px solid #ced4da;background:#fff;color:#495057;cursor:pointer;">▶</button>
  </div>
</div>

<!-- Main Content: Radar + Detail -->
<div style="display:flex;gap:16px;padding:16px 20px;">
  <!-- Radar Chart -->
  <div style="flex:1;background:#fff;border-radius:10px;padding:12px;border:1px solid #e9ecef;">
    <div style="font-size:12px;color:#868e96;margin-bottom:4px;" data-i18n="chart.radar">22维特征</div>
    <canvas id="{instance_id}-radar" width="300" height="280" style="width:100%;"></canvas>
  </div>
  <!-- Frame Detail -->
  <div style="flex:1;background:#fff;border-radius:10px;padding:12px;border:1px solid #e9ecef;" id="{instance_id}-detail-panel">
    <div style="font-size:12px;color:#868e96;margin-bottom:8px;" data-i18n="detail.title">帧详情</div>
    <div id="{instance_id}-detail-content" style="font-size:13px;line-height:1.8;"></div>
  </div>
</div>

<!-- Batch Patch -->
<div style="padding:12px 20px;background:#f1f3f5;">
  <div style="font-size:12px;color:#868e96;margin-bottom:6px;" data-i18n="batch.title">区间批量修正</div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <span data-i18n="batch.frameRange" style="font-size:12px;">帧范围：</span>
    <input type="number" id="{instance_id}-batch-start" style="width:60px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;" placeholder="0">
    <span style="color:#adb5bd;">—</span>
    <input type="number" id="{instance_id}-batch-end" style="width:60px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;">
    <select id="{instance_id}-batch-field" style="border-radius:6px;border:1px solid #ced4da;background:#fff;
            color:#343a40;padding:3px 6px;font-size:12px;">
      <option value="contact" data-i18n-option="field.contact">接触 (0/1)</option>
      <option value="slip_event" data-i18n-option="field.slip_event">滑移事件 (0/1)</option>
      <option value="force_magnitude" data-i18n-option="field.force_magnitude">力度</option>
      <option value="manipulation_phase" data-i18n-option="field.manipulation_phase">操作阶段</option>
    </select>
    <input type="text" id="{instance_id}-batch-value" style="width:80px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;" placeholder="0">
    <button id="{instance_id}-btn-batch"
            style="padding:4px 12px;border-radius:6px;border:none;background:#e85d75;color:#fff;
                   cursor:pointer;font-size:12px;" data-i18n="batch.apply">应用</button>
    <button id="{instance_id}-btn-undo"
            style="padding:4px 10px;border-radius:6px;border:1px solid #ced4da;background:#fff;
                   color:#495057;cursor:pointer;font-size:12px;">↩</button>
  </div>
</div>

<!-- Export -->
<div style="padding:10px 20px;display:flex;justify-content:flex-end;gap:8px;">
  <button id="{instance_id}-btn-export-json"
          style="padding:6px 14px;border-radius:8px;border:none;background:linear-gradient(135deg,#e85d75,#d1495b);
                 color:#fff;cursor:pointer;font-size:12px;" data-i18n="actions.exportFull">导出JSON</button>
  <button id="{instance_id}-btn-export-csv"
          style="padding:6px 14px;border-radius:8px;border:1px solid #e85d75;background:transparent;
                 color:#e85d75;cursor:pointer;font-size:12px;" data-i18n="actions.exportCSV">导出CSV</button>
</div>
</div>

<script>
(function() {{
  // ===== i18n =====
  const I18N = {{
    'zh-CN': {{
      'app.title': 'TLabel 触觉标注工具',
      'stats.frames': '帧数', 'stats.duration': '时长', 'stats.contact': '接触率',
      'stats.slip': '滑移率', 'stats.modified': '已修正',
      'timeline.title': '时间轴',
      'chart.radar': '22维特征',
      'detail.title': '帧详情',
      'batch.title': '区间批量修正', 'batch.frameRange': '帧范围：', 'batch.apply': '应用',
      'actions.exportFull': '导出JSON', 'actions.exportCSV': '导出CSV',
      'contact.yes': '有接触', 'contact.no': '无接触',
      'slip.yes': '滑移', 'slip.no': '无滑移',
      'phase': '操作阶段', 'confidence': '置信度',
      // 雷达图22维中文名
      'dim.contact': '接触', 'dim.deformation_magnitude': '形变', 'dim.force_magnitude': '力度',
      'dim.force_peak': '力峰值', 'dim.force_direction': '力方向', 'dim.slip_entropy': '滑移熵',
      'dim.slip_event': '滑移事件', 'dim.texture_energy': '纹理能量', 'dim.edge_density': '边缘密度',
      'dim.contact_area': '接触面积', 'dim.centroid_x': '质心X',
      'dim.normal_field_magnitude': '法向场强', 'dim.normal_field_variance': '法向方差',
      'dim.shear_field_magnitude': '切向场强', 'dim.shear_field_direction': '切向方向',
      'dim.delta_force_normal': 'δ法向力', 'dim.delta_force_shear': 'δ切向力', 'dim.friction_cone_ratio': '摩擦锥比',
      // 时序4维中文名
      'dim.optical_flow_magnitude': '光流幅度', 'dim.optical_flow_direction': '光流方向',
      'dim.temporal_deformation_rate': '形变速率', 'dim.contact_transition': '接触转换',
      // 批量修正下拉框
      'field.contact': '接触 (0/1)', 'field.slip_event': '滑移事件 (0/1)',
      'field.force_magnitude': '力度', 'field.manipulation_phase': '操作阶段',
    }},
    'en': {{
      'app.title': 'TLabel Tactile Annotator',
      'stats.frames': 'Frames', 'stats.duration': 'Duration', 'stats.contact': 'Contact',
      'stats.slip': 'Slip', 'stats.modified': 'Modified',
      'timeline.title': 'Timeline',
      'chart.radar': '22-Dim Features',
      'detail.title': 'Frame Detail',
      'batch.title': 'Range Batch Patch', 'batch.frameRange': 'Range: ', 'batch.apply': 'Apply',
      'actions.exportFull': 'Export JSON', 'actions.exportCSV': 'Export CSV',
      'contact.yes': 'Contact', 'contact.no': 'No Contact',
      'slip.yes': 'Slip', 'slip.no': 'No Slip',
      'phase': 'Phase', 'confidence': 'Confidence',
      'dim.contact': 'Contact', 'dim.deformation_magnitude': 'Deform', 'dim.force_magnitude': 'Force',
      'dim.force_peak': 'Force Peak', 'dim.force_direction': 'Force Dir', 'dim.slip_entropy': 'Slip Ent',
      'dim.slip_event': 'Slip Ev', 'dim.texture_energy': 'Texture', 'dim.edge_density': 'Edge',
      'dim.contact_area': 'Area', 'dim.centroid_x': 'Centroid',
      'dim.normal_field_magnitude': 'NF Mag', 'dim.normal_field_variance': 'NF Var',
      'dim.shear_field_magnitude': 'SF Mag', 'dim.shear_field_direction': 'SF Dir',
      'dim.delta_force_normal': 'δF_N', 'dim.delta_force_shear': 'δF_S', 'dim.friction_cone_ratio': 'Friction',
      // 时序4维英文名
      'dim.optical_flow_magnitude': 'OF Mag', 'dim.optical_flow_direction': 'OF Dir',
      'dim.temporal_deformation_rate': 'Deform Rate', 'dim.contact_transition': 'Contact Trans',
      'field.contact': 'Contact (0/1)',
      'field.slip_event': 'Slip Event (0/1)',
      'field.force_magnitude': 'Force Magnitude',
      'field.manipulation_phase': 'Phase',
    }},
       'ja': {{
       'app.title': 'TLabel 触覚アノテーター',
       'stats.frames': 'フレーム数',
       'stats.duration': '時間'
      }},
       'ko': {{
       'app.title': 'TLabel 촉각 주석 도구',
       'stats.frames': '프레임 수',
       'stats.duration': '시간'
      }}
  }};

  let currentLang = '{lang}' === 'auto' ? ((navigator.language||'').startsWith('en') ? 'en' : 'zh-CN') : '{lang}';
  const tid = '{instance_id}';
  const data = {data_json};
  let currentFrameIdx = 0;
  let undoStack = [];
  let modifiedCount = 0;

  function t(key) {{
    return (I18N[currentLang] || {{}})[key] || key;
  }}

  function applyI18n() {{
    const root = document.getElementById(tid + '-root');
    root.querySelectorAll('[data-i18n]').forEach(el => {{
      el.textContent = t(el.getAttribute('data-i18n'));
    }});
    root.querySelectorAll('[data-i18n-option]').forEach(el => {{
      el.textContent = t(el.getAttribute('data-i18n-option'));
    }});
    document.getElementById(tid + '-lang-btn').textContent = currentLang === 'zh-CN' ? 'EN' : '中';
  }}

  // ===== Stats =====
  function updateStats() {{
    const ep = data.episode || {{}};
    const stats = ep.stats || {{}};
    document.getElementById(tid + '-stat-frames').textContent = data.frames.length;
    document.getElementById(tid + '-stat-duration').textContent = (ep.duration_s || 0).toFixed(1) + 's';
    document.getElementById(tid + '-stat-contact').textContent = ((stats.contact_ratio || 0) * 100).toFixed(1) + '%';
    document.getElementById(tid + '-stat-slip').textContent = ((stats.slip_ratio || 0) * 100).toFixed(1) + '%';
    document.getElementById(tid + '-stat-modified').textContent = modifiedCount;
    document.getElementById(tid + '-sensor-info').textContent = 
      (data.sensor && data.sensor.type) || '';
  }}

  // ===== Timeline =====
  function drawTimeline() {{
    const canvas = document.getElementById(tid + '-timeline');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const frames = data.frames || [];
    if (!frames.length) return;

    const barH = 24, y = (h - barH) / 2;
    const barW = Math.max(w / frames.length, 1);

    for (let i = 0; i < frames.length; i++) {{
      const f = frames[i];
      const tl = f.tlabel_v2 || {{}};
      if (tl.contact > 0.5) {{
        ctx.fillStyle = tl.slip_event > 0.5 ? '#e85d75' : '#51cf66';
      }} else {{
        ctx.fillStyle = '#dee2e6';
      }}
      ctx.fillRect(i * barW, y, barW + 0.5, barH);
    }}

    // Current frame marker
    const cx = currentFrameIdx * barW + barW / 2;
    ctx.strokeStyle = '#343a40';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, y - 4);
    ctx.lineTo(cx, y + barH + 4);
    ctx.stroke();
  }}

  // Click on timeline
  document.getElementById(tid + '-timeline').addEventListener('click', function(e) {{
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frames = data.frames || [];
    const barW = Math.max(920 / frames.length, 1);
    currentFrameIdx = Math.min(Math.floor(x / barW), frames.length - 1);
    showFrame(currentFrameIdx);
  }});

  // ===== Frame Display =====
  function showFrame(idx) {{
    const frames = data.frames || [];
    if (idx < 0 || idx >= frames.length) return;
    currentFrameIdx = idx;
    const f = frames[idx];
    const tl = f.tlabel_v2 || {{}};

    document.getElementById(tid + '-frame-label').textContent = 
      'Frame ' + f.frame_idx + ' / ' + (frames.length - 1);
    document.getElementById(tid + '-frame-input').value = f.frame_idx;

    // Detail panel
    const contactText = tl.contact > 0.5 ? 
      '<span style="color:#2b8a3e;">✅ ' + t('contact.yes') + '</span>' : 
      '<span style="color:#adb5bd;">⬜ ' + t('contact.no') + '</span>';
    const slipText = tl.slip_event > 0.5 ? 
      '<span style="color:#e85d75;">⚠️ ' + t('slip.yes') + '</span>' : 
      '<span style="color:#adb5bd;">' + t('slip.no') + '</span>';

    document.getElementById(tid + '-detail-content').innerHTML = 
      contactText + ' &nbsp;|&nbsp; ' + slipText + '<br>' +
      '<span style="color:#868e96;">' + t('phase') + ':</span> <b style="color:#e85d75;">' + (f.manipulation_phase || '-') + '</b><br>' +
      '<span style="color:#868e96;">' + t('confidence') + ':</span> ' + ((f.confidence || 1) * 100).toFixed(0) + '%<br>' +
      '<hr style="border-color:#e9ecef;margin:6px 0;">' +
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:11px;">' +
      Object.entries(tl).map(([k, v]) => 
        '<span style="color:#868e96;">' + (t('dim.' + k) || k) + '</span><span style="color:#495057;">' +
        (typeof v === 'number' ? v.toFixed(4) : v) + '</span>'
      ).join('') +
      '</div>';

    drawTimeline();
    drawRadar(tl);
  }}

  // ===== Radar Chart (Canvas, no external lib) =====
  function drawRadar(tl) {{
    const canvas = document.getElementById(tid + '-radar');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const maxVals = {{
      contact: 1, deformation_magnitude: 50, force_magnitude: 50, force_peak: 255,
      force_direction: 180, slip_entropy: 5, slip_event: 1, texture_energy: 100,
      edge_density: 1, contact_area: 1, centroid_x: 1,
      normal_field_magnitude: 50, normal_field_variance: 500,
      shear_field_magnitude: 20, shear_field_direction: 180,
      delta_force_normal: 10, delta_force_shear: 10, friction_cone_ratio: 5,
      // 时序4维
      optical_flow_magnitude: 20, optical_flow_direction: 360,
      temporal_deformation_rate: 50, contact_transition: 1,
    }};

    const dimKeys = Object.keys(maxVals);
    const labels = dimKeys.map(k => t('dim.' + k) || k);

    const keys = dimKeys;
    const values = keys.map(k => Math.min((tl[k] || 0) / maxVals[k], 1));
    const n = keys.length;
    const cx = w / 2, cy = h / 2;
    const R = Math.min(cx, cy) - 30;

    // Grid
    for (let ring = 1; ring <= 4; ring++) {{
      ctx.beginPath();
      const r = R * ring / 4;
      for (let i = 0; i <= n; i++) {{
        const angle = (2 * Math.PI * i / n) - Math.PI / 2;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = 'rgba(0,0,0,0.06)';
      ctx.stroke();
    }}

    // Axes + labels
    ctx.font = '9px sans-serif';
    ctx.fillStyle = '#868e96';
    for (let i = 0; i < n; i++) {{
      const angle = (2 * Math.PI * i / n) - Math.PI / 2;
      const x = cx + R * Math.cos(angle);
      const y = cy + R * Math.sin(angle);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = 'rgba(0,0,0,0.04)';
      ctx.stroke();
      const lx = cx + (R + 14) * Math.cos(angle);
      const ly = cy + (R + 14) * Math.sin(angle);
      ctx.fillText(labels[i], lx - 12, ly + 3);
    }}

    // Data polygon
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {{
      const idx = i % n;
      const angle = (2 * Math.PI * idx / n) - Math.PI / 2;
      const r = R * values[idx];
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.fillStyle = 'rgba(232,93,117,0.18)';
    ctx.fill();
    ctx.strokeStyle = '#e85d75';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Data points
    for (let i = 0; i < n; i++) {{
      const angle = (2 * Math.PI * i / n) - Math.PI / 2;
      const r = R * values[i];
      ctx.beginPath();
      ctx.arc(cx + r * Math.cos(angle), cy + r * Math.sin(angle), 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#e85d75';
      ctx.fill();
    }}
  }}

  // ===== Batch Patch =====
  function batchPatch() {{
    const start = parseInt(document.getElementById(tid + '-batch-start').value) || 0;
    const end = parseInt(document.getElementById(tid + '-batch-end').value) || 0;
    const field = document.getElementById(tid + '-batch-field').value;
    const valStr = document.getElementById(tid + '-batch-value').value;
    const val = (field === 'contact' || field === 'slip_event') ? parseFloat(valStr) : valStr;

    const frames = data.frames || [];
    let count = 0;
    const undoBatch = [];

    for (let i = 0; i < frames.length; i++) {{
      const fi = frames[i].frame_idx;
      if (fi >= start && fi <= end) {{
        if (field === 'manipulation_phase') {{
          undoBatch.push({{idx: i, field: field, old: frames[i].manipulation_phase}});
          frames[i].manipulation_phase = val;
        }} else {{
          const old = (frames[i].tlabel_v2 || {{}})[field];
          if (old !== val) {{
            undoBatch.push({{idx: i, field: field, old: old}});
            if (!frames[i].tlabel_v2) frames[i].tlabel_v2 = {{}};
            frames[i].tlabel_v2[field] = val;
            if (field === 'contact' && val === 0) {{
              if (frames[i].tlabel_v2.slip_event > 0) {{
                undoBatch.push({{idx: i, field: 'slip_event', old: frames[i].tlabel_v2.slip_event}});
                frames[i].tlabel_v2.slip_event = 0;
              }}
              if (frames[i].tlabel_v2.force_magnitude > 0) {{
                undoBatch.push({{idx: i, field: 'force_magnitude', old: frames[i].tlabel_v2.force_magnitude}});
                frames[i].tlabel_v2.force_magnitude = 0;
              }}
              if (frames[i].manipulation_phase !== 'idle') {{
                undoBatch.push({{idx: i, field: 'manipulation_phase', old: frames[i].manipulation_phase}});
                frames[i].manipulation_phase = 'idle';
              }}
            }}
            count++;
          }}
        }}
      }}
    }}
    undoStack.push(undoBatch);
    modifiedCount += count;
    updateStats();
    showFrame(currentFrameIdx);
    return count;
  }}

  function undo() {{
    if (!undoStack.length) return;
    const batch = undoStack.pop();
    const frames = data.frames || [];
    // 统计本次撤销涉及多少个不同帧
    const undoneFrames = new Set();
    for (const op of batch) {{
      if (op.field === 'manipulation_phase') {{
        frames[op.idx].manipulation_phase = op.old;
      }} else {{
        if (frames[op.idx].tlabel_v2) frames[op.idx].tlabel_v2[op.field] = op.old;
      }}
      undoneFrames.add(op.idx);
    }}
    modifiedCount = Math.max(0, modifiedCount - undoneFrames.size);
    updateStats();
    showFrame(currentFrameIdx);
  }}

  // ===== Navigation =====
  function prevFrame() {{
    if (currentFrameIdx > 0) showFrame(currentFrameIdx - 1);
  }}
  function nextFrame() {{
    const frames = data.frames || [];
    if (currentFrameIdx < frames.length - 1) showFrame(currentFrameIdx + 1);
  }}
  function jumpTo(val) {{
    const fi = parseInt(val);
    const frames = data.frames || [];
    const idx = frames.findIndex(f => f.frame_idx === fi);
    if (idx >= 0) showFrame(idx);
  }}

  // ===== Export =====
  function exportJSON() {{
    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tlabel_export.json';
    a.click();
  }}
  function exportCSV() {{
    const frames = data.frames || [];
    const dims = ['contact','deformation_magnitude','force_magnitude','force_peak',
      'force_direction','slip_entropy','slip_event','texture_energy',
      'edge_density','contact_area','centroid_x',
      'normal_field_magnitude','normal_field_variance',
      'shear_field_magnitude','shear_field_direction',
      'delta_force_normal','delta_force_shear','friction_cone_ratio',
      // 时序4维
      'optical_flow_magnitude','optical_flow_direction',
      'temporal_deformation_rate','contact_transition'];
    let csv = 'frame_idx,timestamp_s,manipulation_phase,confidence,' + dims.join(',') + '\\n';
    for (const f of frames) {{
      const tl = f.tlabel_v2 || {{}};
      csv += f.frame_idx + ',' + f.timestamp_s + ',' + f.manipulation_phase + ',' + f.confidence;
      for (const d of dims) csv += ',' + (tl[d] || 0);
      csv += '\\n';
    }}
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tlabel_export.csv';
    a.click();
  }}

  // ===== Lang Toggle =====
  function toggleLang()
  {{
    const langs = ['zh-CN', 'en', 'ja', 'ko'];
    const idx = langs.indexOf(currentLang);
    currentLang = langs[(idx + 1) % langs.length];

    applyI18n();
    showFrame(currentFrameIdx);
  }}

  // ===== Event Listeners (不使用inline onclick，Jupyter可能拦截) =====
  document.getElementById(tid + '-lang-btn').addEventListener('click', toggleLang);
  document.getElementById(tid + '-btn-prev').addEventListener('click', prevFrame);
  document.getElementById(tid + '-btn-next').addEventListener('click', nextFrame);
  document.getElementById(tid + '-frame-input').addEventListener('change', function() {{ jumpTo(this.value); }});
  document.getElementById(tid + '-btn-batch').addEventListener('click', batchPatch);
  document.getElementById(tid + '-btn-undo').addEventListener('click', undo);
  document.getElementById(tid + '-btn-export-json').addEventListener('click', exportJSON);
  document.getElementById(tid + '-btn-export-csv').addEventListener('click', exportCSV);

  // ===== Init =====
  updateStats();
  applyI18n();
  showFrame(0);

  window['_tlabel_' + tid] = {{
    prevFrame, nextFrame, jumpTo, batchPatch, undo,
    exportJSON, exportCSV, toggleLang
  }};
}})();
</script>"""
