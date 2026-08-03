"""
TLabel面板模板生成器 — 生成Jupyter嵌入的HTML+JS+CSS

核心设计：
- Jupyter _repr_html_ 渲染
- Canvas雷达图+彩色时间轴
- 中英文切换
- 帧详情编辑器+批量修正
- Episode级标注表单（v0.4.2）
- 数据质量评分仪表盘（v0.4.2）
- 统计摘要 describe 表格（v0.4.2）
"""

import json
from tlabel._version import __version__
from typing import Optional, Dict


_PANEL_LANGS = ("zh-CN", "en", "ja", "ko")


def _resolve_panel_lang(lang: str, sensor_info: Optional[Dict]) -> str:
    """Normalize the panel language; auto prefers sensor_info.lang."""
    if lang == "auto":
        lang = (sensor_info or {}).get("lang") or "zh-CN"
    if lang in ("zh", "zh-cn"):
        return "zh-CN"
    return lang if lang in _PANEL_LANGS else "zh-CN"


def generate_panel_html(
    data_dict: dict,
    lang: str = "auto",
    instance_id: str = "tlabel",
    episode_info: Optional[Dict] = None,
    quality_score: Optional[Dict] = None,
    describe_stats: Optional[Dict] = None,
    predict_highlights: Optional[Dict] = None,
    auto_label_summary: Optional[Dict] = None,
    tactile_images: Optional[list] = None,  # v0.12: 触觉图像列表（base64）
) -> str:
    """生成完整面板HTML"""
    data_json = json.dumps(data_dict, ensure_ascii=False, default=str)
    # v0.5.0: Inject predict highlight data
    episode_json = json.dumps(episode_info or {}, ensure_ascii=False, default=str)
    quality_json = json.dumps(quality_score or {}, ensure_ascii=False, default=str)
    describe_json = json.dumps(describe_stats or {}, ensure_ascii=False, default=str)
    predict_json = json.dumps(predict_highlights or {}, ensure_ascii=False, default=str)
    autolabel_json = json.dumps(auto_label_summary or {}, ensure_ascii=False, default=str)
    # v0.12: 触觉图像数据
    images_json = json.dumps(tactile_images or [], ensure_ascii=False, default=str)
    tid = instance_id
    initial_lang = _resolve_panel_lang(lang, data_dict.get("sensor_info"))

    return f"""<div id="{tid}-root" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
     max-width: 960px; margin: 0 auto; background: #f8f9fa; color: #343a40;
     border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">

<!-- Header -->
<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;
     background:linear-gradient(135deg,#e9ecef,#f1f3f5);border-bottom:1px solid #dee2e6;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">🦞</span>
    <span style="font-size:16px;font-weight:700;color:#e85d75;" data-i18n="app.title">TLabel 触觉标注工具</span>
    <span style="font-size:10px;color:#868e96;background:#e9ecef;padding:1px 6px;border-radius:4px;">v{__version__}</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:12px;color:#868e96;" id="{tid}-sensor-info"></span>
    <button style="padding:4px 10px;border-radius:6px;border:1px solid #e85d75;background:transparent;
                   color:#e85d75;cursor:pointer;font-size:12px;" id="{tid}-lang-btn">EN</button>
    <span id="{tid}-predict-badge" style="display:none;font-size:11px;color:#e67700;background:#fff3bf;padding:2px 8px;border-radius:4px;cursor:help;" data-i18n-title="predict.badge.title">🤖 Predicted</span>
    <button style="padding:4px 10px;border-radius:6px;border:1px solid #ced4da;background:transparent;
                   color:#868e96;cursor:pointer;font-size:14px;" id="{tid}-dark-btn">🌙</button>
  </div>
</div>

<!-- Tab Navigation -->
<div style="display:flex;border-bottom:2px solid #dee2e6;padding:0 20px;background:#fff;" id="{tid}-tab-nav">
  <button class="{tid}-tab" data-tab="annotate" style="padding:10px 18px;border:none;background:transparent;
          color:#e85d75;font-weight:700;font-size:13px;cursor:pointer;border-bottom:2px solid #e85d75;
          margin-bottom:-2px;" data-i18n="tab.annotate">📝 标注</button>
  <button class="{tid}-tab" data-tab="episode" style="padding:10px 18px;border:none;background:transparent;
          color:#868e96;font-size:13px;cursor:pointer;border-bottom:2px solid transparent;
          margin-bottom:-2px;" data-i18n="tab.episode">🎬 Episode</button>
  <button class="{tid}-tab" data-tab="quality" style="padding:10px 18px;border:none;background:transparent;
          color:#868e96;font-size:13px;cursor:pointer;border-bottom:2px solid transparent;
          margin-bottom:-2px;" data-i18n="tab.quality">📊 质量评分</button>
  <button class="{tid}-tab" data-tab="stats" style="padding:10px 18px;border:none;background:transparent;
          color:#868e96;font-size:13px;cursor:pointer;border-bottom:2px solid transparent;
          margin-bottom:-2px;" data-i18n="tab.stats">📈 统计</button>
  <button class="{tid}-tab" data-tab="export" style="padding:10px 18px;border:none;background:transparent;
          color:#868e96;font-size:13px;cursor:pointer;border-bottom:2px solid transparent;
          margin-bottom:-2px;" data-i18n="tab.export">🚀 导出</button>
</div>

<!-- ==================== TAB 1: Annotate ==================== -->
<div id="{tid}-panel-annotate">

<!-- Stats Bar -->
<div style="display:flex;gap:16px;padding:10px 20px;background:#f1f3f5;font-size:12px;">
  <span>📊 <span data-i18n="stats.frames">帧数</span>: <b id="{tid}-stat-frames">0</b></span>
  <span>⏱ <span data-i18n="stats.duration">时长</span>: <b id="{tid}-stat-duration">0s</b></span>
  <span>🟢 <span data-i18n="stats.contact">接触率</span>: <b id="{tid}-stat-contact">0%</b></span>
  <span>🔴 <span data-i18n="stats.slip">滑移率</span>: <b id="{tid}-stat-slip">0%</b></span>
  <span>✏️ <span data-i18n="stats.modified">已修正</span>: <b id="{tid}-stat-modified">0</b></span>
</div>

<!-- v0.14: Primitive 预标注面板 -->
<div style="padding:8px 20px;background:#fff;border-bottom:1px solid #e9ecef;">
  <div id="{tid}-pp-toggle" style="display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none;padding:4px 0;">
    <span style="font-size:12px;font-weight:600;color:#495057;">🏷️ Primitive 预标注</span>
    <span id="{tid}-pp-arrow" style="font-size:10px;color:#868e96;">▼</span>
  </div>
  <div id="{tid}-pp-content" style="display:none;padding:8px 0 4px;">
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap;">
      <span style="font-size:11px;color:#868e96;">Taxonomy:</span>
      <select id="{tid}-pp-taxonomy" style="border-radius:6px;border:1px solid #ced4da;background:#fff;color:#343a40;padding:2px 6px;font-size:11px;">
        <option value="default">默认 (7种)</option>
        <option value="full">完整 T-Rex (22种)</option>
      </select>
      <span style="font-size:11px;color:#868e96;">最低置信度:</span>
      <input type="number" id="{tid}-pp-minconf" value="0.4" min="0" max="1" step="0.05"
             style="width:50px;border-radius:4px;border:1px solid #ced4da;text-align:center;font-size:11px;padding:2px;">
      <button id="{tid}-pp-run" style="padding:3px 12px;border-radius:6px;border:none;background:#e85d75;color:#fff;cursor:pointer;font-size:11px;font-weight:600;">🤖 运行预标注</button>
    </div>
    <div id="{tid}-pp-result" style="display:none;font-size:11px;color:#495057;background:#f8f9fa;padding:6px 10px;border-radius:6px;margin-top:4px;"></div>
  </div>
</div>

<!-- Timeline -->
<div style="padding:12px 20px;background:#e9ecef;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <span style="font-size:12px;color:#868e96;" data-i18n="timeline.title">时间轴</span>
    <span style="font-size:12px;color:#e85d75;" id="{tid}-frame-label">-</span>
  </div>
  <canvas id="{tid}-timeline" width="920" height="60" 
          style="width:100%;height:60px;border-radius:8px;cursor:pointer;background:#fff;"></canvas>
  <canvas id="{tid}-primitive-track" width="920" height="28" 
          style="width:100%;height:28px;border-radius:6px;background:#f8f9fa;margin-top:4px;display:none;"></canvas>
  <div id="{tid}-primitive-legend" style="display:none;font-size:10px;color:#868e96;margin-top:4px;"></div>
  <div style="display:flex;justify-content:center;gap:8px;margin-top:6px;">
    <button id="{tid}-btn-prev"
            style="padding:4px 12px;border-radius:6px;border:1px solid #ced4da;background:#fff;color:#495057;cursor:pointer;">◀</button>
    <input type="number" id="{tid}-frame-input" min="0" style="width:60px;text-align:center;
           border-radius:6px;border:1px solid #ced4da;background:#fff;color:#343a40;padding:2px 4px;">
    <button id="{tid}-btn-next"
            style="padding:4px 12px;border-radius:6px;border:1px solid #ced4da;background:#fff;color:#495057;cursor:pointer;">▶</button>
  </div>
</div>

<!-- Main Content: Radar + Detail + Image -->
<div style="display:flex;gap:16px;padding:16px 20px;">
  <!-- Radar Chart -->
  <div style="flex:1;background:#fff;border-radius:10px;padding:12px;border:1px solid #e9ecef;">
    <div style="font-size:12px;color:#868e96;margin-bottom:4px;" data-i18n="chart.radar">14维Schema</div>
    <canvas id="{tid}-radar" width="360" height="340" style="width:100%;"></canvas>
  </div>
  <!-- Tactile Image (v0.12) -->
  <div style="flex:1;background:#fff;border-radius:10px;padding:12px;border:1px solid #e9ecef;" id="{tid}-image-panel">
    <div style="font-size:12px;color:#868e96;margin-bottom:4px;">触觉图像</div>
    <div id="{tid}-image-container" style="width:100%;height:340px;display:flex;align-items:center;justify-content:center;background:#f8f9fa;border-radius:8px;overflow:hidden;">
      <img id="{tid}-tactile-img" src="" style="max-width:100%;max-height:100%;object-fit:contain;display:none;" />
      <div id="{tid}-no-image" style="color:#adb5bd;font-size:13px;">无图像数据</div>
    </div>
  </div>
  <!-- Frame Detail -->
  <div style="flex:1;background:#fff;border-radius:10px;padding:12px;border:1px solid #e9ecef;" id="{tid}-detail-panel">
    <div style="font-size:12px;color:#868e96;margin-bottom:8px;" data-i18n="detail.title">帧详情</div>
    <div id="{tid}-detail-content" style="font-size:13px;line-height:1.8;"></div>
  </div>
</div>

<!-- Batch Patch -->
<div style="padding:12px 20px;background:#f1f3f5;">
  <div style="font-size:12px;color:#868e96;margin-bottom:6px;" data-i18n="batch.title">区间批量修正</div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <span data-i18n="batch.frameRange" style="font-size:12px;">帧范围：</span>
    <input type="number" id="{tid}-batch-start" style="width:60px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;" placeholder="0">
    <span style="color:#adb5bd;">—</span>
    <input type="number" id="{tid}-batch-end" style="width:60px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;">
    <select id="{tid}-batch-field" style="border-radius:6px;border:1px solid #ced4da;background:#fff;
            color:#343a40;padding:3px 6px;font-size:12px;">
      <option value="contact" data-i18n="batch.optContact">接触 (0/1)</option>
      <option value="slip_event" data-i18n="batch.optSlip">滑移事件 (0/1)</option>
      <option value="force_magnitude" data-i18n="batch.optForce">力度</option>
      <option value="object_deformation" data-i18n="batch.optDeformation">形变</option>
      <option value="manipulation_phase" data-i18n="batch.optPhase">操作阶段</option>
      <option value="primitive_label" data-i18n="batch.optPrimitive">Primitive 标注</option>
    </select>
    <input type="text" id="{tid}-batch-value" style="width:80px;border-radius:6px;border:1px solid #ced4da;
           background:#fff;color:#343a40;padding:3px 6px;font-size:12px;" placeholder="0">
    <select id="{tid}-batch-prim-value" style="display:none;border-radius:6px;border:1px solid #ced4da;background:#fff;color:#343a40;padding:3px 6px;font-size:12px;">
      <option value="grasp">grasp</option>
      <option value="press">press</option>
      <option value="squeeze">squeeze</option>
      <option value="reach">reach</option>
      <option value="wrap">wrap</option>
      <option value="wipe">wipe</option>
      <option value="lift">lift</option>
      <option value="fold">fold</option>
      <option value="cut">cut</option>
      <option value="insert">insert</option>
      <option value="peel">peel</option>
      <option value="assemble">assemble</option>
      <option value="extract">extract</option>
      <option value="twist">twist</option>
      <option value="shake">shake</option>
      <option value="dispense">dispense</option>
      <option value="disassemble">disassemble</option>
      <option value="pour">pour</option>
      <option value="open">open</option>
      <option value="close">close</option>
      <option value="screw">screw</option>
      <option value="unscrew">unscrew</option>
    </select>
    <button id="{tid}-btn-batch"
            style="padding:4px 12px;border-radius:6px;border:none;background:#e85d75;color:#fff;
                   cursor:pointer;font-size:12px;" data-i18n="batch.apply">应用</button>
    <button id="{tid}-btn-undo"
            style="padding:4px 10px;border-radius:6px;border:1px solid #ced4da;background:#fff;
                   color:#495057;cursor:pointer;font-size:12px;">↩</button>
  </div>
</div>

<!-- Export Section -->
<div style="padding:16px 20px;background:#fff;border-top:2px solid #e9ecef;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <span style="font-size:14px;font-weight:600;color:#343a40;" data-i18n="export.title"> 导出数据</span>
    <span style="font-size:11px;color:#868e96;" data-i18n="export.hint">选择格式并点击对应按钮</span>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <button id="{tid}-btn-export-json"
            style="flex:1;min-width:120px;padding:10px 16px;border-radius:8px;border:none;
                   background:linear-gradient(135deg,#e85d75,#d1495b);color:#fff;
                   cursor:pointer;font-size:13px;font-weight:600;
                   box-shadow:0 2px 6px rgba(232,93,117,0.3);" 
            data-i18n="actions.exportFull">💾 导出 JSON</button>
    <button id="{tid}-btn-export-csv"
            style="flex:1;min-width:120px;padding:10px 16px;border-radius:8px;
                   border:2px solid #e85d75;background:#fff;color:#e85d75;
                   cursor:pointer;font-size:13px;font-weight:600;" 
            data-i18n="actions.exportCSV">📊 导出 CSV</button>
    <button id="{tid}-btn-export-hdf5"
            style="flex:1;min-width:120px;padding:10px 16px;border-radius:8px;
                   border:2px solid #495057;background:#fff;color:#495057;
                   cursor:pointer;font-size:13px;font-weight:600;" 
            data-i18n="actions.exportHDF5">🔬 导出 HDF5</button>
  </div>
  <div id="{tid}-export-status" style="margin-top:8px;font-size:11px;color:#868e96;display:none;"></div>
</div>

</div><!-- end panel-annotate -->

<!-- ==================== TAB 2: Episode ==================== -->
<div id="{tid}-panel-episode" style="display:none;">
<div style="padding:20px;">

  <div style="background:#fff;border-radius:10px;padding:20px;border:1px solid #e9ecef;margin-bottom:16px;">
    <div style="font-size:14px;font-weight:700;color:#343a40;margin-bottom:16px;">
      🎬 <span data-i18n="episode.title">Episode 级标注</span>
    </div>
    <div style="font-size:12px;color:#868e96;margin-bottom:16px;" data-i18n="episode.desc">
      为整个交互Episode添加语义标注，描述操作任务的整体结果和属性。标注结果会写入 episode_info 并随数据一起导出。
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
      <!-- Outcome -->
      <div>
        <label style="font-size:12px;color:#495057;font-weight:600;display:block;margin-bottom:4px;" data-i18n="episode.outcome">操作结果</label>
        <select id="{tid}-ep-outcome" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid #ced4da;
                background:#fff;color:#343a40;font-size:13px;">
          <option value="" data-i18n="episode.outcomeDefault">— 未标注 —</option>
          <option value="success" data-i18n="episode.outSuccess">✅ 成功 (success)</option>
          <option value="failure" data-i18n="episode.outFailure">❌ 失败 (failure)</option>
          <option value="aborted" data-i18n="episode.outAborted">⏹️ 中止 (aborted)</option>
          <option value="partial" data-i18n="episode.outPartial">⚠️ 部分 (partial)</option>
        </select>
      </div>

      <!-- Manipulation Type -->
      <div>
        <label style="font-size:12px;color:#495057;font-weight:600;display:block;margin-bottom:4px;" data-i18n="episode.manipType">操作类型</label>
        <select id="{tid}-ep-manip-type" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid #ced4da;
                background:#fff;color:#343a40;font-size:13px;">
          <option value="" data-i18n="episode.manipTypeDefault">— 未标注 —</option>
          <option value="grasp" data-i18n="episode.mGrasp">🤏 抓取 (grasp)</option>
          <option value="push" data-i18n="episode.mPush">👆 推 (push)</option>
          <option value="pull" data-i18n="episode.mPull">👇 拉 (pull)</option>
          <option value="tap" data-i18n="episode.mTap">👆 轻触 (tap)</option>
          <option value="lift" data-i18n="episode.mLift">⬆️ 提起 (lift)</option>
          <option value="place" data-i18n="episode.mPlace">⬇️ 放置 (place)</option>
          <option value="rotate" data-i18n="episode.mRotate">🔄 旋转 (rotate)</option>
          <option value="insert" data-i18n="episode.mInsert">🔩 插入 (insert)</option>
        </select>
      </div>

      <!-- Difficulty -->
      <div>
        <label style="font-size:12px;color:#495057;font-weight:600;display:block;margin-bottom:4px;" data-i18n="episode.difficulty">难度等级</label>
        <select id="{tid}-ep-difficulty" style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid #ced4da;
                background:#fff;color:#343a40;font-size:13px;">
          <option value="" data-i18n="episode.difficultyDefault">— 未标注 —</option>
          <option value="easy" data-i18n="episode.dEasy">🟢 简单 (easy)</option>
          <option value="medium" data-i18n="episode.dMedium">🟡 中等 (medium)</option>
          <option value="hard" data-i18n="episode.dHard">🔴 困难 (hard)</option>
        </select>
      </div>

      <!-- Operator (optional) -->
      <div>
        <label style="font-size:12px;color:#495057;font-weight:600;display:block;margin-bottom:4px;" data-i18n="episode.operator">标注人</label>
        <input type="text" id="{tid}-ep-operator" data-i18n-placeholder="episode.operatorPh" placeholder="可选" style="width:100%;padding:8px 10px;border-radius:8px;
               border:1px solid #ced4da;background:#fff;color:#343a40;font-size:13px;">
      </div>
    </div>

    <!-- Notes -->
    <div style="margin-top:16px;">
      <label style="font-size:12px;color:#495057;font-weight:600;display:block;margin-bottom:4px;" data-i18n="episode.notes">备注</label>
      <textarea id="{tid}-ep-notes" rows="3" data-i18n-placeholder="episode.notesPh" placeholder="记录操作细节、失败原因等..." 
                style="width:100%;padding:8px 10px;border-radius:8px;border:1px solid #ced4da;
                       background:#fff;color:#343a40;font-size:13px;resize:vertical;font-family:inherit;"></textarea>
    </div>

    <!-- Apply button -->
    <div style="margin-top:16px;display:flex;gap:10px;align-items:center;">
      <button id="{tid}-btn-episode-apply"
              style="padding:8px 24px;border-radius:8px;border:none;background:#e85d75;color:#fff;
                     cursor:pointer;font-size:13px;font-weight:600;
                     box-shadow:0 2px 6px rgba(232,93,117,0.3);" data-i18n="episode.apply">✅ 保存Episode标注</button>
      <span id="{tid}-episode-status" style="font-size:12px;color:#51cf66;display:none;"></span>
    </div>
  </div>

  <!-- Episode Current Info -->
  <div style="background:#fff;border-radius:10px;padding:16px;border:1px solid #e9ecef;">
    <div style="font-size:12px;color:#868e96;margin-bottom:8px;" data-i18n="episode.currentInfo">当前Episode信息</div>
    <div id="{tid}-episode-info-display" style="font-size:13px;line-height:1.8;color:#343a40;">
    </div>
  </div>

</div>
</div><!-- end panel-episode -->

<!-- ==================== TAB 3: Quality ==================== -->
<div id="{tid}-panel-quality" style="display:none;">
<div style="padding:20px;">

  <div style="background:#fff;border-radius:10px;padding:20px;border:1px solid #e9ecef;margin-bottom:16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <div style="font-size:14px;font-weight:700;color:#343a40;">
        📊 <span data-i18n="quality.title">数据质量评分</span>
      </div>
      <div id="{tid}-quality-grade-badge" style="font-size:28px;font-weight:900;width:56px;height:56px;
           border-radius:50%;display:flex;align-items:center;justify-content:center;
           background:#e9ecef;color:#868e96;">-</div>
    </div>

    <div style="font-size:12px;color:#868e96;margin-bottom:16px;" data-i18n="quality.desc">
      基于4个维度自动评估数据质量，对标国标《具身智能数据质量规范》。评分由 Python 引擎预计算。
    </div>

    <!-- Overall Score -->
    <div style="text-align:center;margin-bottom:20px;">
      <div style="font-size:48px;font-weight:900;color:#e85d75;" id="{tid}-quality-overall">--</div>
      <div style="font-size:12px;color:#868e96;" data-i18n="quality.overallLabel">综合评分 (0-100)</div>
    </div>

    <!-- 4 Dimension Bars -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;" id="{tid}-quality-dims">
      <!-- Physical Consistency -->
      <div style="background:#f8f9fa;border-radius:8px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-size:12px;font-weight:600;color:#495057;">🔧 <span data-i18n="quality.physical">物理一致性</span></span>
          <span style="font-size:13px;font-weight:700;color:#e85d75;" id="{tid}-q-physical">--</span>
        </div>
        <div style="height:8px;background:#e9ecef;border-radius:4px;overflow:hidden;">
          <div id="{tid}-q-physical-bar" style="height:100%;background:linear-gradient(90deg,#e85d75,#ff8fa3);border-radius:4px;width:0%;transition:width 0.5s;"></div>
        </div>
        <div style="font-size:10px;color:#868e96;margin-top:4px;" data-i18n="quality.physicalDesc">权重 30% · 联动规则满足度</div>
      </div>

      <!-- Temporal Smoothness -->
      <div style="background:#f8f9fa;border-radius:8px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-size:12px;font-weight:600;color:#495057;">📈 <span data-i18n="quality.temporal">时序平滑度</span></span>
          <span style="font-size:13px;font-weight:700;color:#e85d75;" id="{tid}-q-temporal">--</span>
        </div>
        <div style="height:8px;background:#e9ecef;border-radius:4px;overflow:hidden;">
          <div id="{tid}-q-temporal-bar" style="height:100%;background:linear-gradient(90deg,#4dabf7,#74c0fc);border-radius:4px;width:0%;transition:width 0.5s;"></div>
        </div>
        <div style="font-size:10px;color:#868e96;margin-top:4px;" data-i18n="quality.temporalDesc">权重 25% · 相邻帧突变检测</div>
      </div>

      <!-- Completeness -->
      <div style="background:#f8f9fa;border-radius:8px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-size:12px;font-weight:600;color:#495057;">📋 <span data-i18n="quality.completeness">完整性</span></span>
          <span style="font-size:13px;font-weight:700;color:#e85d75;" id="{tid}-q-completeness">--</span>
        </div>
        <div style="height:8px;background:#e9ecef;border-radius:4px;overflow:hidden;">
          <div id="{tid}-q-completeness-bar" style="height:100%;background:linear-gradient(90deg,#51cf66,#8ce99a);border-radius:4px;width:0%;transition:width 0.5s;"></div>
        </div>
        <div style="font-size:10px;color:#868e96;margin-top:4px;" data-i18n="quality.completenessDesc">权重 25% · 字段缺失/全零比例</div>
      </div>

      <!-- Coverage -->
      <div style="background:#f8f9fa;border-radius:8px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="font-size:12px;font-weight:600;color:#495057;">🎯 <span data-i18n="quality.coverage">覆盖率</span></span>
          <span style="font-size:13px;font-weight:700;color:#e85d75;" id="{tid}-q-coverage">--</span>
        </div>
        <div style="height:8px;background:#e9ecef;border-radius:4px;overflow:hidden;">
          <div id="{tid}-q-coverage-bar" style="height:100%;background:linear-gradient(90deg,#ffd43b,#ffe066);border-radius:4px;width:0%;transition:width 0.5s;"></div>
        </div>
        <div style="font-size:10px;color:#868e96;margin-top:4px;" data-i18n="quality.coverageDesc">权重 20% · 有意义标注占比</div>
      </div>
    </div>
  </div>

  <!-- Warnings -->
  <div style="background:#fff;border-radius:10px;padding:16px;border:1px solid #e9ecef;">
    <div style="font-size:12px;font-weight:600;color:#495057;margin-bottom:8px;">⚠️ <span data-i18n="quality.warnings">质量警告</span></div>
    <div id="{tid}-quality-warnings" style="font-size:12px;line-height:1.8;color:#868e96;"></div>
  </div>

</div>
</div><!-- end panel-quality -->

<!-- ==================== TAB 4: Stats ==================== -->
<div id="{tid}-panel-stats" style="display:none;">
<div style="padding:20px;">

  <div style="background:#fff;border-radius:10px;padding:20px;border:1px solid #e9ecef;margin-bottom:16px;">
    <div style="font-size:14px;font-weight:700;color:#343a40;margin-bottom:4px;">
      📈 <span data-i18n="stats.title">统计摘要 (describe)</span>
    </div>
    <div style="font-size:12px;color:#868e96;margin-bottom:16px;" data-i18n="stats.desc">
      类 pandas.DataFrame.describe() 统计，由 Python 引擎预计算。
    </div>
    <div id="{tid}-describe-table" style="overflow-x:auto;"></div>
  </div>

</div>
</div><!-- end panel-stats -->

<!-- ==================== TAB 5: Export (FTP-1/MTTS) ==================== -->
<div id="{tid}-panel-export" style="display:none;">
<div style="padding:20px;">

  <!-- FTP-1 Section -->
  <div style="background:#fff;border-radius:10px;padding:20px;border:1px solid #e9ecef;margin-bottom:16px;">
    <div style="font-size:14px;font-weight:700;color:#343a40;margin-bottom:4px;">
      🚀 <span data-i18n="export.ftp1_title">FTP-1 / MTTS 格式导出</span>
    </div>
    <div style="font-size:12px;color:#868e96;margin-bottom:16px;" data-i18n="export.ftp1_desc">
      导出为触觉基础模型 FTP-1 兼容的 MTTS Zarr 格式，可直接用于微调或推理。
    </div>

    <!-- Sensor Config -->
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
      <div style="flex:1;min-width:180px;">
        <label style="font-size:11px;color:#868e96;display:block;margin-bottom:4px;" data-i18n="export.sensor_name">传感器名称</label>
        <select id="{tid}-ftp1-sensor" style="width:100%;padding:6px 10px;border:1px solid #ced4da;border-radius:6px;font-size:12px;background:#fff;">
          <option value="GelSightMini" selected>GelSightMini</option>
          <option value="GelSight">GelSight</option>
          <option value="FreeTacMan">FreeTacMan</option>
          <option value="ViTaMIn">ViTaMIn</option>
          <option value="3DViTac">3DViTac (matrix)</option>
          <option value="Contactile">Contactile (matrix)</option>
          <option value="BinaryContact">BinaryContact (binary)</option>
        </select>
      </div>
      <div style="flex:1;min-width:120px;">
        <label style="font-size:11px;color:#868e96;display:block;margin-bottom:4px;" data-i18n="export.side">安装位置</label>
        <select id="{tid}-ftp1-side" style="width:100%;padding:6px 10px;border:1px solid #ced4da;border-radius:6px;font-size:12px;background:#fff;">
          <option value="right" selected>Right (右手)</option>
          <option value="left">Left (左手)</option>
        </select>
      </div>
      <div style="flex:1;min-width:120px;">
        <label style="font-size:11px;color:#868e96;display:block;margin-bottom:4px;" data-i18n="export.group">组名</label>
        <select id="{tid}-ftp1-group" style="width:100%;padding:6px 10px;border:1px solid #ced4da;border-radius:6px;font-size:12px;background:#fff;">
          <option value="gripper" selected>Gripper (夹爪)</option>
          <option value="dexterous">Dexterous (灵巧手)</option>
          <option value="wrist">Wrist (腕部)</option>
        </select>
      </div>
    </div>

    <!-- Functional Area Mapping -->
    <div style="background:#f8f9fa;border-radius:8px;padding:14px;margin-bottom:16px;">
      <div style="font-size:12px;font-weight:600;color:#495057;margin-bottom:8px;" data-i18n="export.area_title">
        🎯 功能区映射 (MTTS Functional Areas)
      </div>
      <div style="font-size:11px;color:#868e96;margin-bottom:10px;" data-i18n="export.area_desc">
        选择此传感器对应的 MTTS 功能区槽位。夹爪通常映射为拇指尖(0)+食指尖(1)。
      </div>
      <div id="{tid}-ftp1-areas" style="display:flex;flex-wrap:wrap;gap:6px;">
        <!-- Dynamically populated -->
      </div>
      <div style="margin-top:10px;display:flex;gap:8px;">
        <button id="{tid}-ftp1-preset-gripper" style="padding:3px 10px;border-radius:4px;border:1px solid #e85d75;background:transparent;color:#e85d75;cursor:pointer;font-size:11px;" data-i18n="export.preset_gripper">预设: 夹爪 [0,1]</button>
        <button id="{tid}-ftp1-preset-three" style="padding:3px 10px;border-radius:4px;border:1px solid #ced4da;background:transparent;color:#868e96;cursor:pointer;font-size:11px;" data-i18n="export.preset_three">预设: 三指 [0,1,2]</button>
        <button id="{tid}-ftp1-preset-five" style="padding:3px 10px;border-radius:4px;border:1px solid #ced4da;background:transparent;color:#868e96;cursor:pointer;font-size:11px;" data-i18n="export.preset_five">预设: 五指 [0-4]</button>
      </div>
    </div>

    <!-- Export Button -->
    <div style="display:flex;align-items:center;gap:12px;">
      <button id="{tid}-ftp1-export-btn" style="padding:8px 24px;border-radius:8px;border:none;background:linear-gradient(135deg,#e85d75,#d63384);color:#fff;font-weight:700;font-size:13px;cursor:pointer;box-shadow:0 2px 8px rgba(232,93,117,0.3);">
        <span data-i18n="export.btn_export">📦 导出为 FTP-1 Zarr</span>
      </button>
      <span id="{tid}-ftp1-status" style="font-size:12px;color:#868e96;"></span>
    </div>

    <!-- Export Result -->
    <div id="{tid}-ftp1-result" style="display:none;margin-top:14px;background:#f1f3f5;border-radius:8px;padding:14px;font-size:12px;">
      <div style="font-weight:600;color:#343a40;margin-bottom:8px;" data-i18n="export.result_title">📋 导出结果</div>
      <pre id="{tid}-ftp1-result-content" style="margin:0;font-family:monospace;font-size:11px;color:#495057;white-space:pre-wrap;"></pre>
    </div>
  </div>

  <!-- FTP-1 Format Reference -->
  <div style="background:#fff;border-radius:10px;padding:20px;border:1px solid #e9ecef;">
    <div style="font-size:13px;font-weight:600;color:#343a40;margin-bottom:8px;" data-i18n="export.format_ref">
      📖 MTTS Zarr 格式说明
    </div>
    <div style="font-size:11px;color:#495057;line-height:1.8;font-family:monospace;background:#f8f9fa;padding:12px;border-radius:6px;">
      <div style="color:#868e96;">// 每个 side + group 包含 4 个 Zarr key:</div>
      <div><span style="color:#e85d75;">right_tactile_data_gripper</span>: (T, N, H, W, 3) uint8 &nbsp;<span style="color:#868e96;">← 触觉图像</span></div>
      <div><span style="color:#e85d75;">right_tactile_area_gripper</span>: (T, N) int32 &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#868e96;">← 功能区ID</span></div>
      <div><span style="color:#e85d75;">right_tactile_sensor_gripper</span>: (T,) string &nbsp;&nbsp;&nbsp;<span style="color:#868e96;">← 传感器名</span></div>
      <div><span style="color:#e85d75;">right_tactile_type_gripper</span>: (T,) string &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#868e96;">← image/matrix/binary</span></div>
    </div>
    <div style="font-size:11px;color:#868e96;margin-top:8px;" data-i18n="export.format_note">
      💡 导出的 .zarr 文件可直接被 FTP-1 的 dataset_zarr.py 加载，用于模型微调。
    </div>
  </div>

</div>
</div><!-- end panel-export -->

</div><!-- end root -->

<script>
(function() {{
  // ===== Data =====
  const data = {data_json};
  const episodeInfo = {episode_json};
  const qualityData = {quality_json};
  const describeData = {describe_json};
  const predictHighlights = {predict_json};
  const autoLabelSummary = {autolabel_json};
  const tactileImages = {images_json};  // v0.12: 触觉图像列表
  const tid = '{tid}';
  let currentFrameIdx = 0;
  let currentLang = '{initial_lang}';
  let modifiedCount = 0;
  const undoStack = [];

  // ===== i18n =====
  const I18N = {{
    'zh-CN': {{
      'app.title': 'TLabel 触觉标注工具',
      'stats.frames': '帧数', 'stats.duration': '时长', 'stats.contact': '接触率',
      'stats.slip': '滑移率', 'stats.modified': '已修正',
      'timeline.title': '时间轴',
      'chart.radar': '14维Schema',
      'detail.title': '帧详情',
      'batch.title': '区间批量修正',
      'batch.frameRange': '帧范围：',
      'batch.apply': '应用',
      'export.title': '导出数据', 'export.hint': '选择格式并点击对应按钮',
      'actions.exportFull': '💾 导出 JSON', 'actions.exportCSV': '📊 导出 CSV', 'actions.exportHDF5': '🔬 导出 HDF5',
      'export.success': '导出成功',
      'tab.annotate': '📝 标注', 'tab.episode': '🎬 Episode', 'tab.quality': '📊 质量评分', 'tab.stats': '📈 统计', 'tab.export': '🚀 导出',
      'export.ftp1_title': 'FTP-1 / MTTS 格式导出', 'export.ftp1_desc': '导出为触觉基础模型 FTP-1 兼容的 MTTS Zarr 格式',
      'export.sensor_name': '传感器名称', 'export.side': '安装位置', 'export.group': '组名',
      'export.area_title': '🎯 功能区映射', 'export.area_desc': '选择传感器对应的 MTTS 功能区槽位',
      'export.btn_export': '📦 导出为 FTP-1 Zarr', 'export.result_title': '📋 导出结果',
      'export.format_ref': '📖 MTTS Zarr 格式说明', 'export.format_note': '💡 导出的 .zarr 文件可直接被 FTP-1 加载',
      'export.preset_gripper': '预设: 夹爪 [0,1]', 'export.preset_three': '预设: 三指 [0,1,2]', 'export.preset_five': '预设: 五指 [0-4]',
      'episode.title': 'Episode 级标注', 'episode.desc': '为整个交互Episode添加语义标注，描述操作任务的整体结果和属性。标注结果会写入 episode_info 并随数据一起导出。',
      'episode.outcome': '操作结果', 'episode.outcomeDefault': '— 未标注 —',
      'episode.manipType': '操作类型', 'episode.manipTypeDefault': '— 未标注 —',
      'episode.difficulty': '难度等级', 'episode.difficultyDefault': '— 未标注 —',
      'episode.operator': '标注人', 'episode.notes': '备注',
      'episode.apply': '✅ 保存Episode标注',
      'episode.currentInfo': '当前Episode信息',
      'quality.title': '数据质量评分',
      'quality.desc': '基于4个维度自动评估数据质量，对标国标《具身智能数据质量规范》。评分由 Python 引擎预计算。',
      'quality.overallLabel': '综合评分 (0-100)',
      'quality.physical': '物理一致性', 'quality.physicalDesc': '权重 30% · 联动规则满足度',
      'quality.temporal': '时序平滑度', 'quality.temporalDesc': '权重 25% · 相邻帧突变检测',
      'quality.completeness': '完整性', 'quality.completenessDesc': '权重 25% · 字段缺失/全零比例',
      'quality.coverage': '覆盖率', 'quality.coverageDesc': '权重 20% · 有意义标注占比',
      'quality.warnings': '质量警告',
      'stats.title': '统计摘要 (describe)', 'stats.desc': '类 pandas.DataFrame.describe() 统计，由 Python 引擎预计算。',
      'stats.noData': '暂无统计数据',
      'batch.optContact': '接触 (0/1)', 'batch.optSlip': '滑移事件 (0/1)',
      'batch.optForce': '力度', 'batch.optDeformation': '形变', 'batch.optPhase': '操作阶段',
      'episode.outSuccess': '✅ 成功 (success)', 'episode.outFailure': '❌ 失败 (failure)',
      'episode.outAborted': '⏹️ 中止 (aborted)', 'episode.outPartial': '⚠️ 部分 (partial)',
      'episode.mGrasp': '🤏 抓取 (grasp)', 'episode.mPush': '👆 推 (push)',
      'episode.mPull': '👇 拉 (pull)', 'episode.mTap': '👆 轻触 (tap)',
      'episode.mLift': '⬆️ 提起 (lift)', 'episode.mPlace': '⬇️ 放置 (place)',
      'episode.mRotate': '🔄 旋转 (rotate)', 'episode.mInsert': '🔩 插入 (insert)',
      'episode.dEasy': '🟢 简单 (easy)', 'episode.dMedium': '🟡 中等 (medium)',
      'episode.dHard': '🔴 困难 (hard)',
      'episode.operatorPh': '可选', 'episode.notesPh': '记录操作细节、失败原因等...',
      'detail.contact': '接触', 'detail.slip': '滑移', 'detail.force': '力度',
      'detail.deformation': '形变', 'detail.area': '面积', 'detail.entropy': '熵',
      'detail.normal': '法向', 'detail.shear': '剪切',
      'quality.noWarnings': '无质量警告',
      'episode.saved': '已保存', 'episode.noLabels': '暂无Episode标注',
      'predict.badge.title': 'AI预标注结果',
      'predict.lowConf': '低置信度',
      'predict.method': '方法',
      'predict.smooth': '时序平滑',
      'predict.hmm': 'HMM解码',
      'predict.cascade': '联动修正',
      // Stats i18n keys for describe table
      'stats.count': '计数', 'stats.mean': '均值', 'stats.std': '标准差',
      'stats.min': '最小值', 'stats.max': '最大值',
      'stats.p25': '25%分位', 'stats.p50': '中位数', 'stats.p75': '75%分位',
      // Radar chart i18n labels (dimension names — Schema V2 14维)
      'dim.contact': '接触', 'dim.contact_centroid': '质心', 'dim.contact_region': '区域',
      'dim.force_magnitude': '力度', 'dim.force_vector': '力向',
      'dim.torque_vector': '扭矩', 'dim.slip_event': '滑移',
      'dim.slip_velocity': '滑速', 'dim.manipulation_phase': '阶段',
      'dim.texture_class': '纹理', 'dim.object_deformation': '形变',
      'dim.temperature': '温度', 'dim.confidence': '置信',
      'dim.compliance_level': '合规',
      // UI messages
      'export.hdf5Msg': 'HDF5 导出需要使用 Python API:\\n\\ndata.export("output.hdf5")',
      'export.selectArea': '⚠️ 请至少选择一个功能区',
      'export.preparing': '⏳ 正在准备导出...',
      'export.previewReady': '✅ 导出预览已生成',
      'episode.savedMsg': 'Episode标注已保存（将随导出数据一起输出）',
      'shortcuts.help': '快捷键:\\n← → 切帧\\n空格 标记/取消接触\\nS 标记/取消滑移\\nD 切换暗色模式\\n? 显示帮助',
    }},
    'en': {{
      'app.title': 'TLabel Tactile Annotation',
      'stats.frames': 'Frames', 'stats.duration': 'Duration', 'stats.contact': 'Contact',
      'stats.slip': 'Slip', 'stats.modified': 'Modified',
      'timeline.title': 'Timeline',
      'chart.radar': '14-Dim Schema',
      'detail.title': 'Frame Detail',
      'batch.title': 'Batch Patch',
      'batch.frameRange': 'Range:',
      'batch.apply': 'Apply',
      'export.title': 'Export', 'export.hint': 'Choose format',
      'actions.exportFull': '💾 JSON', 'actions.exportCSV': '📊 CSV', 'actions.exportHDF5': '🔬 HDF5',
      'export.success': 'Exported',
      'tab.annotate': '📝 Annotate', 'tab.episode': '🎬 Episode', 'tab.quality': '📊 Quality', 'tab.stats': '📈 Stats', 'tab.export': '🚀 Export',
      'export.ftp1_title': 'FTP-1 / MTTS Format Export', 'export.ftp1_desc': 'Export to FTP-1 compatible MTTS Zarr format',
      'export.sensor_name': 'Sensor Name', 'export.side': 'Side', 'export.group': 'Group',
      'export.area_title': '🎯 Functional Area Mapping', 'export.area_desc': 'Select MTTS functional area slots for this sensor',
      'export.btn_export': '📦 Export as FTP-1 Zarr', 'export.result_title': '📋 Export Result',
      'export.format_ref': '📖 MTTS Zarr Format Reference', 'export.format_note': '💡 Exported .zarr can be loaded directly by FTP-1',
      'export.preset_gripper': 'Preset: Gripper [0,1]', 'export.preset_three': 'Preset: 3-Finger [0,1,2]', 'export.preset_five': 'Preset: 5-Finger [0-4]',
      'episode.title': 'Episode Annotation', 'episode.desc': 'Add semantic labels for the entire interaction episode. Results are saved to episode_info and exported with data.',
      'episode.outcome': 'Outcome', 'episode.outcomeDefault': '— Not labeled —',
      'episode.manipType': 'Manipulation Type', 'episode.manipTypeDefault': '— Not labeled —',
      'episode.difficulty': 'Difficulty', 'episode.difficultyDefault': '— Not labeled —',
      'episode.operator': 'Annotator', 'episode.notes': 'Notes',
      'episode.apply': '✅ Save Episode Label',
      'episode.currentInfo': 'Current Episode Info',
      'quality.title': 'Data Quality Score',
      'quality.desc': 'Auto-assessed data quality across 4 dimensions, aligned with national standard for embodied intelligence data quality.',
      'quality.overallLabel': 'Overall Score (0-100)',
      'quality.physical': 'Physical Consistency', 'quality.physicalDesc': '30% · Constraint satisfaction',
      'quality.temporal': 'Temporal Smoothness', 'quality.temporalDesc': '25% · Adjacent frame jump detection',
      'quality.completeness': 'Completeness', 'quality.completenessDesc': '25% · Missing/zero field ratio',
      'quality.coverage': 'Coverage', 'quality.coverageDesc': '20% · Meaningful annotation ratio',
      'quality.warnings': 'Quality Warnings',
      'stats.title': 'Statistics (describe)', 'stats.desc': 'pandas-like describe() stats, pre-computed by Python engine.',
      'stats.noData': 'No stats available',
      'batch.optContact': 'Contact (0/1)', 'batch.optSlip': 'Slip event (0/1)',
      'batch.optForce': 'Force', 'batch.optDeformation': 'Deformation', 'batch.optPhase': 'Phase',
      'episode.outSuccess': '✅ Success', 'episode.outFailure': '❌ Failure',
      'episode.outAborted': '⏹️ Aborted', 'episode.outPartial': '⚠️ Partial',
      'episode.mGrasp': '🤏 Grasp', 'episode.mPush': '👆 Push',
      'episode.mPull': '👇 Pull', 'episode.mTap': '👆 Tap',
      'episode.mLift': '⬆️ Lift', 'episode.mPlace': '⬇️ Place',
      'episode.mRotate': '🔄 Rotate', 'episode.mInsert': '🔩 Insert',
      'episode.dEasy': '🟢 Easy', 'episode.dMedium': '🟡 Medium',
      'episode.dHard': '🔴 Hard',
      'episode.operatorPh': 'Optional', 'episode.notesPh': 'Operation details, failure reasons...',
      'detail.contact': 'Contact', 'detail.slip': 'Slip', 'detail.force': 'Force',
      'detail.deformation': 'Deform', 'detail.area': 'Area', 'detail.entropy': 'Entropy',
      'detail.normal': 'Normal', 'detail.shear': 'Shear',
      'quality.noWarnings': 'No quality warnings',
      'episode.saved': 'Saved', 'episode.noLabels': 'No episode labels yet',
      'predict.badge.title': 'AI Prediction Results',
      'predict.lowConf': 'Low Confidence',
      'predict.method': 'Method',
      'predict.smooth': 'Temporal Smooth',
      'predict.hmm': 'HMM Decode',
      'predict.cascade': 'Cascade Fix',
      // Stats i18n keys for describe table
      'stats.count': 'count', 'stats.mean': 'mean', 'stats.std': 'std',
      'stats.min': 'min', 'stats.max': 'max',
      'stats.p25': '25%', 'stats.p50': '50%', 'stats.p75': '75%',
      // Radar chart i18n labels (dimension names — Schema V2 14维)
      'dim.contact': 'Contact', 'dim.contact_centroid': 'Centroid', 'dim.contact_region': 'Region',
      'dim.force_magnitude': 'Force', 'dim.force_vector': 'Force Vec',
      'dim.torque_vector': 'Torque', 'dim.slip_event': 'Slip',
      'dim.slip_velocity': 'Slip Vel', 'dim.manipulation_phase': 'Phase',
      'dim.texture_class': 'Texture', 'dim.object_deformation': 'Deform',
      'dim.temperature': 'Temp', 'dim.confidence': 'Conf',
      'dim.compliance_level': 'Compliance',
      // UI messages
      'export.hdf5Msg': 'HDF5 export requires Python API:\\n\\ndata.export("output.hdf5")',
      'export.selectArea': '⚠️ Select at least one functional area',
      'export.preparing': '⏳ Preparing export...',
      'export.previewReady': '✅ Export preview ready',
      'episode.savedMsg': 'Episode labels saved (will be exported with data)',
      'shortcuts.help': 'Shortcuts:\\n← → Navigate\\nSpace Toggle contact\\nS Toggle slip\\nD Dark mode\\n? Help',
    }},
    'ja': {{
      'app.title': 'TLabel 触覚アノテーションツール',
      'stats.frames': 'フレーム数', 'stats.duration': '時間', 'stats.contact': '接触率',
      'stats.slip': '滑り率', 'stats.modified': '修正済み',
      'timeline.title': 'タイムライン',
      'chart.radar': '14次元スキーマ',
      'detail.title': 'フレーム詳細',
      'batch.title': '範囲バッチ修正',
      'batch.frameRange': 'フレーム範囲：',
      'batch.apply': '適用',
      'export.title': 'データ出力', 'export.hint': '形式を選択し、対応するボタンをクリック',
      'actions.exportFull': '💾 JSON出力', 'actions.exportCSV': '📊 CSV出力', 'actions.exportHDF5': '🔬 HDF5出力',
      'export.success': '出力成功',
      'tab.annotate': '📝 注釈', 'tab.episode': '🎬 エピソード', 'tab.quality': '📊 品質スコア', 'tab.stats': '📈 統計', 'tab.export': '🚀 出力',
      'export.ftp1_title': 'FTP-1 / MTTS形式の出力', 'export.ftp1_desc': '触覚基盤モデルFTP-1互換のMTTS Zarr形式で出力',
      'export.sensor_name': 'センサー名', 'export.side': '取付位置', 'export.group': 'グループ名',
      'export.area_title': '🎯 機能エリアマッピング', 'export.area_desc': 'このセンサーに対応するMTTS機能エリアのスロットを選択',
      'export.btn_export': '📦 FTP-1 Zarrとして出力', 'export.result_title': '📋 出力結果',
      'export.format_ref': '📖 MTTS Zarr形式リファレンス', 'export.format_note': '💡 出力した.zarrファイルはFTP-1で直接読み込めます',
      'export.preset_gripper': 'プリセット: グリッパー [0,1]', 'export.preset_three': 'プリセット: 3本指 [0,1,2]', 'export.preset_five': 'プリセット: 5本指 [0-4]',
      'episode.title': 'エピソード注釈', 'episode.desc': '操作タスク全体の結果と属性に意味的なラベルを追加します。結果はepisode_infoに保存され、データと一緒に出力されます。',
      'episode.outcome': '操作結果', 'episode.outcomeDefault': '— 未注釈 —',
      'episode.manipType': '操作タイプ', 'episode.manipTypeDefault': '— 未注釈 —',
      'episode.difficulty': '難易度', 'episode.difficultyDefault': '— 未注釈 —',
      'episode.operator': '注釈者', 'episode.notes': '備考',
      'episode.apply': '✅ エピソード注釈を保存',
      'episode.currentInfo': '現在のエピソード情報',
      'quality.title': 'データ品質スコア',
      'quality.desc': '4つの次元でデータ品質を自動評価します（国家標準に準拠）。スコアはPythonエンジンで事前計算されます。',
      'quality.overallLabel': '総合スコア (0-100)',
      'quality.physical': '物理的整合性', 'quality.physicalDesc': '重み30% · 連動ルール充足度',
      'quality.temporal': '時間的平滑性', 'quality.temporalDesc': '重み25% · 隣接フレームの急変検出',
      'quality.completeness': '完全性', 'quality.completenessDesc': '重み25% · フィールド欠落・全ゼロ比率',
      'quality.coverage': 'カバレッジ', 'quality.coverageDesc': '重み20% · 有意な注釈の割合',
      'quality.warnings': '品質警告',
      'stats.title': '統計サマリー (describe)', 'stats.desc': 'pandasのDataFrame.describe()と同様の統計をPythonエンジンで事前計算。',
      'stats.noData': '統計データはありません',
      'batch.optContact': '接触 (0/1)', 'batch.optSlip': '滑りイベント (0/1)',
      'batch.optForce': '力', 'batch.optDeformation': '変形', 'batch.optPhase': '操作フェーズ',
      'episode.outSuccess': '✅ 成功', 'episode.outFailure': '❌ 失敗',
      'episode.outAborted': '⏹️ 中止', 'episode.outPartial': '⚠️ 部分成功',
      'episode.mGrasp': '🤏 掴む', 'episode.mPush': '👆 押す',
      'episode.mPull': '👇 引く', 'episode.mTap': '👆 軽く触れる',
      'episode.mLift': '⬆️ 持ち上げる', 'episode.mPlace': '⬇️ 置く',
      'episode.mRotate': '🔄 回転', 'episode.mInsert': '🔩 挿入',
      'episode.dEasy': '🟢 簡単', 'episode.dMedium': '🟡 普通',
      'episode.dHard': '🔴 難しい',
      'episode.operatorPh': '任意', 'episode.notesPh': '操作の詳細や失敗原因などを記録...',
      'detail.contact': '接触', 'detail.slip': '滑り', 'detail.force': '力',
      'detail.deformation': '変形', 'detail.area': '面積', 'detail.entropy': 'エントロピー',
      'detail.normal': '法線', 'detail.shear': 'せん断',
      'quality.noWarnings': '品質警告なし',
      'episode.saved': '保存済み', 'episode.noLabels': 'エピソード注釈はまだありません',
      'predict.badge.title': 'AI予測結果',
      'predict.lowConf': '低信頼度',
      'predict.method': '手法',
      'predict.smooth': '時系列平滑化',
      'predict.hmm': 'HMMデコード',
      'predict.cascade': '連動修正',
      // Stats i18n keys for describe table
      'stats.count': 'カウント', 'stats.mean': '平均', 'stats.std': '標準偏差',
      'stats.min': '最小', 'stats.max': '最大',
      'stats.p25': '25%分位', 'stats.p50': '中央値', 'stats.p75': '75%分位',
      // Radar chart i18n labels (dimension names — Schema V2 14次元)
      'dim.contact': '接触', 'dim.contact_centroid': '重心', 'dim.contact_region': '領域',
      'dim.force_magnitude': '力', 'dim.force_vector': '力ベクトル',
      'dim.torque_vector': 'トルク', 'dim.slip_event': '滑り',
      'dim.slip_velocity': '滑り速度', 'dim.manipulation_phase': 'フェーズ',
      'dim.texture_class': 'テクスチャ', 'dim.object_deformation': '変形',
      'dim.temperature': '温度', 'dim.confidence': '信頼度',
      'dim.compliance_level': 'コンプライアンス',
      // UI messages
      'export.hdf5Msg': 'HDF5出力にはPython APIを使用してください:\\n\\ndata.export("output.hdf5")',
      'export.selectArea': '⚠️ 機能エリアを1つ以上選択してください',
      'export.preparing': '⏳ 出力を準備中...',
      'export.previewReady': '✅ 出力プレビューを生成しました',
      'episode.savedMsg': 'エピソード注釈を保存しました（データと一緒に出力されます）',
      'shortcuts.help': 'ショートカット:\\n← → フレーム移動\\nスペース 接触を切替\\nS 滑りを切替\\nD ダークモード切替\\n? ヘルプ',
    }},
    'ko': {{
      'app.title': 'TLabel 촉각 어노테이션 도구',
      'stats.frames': '프레임 수', 'stats.duration': '시간', 'stats.contact': '접촉률',
      'stats.slip': '미끄럼률', 'stats.modified': '수정됨',
      'timeline.title': '타임라인',
      'chart.radar': '14차원 스키마',
      'detail.title': '프레임 상세',
      'batch.title': '범위 일괄 수정',
      'batch.frameRange': '프레임 범위:',
      'batch.apply': '적용',
      'export.title': '데이터 내보내기', 'export.hint': '형식을 선택하고 해당 버튼을 클릭하세요',
      'actions.exportFull': '💾 JSON 내보내기', 'actions.exportCSV': '📊 CSV 내보내기', 'actions.exportHDF5': '🔬 HDF5 내보내기',
      'export.success': '내보내기 성공',
      'tab.annotate': '📝 주석', 'tab.episode': '🎬 에피소드', 'tab.quality': '📊 품질 점수', 'tab.stats': '📈 통계', 'tab.export': '🚀 내보내기',
      'export.ftp1_title': 'FTP-1 / MTTS 형식 내보내기', 'export.ftp1_desc': '촉각 기반 모델 FTP-1 호환 MTTS Zarr 형식으로 내보내기',
      'export.sensor_name': '센서 이름', 'export.side': '장착 위치', 'export.group': '그룹 이름',
      'export.area_title': '🎯 기능 영역 매핑', 'export.area_desc': '이 센서에 해당하는 MTTS 기능 영역 슬롯 선택',
      'export.btn_export': '📦 FTP-1 Zarr로 내보내기', 'export.result_title': '📋 내보내기 결과',
      'export.format_ref': '📖 MTTS Zarr 형식 참조', 'export.format_note': '💡 내보낸 .zarr 파일은 FTP-1에서 직접 로드할 수 있습니다',
      'export.preset_gripper': '프리셋: 그리퍼 [0,1]', 'export.preset_three': '프리셋: 3손가락 [0,1,2]', 'export.preset_five': '프리셋: 5손가락 [0-4]',
      'episode.title': '에피소드 주석', 'episode.desc': '상호작용 에피소드 전체에 의미적 라벨을 추가합니다. 결과는 episode_info에 저장되어 데이터와 함께 내보내집니다.',
      'episode.outcome': '작업 결과', 'episode.outcomeDefault': '— 미라벨링 —',
      'episode.manipType': '조작 유형', 'episode.manipTypeDefault': '— 미라벨링 —',
      'episode.difficulty': '난이도', 'episode.difficultyDefault': '— 미라벨링 —',
      'episode.operator': '라벨러', 'episode.notes': '메모',
      'episode.apply': '✅ 에피소드 라벨 저장',
      'episode.currentInfo': '현재 에피소드 정보',
      'quality.title': '데이터 품질 점수',
      'quality.desc': '4개 차원에서 데이터 품질을 자동 평가합니다 (국가 표준 기반). 점수는 Python 엔진에서 미리 계산됩니다.',
      'quality.overallLabel': '종합 점수 (0-100)',
      'quality.physical': '물리적 일관성', 'quality.physicalDesc': '가중치 30% · 연동 규칙 충족도',
      'quality.temporal': '시간적 평활도', 'quality.temporalDesc': '가중치 25% · 인접 프레임 급변 감지',
      'quality.completeness': '완전성', 'quality.completenessDesc': '가중치 25% · 필드 누락/전체 0 비율',
      'quality.coverage': '커버리지', 'quality.coverageDesc': '가중치 20% · 유의미한 라벨 비율',
      'quality.warnings': '품질 경고',
      'stats.title': '통계 요약 (describe)', 'stats.desc': 'pandas DataFrame.describe()와 유사한 통계로, Python 엔진에서 미리 계산됩니다.',
      'stats.noData': '통계 데이터 없음',
      'batch.optContact': '접촉 (0/1)', 'batch.optSlip': '미끄럼 이벤트 (0/1)',
      'batch.optForce': '힘', 'batch.optDeformation': '변형', 'batch.optPhase': '조작 단계',
      'episode.outSuccess': '✅ 성공', 'episode.outFailure': '❌ 실패',
      'episode.outAborted': '⏹️ 중단', 'episode.outPartial': '⚠️ 부분 성공',
      'episode.mGrasp': '🤏 잡기', 'episode.mPush': '👆 밀기',
      'episode.mPull': '👇 당기기', 'episode.mTap': '👆 가볍게 두드리기',
      'episode.mLift': '⬆️ 들어올리기', 'episode.mPlace': '⬇️ 놓기',
      'episode.mRotate': '🔄 회전', 'episode.mInsert': '🔩 삽입',
      'episode.dEasy': '🟢 쉬움', 'episode.dMedium': '🟡 보통',
      'episode.dHard': '🔴 어려움',
      'episode.operatorPh': '선택사항', 'episode.notesPh': '작업 세부사항, 실패 원인 등을 기록...',
      'detail.contact': '접촉', 'detail.slip': '미끄럼', 'detail.force': '힘',
      'detail.deformation': '변형', 'detail.area': '면적', 'detail.entropy': '엔트로피',
      'detail.normal': '법선', 'detail.shear': '전단',
      'quality.noWarnings': '품질 경고 없음',
      'episode.saved': '저장됨', 'episode.noLabels': '에피소드 라벨이 아직 없습니다',
      'predict.badge.title': 'AI 예측 결과',
      'predict.lowConf': '낮은 신뢰도',
      'predict.method': '방법',
      'predict.smooth': '시계열 평활화',
      'predict.hmm': 'HMM 디코딩',
      'predict.cascade': '연동 수정',
      // Stats i18n keys for describe table
      'stats.count': '개수', 'stats.mean': '평균', 'stats.std': '표준편차',
      'stats.min': '최소', 'stats.max': '최대',
      'stats.p25': '25% 분위', 'stats.p50': '중앙값', 'stats.p75': '75% 분위',
      // Radar chart i18n labels (dimension names — Schema V2 14차원)
      'dim.contact': '접촉', 'dim.contact_centroid': '중심', 'dim.contact_region': '영역',
      'dim.force_magnitude': '힘', 'dim.force_vector': '힘 벡터',
      'dim.torque_vector': '토크', 'dim.slip_event': '미끄럼',
      'dim.slip_velocity': '미끄럼 속도', 'dim.manipulation_phase': '단계',
      'dim.texture_class': '질감', 'dim.object_deformation': '변형',
      'dim.temperature': '온도', 'dim.confidence': '신뢰도',
      'dim.compliance_level': '컴플라이언스',
      // UI messages
      'export.hdf5Msg': 'HDF5 내보내기는 Python API를 사용해야 합니다:\\n\\ndata.export("output.hdf5")',
      'export.selectArea': '⚠️ 기능 영역을 하나 이상 선택하세요',
      'export.preparing': '⏳ 내보내기 준비 중...',
      'export.previewReady': '✅ 내보내기 미리보기가 생성되었습니다',
      'episode.savedMsg': '에피소드 라벨이 저장되었습니다 (데이터와 함께 내보내집니다)',
      'shortcuts.help': '단축키:\\n← → 프레임 이동\\n스페이스 접촉 전환\\nS 미끄럼 전환\\nD 다크 모드 전환\\n? 도움말',
    }}
  }};

  function t(key) {{ return (I18N[currentLang] || I18N['zh-CN'])[key] || key; }}

  const LANG_ORDER = ['zh-CN', 'en', 'ja', 'ko'];

  function langBtnLabel(lang) {{
    if (lang === 'zh-CN') return '中文';
    if (lang === 'en') return 'EN';
    if (lang === 'ja') return '日本語';
    return '한국어';
  }}

  function applyI18n() {{
    const lang = I18N[currentLang] ? currentLang : 'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach(el => {{
      const key = el.getAttribute('data-i18n');
      if (I18N[lang] && I18N[lang][key]) el.textContent = I18N[lang][key];
    }});
    // Also update placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {{
      const key = el.getAttribute('data-i18n-placeholder');
      if (I18N[lang] && I18N[lang][key]) el.placeholder = I18N[lang][key];
    }});
    const nextIdx = (LANG_ORDER.indexOf(currentLang) + 1) % LANG_ORDER.length;
    document.getElementById(tid + '-lang-btn').textContent = langBtnLabel(LANG_ORDER[nextIdx]);
  }}

  // ===== Tab Switching =====
  const tabs = document.querySelectorAll('.{tid}-tab');
  const panels = {{
    'annotate': document.getElementById(tid + '-panel-annotate'),
    'episode': document.getElementById(tid + '-panel-episode'),
    'quality': document.getElementById(tid + '-panel-quality'),
    'stats': document.getElementById(tid + '-panel-stats'),
    'export': document.getElementById(tid + '-panel-export'),
  }};

  function switchTab(tabName) {{
    tabs.forEach(btn => {{
      const isActive = btn.getAttribute('data-tab') === tabName;
      btn.style.color = isActive ? '#e85d75' : '#868e96';
      btn.style.fontWeight = isActive ? '700' : '400';
      btn.style.borderBottom = isActive ? '2px solid #e85d75' : '2px solid transparent';
    }});
    Object.keys(panels).forEach(k => {{
      if (panels[k]) panels[k].style.display = k === tabName ? 'block' : 'none';
    }});
  }}

  tabs.forEach(btn => {{
    btn.addEventListener('click', () => switchTab(btn.getAttribute('data-tab')));
  }});

  // ===== Stats =====
  function updateStats() {{
    const frames = data.frames || [];
    document.getElementById(tid + '-stat-frames').textContent = frames.length;
    document.getElementById(tid + '-stat-duration').textContent = data.duration_s ? data.duration_s.toFixed(1) + 's' : '0s';
    const contactCount = frames.filter(f => (f.schema_v2 || {{}}).contact > 0.5).length;
    const slipCount = frames.filter(f => (f.schema_v2 || {{}}).slip_event > 0.5).length;
    document.getElementById(tid + '-stat-contact').textContent = frames.length ? (contactCount / frames.length * 100).toFixed(1) + '%' : '0%';
    document.getElementById(tid + '-stat-slip').textContent = frames.length ? (slipCount / frames.length * 100).toFixed(1) + '%' : '0%';
    document.getElementById(tid + '-stat-modified').textContent = modifiedCount;
    document.getElementById(tid + '-sensor-info').textContent = data.sensor_info ? data.sensor_info.type : '';
  }}

  // ===== Show Frame =====
  function showFrame(idx) {{
    const frames = data.frames || [];
    if (idx < 0 || idx >= frames.length) return;
    currentFrameIdx = idx;
    const f = frames[idx];
    const tv2 = f.schema_v2 || {{}};

    // Frame label
    document.getElementById(tid + '-frame-label').textContent = `Frame ${{f.frame_idx}} / ${{frames.length - 1}}`;

    // Frame input
    document.getElementById(tid + '-frame-input').value = f.frame_idx;

    // Detail content
    const detail = document.getElementById(tid + '-detail-content');
    const phaseColors = {{ 'idle': '#adb5bd', 'initial_contact': '#4dabf7', 'stable_contact': '#51cf66', 'slip': '#ff6b6b', 'grasp': '#ffd43b', 'hold': '#845ef7' }};
    const phaseColor = phaseColors[f.manipulation_phase] || '#adb5bd';
    // v0.13: Find current primitive
    let primLabel = '';
    const primAnns = data.primitive_annotations || [];
    for (const pa of primAnns) {{
      if (f.frame_idx >= pa.start_frame && f.frame_idx <= pa.end_frame) {{
        primLabel = pa.primitive_name;
        break;
      }}
    }}
    let primSource = '', primConfVal = 0;
    for (const pa of primAnns) {{
      if (f.frame_idx >= pa.start_frame && f.frame_idx <= pa.end_frame) {{
        primSource = pa.source || '';
        primConfVal = pa.confidence || 0;
        break;
      }}
    }}
    const sourceIcon = primSource === 'manual' ? '✍️' : (primSource === 'ai_predicted_estimated' ? '🤖⚡' : (primSource === 'ai_predicted' ? '🤖' : ''));
    const confDisplay = primConfVal > 0 ? ` conf:${{(primConfVal * 100).toFixed(0)}}%` : '';
    const primBadge = primLabel
      ? `<span style="background:${{primitiveColors[primLabel] || '#adb5bd'}};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin-left:4px;">🦖 ${{primLabel}}</span><span style="font-size:10px;color:#868e96;margin-left:4px;">${{sourceIcon}}${{confDisplay}}</span>`
      : '';
    detail.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
        <span style="background:${{phaseColor}};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">${{f.manipulation_phase}}</span>
        ${{primBadge}}
        <span style="font-size:11px;color:#868e96;">conf: ${{(f.confidence * 100).toFixed(0)}}%</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;">
        <span>🟢 ${{t('detail.contact')}}: <b>${{typeof tv2.contact === 'boolean' ? (tv2.contact ? '✅' : '—') : (tv2.contact != null ? Number(tv2.contact).toFixed(2) : '-')}}</b></span>
        <span>🔴 ${{t('detail.slip')}}: <b>${{typeof tv2.slip_event === 'boolean' ? (tv2.slip_event ? '✅' : '—') : (tv2.slip_event != null ? Number(tv2.slip_event).toFixed(2) : '-')}}</b></span>
        <span>💪 ${{t('dim.force_magnitude')}}: <b>${{tv2.force_magnitude != null ? Number(tv2.force_magnitude).toFixed(3) : '-'}}</b></span>
        <span>📐 ${{t('dim.object_deformation')}}: <b>${{tv2.object_deformation != null ? Number(tv2.object_deformation).toFixed(3) : '-'}}</b></span>
        <span>🌡️ ${{t('dim.temperature')}}: <b>${{tv2.temperature != null ? Number(tv2.temperature).toFixed(1) : '-'}}</b></span>
        <span>✅ ${{t('dim.confidence')}}: <b>${{tv2.confidence != null ? Number(tv2.confidence).toFixed(2) : '-'}}</b></span>
      </div>
    `;

    // Radar
    drawRadar(tv2);

    // Tactile Image (v0.12)
    const imgEl = document.getElementById(tid + '-tactile-img');
    const noImgEl = document.getElementById(tid + '-no-image');
    if (tactileImages && tactileImages[idx]) {{
      imgEl.src = tactileImages[idx];
      imgEl.style.display = 'block';
      noImgEl.style.display = 'none';
    }} else {{
      imgEl.style.display = 'none';
      noImgEl.style.display = 'block';
    }}

    // Timeline marker
    drawTimeline();
    // v0.13: Primitive track
    drawPrimitiveTrack();
  }}

  // ===== Radar Chart =====
  // i18n keys for radar labels (mapped to dim.* keys in I18N — Schema V2 14维)
  const radarLabelKeys = ['dim.contact','dim.contact_centroid','dim.contact_region',
    'dim.force_magnitude','dim.force_vector','dim.torque_vector',
    'dim.slip_event','dim.slip_velocity','dim.manipulation_phase',
    'dim.texture_class','dim.object_deformation','dim.temperature',
    'dim.confidence','dim.compliance_level'];

  function drawRadar(tv2) {{
    const canvas = document.getElementById(tid + '-radar');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2, R = 130;
    const n = radarLabelKeys.length;
    // v0.17: Schema V2 (14维) only — 直接从 schema_v2 提取
    const sv2 = tv2.schema_v2 || tv2;
    const values = radarLabelKeys.map((lblKey, i) => {{
      const v2Key = lblKey.replace('dim.', '');
      const val = sv2[v2Key];
      if (val === null || val === undefined) return 0;
      if (typeof val === 'boolean') return val ? 1 : 0;
      if (Array.isArray(val)) {{
        const mag = Math.sqrt(val.reduce((s, v) => s + v*v, 0));
        return Math.min(mag, 1);
      }}
      if (typeof val === 'string') {{
        if (v2Key === 'compliance_level') {{
          const lvlMap = {{'L1': 0.25, 'L2': 0.5, 'L3': 0.75, 'L4': 1.0}};
          return lvlMap[val] || 0.25;
        }}
        return val ? 0.5 : 0;
      }}
      return Math.min(Math.max(parseFloat(val) || 0, 0), 1);
    }});

    // Grid
    for (let ring = 1; ring <= 4; ring++) {{
      ctx.beginPath();
      const r = R * ring / 4;
      for (let i = 0; i <= n; i++) {{
        const angle = (2 * Math.PI * (i % n) / n) - Math.PI / 2;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = 'rgba(0,0,0,0.06)';
      ctx.stroke();
    }}

    // Spokes + labels
    for (let i = 0; i < n; i++) {{
      const angle = (2 * Math.PI * i / n) - Math.PI / 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * Math.cos(angle), cy + R * Math.sin(angle));
      ctx.strokeStyle = 'rgba(0,0,0,0.04)';
      ctx.stroke();
      const labelR = R + 18;
      const lx = cx + labelR * Math.cos(angle);
      const ly = cy + labelR * Math.sin(angle);
      const deg = ((angle * 180 / Math.PI) + 360) % 360;
      ctx.textBaseline = 'middle';
      if (deg > 5 && deg < 175) ctx.textAlign = 'left';
      else if (deg > 185 && deg < 355) ctx.textAlign = 'right';
      else ctx.textAlign = 'center';
      ctx.fillText(t(radarLabelKeys[i]), lx, ly);
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

  // ===== Timeline =====
  function drawTimeline() {{
    const canvas = document.getElementById(tid + '-timeline');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const frames = data.frames || [];
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!frames.length) return;
    const barW = W / frames.length;

    for (let i = 0; i < frames.length; i++) {{
      const tv2 = frames[i].schema_v2 || {{}};
      const contact = tv2.contact || 0;
      const slip = tv2.slip_event || 0;
      if (slip > 0.5) ctx.fillStyle = '#ff6b6b';
      else if (contact > 0.5) ctx.fillStyle = '#51cf66';
      else ctx.fillStyle = '#dee2e6';
      ctx.fillRect(i * barW, 0, Math.max(barW, 1), H);
    }}

    // Current frame marker
    ctx.fillStyle = '#e85d75';
    ctx.fillRect(currentFrameIdx * barW, 0, Math.max(barW, 2), H);
  }}

  // ===== v0.13: Primitive Track =====
  const primitiveColors = {{
    'wrap': '#FF6B6B', 'lift': '#4ECDC4', 'grasp': '#45B7D1',
    'fold': '#FFA07A', 'cut': '#98D8C8', 'insert': '#F7DC6F',
    'press': '#BB8FCE', 'wipe': '#85C1E2', 'peel': '#F8B739',
    'assemble': '#82E0AA', 'extract': '#F1948A', 'twist': '#D7BDE2',
    'shake': '#AED6F1', 'dispense': '#A3E4D7', 'disassemble': '#F9E79F',
    'squeeze': '#F5B7B1', 'pour': '#D5F5E3', 'open': '#FADBD8',
    'close': '#D4E6F1', 'screw': '#FCF3CF', 'unscrew': '#E8DAEF',
    'reach': '#D5D8DC'
  }};

  function drawPrimitiveTrack() {{
    const canvas = document.getElementById(tid + '-primitive-track');
    const legend = document.getElementById(tid + '-primitive-legend');
    if (!canvas) return;
    const primAnns = data.primitive_annotations || [];
    if (!primAnns.length) {{
      canvas.style.display = 'none';
      if (legend) legend.style.display = 'none';
      return;
    }}
    canvas.style.display = 'block';
    if (legend) {{
      legend.style.display = 'block';
      const usedPrims = [...new Set(primAnns.map(p => p.primitive_name))];
      legend.innerHTML = '<b>Primitives:</b> ' + usedPrims.map(p =>
        '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:' +
        (primitiveColors[p] || '#adb5bd') + ';margin:0 2px;"></span>' + p
      ).join(' ');
    }}
    const ctx = canvas.getContext('2d');
    const frames = data.frames || [];
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!frames.length) return;
    const totalFrames = frames.length;
    const frameMinIdx = frames[0].frame_idx;
    const frameMaxIdx = frames[frames.length - 1].frame_idx;
    const range = Math.max(frameMaxIdx - frameMinIdx, 1);

    for (const ann of primAnns) {{
      const x1 = ((ann.start_frame - frameMinIdx) / range) * W;
      const x2 = ((ann.end_frame - frameMinIdx) / range) * W;
      ctx.fillStyle = primitiveColors[ann.primitive_name] || '#adb5bd';
      ctx.fillRect(x1, 2, Math.max(x2 - x1, 2), H - 4);
      // Label
      const textW = ctx.measureText(ann.primitive_name).width;
      if ((x2 - x1) > textW + 8) {{
        ctx.fillStyle = '#fff';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.fillText(ann.primitive_name, x1 + 4, H / 2 + 3);
      }}
    }}
    // Current frame marker
    ctx.fillStyle = 'rgba(232, 93, 117, 0.6)';
    const curX = ((frames[currentFrameIdx].frame_idx - frameMinIdx) / range) * W;
    ctx.fillRect(curX, 0, 2, H);
  }}

  // v0.14: Primitive prediction panel toggle
  document.getElementById(tid + '-pp-toggle').addEventListener('click', function() {{
    var content = document.getElementById(tid + '-pp-content');
    var arrow = document.getElementById(tid + '-pp-arrow');
    if (content.style.display === 'none') {{
      content.style.display = 'block';
      arrow.textContent = '▲';
    }} else {{
      content.style.display = 'none';
      arrow.textContent = '▼';
    }}
  }});

  // v0.14: Batch field change - show/hide primitive selector
  document.getElementById(tid + '-batch-field').addEventListener('change', function() {{
    var primSelect = document.getElementById(tid + '-batch-prim-value');
    var textInput = document.getElementById(tid + '-batch-value');
    if (this.value === 'primitive_label') {{
      primSelect.style.display = 'inline-block';
      textInput.style.display = 'none';
    }} else {{
      primSelect.style.display = 'none';
      textInput.style.display = 'inline-block';
    }}
  }});

  // Timeline click
  document.getElementById(tid + '-timeline').addEventListener('click', function(e) {{
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    const frames = data.frames || [];
    const idx = Math.floor(ratio * frames.length);
    if (idx >= 0 && idx < frames.length) showFrame(idx);
  }});

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
        }} else if (field === 'primitive_label') {{
          var primVal = document.getElementById(tid + '-batch-prim-value').value;
          frames[i].primitive_label = primVal;
          frames[i].primitive_confidence = 1.0;
          // 更新 primitive_annotations
          var primAnns = data.primitive_annotations || [];
          var found = false;
          for (var pi = 0; pi < primAnns.length; pi++) {{
            if (primAnns[pi].start_frame <= fi && fi <= primAnns[pi].end_frame) {{
              undoBatch.push({{idx: i, field: 'primitive_annotation', old: JSON.parse(JSON.stringify(primAnns[pi]))}});
              primAnns[pi].primitive_name = primVal;
              primAnns[pi].source = 'manual';
              primAnns[pi].confidence = 1.0;
              found = true;
              break;
            }}
          }}
          if (!found) {{
            var newAnn = {{primitive_name: primVal, start_frame: fi, end_frame: fi, confidence: 1.0, source: 'manual'}};
            data.primitive_annotations.push(newAnn);
            undoBatch.push({{idx: i, field: 'primitive_annotation_new', old: newAnn}});
          }}
          count++;
        }} else {{
          const old = (frames[i].schema_v2 || {{}})[field];
          if (old !== val) {{
            undoBatch.push({{idx: i, field: field, old: old}});
            if (!frames[i].schema_v2) frames[i].schema_v2 = {{}};
            frames[i].schema_v2[field] = val;
            if (field === 'contact' && (val === 0 || val === false)) {{
              const curSlip = frames[i].schema_v2.slip_event;
              if (curSlip === true || curSlip > 0) {{
                undoBatch.push({{idx: i, field: 'slip_event', old: curSlip}});
                frames[i].schema_v2.slip_event = false;
              }}
              if (frames[i].schema_v2.force_magnitude > 0) {{
                undoBatch.push({{idx: i, field: 'force_magnitude', old: frames[i].schema_v2.force_magnitude}});
                frames[i].schema_v2.force_magnitude = 0;
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
    const undoneFrames = new Set();
    for (const op of batch) {{
      if (op.field === 'manipulation_phase') {{
        frames[op.idx].manipulation_phase = op.old;
      }} else {{
        if (frames[op.idx].schema_v2) frames[op.idx].schema_v2[op.field] = op.old;
      }}
      undoneFrames.add(op.idx);
    }}
    modifiedCount = Math.max(0, modifiedCount - undoneFrames.size);
    updateStats();
    showFrame(currentFrameIdx);
  }}

  // ===== Navigation =====
  function prevFrame() {{ if (currentFrameIdx > 0) showFrame(currentFrameIdx - 1); }}
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

  // v0.14: Run primitive prediction (client-side simplified)
  function runPrimitivePrediction() {{
    var taxonomyVal = document.getElementById(tid + '-pp-taxonomy').value;
    var minConf = parseFloat(document.getElementById(tid + '-pp-minconf').value) || 0.4;
    var frames = data.frames || [];
    if (!frames.length) return;
    
    // Simple client-side heuristic prediction
    var predPrimAnns = [];
    var currentPrim = null;
    
    for (var i = 0; i < frames.length; i++) {{
      var tv2 = frames[i].schema_v2 || {{}};
      // v0.17: Schema V2 字段优先，回退旧字段名
      var sv2 = frames[i].schema_v2 || null;
      var force = sv2 ? (sv2.force_magnitude || 0) : (tv2.force_magnitude || 0);
      var contact = sv2 ? (sv2.contact || 0) : (tv2.contact || 0);
      var deform = sv2 ? (sv2.object_deformation || 0) : 0;
      var shear = sv2 && sv2.force_vector ? Math.sqrt(sv2.force_vector[0]**2 + sv2.force_vector[1]**2) : 0;  // V2: shear from force_vector horizontal components
      
      // Compute force delta
      var prevForce = i > 0 ? ((frames[i-1].schema_v2 || {{}}).force_magnitude || 0) : 0;
      var fd = force - prevForce;
      // Short window average
      var windowSize = Math.min(5, i + 1);
      var avgFd = 0;
      for (var w = Math.max(0, i - windowSize + 1); w <= i; w++) {{
        var pf = w > 0 ? ((frames[w-1].schema_v2 || {{}}).force_magnitude || 0) : 0;
        var cf = (frames[w].schema_v2 || {{}}).force_magnitude || 0;
        avgFd += (cf - pf);
      }}
      avgFd /= windowSize;
      
      var primitive = null;
      var conf = 0;
      
      if (contact < 0.3 && force < 0.1) {{
        primitive = 'reach'; conf = 0.65;
      }} else if (contact > 0.5) {{
        if (avgFd > 0.05) {{
          if (deform > 0.2) {{ primitive = 'grasp'; conf = 0.6; }}
          else {{ primitive = 'press'; conf = 0.55; }}
        }} else if (Math.abs(avgFd) < 0.02 && force > 0.2) {{
          if (shear > 0.1) {{ primitive = 'wipe'; conf = 0.5; }}
          else {{ primitive = 'wrap'; conf = 0.5; }}
        }} else if (avgFd < -0.05) {{
          primitive = 'squeeze'; conf = 0.45;
        }} else {{
          primitive = 'grasp'; conf = 0.35;
        }}
      }}
      
      // Filter by taxonomy
      var defaultPrims = ['reach','grasp','press','squeeze','wrap','wipe','lift'];
      var allowedPrims = taxonomyVal === 'full' ? 
        ['wrap','lift','grasp','fold','cut','insert','press','wipe','peel','assemble','extract','twist','shake','dispense','disassemble','squeeze','pour','open','close','screw','unscrew','reach'] :
        defaultPrims;
      if (primitive && allowedPrims.indexOf(primitive) === -1) {{
        primitive = null; conf = 0;
      }}
      
      if (conf < minConf) {{ primitive = null; conf = 0; }}
      
      // Merge consecutive same primitives
      if (primitive === null) {{
        if (currentPrim) {{ predPrimAnns.push(currentPrim); currentPrim = null; }}
      }} else if (currentPrim && currentPrim.primitive_name === primitive) {{
        currentPrim.end_frame = frames[i].frame_idx;
        currentPrim.confidence = Math.min(currentPrim.confidence, conf);
      }} else {{
        if (currentPrim) predPrimAnns.push(currentPrim);
        currentPrim = {{
          primitive_name: primitive,
          start_frame: frames[i].frame_idx,
          end_frame: frames[i].frame_idx,
          confidence: conf,
          source: 'ai_predicted'
        }};
      }}
    }}
    if (currentPrim) predPrimAnns.push(currentPrim);
    
    // Filter short segments
    predPrimAnns = predPrimAnns.filter(function(a) {{ return a.end_frame - a.start_frame >= 2; }});
    
    data.primitive_annotations = predPrimAnns;
    
    // Show results
    var resultDiv = document.getElementById(tid + '-pp-result');
    if (predPrimAnns.length) {{
      var counts = {{}};
      predPrimAnns.forEach(function(a) {{ counts[a.primitive_name] = (counts[a.primitive_name] || 0) + 1; }});
      var parts = Object.keys(counts).map(function(k) {{ return k + '×' + counts[k]; }});
      var lowCount = predPrimAnns.filter(function(a) {{ return a.confidence < 0.5; }}).length;
      resultDiv.innerHTML = '<b>✅ 预标注完成：</b>' + parts.join(', ') +
        (lowCount > 0 ? '<br><span style="color:#e85d75;">⚠ ' + lowCount + '个低置信度区间，请人工审核</span>' : '');
    }} else {{
      resultDiv.innerHTML = '<span style="color:#868e96;">未检测到有效的 primitive 区间</span>';
    }}
    resultDiv.style.display = 'block';
    
    // Refresh display
    drawPrimitiveTrack();
    showFrame(currentFrameIdx);
  }}

  document.getElementById(tid + '-pp-run').addEventListener('click', runPrimitivePrediction);

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
    // v0.17: Schema V2 14维字段名（向量展开为多列）
    // v0.17: Schema V2 only — 14维展开
    const dims = ['contact','centroid_x','centroid_y','contact_region',
      'force_magnitude','force_x','force_y','force_z',
      'torque_x','torque_y','torque_z',
      'slip_event','slip_vx','slip_vy',
      'manipulation_phase','texture_class',
      'object_deformation','temperature',
      'confidence','compliance_level'];
    let csv = 'frame_idx,timestamp_s,manipulation_phase,confidence,primitive_label,primitive_source,primitive_confidence,' + dims.join(',') + '\\n';
    const primAnns = data.primitive_annotations || [];
    for (const f of frames) {{
      // Schema V2: flatten structured fields
      const sv = f.schema_v2 || {{}};
      const tl = {{}};
      tl.contact = sv.contact || 0;
        tl.centroid_x = (sv.contact_centroid && sv.contact_centroid[0]) || 0;
        tl.centroid_y = (sv.contact_centroid && sv.contact_centroid[1]) || 0;
        tl.contact_region = sv.contact_region || '';
        tl.force_magnitude = sv.force_magnitude || 0;
        tl.force_x = (sv.force_vector && sv.force_vector[0]) || 0;
        tl.force_y = (sv.force_vector && sv.force_vector[1]) || 0;
        tl.force_z = (sv.force_vector && sv.force_vector[2]) || 0;
        tl.torque_x = (sv.torque_vector && sv.torque_vector[0]) || 0;
        tl.torque_y = (sv.torque_vector && sv.torque_vector[1]) || 0;
        tl.torque_z = (sv.torque_vector && sv.torque_vector[2]) || 0;
        tl.slip_event = sv.slip_event || 0;
        tl.slip_vx = (sv.slip_velocity && sv.slip_velocity[0]) || 0;
        tl.slip_vy = (sv.slip_velocity && sv.slip_velocity[1]) || 0;
        tl.manipulation_phase = sv.manipulation_phase || '';
        tl.texture_class = sv.texture_class || '';
        tl.object_deformation = sv.object_deformation || 0;
        tl.temperature = sv.temperature || 0;
        tl.confidence = sv.confidence || 1;
      tl.compliance_level = sv.compliance_level || '';
      // find primitive for this frame
      let prim = '';
      for (const pa of primAnns) {{
        if (f.frame_idx >= pa.start_frame && f.frame_idx <= pa.end_frame) {{ prim = pa.primitive_name; break; }}
      }}
      var primSource = '', primConf = '';
      for (var pa2 of primAnns) {{
        if (f.frame_idx >= pa2.start_frame && f.frame_idx <= pa2.end_frame) {{
          primSource = pa2.source || '';
          primConf = pa2.confidence !== undefined ? pa2.confidence : '';
          break;
        }}
      }}
      csv += f.frame_idx + ',' + f.timestamp_s + ',' + f.manipulation_phase + ',' + f.confidence + ',' + prim + ',' + primSource + ',' + primConf;
      for (const d of dims) csv += ',' + (tl[d] || 0);
      csv += '\\n';
    }}
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tlabel_export.csv';
    a.click();
    showExportStatus(t('export.success') + ' CSV');
  }}

  function exportHDF5() {{
    alert(t('export.hdf5Msg'));
  }}

  function showExportStatus(msg) {{
    const el = document.getElementById(tid + '-export-status');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(() => {{ el.style.display = 'none'; }}, 3000);
  }}

  // ===== FTP-1/MTTS Export Panel =====
  const FTP1_AREAS = {{
    0: 'Thumb Tip', 1: 'Index Tip', 2: 'Middle Tip', 3: 'Ring Tip', 4: 'Pinky Tip',
    5: 'Thumb Pad', 6: 'Index Pad', 7: 'Middle Pad', 8: 'Ring Pad', 9: 'Pinky Pad',
    10: 'Thenar', 11: 'Hypothenar', 12: 'Palm Center', 13: 'Proximal Phalanx', 14: 'Dorsum',
    15: 'Wrist FX', 16: 'Wrist FY', 17: 'Wrist FZ', 18: 'Wrist TX', 19: 'Wrist TY', 20: 'Wrist TZ'
  }};
  let selectedAreas = [0, 1]; // Default: gripper

  function initFTP1Areas() {{
    const container = document.getElementById(tid + '-ftp1-areas');
    if (!container) return;
    container.innerHTML = '';
    for (let id = 0; id <= 20; id++) {{
      const checked = selectedAreas.includes(id) ? 'checked' : '';
      const active = selectedAreas.includes(id) ? 'background:#e85d75;color:#fff;border-color:#e85d75;' : '';
      const label = document.createElement('label');
      label.style.cssText = 'display:inline-flex;align-items:center;gap:3px;padding:3px 8px;border-radius:4px;border:1px solid #ced4da;font-size:11px;cursor:pointer;user-select:none;transition:all 0.15s;' + active;
      label.innerHTML = '<input type="checkbox" value="' + id + '" ' + checked + ' style="display:none;">' + id + ':' + FTP1_AREAS[id];
      const cb = label.querySelector('input');
      cb.addEventListener('change', function() {{
        if (this.checked) {{
          if (!selectedAreas.includes(id)) selectedAreas.push(id);
          label.style.background = '#e85d75'; label.style.color = '#fff'; label.style.borderColor = '#e85d75';
        }} else {{
          selectedAreas = selectedAreas.filter(a => a !== id);
          label.style.background = ''; label.style.color = ''; label.style.borderColor = '#ced4da';
        }}
        selectedAreas.sort((a,b) => a-b);
      }});
      container.appendChild(label);
    }}
  }}

  function setAreaPreset(areas, btnId) {{
    selectedAreas = [...areas];
    initFTP1Areas();
    // Update preset button styles
    ['gripper','three','five'].forEach(p => {{
      const b = document.getElementById(tid + '-ftp1-preset-' + p);
      if (b) {{ b.style.borderColor = '#ced4da'; b.style.color = '#868e96'; b.style.background = 'transparent'; }}
    }});
    const activeBtn = document.getElementById(tid + '-ftp1-preset-' + btnId);
    if (activeBtn) {{ activeBtn.style.borderColor = '#e85d75'; activeBtn.style.color = '#e85d75'; activeBtn.style.background = 'transparent'; }}
  }}

  function handleFTP1Export() {{
    const sensor = document.getElementById(tid + '-ftp1-sensor').value;
    const side = document.getElementById(tid + '-ftp1-side').value;
    const group = document.getElementById(tid + '-ftp1-group').value;
    const status = document.getElementById(tid + '-ftp1-status');
    const resultDiv = document.getElementById(tid + '-ftp1-result');
    const resultContent = document.getElementById(tid + '-ftp1-result-content');

    if (selectedAreas.length === 0) {{
      status.textContent = t('export.selectArea');
      return;
    }}

    status.textContent = t('export.preparing');

    // Generate export summary (actual Zarr writing requires Python backend)
    const nFrames = (data.frames || []).length;
    const areaNames = selectedAreas.map(a => FTP1_AREAS[a] || ('Area_' + a));
    const summary = {{
      sensor_name: sensor,
      side: side,
      group: group,
      functional_areas: selectedAreas,
      functional_area_names: areaNames,
      time_steps: nFrames,
      num_slots: selectedAreas.length,
      data_shape: '[' + nFrames + ', ' + selectedAreas.length + ', 224, 224, 3]',
      zarr_keys: [
        side + '_tactile_data_' + group,
        side + '_tactile_area_' + group,
        side + '_tactile_sensor_' + group,
        side + '_tactile_type_' + group,
      ],
      python_command: 'data.export_ftp1("output.zarr",\\n  sensor_name="' + sensor + '",\\n  functional_areas=' + JSON.stringify(selectedAreas) + ',\\n  side="' + side + '",\\n  group="' + group + '")'
    }};

    status.textContent = t('export.previewReady');
    resultDiv.style.display = 'block';
    resultContent.textContent = JSON.stringify(summary, null, 2);
  }}

  // ===== Lang Toggle =====
  function toggleLang() {{
    const idx = LANG_ORDER.indexOf(currentLang);
    currentLang = LANG_ORDER[(idx + 1) % LANG_ORDER.length];
    applyI18n();
    showFrame(currentFrameIdx);
    // Re-render quality and episode info
    renderQuality();
    renderEpisodeInfo();
    renderDescribe();
  }}

  // ===== Episode Labeling =====
  function initEpisodeForm() {{
    // Populate from existing episode_info
    if (episodeInfo.outcome) document.getElementById(tid + '-ep-outcome').value = episodeInfo.outcome;
    if (episodeInfo.manipulation_type) document.getElementById(tid + '-ep-manip-type').value = episodeInfo.manipulation_type;
    if (episodeInfo.difficulty) document.getElementById(tid + '-ep-difficulty').value = episodeInfo.difficulty;
    if (episodeInfo.operator) document.getElementById(tid + '-ep-operator').value = episodeInfo.operator;
    if (episodeInfo.notes) document.getElementById(tid + '-ep-notes').value = episodeInfo.notes;
    renderEpisodeInfo();
  }}

  function applyEpisodeLabel() {{
    const outcome = document.getElementById(tid + '-ep-outcome').value;
    const manipType = document.getElementById(tid + '-ep-manip-type').value;
    const difficulty = document.getElementById(tid + '-ep-difficulty').value;
    const operator = document.getElementById(tid + '-ep-operator').value;
    const notes = document.getElementById(tid + '-ep-notes').value;

    // Update data.episode_info
    if (!data.episode_info) data.episode_info = {{}};
    if (outcome) data.episode_info.outcome = outcome;
    if (manipType) data.episode_info.manipulation_type = manipType;
    if (difficulty) data.episode_info.difficulty = difficulty;
    if (operator) data.episode_info.operator = operator;
    if (notes) data.episode_info.notes = notes;

    // Also update local episodeInfo for display
    Object.assign(episodeInfo, data.episode_info);

    // Disable button to prevent double-click
    const btn = document.getElementById(tid + '-btn-episode-apply');
    btn.disabled = true;
    btn.style.opacity = '0.6';
    btn.style.cursor = 'not-allowed';

    // Show status with detailed message
    const statusEl = document.getElementById(tid + '-episode-status');
    const savedMsg = t('episode.savedMsg');
    statusEl.textContent = '✅ ' + savedMsg;
    statusEl.style.color = '#51cf66';
    statusEl.style.display = 'inline';

    // Re-enable button and hide status after 4 seconds
    setTimeout(() => {{
      statusEl.style.display = 'none';
      btn.disabled = false;
      btn.style.opacity = '1';
      btn.style.cursor = 'pointer';
    }}, 4000);

    renderEpisodeInfo();
  }}

  function renderEpisodeInfo() {{
    const display = document.getElementById(tid + '-episode-info-display');
    if (!display) return;
    const info = data.episode_info || episodeInfo;
    const keys = Object.keys(info);
    if (keys.length === 0) {{
      display.innerHTML = `<span style="color:#adb5bd;">${{t('episode.noLabels')}}</span>`;
      return;
    }}
    const outcomeEmoji = {{ 'success': '✅', 'failure': '❌', 'aborted': '⏹️', 'partial': '⚠️' }};
    const diffEmoji = {{ 'easy': '🟢', 'medium': '🟡', 'hard': '🔴' }};
    let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;">';
    for (const [k, v] of Object.entries(info)) {{
      let displayVal = v;
      if (k === 'outcome') displayVal = (outcomeEmoji[v] || '') + ' ' + v;
      if (k === 'difficulty') displayVal = (diffEmoji[v] || '') + ' ' + v;
      if (k === 'manipulation_type') displayVal = '🤖 ' + v;
      if (k === 'notes') {{
        html += `</div><div style="margin-top:8px;font-size:12px;color:#495057;"><b>${{k}}</b>: ${{v}}</div>`;
        continue;
      }}
      html += `<div><span style="color:#868e96;font-size:11px;">${{k}}</span><br><b style="font-size:13px;">${{displayVal}}</b></div>`;
    }}
    if (!html.includes('</div></div>')) html += '</div>';
    display.innerHTML = html;
  }}

  // ===== Quality Score =====
  function renderQuality() {{
    const q = qualityData;
    if (!q || !q.overall && q.overall !== 0) {{
      document.getElementById(tid + '-quality-overall').textContent = '--';
      document.getElementById(tid + '-quality-grade-badge').textContent = '-';
      return;
    }}

    // Colors for light/dark mode
    const textColor = isDark ? '#c0caf5' : '#343a40';
    const borderColor = isDark ? '#3b4261' : '#f1f3f5';

    // Overall
    const overallEl = document.getElementById(tid + '-quality-overall');
    overallEl.textContent = q.overall.toFixed(1);
    overallEl.style.color = textColor;

    // Grade badge
    const grade = q.grade || '-';
    const gradeColors = {{ 'A': '#51cf66', 'B': '#4dabf7', 'C': '#ffd43b', 'D': '#ff922b', 'F': '#ff6b6b' }};
    const badge = document.getElementById(tid + '-quality-grade-badge');
    badge.textContent = grade;
    badge.style.background = gradeColors[grade] || (isDark ? '#3b4261' : '#e9ecef');
    badge.style.color = (grade === 'C' || grade === 'D') ? textColor : '#fff';

    // Dimension bars
    const dims = [
      {{ key: 'physical_consistency', el: 'physical' }},
      {{ key: 'temporal_smoothness', el: 'temporal' }},
      {{ key: 'completeness', el: 'completeness' }},
      {{ key: 'coverage', el: 'coverage' }},
    ];
    for (const d of dims) {{
      const val = q[d.key] || 0;
      const valEl = document.getElementById(tid + '-q-' + d.el);
      if (valEl) valEl.textContent = val.toFixed(1);
      const barEl = document.getElementById(tid + '-q-' + d.el + '-bar');
      if (barEl) barEl.style.width = val + '%';
    }}

    // Warnings
    const warnEl = document.getElementById(tid + '-quality-warnings');
    if (q.warnings && q.warnings.length > 0) {{
      warnEl.innerHTML = q.warnings.map(w => `<div style="padding:4px 0;border-bottom:1px solid ${{borderColor}};color:${{textColor}};">⚠️ ${{w}}</div>`).join('');
    }} else {{
      warnEl.innerHTML = `<span style="color:#51cf66;">✅ ${{t('quality.noWarnings')}}</span>`;
    }}
  }}

  // ===== Describe Stats =====
  function renderDescribe() {{
    const container = document.getElementById(tid + '-describe-table');
    if (!container) return;
    const dd = describeData;
    if (!dd || Object.keys(dd).length === 0) {{
      container.innerHTML = `<span style="color:#adb5bd;">${{t('stats.noData')}}</span>`;
      return;
    }}

    // Colors for light/dark mode
    const labelColor = isDark ? '#565f89' : '#868e96';
    const borderColor = isDark ? '#3b4261' : '#dee2e6';
    const headerColor = isDark ? '#c0caf5' : '#495057';
    const textColor = isDark ? '#c0caf5' : '#343a40';
    const rowBg = isDark ? '#1f2335' : '#f1f3f5';

    // Build table like pandas describe() with i18n
    // i18n keys and their corresponding data keys
    const statKeyMap = [
      {{ i18n: 'stats.count', data: 'count' }},
      {{ i18n: 'stats.mean', data: 'mean' }},
      {{ i18n: 'stats.std', data: 'std' }},
      {{ i18n: 'stats.min', data: 'min' }},
      {{ i18n: 'stats.p25', data: '25%' }},
      {{ i18n: 'stats.p50', data: '50%' }},
      {{ i18n: 'stats.p75', data: '75%' }},
      {{ i18n: 'stats.max', data: 'max' }},
    ];
    const fields = Object.keys(dd);
    
    let html = `<table style="width:100%;border-collapse:collapse;font-size:12px;color:${{textColor}};">`;
    html += `<thead><tr><th style="padding:6px 10px;text-align:left;border-bottom:2px solid ${{borderColor}};color:${{labelColor}};"></th>`;
    for (const f of fields) {{
      const fieldName = t('dim.' + f) || f;
      html += `<th style="padding:6px 10px;text-align:right;border-bottom:2px solid ${{borderColor}};color:${{headerColor}};">${{fieldName}}</th>`;
    }}
    html += '</tr></thead><tbody>';
    
    for (const skm of statKeyMap) {{
      html += `<tr><td style="padding:5px 10px;border-bottom:1px solid ${{borderColor}};color:${{labelColor}};font-weight:600;">${{t(skm.i18n)}}</td>`;
      for (const f of fields) {{
        const val = dd[f][skm.data];
        const display = (val !== undefined && val !== null) ? (typeof val === 'number' ? val.toFixed(4) : val) : '-';
        html += `<td style="padding:5px 10px;text-align:right;border-bottom:1px solid ${{rowBg}};color:${{textColor}};">${{display}}</td>`;
      }}
      html += '</tr>';
    }}
    html += '</tbody></table>';
    container.innerHTML = html;
  }}

  // ===== Event Listeners =====
  document.getElementById(tid + '-lang-btn').addEventListener('click', toggleLang);
  document.getElementById(tid + '-btn-prev').addEventListener('click', prevFrame);
  document.getElementById(tid + '-btn-next').addEventListener('click', nextFrame);
  document.getElementById(tid + '-frame-input').addEventListener('change', function() {{ jumpTo(this.value); }});
  document.getElementById(tid + '-btn-batch').addEventListener('click', batchPatch);
  document.getElementById(tid + '-btn-undo').addEventListener('click', undo);
  document.getElementById(tid + '-btn-export-json').addEventListener('click', exportJSON);
  document.getElementById(tid + '-btn-export-csv').addEventListener('click', exportCSV);
  document.getElementById(tid + '-btn-export-hdf5').addEventListener('click', exportHDF5);
  document.getElementById(tid + '-btn-episode-apply').addEventListener('click', applyEpisodeLabel);

  // FTP-1 Export Panel events
  const ftp1ExportBtn = document.getElementById(tid + '-ftp1-export-btn');
  if (ftp1ExportBtn) ftp1ExportBtn.addEventListener('click', handleFTP1Export);
  const ftp1PresetGripper = document.getElementById(tid + '-ftp1-preset-gripper');
  if (ftp1PresetGripper) ftp1PresetGripper.addEventListener('click', () => setAreaPreset([0, 1], 'gripper'));
  const ftp1PresetThree = document.getElementById(tid + '-ftp1-preset-three');
  if (ftp1PresetThree) ftp1PresetThree.addEventListener('click', () => setAreaPreset([0, 1, 2], 'three'));
  const ftp1PresetFive = document.getElementById(tid + '-ftp1-preset-five');
  if (ftp1PresetFive) ftp1PresetFive.addEventListener('click', () => setAreaPreset([0, 1, 2, 3, 4], 'five'));

  // ===== Dark Mode =====
  let isDark = false;
  const rootEl = document.getElementById(tid + '-root');

  function toggleDark() {{
    isDark = !isDark;
    const btn = document.getElementById(tid + '-dark-btn');
    if (isDark) {{
      rootEl.style.background = '#1a1b26';
      rootEl.style.color = '#c0caf5';
      btn.textContent = '☀️';
      rootEl.querySelectorAll('div[style*="background: #fff"], div[style*="background:#fff"]').forEach(el => {{
        el.style.background = '#24283b'; el.style.borderColor = '#3b4261'; el.style.color = '#c0caf5';
      }});
      rootEl.querySelectorAll('div[style*="background: #f1f3f5"], div[style*="background:#f1f3f5"]').forEach(el => {{
        el.style.background = '#1f2335';
      }});
      rootEl.querySelectorAll('div[style*="background: #e9ecef"], div[style*="background:#e9ecef"]').forEach(el => {{
        el.style.background = '#1f2335';
      }});
      rootEl.querySelectorAll('div[style*="background: #f8f9fa"], div[style*="background:#f8f9fa"]').forEach(el => {{
        el.style.background = '#1f2335';
      }});
      rootEl.querySelectorAll('span[style*="color: #868e96"]').forEach(el => {{
        el.style.color = '#565f89';
      }});
      rootEl.querySelectorAll('span[style*="color: #343a40"]').forEach(el => {{
        el.style.color = '#c0caf5';
      }});
      rootEl.querySelectorAll('input, select, textarea').forEach(el => {{
        el.style.background = '#24283b'; el.style.color = '#c0caf5'; el.style.borderColor = '#3b4261';
      }});
      rootEl.querySelectorAll('canvas').forEach(el => {{
        el.style.background = '#24283b';
      }});
      rootEl.querySelectorAll('button').forEach(el => {{
        if (!el.style.background.includes('gradient') && !el.style.background.includes('#e85d75')) {{
          el.style.background = '#24283b'; el.style.color = '#c0caf5'; el.style.borderColor = '#3b4261';
        }}
      }});
      rootEl.querySelectorAll('table').forEach(el => {{
        el.style.color = '#c0caf5';
      }});
      rootEl.querySelectorAll('th, td').forEach(el => {{
        el.style.borderColor = '#3b4261'; el.style.color = '#c0caf5';
      }});
      // Re-render quality and stats tabs for dark mode colors
      renderQuality();
      renderDescribe();
    }} else {{
      btn.textContent = '🌙';
      rootEl.style.background = '#f8f9fa';
      rootEl.style.color = '#343a40';
      updateStats();
      showFrame(currentFrameIdx);
      // Re-render quality and stats tabs for light mode colors
      renderQuality();
      renderDescribe();
    }}
  }}

  document.getElementById(tid + '-dark-btn').addEventListener('click', toggleDark);

  // ===== Keyboard Shortcuts =====
  document.addEventListener('keydown', function(e) {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    switch(e.key) {{
      case 'ArrowLeft': e.preventDefault(); prevFrame(); break;
      case 'ArrowRight': e.preventDefault(); nextFrame(); break;
      case ' ':
        e.preventDefault();
        const frame = data.frames[currentFrameIdx];
        if (frame) {{
          const wasContact = typeof frame.schema_v2.contact === 'boolean' ? frame.schema_v2.contact : (frame.schema_v2.contact > 0.5);
          frame.schema_v2.contact = !wasContact;
          if (wasContact) {{
            frame.schema_v2.slip_event = false;
            frame.schema_v2.force_magnitude = 0.0;
            frame.manipulation_phase = 'idle';
          }}
          modifiedCount++; updateStats(); showFrame(currentFrameIdx);
        }}
        break;
      case 's': case 'S':
        const f2 = data.frames[currentFrameIdx];
        const f2Contact = typeof f2.schema_v2.contact === 'boolean' ? f2.schema_v2.contact : (f2.schema_v2.contact > 0.5);
        if (f2 && f2Contact) {{
          const wasSlip = typeof f2.schema_v2.slip_event === 'boolean' ? f2.schema_v2.slip_event : (f2.schema_v2.slip_event > 0.5);
          f2.schema_v2.slip_event = !wasSlip;
          if (!wasSlip) f2.manipulation_phase = 'slip';
          modifiedCount++; updateStats(); showFrame(currentFrameIdx);
        }}
        break;
      case 'd': case 'D': toggleDark(); break;
      case '?':
        alert(t('shortcuts.help'));
        break;
    }}
  }});

  // ===== Init =====
  updateStats();
  applyI18n();
  showFrame(0);
  initEpisodeForm();
  renderQuality();
  renderDescribe();
  initFTP1Areas();

  window['_tlabel_' + tid] = {{
    prevFrame, nextFrame, jumpTo, batchPatch, undo,
    exportJSON, exportCSV, exportHDF5, toggleLang,
    applyEpisodeLabel, switchTab, handleFTP1Export, setAreaPreset
  }};
}})();
</script>"""
