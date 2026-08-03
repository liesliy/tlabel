"""
TLabelBatchPanel — 批处理仪表盘

在Jupyter中显示BatchProcessor的运行结果，支持：
- 批量Episode列表（含质量评分+等级）
- 平均质量评分仪表
- 每个Episode的快速预览
- 中英文切换

用法:
    bp = tlabel.BatchProcessor("episodes_dir/")
    bp.load_all().auto_label().quality_check()
    bp.review()  # 渲染批处理仪表盘
"""

import json
import uuid
from typing import Optional, Dict

from tlabel.batch.processor import BatchProcessor


# i18n dictionaries for batch panel
_BATCH_I18N = {
    "zh-CN": {
        "title": "TLabel 批处理仪表盘",
        "episodes": "Episodes",
        "totalFrames": "总帧数",
        "avgQuality": "平均质量",
        "gradeDist": "等级分布",
        "episodeList": "📁 Episode 列表",
        "colFile": "文件",
        "colFrames": "帧数",
        "colDuration": "时长(s)",
        "colSensor": "传感器",
        "colQuality": "质量",
        "colGrade": "等级",
        "colModified": "已修正",
        "noData": "暂无Episode数据",
    },
    "en": {
        "title": "TLabel Batch Dashboard",
        "episodes": "Episodes",
        "totalFrames": "Total Frames",
        "avgQuality": "Avg Quality",
        "gradeDist": "Grade Dist",
        "episodeList": "📁 Episode List",
        "colFile": "File",
        "colFrames": "Frames",
        "colDuration": "Duration(s)",
        "colSensor": "Sensor",
        "colQuality": "Quality",
        "colGrade": "Grade",
        "colModified": "Modified",
        "noData": "No episode data",
    },
    "ja": {
        "title": "TLabel バッチダッシュボード",
        "episodes": "エピソード数",
        "totalFrames": "総フレーム数",
        "avgQuality": "平均品質",
        "gradeDist": "グレード分布",
        "episodeList": "📁 エピソードリスト",
        "colFile": "ファイル",
        "colFrames": "フレーム数",
        "colDuration": "時間(s)",
        "colSensor": "センサー",
        "colQuality": "品質",
        "colGrade": "グレード",
        "colModified": "修正済み",
        "noData": "エピソードデータはありません",
    },
    "ko": {
        "title": "TLabel 일괄 처리 대시보드",
        "episodes": "에피소드 수",
        "totalFrames": "총 프레임 수",
        "avgQuality": "평균 품질",
        "gradeDist": "등급 분포",
        "episodeList": "📁 에피소드 목록",
        "colFile": "파일",
        "colFrames": "프레임 수",
        "colDuration": "시간(s)",
        "colSensor": "센서",
        "colQuality": "품질",
        "colGrade": "등급",
        "colModified": "수정됨",
        "noData": "에피소드 데이터 없음",
    },
}


class TLabelBatchPanel:
    """批处理仪表盘"""

    def __init__(self, batch_processor: BatchProcessor, lang: str = "auto", **kwargs):
        self.bp = batch_processor
        self.lang = lang
        self.instance_id = f"tlabel_batch_{uuid.uuid4().hex[:6]}"

    def _t(self, key: str) -> str:
        """Get i18n text."""
        lang = self.lang if self.lang in _BATCH_I18N else "en"
        return _BATCH_I18N[lang].get(key, key)

    def _repr_html_(self):
        """Jupyter自动调用渲染"""
        summary = self.bp.summary()
        summary_json = json.dumps(summary, ensure_ascii=False, default=str)
        tid = self.instance_id

        return f"""<div id="{tid}-root" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
     max-width: 960px; margin: 0 auto; background: #f8f9fa; color: #343a40;
     border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">

<!-- Header -->
<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 20px;
     background:linear-gradient(135deg,#e9ecef,#f1f3f5);border-bottom:1px solid #dee2e6;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:20px;">🦞</span>
    <span style="font-size:16px;font-weight:700;color:#e85d75;">{self._t("title")}</span>
    <span style="font-size:10px;color:#868e96;background:#e9ecef;padding:1px 6px;border-radius:4px;">v0.4.2</span>
  </div>
</div>

<!-- Summary Stats -->
<div style="display:flex;gap:16px;padding:16px 20px;background:#fff;border-bottom:1px solid #dee2e6;">
  <div style="text-align:center;flex:1;">
    <div style="font-size:28px;font-weight:900;color:#e85d75;">{summary.get("total_episodes", 0)}</div>
    <div style="font-size:11px;color:#868e96;">{self._t("episodes")}</div>
  </div>
  <div style="text-align:center;flex:1;">
    <div style="font-size:28px;font-weight:900;color:#495057;">{summary.get("total_frames", 0)}</div>
    <div style="font-size:11px;color:#868e96;">{self._t("totalFrames")}</div>
  </div>
  <div style="text-align:center;flex:1;">
    <div style="font-size:28px;font-weight:900;color:#4dabf7;">{summary.get("avg_quality", 0):.1f}</div>
    <div style="font-size:11px;color:#868e96;">{self._t("avgQuality")}</div>
  </div>
  <div style="text-align:center;flex:1;">
    <div style="font-size:12px;font-weight:700;line-height:1.6;">
    {self._grade_bar_html(summary.get("quality_grades", {}))}
    </div>
    <div style="font-size:11px;color:#868e96;">{self._t("gradeDist")}</div>
  </div>
</div>

<!-- Episode Table -->
<div style="padding:16px 20px;background:#fff;">
  <div style="font-size:13px;font-weight:600;color:#343a40;margin-bottom:10px;">{self._t("episodeList")}</div>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="border-bottom:2px solid #dee2e6;">
          <th style="padding:8px 10px;text-align:left;color:#868e96;">{self._t("colFile")}</th>
          <th style="padding:8px 10px;text-align:right;color:#868e96;">{self._t("colFrames")}</th>
          <th style="padding:8px 10px;text-align:right;color:#868e96;">{self._t("colDuration")}</th>
          <th style="padding:8px 10px;text-align:left;color:#868e96;">{self._t("colSensor")}</th>
          <th style="padding:8px 10px;text-align:right;color:#868e96;">{self._t("colQuality")}</th>
          <th style="padding:8px 10px;text-align:center;color:#868e96;">{self._t("colGrade")}</th>
          <th style="padding:8px 10px;text-align:right;color:#868e96;">{self._t("colModified")}</th>
        </tr>
      </thead>
      <tbody>
        {self._episode_rows_html(summary.get("episodes", []))}
      </tbody>
    </table>
  </div>
</div>

</div>"""

    def _grade_bar_html(self, grades: Dict) -> str:
        if not grades:
            return '<span style="color:#adb5bd;">—</span>'
        parts = []
        for g in ['A', 'B', 'C', 'D', 'F']:
            if g in grades:
                colors = {'A': '#51cf66', 'B': '#4dabf7', 'C': '#ffd43b', 'D': '#ff922b', 'F': '#ff6b6b'}
                parts.append(f'<span style="color:{colors.get(g, "#868e96")};">{g}:{grades[g]}</span>')
        return ' '.join(parts)

    def _episode_rows_html(self, episodes: list) -> str:
        if not episodes:
            return f'<tr><td colspan="7" style="padding:20px;text-align:center;color:#adb5bd;">{self._t("noData")}</td></tr>'
        rows = []
        for ep in episodes:
            grade = ep.get('grade', '-')
            quality = ep.get('quality', 0)
            grade_colors = {'A': '#51cf66', 'B': '#4dabf7', 'C': '#ffd43b', 'D': '#ff922b', 'F': '#ff6b6b'}
            gcolor = grade_colors.get(grade, '#adb5bd')
            rows.append(f"""<tr style="border-bottom:1px solid #f1f3f5;">
              <td style="padding:6px 10px;color:#495057;font-weight:500;">{ep.get('file', '-')}</td>
              <td style="padding:6px 10px;text-align:right;">{ep.get('frames', 0)}</td>
              <td style="padding:6px 10px;text-align:right;">{ep.get('duration_s', 0)}</td>
              <td style="padding:6px 10px;">{ep.get('sensor', '-')}</td>
              <td style="padding:6px 10px;text-align:right;font-weight:600;color:#e85d75;">{quality:.1f if quality else '-'}</td>
              <td style="padding:6px 10px;text-align:center;"><span style="background:{gcolor};color:#fff;padding:1px 8px;border-radius:4px;font-weight:700;font-size:11px;">{grade}</span></td>
              <td style="padding:6px 10px;text-align:right;">{ep.get('modified', 0)}</td>
            </tr>""")
        return '\n'.join(rows)

    def __repr__(self):
        return (f"TLabelBatchPanel(episodes={len(self.bp)}, "
                f"dir={self.bp.source_dir})")
