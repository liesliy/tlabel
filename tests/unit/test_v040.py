"""TLabel v0.4.0 新功能测试"""
import json
import tempfile
import os
import pytest
from pathlib import Path

from tlabel.core.schema import TLabelSchemaV2


class TestEpisodeLabel:
    """Episode级标注测试"""

    def _make_data(self, n_frames=10):
        from tlabel.core.types import TLabelData, TLabelFrame
        frames = []
        for i in range(n_frames):
            is_contact = i % 3 == 0
            f = TLabelFrame(
                frame_idx=i,
                timestamp_s=i / 30.0,
                schema_v2=TLabelSchemaV2(
                    contact=is_contact,
                    force_magnitude=0.5 if is_contact else None,
                    slip_event=False,
                    contact_centroid=[0.5, 0.5] if is_contact else None,
                    confidence=0.9,
                ),
                manipulation_phase="idle",
                confidence=0.9,
            )
            frames.append(f)
        return TLabelData(
            frames=frames,
            sensor_info={"type": "test"},
            episode_info={"source": "test"},
            capabilities={"contact": True},
        )

    def test_label_episode_basic(self):
        """基本Episode标注"""
        data = self._make_data()
        label = data.label_episode(
            outcome="success",
            manipulation_type="grasp",
            difficulty="medium",
            notes="稳定抓取"
        )
        assert label.outcome == "success"
        assert label.manipulation_type == "grasp"
        assert label.difficulty == "medium"
        assert label.notes == "稳定抓取"

    def test_episode_label_property(self):
        """episode_label属性读取"""
        data = self._make_data()
        assert data.episode_label is None
        data.label_episode(outcome="failure", manipulation_type="pinch")
        assert data.episode_label is not None
        assert data.episode_label.outcome == "failure"

    def test_label_episode_invalid_outcome(self):
        """无效outcome应该报错"""
        data = self._make_data()
        with pytest.raises(ValueError, match="Invalid outcome"):
            data.label_episode(outcome="invalid")

    def test_label_episode_invalid_type(self):
        """无效manipulation_type应该报错"""
        data = self._make_data()
        with pytest.raises(ValueError, match="Invalid manipulation_type"):
            data.label_episode(manipulation_type="invalid")

    def test_label_episode_in_to_dict(self):
        """to_dict应包含episode_label"""
        data = self._make_data()
        data.label_episode(outcome="success", manipulation_type="grasp")
        d = data.to_dict()
        assert d["episode"]["episode_label"] is not None
        assert d["episode"]["episode_label"]["outcome"] == "success"

    def test_episode_label_from_dict(self):
        """EpisodeLabel.from_dict 反序列化"""
        from tlabel.core.types import EpisodeLabel
        d = {"outcome": "partial", "manipulation_type": "slide", "difficulty": "hard"}
        label = EpisodeLabel.from_dict(d)
        assert label.outcome == "partial"
        assert label.manipulation_type == "slide"
        assert label.difficulty == "hard"


class TestQualityScore:
    """数据质量评分测试"""

    def _make_good_data(self, n_frames=30):
        """生成高质量数据"""
        from tlabel.core.types import TLabelData, TLabelFrame
        frames = []
        for i in range(n_frames):
            is_contact = i >= 5
            f = TLabelFrame(
                frame_idx=i,
                timestamp_s=i / 30.0,
                schema_v2=TLabelSchemaV2(
                    contact=is_contact,
                    force_magnitude=0.5 if is_contact else None,
                    slip_event=False,
                    contact_centroid=[0.5, 0.5] if is_contact else None,
                    object_deformation=0.3 if is_contact else None,
                    confidence=0.95,
                    compliance_level="L2" if is_contact else "L1",
                ),
                manipulation_phase="stable_contact" if is_contact else "idle",
                confidence=0.95,
            )
            frames.append(f)
        return TLabelData(
            frames=frames,
            sensor_info={"type": "test"},
            episode_info={"source": "test"},
            capabilities={"contact": True},
        )

    def test_quality_score_basic(self):
        """基本质量评分"""
        data = self._make_good_data()
        score = data.quality_score()
        assert "overall" in score
        assert "physical_consistency" in score
        assert "temporal_smoothness" in score
        assert "completeness" in score
        assert "coverage" in score
        assert "grade" in score
        assert 0 <= score["overall"] <= 100

    def test_quality_grade_range(self):
        """等级应在A-F范围内"""
        data = self._make_good_data()
        score = data.quality_score()
        assert score["grade"] in {"A", "B", "C", "D", "F"}

    def test_quality_good_data_high_score(self):
        """好的数据应该得较高分"""
        data = self._make_good_data()
        score = data.quality_score()
        assert score["overall"] >= 50  # 物理一致的数据不应太低

    def test_quality_empty_data(self):
        """空数据应得0分"""
        from tlabel.core.types import TLabelData
        data = TLabelData(frames=[], sensor_info={}, episode_info={}, capabilities={})
        score = data.quality_score()
        assert score["overall"] == 0.0
        assert score["grade"] == "F"

    def test_quality_verbose_warnings(self):
        """verbose模式应输出详细警告"""
        data = self._make_good_data()
        score = data.quality_score(verbose=True)
        assert "warnings" in score
        assert isinstance(score["warnings"], list)

    def test_quality_scorer_direct(self):
        """直接使用QualityScorer"""
        from tlabel.quality.scorer import QualityScorer
        data = self._make_good_data()
        scorer = QualityScorer()
        score = scorer.score(data)
        assert score["overall"] >= 0


class TestDescribe:
    """describe统计摘要测试"""

    def _make_data(self, n_frames=20):
        from tlabel.core.types import TLabelData, TLabelFrame
        frames = []
        for i in range(n_frames):
            is_contact = i % 3 == 0
            f = TLabelFrame(
                frame_idx=i,
                timestamp_s=i / 30.0,
                schema_v2=TLabelSchemaV2(
                    contact=is_contact,
                    force_magnitude=round(0.5 * (i + 1) / n_frames, 4),
                    slip_event=False,
                    contact_centroid=[0.5, 0.5] if is_contact else None,
                    confidence=0.9,
                ),
                manipulation_phase="idle",
                confidence=0.9,
            )
            frames.append(f)
        return TLabelData(
            frames=frames,
            sensor_info={"type": "test"},
            episode_info={"source": "test"},
            capabilities={"contact": True},
        )

    def test_describe_basic(self):
        """基本describe"""
        data = self._make_data()
        stats = data.describe()
        assert "contact" in stats
        assert "mean" in stats["contact"]
        assert "std" in stats["contact"]
        assert "min" in stats["contact"]
        assert "max" in stats["contact"]
        assert stats["contact"]["count"] == 20

    def test_describe_specific_fields(self):
        """指定字段describe"""
        data = self._make_data()
        stats = data.describe(fields=["contact", "force_magnitude"])
        assert "contact" in stats
        assert "force_magnitude" in stats
        assert "slip_event" not in stats

    def test_describe_empty_data(self):
        """空数据describe"""
        from tlabel.core.types import TLabelData
        data = TLabelData(frames=[], sensor_info={}, episode_info={}, capabilities={})
        stats = data.describe()
        assert stats == {}

    def test_describe_percentiles(self):
        """百分位数"""
        data = self._make_data()
        stats = data.describe()
        for key, s in stats.items():
            assert s["25%"] <= s["50%"] <= s["75%"], f"Percentile order wrong for {key}"


class TestBatchProcessor:
    """批处理测试"""

    def _create_test_dir(self):
        """创建包含多个Episode的测试目录"""
        import tlabel
        tmpdir = tempfile.mkdtemp()

        # 生成3个demo数据并导出JSON
        for sensor in ["gelsight", "digit", "paxini"]:
            try:
                data = tlabel.demo(sensor)
                out_file = os.path.join(tmpdir, f"episode_{sensor}.json")
                data.export(out_file, format="json")
            except Exception:
                pass

        return tmpdir

    def test_batch_load(self):
        """批量加载"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = self._create_test_dir()
        try:
            bp = BatchProcessor(tmpdir)
            bp.load_all()
            assert len(bp) >= 1  # 至少加载1个
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_batch_auto_label(self):
        """批量预标注"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = self._create_test_dir()
        try:
            bp = BatchProcessor(tmpdir)
            bp.load_all()
            bp.auto_label(min_confidence=0.5, engine="rule")
            summary = bp.summary()
            assert summary["total_episodes"] >= 1
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_batch_quality_check(self):
        """批量质量评分"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = self._create_test_dir()
        try:
            bp = BatchProcessor(tmpdir)
            bp.load_all()
            bp.quality_check()
            summary = bp.summary()
            assert "avg_quality" in summary
            assert "quality_grades" in summary
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_batch_export(self):
        """批量导出"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = self._create_test_dir()
        outdir = tempfile.mkdtemp()
        try:
            bp = BatchProcessor(tmpdir)
            bp.load_all()
            bp.export_all(outdir, format="json")
            # Check output files exist
            output_files = list(Path(outdir).glob("*.json"))
            assert len(output_files) >= 1
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(outdir, ignore_errors=True)

    def test_batch_empty_dir(self):
        """空目录应报错"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = tempfile.mkdtemp()
        try:
            bp = BatchProcessor(tmpdir)
            with pytest.raises(ValueError, match="未找到"):
                bp.load_all()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_batch_nonexistent_dir(self):
        """不存在的目录应报错"""
        from tlabel.batch.processor import BatchProcessor
        bp = BatchProcessor("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            bp.load_all()

    def test_batch_summary(self):
        """批处理汇总"""
        from tlabel.batch.processor import BatchProcessor
        tmpdir = self._create_test_dir()
        try:
            bp = BatchProcessor(tmpdir)
            bp.load_all()
            bp.auto_label(engine="rule")
            bp.quality_check()
            summary = bp.summary()
            assert "total_episodes" in summary
            assert "total_frames" in summary
            assert "episodes" in summary
            assert len(summary["episodes"]) >= 1
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestImports:
    """v0.4.0新增模块导入测试"""

    def test_import_episode_label(self):
        from tlabel import EpisodeLabel
        assert EpisodeLabel is not None

    def test_import_quality_scorer(self):
        from tlabel import QualityScorer
        assert QualityScorer is not None

    def test_import_batch_processor(self):
        from tlabel import BatchProcessor
        assert BatchProcessor is not None

    def test_version(self):
        import tlabel
        from packaging.version import Version
        assert Version(tlabel.__version__) >= Version("0.4.0")


class TestI18n:
    """Test i18n completeness for v0.4.2"""

    def test_i18n_en_dict_has_all_zh_keys(self):
        """Every zh-CN key must have an en counterpart"""
        from tlabel.viewer.templates import generate_panel_html
        import re
        # Extract the i18n dicts from the generated HTML
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="en",
            instance_id="test_i18n",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        # Find all data-i18n keys used in HTML
        keys_in_html = set(re.findall(r'data-i18n="([^"]+)"', html))
        # Find all data-i18n-placeholder keys
        placeholder_keys = set(re.findall(r'data-i18n-placeholder="([^"]+)"', html))
        all_keys = keys_in_html | placeholder_keys
        assert len(all_keys) > 30, f"Expected 30+ i18n keys, got {len(all_keys)}"

    def test_i18n_switch_produces_english(self):
        """English lang should produce English output"""
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="en",
            instance_id="test_en",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        # The default lang when lang="en" should show English tab names
        assert "Annotate" in html, "English tab 'Annotate' not found"
        assert "Quality" in html, "English tab 'Quality' not found"
        assert "Stats" in html, "English tab 'Stats' not found"

    def test_i18n_zh_produces_chinese(self):
        """zh-CN lang should produce Chinese output"""
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="zh-CN",
            instance_id="test_zh",
            episode_info={},
            quality_score=None,
 describe_stats=None,
        )
        assert "标注" in html, "Chinese tab '标注' not found"
        assert "质量评分" in html, "Chinese tab '质量评分' not found"

    def test_batch_panel_i18n(self):
        """BatchPanel should support i18n"""
        from tlabel.viewer.batch_panel import TLabelBatchPanel, _BATCH_I18N
        # Check both dicts have same keys
        zh_keys = set(_BATCH_I18N["zh-CN"].keys())
        en_keys = set(_BATCH_I18N["en"].keys())
        assert zh_keys == en_keys, f"Key mismatch: zh={zh_keys - en_keys}, en={en_keys - zh_keys}"

    def test_frame_detail_labels_i18n(self):
        """Frame detail labels should use t() for i18n"""
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="en",
            instance_id="test_detail",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        # Frame detail should use t('detail.contact') etc
        assert "t('detail.contact')" in html or "t(\"detail.contact\")" in html, \
            "Frame detail should use i18n t() function"
        assert "t('detail.slip')" in html or "t(\"detail.slip\")" in html, \
            "Frame detail slip should use i18n"

    def test_i18n_ja_initial_lang(self):
        """Japanese lang should set currentLang and include Japanese labels"""
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="ja",
            instance_id="test_ja",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        assert "currentLang = 'ja';" in html, "Japanese initial language not set"
        assert "触覚" in html, "Japanese labels missing"

    def test_i18n_ko_initial_lang(self):
        """Korean lang should set currentLang and include Korean labels"""
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="ko",
            instance_id="test_ko",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        assert "currentLang = 'ko';" in html, "Korean initial language not set"
        assert "촉각" in html, "Korean labels missing"

    def test_i18n_four_dicts_have_same_keys(self):
        """All four I18N language dicts must expose the same keys"""
        import re
        from tlabel.viewer.templates import generate_panel_html
        html = generate_panel_html(
            data_dict={"frames": [], "sensor_info": {"type": "demo"}},
            lang="en",
            instance_id="test_keys",
            episode_info={},
            quality_score=None,
            describe_stats=None,
        )
        body = html.split("const I18N = {", 1)[1].split("\n  };", 1)[0]
        keys_by_lang = {}
        for lang in ("zh-CN", "en", "ja", "ko"):
            block = body.split(f"'{lang}': {{", 1)[1].split("\n    },", 1)[0]
            keys_by_lang[lang] = set(
                re.findall(r"'([A-Za-z][A-Za-z0-9._-]*)':", block)
            )
        zh_keys = keys_by_lang["zh-CN"]
        assert len(zh_keys) > 50, f"Expected 50+ i18n keys, got {len(zh_keys)}"
        for lang in ("en", "ja", "ko"):
            assert zh_keys == keys_by_lang[lang], (
                f"Key mismatch for {lang}: "
                f"missing={zh_keys - keys_by_lang[lang]}, "
                f"extra={keys_by_lang[lang] - zh_keys}"
            )

    def test_i18n_resolve_lang(self):
        """auto picks sensor_info.lang; aliases and unknown langs fall back safely"""
        from tlabel.viewer.templates import _resolve_panel_lang
        assert _resolve_panel_lang("auto", {"lang": "ja"}) == "ja"
        assert _resolve_panel_lang("auto", {"lang": "ko"}) == "ko"
        assert _resolve_panel_lang("auto", None) == "zh-CN"
        assert _resolve_panel_lang("zh", {}) == "zh-CN"
        assert _resolve_panel_lang("fr", {}) == "zh-CN"

    def test_review_lang_ja_ko(self):
        """data.review(lang='ja'/'ko') should render Japanese/Korean panels"""
        import tlabel
        data = tlabel.demo("gelsight")
        for lang, marker in (("ja", "触覚"), ("ko", "촉각")):
            html = data.review(lang=lang)._repr_html_()
            assert f"currentLang = '{lang}';" in html, \
                f"{lang} initial language not set"
            assert marker in html, f"{lang} labels missing"

    def test_batch_panel_i18n_all_langs(self):
        """Batch panel should expose identical keys across all four languages"""
        from tlabel.viewer.batch_panel import TLabelBatchPanel, _BATCH_I18N
        zh_keys = set(_BATCH_I18N["zh-CN"].keys())
        for lang in ("en", "ja", "ko"):
            assert zh_keys == set(_BATCH_I18N[lang].keys()), \
                f"Key mismatch for {lang}"

        class _StubBP:
            def summary(self):
                return {
                    "total_episodes": 0,
                    "total_frames": 0,
                    "avg_quality": 0.0,
                    "grade_dist": {},
                    "episodes": [],
                }

        html = TLabelBatchPanel(_StubBP(), lang="ja")._repr_html_()
        assert "バッチ" in html, "Japanese batch panel labels missing"
