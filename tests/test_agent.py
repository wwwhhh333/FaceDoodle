"""Test agent module — module-level functions and class methods via instance."""

import pytest

from app.ai.agent import (
    _estimate_scale, FaceDoodleAgent,
    REGION_GROUPS, REGION_ALIASES, KEYWORD_REGION_MAP,
)


@pytest.fixture
def agent():
    """Agent with no API key — always uses keyword fallback."""
    return FaceDoodleAgent(api_key=None)


# ── _estimate_scale (module-level) ──

def test_estimate_scale_large():
    assert _estimate_scale("head_top", "猫耳") == 1.4
    assert _estimate_scale("head_top", "帽子") == 1.4
    assert _estimate_scale("head_top", "皇冠") == 1.4


def test_estimate_scale_small():
    assert _estimate_scale("nose", "鼻子") == 0.5
    assert _estimate_scale("nose", "雀斑") == 0.5
    assert _estimate_scale("cheek_left", "爱心") == 0.5


def test_estimate_scale_default():
    assert _estimate_scale("eyes", "眼镜") == 1.0
    assert _estimate_scale("mouth", "口罩") == 1.0


# ── _parse_json_text ──

def test_parse_valid_json(agent):
    assert agent._parse_json_text('{"a": 1}') == {"a": 1}


def test_parse_json_in_markdown(agent):
    result = agent._parse_json_text('```json\n{"prompt": "cat", "region": "eyes"}\n```')
    assert result["prompt"] == "cat"
    assert result["region"] == "eyes"


def test_parse_json_with_surrounding_text(agent):
    result = agent._parse_json_text('hello {"key": "value"} world')
    assert result["key"] == "value"


def test_parse_empty_raises(agent):
    with pytest.raises(ValueError, match="返回为空"):
        agent._parse_json_text("")


def test_parse_garbage_raises(agent):
    with pytest.raises(ValueError, match="无法从模型响应中提取 JSON"):
        agent._parse_json_text("no json here at all")


# ── _normalize_region ──

def test_normalize_valid_group_passthrough(agent):
    assert agent._normalize_region("eyes") == "eyes"
    assert agent._normalize_region("nose") == "nose"
    assert agent._normalize_region("mouth") == "mouth"


def test_normalize_alias_mapping(agent):
    assert agent._normalize_region("额头") == "forehead_top"
    assert agent._normalize_region("眼睛") == "eyes"
    assert agent._normalize_region("鼻子") == "nose"
    assert agent._normalize_region("嘴巴") == "mouth"
    assert agent._normalize_region("下巴") == "chin"
    assert agent._normalize_region("头顶") == "head_top"


def test_normalize_english_alias(agent):
    assert agent._normalize_region("forehead") == "forehead_top"
    assert agent._normalize_region("cheek") == "cheek_left"


def test_normalize_unknown_falls_back_to_eyes(agent):
    assert agent._normalize_region("nonexistent_place") == "eyes"
    assert agent._normalize_region("") == "eyes"


# ── _keyword_fallback ──

def test_keyword_fallback_match(agent):
    region, kw = agent._keyword_fallback("我想要一副眼镜")
    assert region == "eyes"
    assert kw == "眼镜"


def test_keyword_fallback_cat_ear(agent):
    region, kw = agent._keyword_fallback("加个猫耳")
    assert region == "head_top"
    assert kw == "猫耳"


def test_keyword_fallback_no_match(agent):
    region, kw = agent._keyword_fallback("xyzzy12345")
    assert region is None
    assert kw is None


def test_keyword_fallback_first_match_wins(agent):
    region, kw = agent._keyword_fallback("给我一顶帽子")
    assert region == "head_top"


# ── _build_fallback_result ──

def test_build_fallback_result_with_match(agent):
    result = agent._build_fallback_result("猫耳")
    assert result["action"] == "generate"
    assert len(result["tasks"]) == 1
    task = result["tasks"][0]
    assert task["region"] == "head_top"
    assert task["scale"] == 1.4
    assert len(task["prompt"]) > 0
    assert result["workflow"] == "transparent_workflow_api.json"


def test_build_fallback_result_no_match(agent):
    result = agent._build_fallback_result("xyzzy nothing")
    assert result["action"] == "generate"
    assert len(result["tasks"]) == 1
    task = result["tasks"][0]
    assert task["region"] == "eyes"
    assert task["scale"] == 1.0


# ── parse_command (main entry point, no API key → fallback) ──

def test_parse_command_uses_fallback(agent):
    result = agent.parse_command("猫耳")
    assert result["target_location"] == "head_top"
    assert result["scale"] == 1.4
    assert "workflow" in result


# ── Constant integrity ──

def test_region_groups_nonempty():
    assert len(REGION_GROUPS) > 5


def test_region_aliases_cover_common_terms():
    assert "眼睛" in REGION_ALIASES
    assert "鼻子" in REGION_ALIASES
    assert "嘴巴" in REGION_ALIASES


def test_keyword_region_map_keys_are_valid_regions():
    for region in KEYWORD_REGION_MAP:
        assert region in REGION_GROUPS, f"{region} not in REGION_GROUPS"
