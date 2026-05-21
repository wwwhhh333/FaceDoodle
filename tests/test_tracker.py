"""Tests for app/core/tracker.py — isolated logic and state classes."""

import time
import pytest

from app.core.tracker import GenerationState
from app.core.sticker_registry import StickerRegistry
from app.core.protocol import StickerInstance


# ── GenerationState ──

class TestGenerationState:
    def test_start_sets_generating(self):
        gs = GenerationState()
        gs.start()
        assert gs.is_generating is True

    def test_finish_clears_generating(self):
        gs = GenerationState()
        gs.start()
        gs.finish()
        assert gs.is_generating is False

    def test_initial_state_is_idle(self):
        gs = GenerationState()
        assert gs.is_generating is False

    def test_get_elapsed_returns_zero_when_idle(self):
        gs = GenerationState()
        elapsed = gs.get_elapsed()
        assert elapsed == 0.0

    def test_get_elapsed_increases_while_generating(self):
        gs = GenerationState()
        gs.start()
        time.sleep(0.05)
        elapsed = gs.get_elapsed()
        assert elapsed > 0.0
        gs.finish()

    def test_get_elapsed_returns_zero_after_finish(self):
        gs = GenerationState()
        gs.start()
        gs.finish()
        assert gs.get_elapsed() == 0.0

    def test_multiple_start_noop(self):
        gs = GenerationState()
        gs.start()
        gs.start()  # second start should not reset
        assert gs.is_generating is True
        ts = gs._start_time
        gs.start()  # should be no-op
        assert gs._start_time == ts


# ── _get_face_width ──

class TestGetFaceWidth:
    def test_normal_value(self):
        from app.core.tracker import ConsumerProcessor
        result = ConsumerProcessor._get_face_width({"face_width": 200.0})
        assert result == 200.0

    def test_none_face_data(self):
        from app.core.tracker import ConsumerProcessor
        result = ConsumerProcessor._get_face_width(None)
        assert result == 200.0

    def test_missing_face_width_key(self):
        from app.core.tracker import ConsumerProcessor
        result = ConsumerProcessor._get_face_width({"nose_tip": [0.5, 0.5]})
        assert result == 200.0

    def test_minimum_width_clamped(self):
        from app.core.tracker import ConsumerProcessor
        result = ConsumerProcessor._get_face_width({"face_width": 0.0})
        assert result == 1.0  # clamped to 1.0

    def test_negative_width_clamped(self):
        from app.core.tracker import ConsumerProcessor
        result = ConsumerProcessor._get_face_width({"face_width": -5.0})
        assert result == 1.0  # max(-5, 1) = 1


# ── _face_data_changed ──

class TestFaceDataChanged:
    def test_first_call_returns_true(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp._last_face_fp = None
        assert cp._face_data_changed(100, 100, 50) is True

    def test_same_values_returns_false(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp._last_face_fp = (100.0, 100.0, 50.0)
        assert cp._face_data_changed(101.0, 100.0, 50.0) is False

    def test_large_x_change_returns_true(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp._last_face_fp = (100.0, 100.0, 50.0)
        assert cp._face_data_changed(120.0, 100.0, 50.0) is True

    def test_large_fw_change_returns_true(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp._last_face_fp = (100.0, 100.0, 50.0)
        assert cp._face_data_changed(100.0, 100.0, 60.0) is True

    def test_custom_threshold(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp._last_face_fp = (100.0, 100.0, 50.0)
        assert cp._face_data_changed(102.5, 100.0, 50.0, threshold=3.0) is False
        assert cp._face_data_changed(103.1, 100.0, 50.0, threshold=3.0) is True


# ── _append_conversation ──

class TestAppendConversation:
    def test_appends_user_message(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = []
        cp.max_conversation_turns = 6
        cp._had_stickers = False
        cp._append_conversation("user", "hello")
        assert len(cp.conversation_history) == 1
        assert cp.conversation_history[0]["role"] == "user"

    def test_appends_assistant_message(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = []
        cp.max_conversation_turns = 6
        cp._had_stickers = False
        cp._append_conversation("assistant", "generated")
        assert cp.conversation_history[0]["role"] == "assistant"

    def test_empty_content_skipped(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = []
        cp.max_conversation_turns = 6
        cp._had_stickers = False
        cp._append_conversation("user", "")
        assert len(cp.conversation_history) == 0

    def test_none_content_skipped(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = []
        cp.max_conversation_turns = 6
        cp._had_stickers = False
        cp._append_conversation("user", None)
        assert len(cp.conversation_history) == 0

    def test_truncates_at_max_turns(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = []
        cp.max_conversation_turns = 4
        cp._had_stickers = False
        for i in range(6):
            cp._append_conversation("user", f"msg {i}")
        assert len(cp.conversation_history) == 4
        assert cp.conversation_history[-1]["content"] == "msg 5"
        assert cp.conversation_history[0]["content"] == "msg 2"

    def test_auto_clear_when_stickers_removed(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = [{"role": "user", "content": "hello"}]
        cp.max_conversation_turns = 6
        cp._had_stickers = True   # but were present before
        cp._append_conversation("user", "new message")
        # Auto-clear triggers after append: no active stickers, was had_stickers
        assert len(cp.conversation_history) == 0  # cleared entirely
        assert cp._had_stickers is False

    def test_auto_clear_not_triggered_if_stickers_still_active(self):
        from app.core.tracker import ConsumerProcessor
        cp = ConsumerProcessor.__new__(ConsumerProcessor)
        cp.registry = StickerRegistry(max_stickers=20)
        cp.conversation_history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        cp.max_conversation_turns = 6
        cp.registry.add(StickerInstance(instance_id="1"))
        cp._had_stickers = True
        cp._append_conversation("user", "again")
        # Active stickers exist, so no auto-clear
        assert len(cp.conversation_history) == 3
