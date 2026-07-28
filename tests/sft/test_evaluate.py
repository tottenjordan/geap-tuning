"""Tests for SFT evaluation (pure scoring + mocked predictor)."""

from geap_tuning.sft.evaluate import run_eval, score_classification


def test_score_classification() -> None:
    m = score_classification(y_true=["billing", "technical"], y_pred=["billing", "billing"])
    assert m["accuracy"] == 0.5
    assert "report" in m


def test_run_eval_uses_predict_fn() -> None:
    records = [
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Q1"}]},
                {"role": "model", "parts": [{"text": "billing"}]},
            ],
        },
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Q2"}]},
                {"role": "model", "parts": [{"text": "technical"}]},
            ],
        },
    ]
    m = run_eval(records, predict_fn=lambda _user_text: "billing")
    assert m["accuracy"] == 0.5


def test_run_eval_all_correct() -> None:
    records = [
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Q"}]},
                {"role": "model", "parts": [{"text": "billing"}]},
            ],
        },
    ]
    m = run_eval(records, predict_fn=lambda _user_text: "billing")
    assert m["accuracy"] == 1.0
