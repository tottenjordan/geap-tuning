"""Tests for the DPO win-rate evaluation (injected generate/judge fns)."""

from geap_tuning.preference.evaluate import run_preference_eval, score_winrate


def test_score_winrate() -> None:
    metrics = score_winrate(["A", "A", "B", "A"])
    assert metrics["win_rate"] == 0.75
    assert metrics["n"] == 4
    assert metrics["wins"] == 3
    assert metrics["losses"] == 1
    assert metrics["ties"] == 0


def test_score_winrate_counts_ties() -> None:
    metrics = score_winrate(["A", "B", "TIE"])
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["ties"] == 1
    assert metrics["wins"] + metrics["losses"] + metrics["ties"] == metrics["n"]


def test_score_winrate_empty() -> None:
    metrics = score_winrate([])
    assert metrics["win_rate"] == 0.0
    assert metrics["n"] == 0
    assert metrics["wins"] == metrics["losses"] == metrics["ties"] == 0


def test_run_preference_eval_uses_injected_fns() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "Where is my order?"}]}],
            "completions": [
                {"score": 1, "completion": {"role": "model", "parts": [{"text": "On it!"}]}},
                {"score": 0, "completion": {"role": "model", "parts": [{"text": "Idk."}]}},
            ],
        },
    ]
    metrics = run_preference_eval(
        records,
        generate_fn=lambda _user: "On it!",
        judge_fn=lambda _user, _a, _b: "A",
    )
    assert metrics["win_rate"] == 1.0
    assert metrics["n"] == 1


def test_run_preference_eval_passes_tuned_reply_and_dispreferred_ref() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "Q"}]}],
            "completions": [
                {"score": 1, "completion": {"role": "model", "parts": [{"text": "good"}]}},
                {"score": 0, "completion": {"role": "model", "parts": [{"text": "bad"}]}},
            ],
        },
    ]
    seen: dict[str, str] = {}

    def judge(user: str, cand_a: str, cand_b: str) -> str:
        seen.update(user=user, a=cand_a, b=cand_b)
        return "B"  # dispreferred wins → tuned loses

    metrics = run_preference_eval(records, generate_fn=lambda _u: "tuned reply", judge_fn=judge)
    assert seen == {"user": "Q", "a": "tuned reply", "b": "bad"}
    assert metrics["win_rate"] == 0.0
