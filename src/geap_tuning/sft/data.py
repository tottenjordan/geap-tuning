"""Support-ticket intent classification dataset for the SFT example.

A small, self-contained, text-only dataset: each example is a customer support
message labeled with one of five intents. It is deliberately tiny so the example
is cheap to run and fully reproducible (no external downloads). The records use
the shared ``contents``-format builders in :mod:`geap_tuning.schemas`.
"""

from __future__ import annotations

import random
from pathlib import Path

from geap_tuning.schemas import Record, sft_example, write_jsonl

LABELS = ("billing", "technical", "account", "shipping", "other")

PROMPT = (
    "Classify this customer support ticket into exactly one of these intents: "
    "billing, technical, account, shipping, other. "
    "Respond with only the intent label."
)

# (ticket_text, label) pairs — roughly balanced across the five intents.
SUPPORT_TICKETS: list[tuple[str, str]] = [
    ("My card was charged twice for the same order.", "billing"),
    ("I was billed after cancelling my subscription.", "billing"),
    ("Can I get a refund for last month's payment?", "billing"),
    ("Why is there a $5 fee on my invoice?", "billing"),
    ("The discount code did not apply at checkout.", "billing"),
    ("I need an itemized receipt for my purchase.", "billing"),
    ("You charged me in the wrong currency.", "billing"),
    ("My payment failed but the money left my account.", "billing"),
    ("The app crashes every time I open the reports tab.", "technical"),
    ("I'm getting a 500 error when I try to log in.", "technical"),
    ("The export button does nothing when I click it.", "technical"),
    ("Video playback keeps buffering on the dashboard.", "technical"),
    ("The mobile app won't sync with the web version.", "technical"),
    ("Search returns no results even for valid queries.", "technical"),
    ("Notifications stopped working after the update.", "technical"),
    ("The page loads blank on Safari but works on Chrome.", "technical"),
    ("I forgot my password and can't reset it.", "account"),
    ("How do I change the email on my account?", "account"),
    ("I want to delete my account permanently.", "account"),
    ("Two-factor authentication is locking me out.", "account"),
    ("Can I merge two accounts into one?", "account"),
    ("My username was changed without my permission.", "account"),
    ("How do I add a teammate to my workspace?", "account"),
    ("I can't update my profile picture.", "account"),
    ("Where is my package? It was due three days ago.", "shipping"),
    ("The tracking number you sent is invalid.", "shipping"),
    ("My order arrived damaged in the box.", "shipping"),
    ("Can I change the delivery address on my order?", "shipping"),
    ("I received the wrong item in my shipment.", "shipping"),
    ("Do you ship internationally to Canada?", "shipping"),
    ("Part of my order is missing from the package.", "shipping"),
    ("How long does standard delivery usually take?", "shipping"),
    ("Do you have a mobile app for Android?", "other"),
    ("What are your customer support hours?", "other"),
    ("I'd like to give feedback about your service.", "other"),
    ("Is there a student discount available?", "other"),
    ("Can you tell me more about your enterprise plan?", "other"),
    ("I love the new design, just wanted to say thanks!", "other"),
    ("Where can I find your privacy policy?", "other"),
    ("Do you partner with other companies?", "other"),
    # Additional examples for more phrasings and edge cases (still balanced).
    ("I was double-charged after a failed retry.", "billing"),
    ("My annual plan renewed even though I opted out.", "billing"),
    ("The invoice total doesn't match my order summary.", "billing"),
    ("Please explain the proration charge on this bill.", "billing"),
    ("I upgraded but was charged the old and new price.", "billing"),
    ("The dashboard freezes when I filter by date range.", "technical"),
    ("File uploads fail silently over 10 MB.", "technical"),
    ("I keep getting logged out every few minutes.", "technical"),
    ("The API returns a 403 with a valid token.", "technical"),
    ("Dark mode doesn't persist between sessions.", "technical"),
    ("How do I transfer ownership of my workspace?", "account"),
    ("My account shows the wrong subscription tier.", "account"),
    ("I need to update my billing contact email.", "account"),
    ("Can I set up single sign-on for my team?", "account"),
    ("My verification email never arrives.", "account"),
    ("The courier marked it delivered but nothing arrived.", "shipping"),
    ("Can I upgrade to expedited shipping after ordering?", "shipping"),
    ("My shipment has been stuck in transit for a week.", "shipping"),
    ("The label says a different weight than my order.", "shipping"),
    ("Do you offer carbon-neutral delivery options?", "shipping"),
    ("What integrations do you support?", "other"),
    ("Are you hiring for engineering roles?", "other"),
    ("Can I book a demo with your sales team?", "other"),
    ("Is my data stored in the EU?", "other"),
    ("Where do I submit a feature request?", "other"),
]


def build_records(pairs: list[tuple[str, str]]) -> list[Record]:
    """Turn ``(ticket, label)`` pairs into SFT ``contents`` records."""
    return [
        sft_example(user_text=f"{PROMPT}\n\nTicket: {ticket}", model_text=label)
        for ticket, label in pairs
    ]


def split_dataset(
    pairs: list[tuple[str, str]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministically shuffle and split ``pairs`` into (train, val, test)."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic split, not cryptographic
    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_sft_dataset(out_dir: str | Path) -> dict[str, str]:
    """Write train/val/test JSONL under ``out_dir``; return a name→path mapping."""
    out_dir = Path(out_dir)
    train, val, test = split_dataset(SUPPORT_TICKETS)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_records(split), path)
        paths[name] = str(path)
    return paths
