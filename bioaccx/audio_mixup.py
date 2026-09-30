"""Audio-level mixup: synthesise overlapping calls for multi-label sigmoid heads.

Datasets built from separated sounds hold one class per window, so a sigmoid
head never sees two calls at once. Audio mixup sums the audio of windows from
different classes, embeds the mixture with the foundation model, and trains on
the union of their labels.

A mix is an ordinary :class:`~bioaccx.dataset.AudioSample` whose
``mix_sources`` lists the windows summed into it; the embedding stage renders
the audio (:func:`bioaccx.dataset.mix_audio`) and caches the embedding under
:func:`bioaccx.dataset.mix_key`. Mixes are drawn from one split only — train
mixes from train windows, test mixes from test windows — so no test audio ever
reaches training.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from bioaccx.config import AudioMixupConfig
from bioaccx.dataset import AudioSample, MixSource, mix_key, sample_labels

# Draws allowed per requested mix before giving up on finding compatible sources.
_MAX_ATTEMPTS = 20


class _Rules:
    """Which label sets may be mixed, and what target a mix gets."""

    def __init__(self, cfg: AudioMixupConfig) -> None:
        self.group_of = {l: g for g, members in cfg.exclusive_groups.items() for l in members}
        self.background = set(cfg.background_labels)
        self.drop_background = cfg.background_target == "drop"

    def compatible(self, chosen: set[str], candidate: tuple[str, ...]) -> bool:
        """Whether a window labelled *candidate* can join a mix already holding *chosen*.

        Never the same label twice, never two labels of one exclusive group,
        and at most one background label per mix.
        """
        if chosen & set(candidate):
            return False
        groups = {self.group_of[l] for l in chosen if l in self.group_of}
        if any(self.group_of.get(l) in groups for l in candidate if l in self.group_of):
            return False
        n_background = len(chosen & self.background) + len(set(candidate) & self.background)
        return n_background <= 1

    def is_foreground(self, labels: tuple[str, ...]) -> bool:
        return not set(labels) <= self.background

    def target(self, labels: list[str]) -> list[str]:
        """The mix's labels, first-seen order; background dropped when configured."""
        out = list(dict.fromkeys(labels))
        if self.drop_background:
            out = [l for l in out if l not in self.background]
        return out


def mixable(s: AudioSample) -> bool:
    """Whether *s* can give a mix source: real, local audio (not a mix, SSH or .npy)."""
    return s.mix_sources is None and not s.ssh_path and s.path.suffix.lower() != ".npy"


def clean_windows(samples: list[AudioSample]) -> list[AudioSample]:
    """The distinct clean windows behind *samples*, in first-seen order.

    A noise-augmented copy contributes the window it was cut from, without its
    noise: mixes are calls over calls, and noise augmentation stays a separate
    step. Several copies of one window (and the original, with
    ``keep_original``) give one source, so augmentation neither multiplies a
    window's chances of being drawn nor, with ``keep_original: false``, leaves
    nothing to mix.
    """
    seen: set[tuple] = set()
    out: list[AudioSample] = []
    for s in samples:
        if not mixable(s):
            continue
        key = (str(s.path), s.start_time, s.end_time)
        if key in seen:
            continue
        seen.add(key)
        out.append(s if s.noise_path is None
                   else dataclasses.replace(s, noise_path=None, snr=None, noise_start_time=None))
    return out


def build_audio_mixes(
    samples: list[AudioSample],
    cfg: AudioMixupConfig,
    seed: int,
    ratio: Optional[float] = None,
    split: str = "train",
) -> list[AudioSample]:
    """Draw mixes from *samples* — all of one split — and return them as new samples.

    The number of mixes is *ratio* × the windows that may be mixed (only those
    of ``cfg.labels``, when set); without *ratio* it is ``cfg.n_mixes``, else
    ``cfg.ratio`` × those windows. Every mix has at least one non-background source; further
    sources are drawn among the windows compatible with the ones already chosen
    (see :class:`_Rules`). Each added source gets a level relative to the first
    drawn uniformly from ``cfg.snr_db``. The draw is fully determined by *seed*
    and the order of *samples*, and a mix drawn twice is kept once.
    """
    rules = _Rules(cfg)
    pool = clean_windows(samples)
    skipped = sum(1 for s in samples if not mixable(s))
    if cfg.labels:
        # Only windows whose every label is listed: a window also carrying an
        # unlisted label would put that label into the mix target.
        allowed = set(cfg.labels)
        present = {l for s in pool for l in sample_labels(s)}
        missing = sorted(allowed - present)
        if missing:
            print(f"  [audio_mixup] labels not found in the {split} windows: {missing}")
        ignored = sorted(present - allowed)
        pool = [s for s in pool if set(sample_labels(s)) <= allowed]
        if ignored:
            print(f"  [audio_mixup] never mixed (not in audio_mixup.labels): {ignored}")
    if ratio is not None:
        n_mixes = round(ratio * len(pool))
    else:
        n_mixes = cfg.n_mixes if cfg.n_mixes is not None else round(cfg.ratio * len(pool))
    if n_mixes <= 0 or not pool:
        return []

    # Windows grouped by their label set: compatibility is a property of the
    # labels, so it is checked per set rather than per window.
    by_set: dict[tuple[str, ...], list[int]] = {}
    for i, s in enumerate(pool):
        by_set.setdefault(sample_labels(s), []).append(i)
    first_sets = [ls for ls in by_set if rules.is_foreground(ls)]
    if not first_sets:
        print(f"  [audio_mixup] no non-background {split} windows to mix — skipped")
        return []

    rng = np.random.default_rng(seed)

    def _draw(sets: list[tuple[str, ...]]) -> int:
        """One window among *sets*: balanced over labels, or uniform over windows."""
        if cfg.pairing == "balanced":
            labels = sorted({ls[0] for ls in sets})
            label = labels[int(rng.integers(len(labels)))]
            sets = [ls for ls in sets if ls[0] == label]
        sizes = np.array([len(by_set[ls]) for ls in sets], dtype=float)
        ls = sets[int(rng.choice(len(sets), p=sizes / sizes.sum()))]
        return by_set[ls][int(rng.integers(len(by_set[ls])))]

    mixes: list[AudioSample] = []
    seen: set[str] = set()
    attempts = 0
    while len(mixes) < n_mixes and attempts < n_mixes * _MAX_ATTEMPTS:
        attempts += 1
        n_sources = 3 if cfg.max_sources == 3 and rng.random() < cfg.p_three_sources else 2
        picks = [_draw(first_sets)]
        chosen = set(sample_labels(pool[picks[0]]))
        while len(picks) < n_sources:
            candidates = [ls for ls in by_set if rules.compatible(chosen, ls)]
            if not candidates:
                break
            picks.append(_draw(candidates))
            chosen |= set(sample_labels(pool[picks[-1]]))
        if len(picks) < 2:
            continue

        sources = tuple(
            MixSource(
                path=pool[i].path, label=pool[i].label,
                start_time=pool[i].start_time, end_time=pool[i].end_time,
                snr_db=0.0 if k == 0 else round(float(rng.uniform(*cfg.snr_db)), 2),
                signal_duration_seconds=pool[i].signal_duration_seconds,
                signal_offset_samples=pool[i].signal_offset_samples,
                original_path=pool[i].original_path,
            )
            for k, i in enumerate(picks)
        )
        key = mix_key(sources)
        if key in seen:
            continue
        seen.add(key)
        target = rules.target([l for i in picks for l in sample_labels(pool[i])])
        mixes.append(AudioSample(
            path=Path(key), label=target[0], extra_labels=tuple(target[1:]),
            split=split, mix_sources=sources,
        ))

    n3 = sum(1 for m in mixes if len(m.mix_sources) == 3)
    print(f"  [audio_mixup] {len(mixes)} {split} mixes from {len(pool)} windows"
          + (f" ({n3} with 3 sources)" if n3 else "")
          + (f"; {skipped} window(s) not mixable (SSH or .npy)" if skipped else ""))
    if len(mixes) < n_mixes:
        print(f"  [audio_mixup] asked for {n_mixes}, only {len(mixes)} distinct compatible "
              f"mixes found")
    pairs = Counter(" + ".join(sorted(sample_labels(m))) for m in mixes)
    for combo, n in pairs.most_common(5):
        print(f"    {combo}: {n}")
    return mixes


def check_audio_mixup_head(classifier: str, output_activation: Optional[str],
                           label_groups: dict) -> None:
    """Raise unless the run trains a head that can learn from multi-label targets.

    Only a keras sigmoid head treats classes independently; a softmax (or
    logits, trained with softmax) or grouped head forces the classes of a mix to
    compete, which is exactly what the union target contradicts.
    """
    if classifier == "sklearn":
        raise ValueError("audio_mixup needs a keras head (training.classifier: keras or both)")
    if label_groups or output_activation != "sigmoid":
        raise ValueError(
            "audio_mixup requires training.keras.output_activation: sigmoid — a mix is "
            "labelled with every class it holds, which a softmax, logits or grouped head "
            f"cannot represent (got {output_activation!r}"
            f"{' with label_groups' if label_groups else ''})."
        )


def training_composition(train_samples: list[AudioSample], test_samples: list[AudioSample],
                         label_names: list[str]) -> dict:
    """Per label: real train windows, mixes carrying it, its mix partners, real test windows.

    ``in_mixes`` counts the mixes whose *target* holds the label; ``as_source``
    the mixes it was summed into, which differ for a background label dropped
    from mix targets. ``mix_share`` is the synthetic part of the label's
    positive train samples. ``partners`` counts the other source labels it was
    mixed with, most frequent first.
    """
    mixes = [s for s in train_samples if s.mix_sources is not None]
    real = [s for s in train_samples if s.mix_sources is None]
    labels: dict[str, dict] = {}
    for label in label_names:
        n_real = sum(1 for s in real if label in sample_labels(s))
        carrying = sum(1 for m in mixes if label in sample_labels(m))
        partners: Counter = Counter()
        n_source = 0
        for m in mixes:
            sources = [src.label for src in m.mix_sources]
            if label in sources:
                n_source += 1
                partners.update(l for l in sources if l != label)
        labels[label] = {
            "real_train": n_real,
            "in_mixes": carrying,
            "as_source": n_source,
            "mix_share": round(carrying / (n_real + carrying), 4) if n_real + carrying else 0.0,
            "partners": dict(partners.most_common()),
            "test": sum(1 for s in test_samples if label in sample_labels(s)),
        }
    return {
        "n_mixes": len(mixes),
        "n_real_train": len(real),
        "n_sources": dict(sorted(Counter(len(m.mix_sources) for m in mixes).items())),
        "labels": labels,
    }
