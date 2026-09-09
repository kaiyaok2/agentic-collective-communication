#!/usr/bin/env python3
"""Plot the OLMoE-10B real-OpenWebText 2500-step loss-curve (Figure 2).

Reads the per-step losses field from two olmoe10b_*.json files (baseline and
agent) produced by train_olmoe10b.py --realtok, overlays them on one axis,
and emits figures/loss_curve.pdf.
"""
import argparse, json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(xs, w=20):
    if w <= 1: return xs
    out, acc = [], []
    for x in xs:
        if x is None: continue
        acc.append(x)
        if len(acc) > w: acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='JSON from --backend baseline run')
    ap.add_argument('--agent', required=True, help='JSON from --backend agent run')
    ap.add_argument('--out', default='figures/loss_curve.pdf')
    ap.add_argument('--smooth-window', type=int, default=20)
    args = ap.parse_args()

    with open(args.baseline) as f: bjson = json.load(f)
    with open(args.agent) as f:    ajson = json.load(f)
    blosses = [l for l in bjson['losses'] if l is not None]
    alosses = [l for l in ajson['losses'] if l is not None]
    bs = smooth(blosses, args.smooth_window)
    as_ = smooth(alosses, args.smooth_window)
    n = min(len(bs), len(as_))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(range(n), bs[:n], label='baseline (developer)', linewidth=1.6, color='#0066aa')
    ax.plot(range(n), as_[:n], label='agent (strategy-enum)', linewidth=1.6, color='#cc6622')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('OLMoE-10B real-OpenWebText, 224 ranks (bf16 AdamW)')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f'wrote {args.out} (baseline n={len(blosses)}, agent n={len(alosses)})')


if __name__ == '__main__':
    main()
