"""
prepare_submission_csv.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/prepare_submission_csv.py --help` for options where applicable.
"""
import json, csv, argparse, os

CAT_FILE = './data/MMA-52/Annotations/adatad/category_idx.txt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='predictions/large_test.json')
    parser.add_argument('--output', default='predictions/submission.csv')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--max_per_video', type=int, default=200)
    args = parser.parse_args()

    # Load category name -> index mapping
    cats = [l.strip() for l in open(CAT_FILE) if l.strip()]
    cat2idx = {c: i for i, c in enumerate(cats)}
    print(f"Categories: {len(cats)}")

    data = json.load(open(args.input))
    results = data.get('results', data)

    rows = []
    row_id = 0
    for vid in sorted(results.keys()):
        preds = results[vid]
        # Filter by threshold
        preds = [p for p in preds if p['score'] >= args.threshold]
        # Sort by score descending, keep top N
        preds = sorted(preds, key=lambda x: -x['score'])[:args.max_per_video]
        for p in preds:
            label_idx = cat2idx.get(p['label'], -1)
            if label_idx == -1:
                print(f"WARNING: unknown label {p['label']}")
                continue
            rows.append({
                'ID': row_id,
                'video-id': vid,
                't-start': round(p['segment'][0], 6),
                't-end': round(p['segment'][1], 6),
                'label': label_idx,
                'score': round(p['score'], 6)
            })
            row_id += 1

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID','video-id','t-start','t-end','label','score'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows for {len(results)} videos")
    print(f"Saved: {args.output}")
    print(f"Sample rows:")
    for r in rows[:3]:
        print(f"  {r}")

if __name__ == '__main__':
    main()
