"""
Convert AdaTAD test predictions to CodaLab submission format for MAC 2026 Track 2.
Usage: python prepare_submission.py --input predictions/large_test.json --output submission.zip
Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
"""
import json, zipfile, os, argparse

def convert(input_path, output_path):
    data = json.load(open(input_path))
    results = data.get('results', data)
    
    # CodaLab expects: {results: {video_id: [{segment, label, score}]}}
    # which is exactly what we already have — just needs zipping
    submission = {'results': {}}
    
    for vid, preds in results.items():
        # Sort by score descending, keep top 100 per video
        sorted_preds = sorted(preds, key=lambda x: -x['score'])[:100]
        submission['results'][vid] = [
            {
                'segment': p['segment'],
                'label': p['label'],
                'score': round(float(p['score']), 6)
            }
            for p in sorted_preds
        ]
    
    # Save JSON
    json_path = output_path.replace('.zip', '.json')
    json.dump(submission, open(json_path, 'w'), indent=2)
    print(f"Saved JSON: {json_path} ({len(submission['results'])} videos)")
    
    # Zip it
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, 'results.json')
    print(f"Saved ZIP: {output_path}")
    
    # Stats
    all_scores = [p['score'] for preds in submission['results'].values() for p in preds]
    print(f"Total predictions: {len(all_scores)}")
    print(f"Avg predictions/video: {len(all_scores)/len(submission['results']):.1f}")
    print(f"Score range: {min(all_scores):.4f} - {max(all_scores):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='predictions/large_test.json')
    parser.add_argument('--output', default='predictions/submission_large.zip')
    args = parser.parse_args()
    convert(args.input, args.output)
