#!/bin/bash
# Run this on the server to reorganize the repo
cd /home/woody/iwso/iwso226h/ma52

# 1. Create new hand directory structure
mkdir -p src/hand/legacy

# 2. Move new hand scripts into src/hand/
cp src/hand/prepare_twostage.py src/hand/prepare_twostage.py 2>/dev/null || true
cp src/hand/train_stage1.py     src/hand/train_stage1.py     2>/dev/null || true
cp src/hand/train_stage2.py     src/hand/train_stage2.py     2>/dev/null || true
cp src/hand/evaluate_twostage.py src/hand/evaluate_twostage.py 2>/dev/null || true

# 3. Move old hand_fine scripts to legacy
cp src/hand_fine/train_hand_fine.py       src/hand/legacy/
cp src/hand_fine/train_hand_fine_crop.py  src/hand/legacy/
cp src/hand_fine/hand_fine_model.py       src/hand/legacy/
cp src/hand_fine/hand_fine_dataset.py     src/hand/legacy/
cp src/hand_fine/eval_hand_fine.py        src/hand/legacy/
cp src/hand_fine/prepare_hand_fine_dataset.py src/hand/legacy/

# 4. Keep useful scripts in src/hand/
cp src/hand_fine/crop_upperbody.py src/hand/crop_upperbody.py

# 5. Create __init__.py
touch src/hand/__init__.py
touch src/hand/legacy/__init__.py

# 6. Organize jobs into subfolders
mkdir -p jobs/head jobs/hand jobs/skeleton jobs/body_leg

# Head jobs
mv jobs/job_crop_head.sh         jobs/head/ 2>/dev/null || true
mv jobs/job_train_head.sh        jobs/head/ 2>/dev/null || true
mv jobs/job_train_head_dynamic.sh jobs/head/ 2>/dev/null || true
mv jobs/job_evaluate_head.sh     jobs/head/ 2>/dev/null || true
mv jobs/job_extract_pose.sh      jobs/head/ 2>/dev/null || true

# Hand jobs
mv jobs/job_hand_*.sh            jobs/hand/ 2>/dev/null || true
mv jobs/job_eval_hand_fine.sh    jobs/hand/ 2>/dev/null || true

# Skeleton jobs
mv jobs/job_body_*.sh            jobs/skeleton/ 2>/dev/null || true
mv jobs/job_leg_*.sh             jobs/skeleton/ 2>/dev/null || true

# Body/leg VideoMAE jobs
mv jobs/skeleton/job_body_videomae.sh jobs/body_leg/ 2>/dev/null || true
mv jobs/skeleton/job_leg_videomae.sh  jobs/body_leg/ 2>/dev/null || true

# Pose extraction
mv jobs/job_extract_pose_new.sh  jobs/body_leg/ 2>/dev/null || true

echo "Reorganization done!"
echo ""
echo "New structure:"
find jobs/ -name "*.sh" | sort
echo ""
find src/ -name "*.py" | grep -v BlockGCN | grep -v CTR-GCN | grep -v MMN | sort
