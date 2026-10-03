# SCRUM: Social Contact Recognition Using Motion

SCRUM, Social Contact Recognition Using Motion, is a tool that has frame-by-frame classification of rat social and aggressive behavior (sniffing, biting, mounting).

There are two separate ways to get from video to behavior labels: the first way includes a pipeline that runs on top of SLEAP / DeepLabCut; the second is a path that is pose-free for raw video when there is no tracking model that exists yet. This project was first trained / validated on CalMS21, a public dataset with annotated mouse social behavior provided by Caltech.


## How It Works

Pose tracking is handled by SLEAP and DeepLabCut in which they include keypoints per animal, annotated per frame, and this in turn enables classification to turn these tracked keypoints into behavior labels. 

```
SLEAP / DeepLabCut Export
        |
        v
pipeline/io_utils.py            
        |
        v
pipeline/identity_resolution.py -> detects occlusion/contact, repairs tracker identity swaps,
        |                           
        v
pipeline/features.py            -> per-frame geometric/kinematic features (distances, orientation,
        |                           overlap, speed)
        v
pipeline/behavior_classifier.py -> trained on hand-labeled frames (random forest or gradient boosting)
        |
        v
pipeline/smoothing.py           -> removes single-frame flicker, drops bouts shorter than a minimum duration
        |
        v
pipeline/evaluate.py            -> frame-level and event-level (bout) scoring against ground truth
        |
        v
pipeline/visualize.py           -> annotated video overlay 
```

When no pose tracking exists yet, `pipeline/blob_tracker.py` finds each rat directly from raw video as a bright blob against the dark arena floor, and tracks identity across frames with the same motion-prediction and assignment logic that `identity_resolution.py` uses. Since it has no keypoints, it can't drive the classifier.

`pipeline/calms21_import.py` loads CalMS21's own keypoints and real frame-level labels directly, so the same classifier and features can train and score against ground-truth mounting, attack, and investigation data.

## Running It

### Raw Video

```
python scripts/detect_and_track.py \
    --video TwoRats.mp4 --out-dir outputs/tworats_demo --export-frames
```

Writes `tracks.csv`, `contact_bouts.csv`, and `annotated.mp4`. Each rat's outline is traced in green and there is a contact label at the top of the screen that will display when contact is detected.

### SLEAP/DeepLabCut 

```
# 1. Track a video in SLEAP or DeepLabCut, export the analysis in a .h5 format
# 2. Hand-label a training clip as a CSV: frame,behavior

python scripts/train_classifier.py \
    --tracking clip01.h5 --labels clip01_labels.csv --model-out models/rf.joblib

python scripts/run_pipeline.py \
    --tracking clip02.h5 --model models/rf.joblib --out-dir outputs/clip02 \
    --video clip02.mp4 --export-frames --ground-truth clip02_labels.csv
```

The second command will write an annotated video, per-frame images, predictions, and (with `--ground-truth`) a scored report.

### CalMS21

```
python scripts/inspect_calms21.py calms21_task1_train.json

python scripts/train_on_calms21.py \
    --train-json calms21_task1_train.json --test-json calms21_task1_test.json \
    --model-out models/calms21_sbm_hgb.joblib

python scripts/render_calms21_demo.py \
    --video mouse050_task1_annotator1.mp4 --sequence-json mouse050_seq.json \
    --model models/calms21_sbm_hgb.joblib --out-dir outputs/mouse050_demo
```
If you don't have a dataset, CalMS21 can provide sample videos to display the model's functionality and for your own training/testing use.
If interested in this dataset, please use the original dataset found at [CalMS21](https://data.caltech.edu/records/s0vdx-0k302). 
Additionally, please credit the original authors:
```bibtex
@article{calms21,
  title={The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions},
  author={Sun, Jennifer J and Karigo, Tomomi and Chakraborty, Dipam and Mohanty, Sharada P and Wild, Benjamin and Sun, Quan and Chen, Chen and Anderson, David J and Perona, Pietro and Yue, Yisong and Kennedy, Ann},
  journal={arXiv preprint arXiv:2104.02710},
  year={2021}
}
```
