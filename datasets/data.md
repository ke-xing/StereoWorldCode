# Dataset Processing

## Data Source Notice

Due to copyright and legal restrictions, we cannot distribute the original movie files, video clips, frames, audio, or any other source media. This directory only provides the list of movie titles used for our research: [movies.csv](./movies.csv).

To reproduce the dataset construction pipeline, users must lawfully obtain access to the corresponding SBS (Side-by-Side) 3D movies according to `movies.csv`, and place their local source video files under:

```text
datasets/raw_movies/
```

This repository does not provide, host, link to, or facilitate access to copyrighted movie files. Source videos should not be committed to version control or redistributed with this repository.

## Copyright and Legal Compliance

Users are solely responsible for ensuring that their use of any source videos complies with applicable copyright law, license terms, and platform terms of service. This includes obtaining any required permissions and avoiding unauthorized copying, redistribution, or public sharing of copyrighted content.

This repository only provides processing scripts and metadata needed to reproduce the pipeline. Generated clips, frames, captions, depth maps, disparity maps, or other annotations may still be derived from copyrighted source works. Before sharing any generated outputs, users should independently confirm that such sharing is permitted under the applicable law and rights holder terms.

This repository does not provide instructions for bypassing DRM, access controls, encryption, paywalls, or other technological protection measures. Nothing in this README should be interpreted as legal advice.

## Directory Structure

Before processing, the directory structure is expected to look like this:

```text
datasets/
+-- movies.csv
+-- raw_movies/                 # Lawfully obtained local SBS source videos
+-- split.py
+-- caption.py
+-- Video-Depth-Anything/       # DepthAnything/Video-Depth-Anything repository
|   +-- depth.py                # Depth annotation script provided by this repository
+-- stereoanyvideo/             # TomTomTommi/stereoanyvideo repository
    +-- disparity.py            # Disparity annotation script provided by this repository
    +-- demo_video.py           # Script required by stereoanyvideo inference
```

After processing, the main outputs will be written to:

```text
datasets/split_movies/
+-- left/                       # Left-view video clips
+-- right/                      # Right-view video clips
+-- metadata_caption.csv        # Caption annotations
+-- depth/                      # Depth annotations
+-- disp/                       # Disparity annotations
```

## 1. Split and Resize SBS Movies

First, enter the `datasets` directory:

```bash
cd datasets
```

Run `split.py` to split the SBS movies in `raw_movies` into left-view and right-view clips, and resize them to the target resolution specified by the script:

```bash
python split.py --input_path ./raw_movies --output ./split_movies
```

By default, each clip contains 81 frames, and the script samples frames with `interval=2`. You can adjust the parameters as needed:

```bash
python split.py \
  --input_path ./raw_movies \
  --output ./split_movies \
  --height 480 \
  --frames 81 \
  --interval 2 \
  --workers 8
```

The output files will be saved to:

```text
split_movies/left/
split_movies/right/
```

## 2. Generate Caption Annotations

Caption annotations are generated with [damo-nlp-sg/videollama3](https://github.com/damo-nlp-sg/videollama3). Please follow the upstream repository instructions to configure the VideoLLaMA3 inference environment first, including dependencies, model weights, CUDA/PyTorch versions, and GPU memory requirements.

After the environment is ready, run the following command under the `datasets` directory:

```bash
python caption.py
```

The script reads videos from:

```text
split_movies/left/
```

and writes caption results to:

```text
split_movies/metadata_caption.csv
```

`caption.py` automatically detects available GPUs and distributes videos according to the multiprocessing configuration in the script. To change the model path, the number of processes per GPU, or the prompt, modify the corresponding settings in `caption.py`.

## 3. Generate Depth Annotations

Depth annotations are generated with [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything).

Please download the upstream repository to:

```text
datasets/Video-Depth-Anything/
```

Then follow the upstream instructions to configure the runtime environment and download the required checkpoint. The current script expects the default checkpoint path to be:

```text
datasets/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth
```

Place the `depth.py` script provided by this repository under `datasets/Video-Depth-Anything/`, then run:

```bash
cd datasets/Video-Depth-Anything
python depth.py \
  --input_dir ../split_movies/right \
  --output_dir ../split_movies/depth \
  --encoder vitl \
  --gpu_ids 0 \
  --processes_per_gpu 1
```

For multi-GPU environments, set `--gpu_ids` to a comma-separated list of GPU IDs, for example:

```bash
python depth.py --gpu_ids 0,1,2,3 --processes_per_gpu 1
```

Depth results are saved as `.npz` files under:

```text
datasets/split_movies/depth/
```

## 4. Generate Disparity Annotations

Disparity annotations are generated with [TomTomTommi/stereoanyvideo](https://github.com/TomTomTommi/stereoanyvideo).

Please download the upstream repository to:

```text
datasets/stereoanyvideo/
```

Then follow the upstream instructions to configure the runtime environment, dependencies, and required weights. Place the `disparity.py` and `demo_video.py` script provided by this repository under `datasets/stereoanyvideo/`, and make sure `demo_video.py` can run correctly in the same directory.

Run the following command under `datasets/stereoanyvideo`:

```bash
cd datasets/stereoanyvideo
python disparity.py
```

The script reads videos from:

```text
../split_movies/left/
../split_movies/right/
```

and writes disparity results to:

```text
../split_movies/disp/
```

## Full Pipeline

```bash
# 1. Prepare SBS source videos
# Place lawfully obtained local SBS movie files under datasets/raw_movies/

# 2. Split left/right views and resize videos
cd datasets
python split.py --input_path ./raw_movies --output ./split_movies

# 3. Generate captions
python caption.py

# 4. Generate depth annotations
cd Video-Depth-Anything
python depth.py --input_dir ../split_movies/right --output_dir ../split_movies/depth --encoder vitl --gpu_ids 0

# 5. Generate disparity annotations
cd ../stereoanyvideo
python disparity.py
```

Please note that VideoLLaMA3, Video-Depth-Anything, and stereoanyvideo each require their own environment setup and model weights. Dependency versions, checkpoint download instructions, and license requirements should follow the corresponding upstream repositories.
