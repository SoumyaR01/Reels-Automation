# AI Movie Analysis & Reel Generation System: Code Review and Architecture Plan

## 1. Review of the existing code

The existing script is a solid prototype. The ffmpeg piping, feature design, CLI and error handling are all reusable. But it is a heuristic shot-compiler, not yet a movie-understanding system.

### Bugs and correctness problems

| # | Location | Issue |
|---|---|---|
| 1 | `main()` | **Crash:** `score_scene` returns `(0, 0, {})` when `end_idx <= start_idx`, then `details["start"]` raises `KeyError`. It isn't in the `except` tuple, so you get a raw traceback. Skip scenes with empty details. |
| 2 | `extract_audio_features` | `hop_length=sample_rate` with `n_fft=2048` analyzes only a 128 ms slice out of every 1000 ms. Chroma and minor-key estimates from 128 ms of mixed audio are mostly noise. Use ~0.5 s windows, and estimate key over 5-10 s, not per second. |
| 3 | `analyze_video_stream` | Everything runs at 2 fps, so **cut times are quantized to +/-0.5 s**. Exported clips include frames from the previous shot or lose the first frames of the next. |
| 4 | Same | `phaseCorrelate`, "jerk" and "slow/frozen" are computed between frames 0.5 s apart. Camera motion at that gap is meaningless, so stability and slow-motion scores are essentially random. Needs >=8-12 fps, or optical flow on short bursts. |
| 5 | Cut detection | A fixed `0.35` threshold on a histogram+pixel metric misses dissolves and fades and over-fires on flashes, explosions and strobes. Not adaptive per film. |
| 6 | `find_scene_boundaries` | These are **shots** (1-3 consecutive shots, <=9 s), not **scenes**. A real scene is minutes long. Biggest conceptual gap for "complete scenes". |
| 7 | `collect_qualifying_scenes` | The last clip can push the total past 30 s, then `-t 30` chops it mid-scene, contradicting "complete scenes". `max_overlap_iou` is never used. |
| 8 | Thresholds | An absolute `0.40` isn't comparable across films. Use per-movie percentile or z-score normalization. |
| 9 | Sadness model | Dark is not sad (night fight scenes score "sad"), quiet is not sad, and minor key on mixed audio is unreliable. Needs learned models (see section 3). |
| 10 | `export_reel` | Center-crop 16:9 to 9:16 discards ~68% of the frame and will crop out faces and action. It also double-encodes (per clip, then concat), has hard audio cuts (pops), and the concat list breaks on filenames containing `'`. |
| 11 | `stderr=PIPE` | Only stdout is read, so a chatty ffmpeg can deadlock the pipe. Use `DEVNULL` or a reader thread. |
| 12 | Misc | Timestamps are `float32` (use `float64`). The final cut is dropped if the last shot is under 2 s. `video_duration` comes from the last sampled frame, not the container. No caching, so a crash at minute 90 restarts everything. No tests, no logging. |

### Performance issues
- The movie is decoded three times (audio, video, export). Decode once into a proxy (360p video plus audio WAV) and reuse it.
- Nothing is persisted, so there is no resumability and no re-scoring without re-analysis.
- No GPU use anywhere (decode, encode or inference).

### What to keep
`probe_video`, `require_tool`, `create_rule_of_thirds_mask`, the color/brightness/composition features, and the RMS/centroid audio features all stay. They become cheap pre-filter features in a larger pipeline rather than the final judge.

### Quick fix for bug #1

```python
cinematic_score, sadness_score, details = score_scene(scene, video_feats, audio_feats)
if not details:
    continue
```

---

## 2. Target architecture

The key design choice is a **two-pass funnel**. Cheap features run on the whole film, then expensive AI (VLM, LLM) runs only on the top ~5-10% of candidates. This keeps a 2.5-hour film tractable on one GPU.

```
Movie -> [0 Ingest] -> [1 Shots] -> [2 Scenes] -> [3 Audio AI] -> [4 Visual AI]
          proxy+WAV     TransNetV2    cluster       ASR+prosody    faces+actions
                                      shots         +emotion       +embeddings
                                                         |              |
                                                         v              v
                                              [5 Cheap scoring -> shortlist]
                                                         |
                                              [6 VLM + LLM judge on shortlist]
                                                         |
                                       [7 Moment selection (30 s windows)]
                                                         |
                                           [8 Human review UI]
                                                         |
                       [9 Edit: reframe, subtitles, shayari, music, FX, render]
                                                         |
                                      [10 Export + metadata + rights log]
```

Every stage is idempotent, reads and writes artifacts to `workdir/<movie_id>/`, and records status in a manifest. A crash resumes at the failed stage.

### Repo layout (existing code mapped in)

```
reelforge/
  core/        ffmpeg.py (probe_video, require_tool), config.py, manifest.py, schemas.py
  ingest/      proxy.py, audio_extract.py
  shots/       transnet.py            # replaces existing cut detection
  scenes/      grouping.py            # shots -> scenes (embeddings + audio + dialogue gaps)
  audio/       features.py            # existing librosa code, fixed windows
               asr.py, prosody.py, emotion.py, events.py, separation.py
  video/       aesthetics.py          # existing thirds/color/brightness code
               faces.py, actions.py, vlm.py
  scoring/     heuristics.py          # existing score_scene -> baseline/fallback + features
               fusion.py, llm_judge.py
  selection/   windows.py, diversity.py   # fixed collect_qualifying_scenes
  edit/        reframe.py, subtitles.py, shayari.py, music.py, fx.py, render.py
  rights/      registry.py            # rights status per source; publish gate
  api/         FastAPI + review UI
  workers/     GPU worker, CPU worker
  cli.py                              # existing argparse CLI, extended
```

### Core data contract

```json
{
  "movie_id": "m_001", "scene_id": "s_0412",
  "start": 5231.84, "end": 5262.10,
  "labels": {"sad": 0.91, "romantic": 0.12, "action": 0.02,
             "powerful_dialogue": 0.64, "motivational": 0.05, "suspense": 0.10},
  "signals": {"arousal": 0.71, "valence": -0.62, "speech_rate": 2.1,
              "f0_range": 148, "pause_ratio": 0.34, "face_emotion": "sad",
              "music_mood": "melancholic", "audio_events": ["crying"]},
  "dialogue": {"text": "...", "standalone_score": 0.8, "quotability": 0.7},
  "spoiler_risk": 0.3, "hook_score": 0.8, "final_score": 0.87,
  "rights": {"source_status": "licensed|public_domain|own|unknown"}
}
```

---

## 3. Recommended models and tools

Versions move fast, so check for newer releases before pinning anything.

| Task | Recommendation | Notes |
|---|---|---|
| Shot boundaries | **TransNetV2** (fallback: PySceneDetect AdaptiveDetector) | Handles dissolves; frame-accurate |
| Scene grouping | SigLIP/CLIP keyframe embeddings + audio-change + dialogue gaps | Agglomerative clustering over shots |
| Stem separation | **Demucs (htdemucs)** | Separate dialogue from music/effects before prosody |
| VAD | Silero VAD | Cheap speech gating |
| ASR | **WhisperX** (faster-whisper large-v3 / turbo) | Word timestamps, Hindi/English code-switching; diarization via pyannote 3.1 |
| Prosody | **Parselmouth (Praat)**, openSMILE eGeMAPS, torchcrepe/pyin | F0, jitter/shimmer, energy, speech rate, pauses |
| Speech emotion | **emotion2vec+** (FunASR) and/or wav2vec2 arousal/valence/dominance | Dimensional output is more robust than discrete labels |
| Audio events | **PANNs / BEATs / AST** (AudioSet) | Gunshots, explosions, crying, screaming, applause |
| Music mood | CLAP or MERT embeddings, Essentia | Replaces the chroma "minor key" heuristic |
| Faces | **InsightFace (SCRFD)** detection/tracking, **HSEmotion** expression | Detection and expression only, **no identity recognition** |
| Action | VideoMAE-v2 / InternVideo2 or X-CLIP (zero-shot) | Fight/chase/kiss classification |
| Visual understanding | **Qwen-VL family** (or InternVL) served via **vLLM** | Run only on shortlisted candidates |
| Dialogue semantics / judging | Local LLM via vLLM (Qwen/Llama class), structured JSON output | Classify, rate quotability, standalone-ness, spoiler risk |
| Reframing | InsightFace + saliency, Kalman-smoothed virtual camera; fallback: blurred-background fit | |
| Subtitles | WhisperX word timings -> **ASS + libass** (karaoke-style highlight) | Needs HarfBuzz for correct Devanagari shaping; Noto Sans Devanagari / Hind (OFL) |
| Transitions/FX | ffmpeg `xfade`, `acrossfade`, LUTs, grain, vignette | Mostly ffmpeg filters; no AI needed |
| Music generation | ACE-Step, Stable Audio Open, MusicGen | **Check licenses.** MusicGen weights are non-commercial (CC-BY-NC), a problem for monetization. A licensed library is often safer. |
| Loudness | ffmpeg `loudnorm` (~-14 LUFS), `sidechaincompress` for ducking | |
| Video generation | Wan / LTX-Video / HunyuanVideo | Optional, for abstract backgrounds or intro bumpers only. Not needed for the core product. |

**Hindi shayari:** use an LLM with a rhyme/meter validator (check qafiya/radif and approximate matra count). Generate several drafts and let the review UI pick. Have the model write original lines only, and don't let it reproduce existing poets' work.

**Emotion honesty:** facial-emotion models are noisy on cinematic lighting and stylized acting. Treat them as one signal among several, not ground truth. The best long-term upgrade is a small fusion model trained on **your own channel's retention/share data**, which also becomes your monetization moat.

---

## 4. Moment selection (the biggest upgrade)

The current logic picks *scenes* and trims to 30 s. A good reel is a *moment*:
- Candidate 25-35 s windows snapped to sentence boundaries and shot cuts. Never cut mid-word.
- A hook in the first 3 s (strong line, striking face, sudden audio event), and an ending on a beat.
- Penalize spoiler risk and context-dependence (`standalone_score`).
- Diversity re-ranking (MMR) so you don't get 10 reels of the same scene.
- Per-category quotas, so one film yields a few sad, a few action, a few dialogue reels, etc.

---

## 5. Server design

- **Orchestration:** start with a simple stage runner and a SQLite/Postgres manifest. Move to Prefect or Dagster (or Celery+Redis) when you need multi-movie queues.
- **GPU worker:** loads one model family at a time (Demucs -> Whisper -> emotion -> VLM -> LLM) and unloads between stages. A 24 GB card (4090 / L4 / A10-class) handles this comfortably, run sequentially.
- **Decode/encode:** `-hwaccel cuda` for decode and `h264_nvenc` for drafts. Use CPU `libx264` for final-quality exports.
- **Serving:** vLLM for the VLM and LLM. Batch ASR and embeddings.
- **Packaging:** Docker plus NVIDIA Container Toolkit, FastAPI for the API, MinIO/S3 for artifacts, and a small review UI (Streamlit/Gradio first).
- **Rough throughput goal:** a full 2.5 h film analyzed in roughly 30-60 min and each reel rendered in about a minute, on one 24 GB GPU. Treat this as a target to benchmark, not a promise.
- **Single-pass render:** build one `filter_complex` (reframe, subtitles, xfade, FX, audio mix) instead of encode -> concat -> re-encode.

---

## 6. Copyright and platform policy

This affects the whole business model.

- **AI editing does not make copyrighted footage safe.** Cropping, subtitles, effects, new music and shayari overlays don't change who owns the footage, music or dialogue. Transformation is one factor in a fair-use or fair-dealing argument, not an exemption.
- **Fair use is a US defense, not a right.** In India, the relevant concept is *fair dealing* under Section 52 of the Copyright Act, which is narrower (criticism, review, reporting). Entertainment compilations of film clips generally don't fit it. This is not legal advice; consult a lawyer before building a revenue model on this.
- **In practice:** Content ID and Meta Rights Manager will claim, mute, demonetize or remove, and repeated strikes can take down the channel. Instagram and YouTube have also been tightening policies on unoriginal or reused content, so check their current rules.
- **No evasion techniques** (flipping, pitch-shifting, watermark removal) will be designed, since they are both policy-violating and legally riskier.

**Safer and more commercial paths, all supported by this same architecture:**
1. **Licensed or permitted content:** studio/distributor agreements, official clip programs, or content you own or commission.
2. **Public-domain or openly licensed films** (verify the film's status in your jurisdiction, and check that music, restorations and subtitle tracks are also clear).
3. **Real commentary/review formats**, where the clips support your own analysis.
4. **Sell it as a tool.** Studios, OTT platforms and production houses *own* the rights and want automated promos and reels. This is probably the strongest commercial route.

**Built-in safeguards:**
- A `rights/registry` module stores a rights status per source, and **export is blocked while status is `unknown`**.
- An audit log records which source, timestamps and generated assets went into each reel.
- A toggle for **AI-disclosure labels** where platforms require them.
- No face *identification*, so no biometric/likeness issue.
- Original shayari only, with clearly licensed music and fonts.

---

## 7. Roadmap

| Phase | Deliverable |
|---|---|
| **0** | Fix bugs #1-12, add a proxy/cache layer, make thresholds percentile-based. The existing script becomes reliable. |
| **1** | TransNetV2 shots, scene grouping, WhisperX transcripts, prosody and emotion features, persisted features. |
| **2** | Audio events, face/action models, the 6-category classifier, LLM judge, moment-window selection. |
| **3** | Smart reframing, ASS subtitles with Devanagari, shayari generation, music and ducking, single-pass render. |
| **4** | API, queue, review UI, Docker/GPU deployment, rights gate, metrics, feedback loop from channel analytics. |

## 8. Open question

Which GPU, how much VRAM and how much RAM does the server have? That decides which model sizes get scaffolded in Phase 0-1, followed by the restructured package with the fixed code and the new stages.
