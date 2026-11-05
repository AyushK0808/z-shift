# Response to reviewers: CICT-2026, Paper 760

We thank both reviewers. Each numbered item below is a requested revision, with
what was done and where the evidence is. Every number cited traces to a CSV in
`bench/results/`, produced by `scripts/run-reviewer-response.sh` (B10-B13) or
the earlier benchmark scripts.

## Reviewer 1

**1. Roadmap for the missing real-time delivery protocols (RTSP/WebRTC).**
This is a roadmap, so it lives here and not in the paper, which keeps its
limitation statement. The path from where the prototype is:

1. *RTSP ingestion.* The live track already takes WebSocket frames into a
   bounded ring buffer with backpressure. RTSP is another producer into the same
   buffer: an FFmpeg-backed reader that decodes a stream and pushes
   `FrameReference`s, so the buffer, priority tagging and schema are unchanged.
2. *WebRTC ingestion.* A signalling endpoint on the existing gateway (reusing
   its authentication and rate limiting) and a receiver that terminates the
   peer connection and feeds the same buffer.
3. *Live reconstruction.* The current reconstruction stage needs at least two
   views and an offline alignment, so a live path needs an incremental one: a
   sliding window over the buffer, aligned with the existing windowed pairing
   and re-fused as frames arrive. This is the largest piece and is future work.
4. *Live delivery.* The live outcome route, which the routing contract rejects
   today, is added once step 3 produces geometry; the contract already
   isolates it, so the other routes are unaffected.

**2. Future work should prioritise quantitative comparison with classical MVS
and MASt3R baselines.** Kept as the first priority in the Limitations section.
The paper now reports ground-truth accuracy for the pipeline itself (see
Reviewer 2, item 1), and says plainly that no baseline ran under the same
protocol.

## Reviewer 2

**1. Ground-truth accuracy (Chamfer, F-score, normal consistency) on DTU and
CO3D; baselines; the effect of retention order.**

- *DTU.* Five scans (scan24, 37, 40, 55, 63), DTU's own evaluation protocol,
  with a similarity fit to the calibrated cameras that is trimmed of lost views
  (B2; table in the paper). The effect of retention order is B4: restoring
  capture order did not improve reconstruction.
- *Normal consistency on DTU (new).* DTU's ground truth has no normals, so they
  are estimated from the structured-light points and compared with the
  reconstructed meshes' normals (`bench/dtu_eval.py::mesh_normal_consistency`,
  B10). The measured value is about 0.52 mean |cos| for every variant, close
  to the 0.5 of unrelated normals, so it does not separate the variants. We
  therefore report Chamfer, F-score and DTU accuracy/completeness in the paper
  and leave this metric in the repository, not as a result.
- *CO3D (new).* CO3D's point cloud is a COLMAP reconstruction, not an
  independent ground truth, and the single-sequence release has no camera
  annotations. We therefore score agreement with CO3D's reference after
  registering to it (a fixed scale, then ICP), report the registration residual (1.4-1.5% of the bounding-box diagonal),
  and call it a
  pseudo-reference (B11). Refined and isolated, the mesh scores F@2% 0.745 at
  8 views and 0.769 at 24, against 0.586 and 0.502 for the raw point cloud
  (new subsection "Further Evaluation" in the paper).
- *Baselines.* Not done in this revision. We report that no baseline ran under
  the same protocol and do not compare against published MVS scores, which are
  given DTU's cameras. The second backend below (DUSt3R) is scored under the
  same protocol, so one comparison now exists: at ten views MASt3R reaches
  mean F-score 0.210 and 5.85 mm overall error, DUSt3R 0.069 and 8.3 mm.

**2. Validate frame selection and synchronization on real multi-camera data.**
We used five real DTU scans (scan24, 37, 40, 55, 63), each a real 49-camera
capture. Frame selection and its effect on reconstruction are B4 on those
rigs. For synchronization (B12), the images and the motion signal are real: the
production sampler's motion score is computed on consecutive real frames, and a
second camera's scores come from its own perturbed pixels (gain, offset, blur,
sensor noise). What is injected is the clocks, since DTU's cameras have no
timestamps; recovery is scored against the injected offset. Over 1,100 trials the median
absolute error is 12-30 ms per scan against a 40 ms frame interval, the sign is
right in 82-95% of trials and groups form in 83-90%; a minority of trials lock
onto the wrong motion peak, so the mean error (31-145 ms) is larger. This is a
validation of the synchronizer on a real scene's signal, not a phone-rig
experiment, and the paper says so. We did not collect new multi-camera
recordings or test mixed-media payloads, which the ingestion router rejects by
design.

**3. Batched component filter, rerun on several real meshes with variance, and
justification for dropping 78% of cells.**

- *Batched filter.* Implemented (`component_filter="labels"`, on the GPU) and
  checked against the original on the synthetic ladder and the CO3D trace.
- *Real meshes with variance (new, B10).* Both filters timed on the raw meshes
  of scan24, 37, 40, 55 and 63 (1.3-2.6M cells each) with five repeats. The
  batched filter takes 0.30-0.84 s (SD at most 0.09 s). The per-body filter
  finished only on scan37 (1,259 s, with identical output); on the other four
  it exceeded the 30-minute budget. Refinement is deterministic, so the
  variance is timing only.
- *The dropped cells (new, B10).* On DTU, object mode keeps 11-26% of the
  faces and scores lower than the raw mesh, so we do not justify the drop as an
  accuracy gain. With object isolation on, the dominant support plane is cut
  first and the subject's fragments gathered; B10 reports what that keeps and
  how it scores, next to the plain largest-component rule. Isolation cuts the
  support plane (32-60% of faces) and keeps 15-43% of the faces on four scans,
  where its mean F-score is 0.239 against 0.229 raw and 0.217 for object mode.
  On scan37, a pose failure throughout, the plane cut removes the subject and
  the score collapses; we report this and have not tuned it away.

**4. Demonstrate stage replaceability with a second reconstruction backend.**
Reconstruction now sits behind a backend interface
(`spatial_ingestion/reconstruction/backends.py`), selected by the job. DUSt3R,
a different network with a different optimiser and no TSDF fusion, runs behind
it. B13 runs it on the five DTU scans and records, per run, that the stage wrote
the same artifacts and that refinement consumed its mesh unchanged. A contract
test (`tests/test_backends.py`) states what any backend must produce.
Replaceability is shown for the reconstruction stage; we do not claim it for
every stage. DUSt3R met the artifact contract and the refinement hand-off on
all five scans. Its geometry is worse under our protocol (mean F-score 0.069,
all five flagged as pose failures), which shows the swap changes the
measurement; it is evidence of replaceability, not of an equally accurate
backend.

**5. Align the title and abstract with the implemented scope.** The title now
reads "Multi-View 2D Media". The abstract and introduction state that
reconstruction accepts image folders, videos and synchronized captures, and
that single images and live streams are rejected and are future work.

**6. Release code, configurations, seeds, and hardware/thresholds.** The code,
experiment scripts and every result CSV are in the repository; each result row
records its seed, the library versions, the GPU, and the commit. The thresholds
are the named constants in the source (for example the motion thresholds in
`video_sampler.py` and the synchronizer's tolerances in `multi_source.py`).
Repository: https://github.com/AyushK0808/z-shift. Hardware for every
reported run: an NVIDIA GeForce RTX 5060 Laptop GPU (8 GB) under WSL2 on
Windows 11, 512-pixel input; the camera-ready version will carry the same
statement.
