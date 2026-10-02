# Bullet time local

Serve the repository over HTTP (localhost) or HTTPS; opening the HTML with
`file://` does not support the module worker reliably.

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/demo/bullet-time/bullet-time.html`.
The portfolio links to the same page. Drop/select a video, choose or suggest an
instant, prepare the effect, then pick the subject in the original-frame preview.
Choose a lateral move, a small arc or a diagonal approach, adjust its intensity,
and export the silent video. The point can be reset to the image centre at any time.

## Pipeline and limits

- Static site, no backend and no media upload. Libraries and ONNX weights are
  downloaded from their public hosts. Browser caching is best effort; this is
  not a packaged offline application.
- Same Three.js 0.180.0, Spark 2.0.0 and Transformers.js 4.2.0 as the capture demo;
  Mediabunny 1.61.0 handles demuxing, encoding and muxing with WebCodecs.
- Depth Anything V2 Small: FP32 WebGPU when available, Q8 WASM fallback. A worker
  keeps inference off the UI thread and is terminated after inference to release
  model memory before allocating the renderer and encoder.
- Relative inverse depth is converted to bounded camera distances and projected
  RGB surface splats at output resolution (capped at one million splats), separately
  from depth inference at 384 pixels on the long side. This is 2.5D
  reconstruction, not learned multi-view 3DGS. Gaussian coverage is resolved to
  an opaque view on the GPU; only small holes are filled from neighbours in that
  moved view. The original image appears only in the short transition at each end.
  Large camera movements can reveal missing regions or stretched depth boundaries.
- A contour-guidance refinement cleans the depth boundaries before surface
  construction, reducing foreground/background leakage around the selected subject.
  Its optional timing is shown with the other preparation timings.
- Suggestion scans small images at 8 Hz, scores activity and sharpness, and
  excludes half a second around unusually large changes. It is a heuristic and
  does not understand sporting events or perfectly distinguish pans from cuts.
- Input limit: 60 seconds and 100 MiB. Export: 30 fps, even dimensions, no upscale,
  long side at most 1280 pixels. Exactly 60 frames are inserted after the selected
  source frame, following the chosen path around the selected point with eased
  camera motion, zoom and fades. The result has no audio track. Duration is rounded
  up to the nearest 1/30 second, then extended by two seconds.
- MP4/H.264 preferred, WebM/VP9 or VP8 fallback. Unsupported input codecs or missing
  WebGL 2/WebCodecs produce an error before depth inference.
- Decoded canvases are pooled and encode submissions are awaited. The encoded
  output is held in memory (`BufferTarget` + final Blob); memory is bounded by the
  file and frame buffers, not by an array of all decoded frames.
- Cancel closes pending decoding/encoding and releases the depth worker; an
  already prepared preview remains available for retry. Replacing the input also
  disposes its splat renderer and revokes source/result URLs.

## Validation

The existing Python test environment supplies Playwright, Chrome, OpenCV, NumPy
and imageio-ffmpeg. Run:

```powershell
python demo/bullet-time/validate.py --clip "C:/path/example.mp4"
python demo/bullet-time/validate.py --clip "C:/path/example.mp4" --wasm
```

`--smoke` skips real model inference and the full clip export. `--wasm` disables
GPU adapter detection in the **test HTTP response**, without modifying production
files, and exercises real Q8 CPU inference. Artifacts are written only under the
ignored `test/out/`: reports, preview screenshots, locally generated fixtures and
the exported video. The supplied video is read from its original location.

Checks include RGB colour preservation and exact start/end images, motion in the
middle, selection, responsive layout, model/export cancellation and restart,
source-frame parity across the insertion, silent 30 fps output, portrait export,
file limits, invalid containers, missing WebCodecs, and WebM codec negotiation.
The full parity check is intended for a 30 fps example video longer than 10 s.

Validated on local headless Chrome with the supplied 848×480 football clip:
1027 output frames, 34.233 seconds, H.264, no audio. Both WebGPU and WASM inference
passed; screenshots were inspected. Phone-sized layout is tested, but no real
phone, Safari or Firefox has been tested. Timings are machine/cache-specific and
are available in the UI and the generated reports; no memory peak is claimed.

Model: [Depth Anything V2 Small ONNX](https://huggingface.co/onnx-community/depth-anything-v2-small)
(Apache-2.0). Media API: [Mediabunny](https://mediabunny.dev/guide/quick-start).
