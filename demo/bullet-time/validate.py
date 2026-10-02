"""Browser pilot and full export. Artifacts stay in ignored test/out/.

python demo/bullet-time/validate.py --clip "C:/path/video.mp4"
Requires the existing test environment: playwright, imageio_ffmpeg, numpy, cv2.
No real phone is emulated as a substitute for hardware validation.
"""
import argparse
import functools
import http.server
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def ready(page, selector, timeout=300_000):
    page.wait_for_function(
        "selector => !document.querySelector(selector).disabled || document.querySelector('#error').textContent.length > 0",
        arg=selector, timeout=timeout,
    )
    error = page.locator('#error').inner_text()
    assert not error, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Skip model and full export; validate media/UI and synthetic splats.')
    parser.add_argument('--wasm', action='store_true', help='Exercise real CPU inference with GPU adapter detection disabled in the test response.')
    args = parser.parse_args()
    out = ROOT / 'test' / 'out' / ('bullet-time-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    report = {'clip': str(args.clip), 'real_phone_tested': False, 'smoke': args.smoke, 'forced_wasm': args.wasm}
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, requests = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome', headless=True, args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1200, 'height': 900}, accept_downloads=True)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda req: requests.append({'url': req.url, 'method': req.method}))
            if args.wasm:
                def without_gpu(route):
                    response = route.fetch()
                    text = response.text()
                    original = 'gpu = !!(await navigator.gpu?.requestAdapter())'
                    assert original in text
                    route.fulfill(response=response, body=text.replace(original, 'gpu = false'))
                page.route('**/depth-worker.js', without_gpu)
            page.goto(f'http://127.0.0.1:{server.server_port}/demo/bullet-time/bullet-time.html')
            page.wait_for_function("() => document.querySelector('#file') && !document.querySelector('#file').disabled")
            # Wait for actual module graph rather than merely the static HTML.
            page.wait_for_function("() => document.documentElement.dataset.ready === 'true'", timeout=60_000)
            page.set_input_files('#file', str(args.clip))
            ready(page, '#prepare')
            report['metadata'] = page.locator('#metadata').inner_text()
            page.screenshot(path=str(out / '01-loaded.png'), full_page=True)
            # Manual selection, then automatic suggestion.
            page.locator('#timestamp').evaluate("el => { el.value = 10; el.dispatchEvent(new Event('input', {bubbles:true})); }")
            page.wait_for_timeout(300)
            assert '10,00' in page.locator('#time-label').inner_text()
            page.click('#suggest')
            ready(page, '#prepare')
            report['suggested_time'] = float(page.locator('#timestamp').input_value())
            # Responsive layout is only a viewport check, not a real phone test.
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(out / '02-mobile-layout.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.set_viewport_size({'width': 1200, 'height': 900})
            report['depth_refinement'] = page.evaluate("""async () => {
              const {refineDepth}=await import('./depth-refinement.js');
              const rgba=new Uint8ClampedArray(64*48*4);
              for(let y=0;y<48;y++) for(let x=0;x<64;x++) {
                const i=(y*64+x)*4;
                rgba.set(x<31?[240,50,40,255]:[30,60,220,255],i);
              }
              const depth=new Float32Array(16*12);
              for(let y=0;y<12;y++) for(let x=0;x<16;x++) depth[y*16+x]=x<8?.2:.9;
              const signal=new AbortController().signal;
              const refined=await refineDepth(depth,16,12,rgba,64,48,signal);
              const step=refined.values[24*64+31];
              depth.fill(.5);
              const flat=await refineDepth(depth,16,12,rgba,64,48,signal);
              const controller=new AbortController();controller.abort();let aborted=false;
              try {await refineDepth(depth,16,12,rgba,64,48,controller.signal);}catch(e){aborted=e.name==='AbortError';}
              return {step,baseline:.2*.625+.9*.375,flat:flat.values.every(v=>v===.5),
                flatMask:flat.edgeMask.every(v=>v===0),bounded:refined.values.every(v=>v>=.199999&&v<=.900001),aborted};
            }""")
            refinement = report['depth_refinement']
            assert refinement['flat'] and refinement['flatMask'] and refinement['bounded'] and refinement['aborted']
            assert abs(refinement['step']-.9) < abs(refinement['baseline']-.9)
            # Synthetic depth tests geometry, colour and boundaries without AI mocking.
            report['synthetic_effect'] = page.evaluate("""async () => {
              const { createEffect } = await import('./effect.js');
              const frame = document.createElement('canvas'); frame.width=240; frame.height=320;
              const ctx=frame.getContext('2d'); ctx.fillStyle='#2060a0'; ctx.fillRect(0,0,240,320);
              ctx.fillStyle='#e05030'; ctx.fillRect(70,90,100,160);
              const depth=new Float32Array(240*320); depth.fill(.1);
              for(let y=90;y<250;y++) for(let x=70;x<170;x++) depth[y*240+x]=1;
              const e=await createEffect(frame,{grid:frame,depth,width:240,height:320},new AbortController().signal);
              const first=await e.render(0); const a=first.getContext('2d').getImageData(0,0,240,320).data.slice();
              const middle=await e.render(.5); const b=middle.getContext('2d').getImageData(0,0,240,320).data.slice();
              const last=await e.render(1); const c=last.getContext('2d').getImageData(0,0,240,320).data;
              const same=a.every((v,i)=>v===c[i]); const changed=a.some((v,i)=>v!==b[i]);
              const rgb=[...b.slice((30*240+30)*4,(30*240+30)*4+3)];
              const trajectories={};
              for(const trajectory of ['lateral','arc','dolly']) {
                e.trajectory=trajectory;e.pivot={u:.3,v:.4};
                await e.render(.5);trajectories[trajectory]=e.lastView;
                const endpoint=await e.render(1);
                const endpointPixels=endpoint.getContext('2d').getImageData(0,0,240,320).data;
                trajectories[trajectory].exactEnd=a.every((v,i)=>v===endpointPixels[i]);
              }
              e.trajectory='lateral';e.pivot={u:.5,v:.5};
              const steady=await e.render(.5);
              const beforeEdit=steady.getContext('2d').getImageData(0,0,240,320).data.slice();
              // Changing the unmoved source after building the splats must have
              // no influence on the middle of the camera move (ghosting regression).
              ctx.fillStyle='#00ff00';ctx.fillRect(0,0,240,320);
              const moved=await e.render(.5);
              const after=moved.getContext('2d').getImageData(0,0,240,320).data;
              const independentSource=beforeEdit.every((v,i)=>v===after[i]);
              const opaque=after.every((v,i)=>i%4!==3||v===255);
              const count=e.count; e.dispose(); return {same,changed,count,rgb,independentSource,opaque,trajectories};
            }""")
            assert report['synthetic_effect']['same'] and report['synthetic_effect']['changed']
            assert report['synthetic_effect']['independentSource'] and report['synthetic_effect']['opaque']
            for trajectory, view in report['synthetic_effect']['trajectories'].items():
                assert view['exactEnd']
                assert abs(view['pivot']['u']-.3) < 1e-6 and abs(view['pivot']['v']-.4) < 1e-6
                assert view['position'][0] > 0
                if trajectory == 'lateral':
                    assert view['position'][1:] == [0, 0]
                elif trajectory == 'arc':
                    initial_radius = sum(n*n for n in view['target'])
                    moved_radius = sum((a-b)**2 for a,b in zip(view['target'],view['position']))
                    assert abs(initial_radius-moved_radius) < 1e-6
                else:
                    assert view['position'][1] > 0 and view['position'][2] < 0
            assert max(abs(a-b) for a,b in zip(report['synthetic_effect']['rgb'],[32,96,160])) < 12
            # Fine RGB detail must survive a nearly stationary render even when
            # inference uses a much smaller depth grid.
            report['fine_detail'] = page.evaluate("""async () => {
              const {createEffect}=await import('./effect.js');
              const frame=document.createElement('canvas'); frame.width=848; frame.height=480;
              const ctx=frame.getContext('2d');
              for(let x=0;x<848;x++) {ctx.fillStyle=Math.floor(x/2)%2?'white':'black';ctx.fillRect(x,0,1,480);}
              const original=ctx.getImageData(0,0,848,480).data;
              const grid=document.createElement('canvas');grid.width=384;grid.height=217;
              const depth=new Float32Array(384*217);depth.fill(.5);
              const e=await createEffect(frame,{grid,depth,width:384,height:217},new AbortController().signal);
              e.intensity=0;
              let rendered=await e.render(.5);
              const zero=rendered.getContext('2d').getImageData(0,0,848,480).data;
              const exactZero=original.every((v,i)=>v===zero[i]);
              e.intensity=.00001;
              rendered=await e.render(.5);
              const small=rendered.getContext('2d').getImageData(0,0,848,480).data;
              let error=0,originalEdges=0,renderedEdges=0,n=0;
              for(let y=20;y<460;y++) for(let x=20;x<827;x++) {
                const i=(y*848+x)*4;
                error+=Math.abs(original[i]-small[i]);
                originalEdges+=Math.abs(original[i+4]-original[i]);
                renderedEdges+=Math.abs(small[i+4]-small[i]);n++;
              }
              const result={exactZero,count:e.count,mae:error/n,contrastRatio:renderedEdges/originalEdges};
              e.dispose();return result;
            }""")
            assert report['fine_detail']['exactZero']
            assert report['fine_detail']['count'] == 848 * 480
            assert report['fine_detail']['contrastRatio'] > .7, report['fine_detail']
            assert report['fine_detail']['mae'] < 25, report['fine_detail']
            if not args.smoke:
                # Stop a model load/inference, then restart with a fresh worker.
                page.click('#prepare')
                page.wait_for_timeout(200)
                page.click('#cancel')
                ready(page, '#prepare')
                start = time.perf_counter()
                page.click('#prepare')
                ready(page, '#generate', timeout=600_000)
                report['pilot_wall_seconds'] = time.perf_counter() - start
                page.wait_for_timeout(900)
                page.screenshot(path=str(out / '03-pilot.png'), full_page=True)
                page.locator('#preview').screenshot(path=str(out / 'pilot-frame.png'))
                page.wait_for_timeout(1500)
                # Path and pivot changes reuse the same geometry/model result.
                for trajectory in ['arc', 'dolly']:
                    page.select_option('#trajectory', trajectory)
                    page.wait_for_timeout(900)
                    page.locator('#preview').screenshot(path=str(out / f'path-{trajectory}.png'))
                    page.wait_for_timeout(1300)
                page.click('#pivot-pick')
                page.wait_for_selector('#pivot-overlay:not([hidden])')
                assert page.locator('#generate').is_disabled()
                bounds = page.locator('#preview').bounding_box()
                page.mouse.click(bounds['x'] + .4 * bounds['width'], bounds['y'] + .45 * bounds['height'])
                page.wait_for_function("() => document.querySelector('#pivot-overlay').hidden")
                pivot = page.locator('#pivot-label').evaluate("el=>({u:Number(el.dataset.u),v:Number(el.dataset.v)})")
                assert abs(pivot['u']-.4) < .01 and abs(pivot['v']-.45) < .01, pivot
                report['picked_pivot'] = pivot
                page.wait_for_timeout(900)
                page.screenshot(path=str(out / 'pivot-dolly.png'), full_page=True)
                page.wait_for_timeout(1300)
                page.click('#pivot-pick')
                page.wait_for_selector('#pivot-overlay:not([hidden])')
                page.keyboard.press('Escape')
                page.wait_for_function("() => document.querySelector('#pivot-overlay').hidden")
                page.click('#pivot-reset')
                page.wait_for_function("() => Number(document.querySelector('#pivot-label').dataset.u) === .5 && Number(document.querySelector('#pivot-label').dataset.v) === .5")
                page.wait_for_timeout(2200)
                report['export_trajectory'] = page.locator('#trajectory').input_value()
                # Cancel an export and ensure a fresh export can run.
                page.click('#generate')
                page.wait_for_selector('#cancel:not([hidden])')
                page.wait_for_function("() => document.querySelector('#progress').value > 10", timeout=30_000)
                page.click('#cancel')
                ready(page, '#generate')
                start = time.perf_counter()
                page.click('#generate')
                page.wait_for_function("() => !document.querySelector('#download').hidden || document.querySelector('#error').textContent.length > 0", timeout=600_000)
                assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()
                report['export_wall_seconds'] = time.perf_counter() - start
                report['timings'] = page.locator('#timings').text_content()
                if args.wasm:
                    assert 'WASM' in report['timings']
                with page.expect_download() as pending:
                    page.click('#download')
                downloaded = pending.value
                export_path = out / downloaded.suggested_filename
                downloaded.save_as(export_path)
                page.screenshot(path=str(out / '04-result.png'), full_page=True)
                cap = cv2.VideoCapture(str(export_path))
                report['export'] = {'width': cap.get(3), 'height': cap.get(4), 'fps': cap.get(5), 'frames': cap.get(7)}
                cap.release()
                report['export']['duration'] = report['export']['frames'] / report['export']['fps']
                original = cv2.VideoCapture(str(args.clip))
                duration = original.get(7) / original.get(5)
                original.release()
                assert abs(report['export']['duration'] - (duration + 2)) < .05
                assert report['export']['fps'] == 30
                original = cv2.VideoCapture(str(args.clip))
                exported = cv2.VideoCapture(str(export_path))
                selected = round(report['suggested_time'] * 30)
                last_index = round(duration * 30) - 1
                comparisons = []
                for index in sorted(set([0, max(0, selected - 1), selected, min(last_index, selected + 1), last_index])):
                    original.set(cv2.CAP_PROP_POS_FRAMES, index)
                    exported.set(cv2.CAP_PROP_POS_FRAMES, index + (60 if index > selected else 0))
                    ok_a, a = original.read(); ok_b, b = exported.read()
                    assert ok_a and ok_b
                    error = float(np.abs(a.astype(float) - b.astype(float)).mean())
                    comparisons.append({'source_frame': index, 'mae': error})
                    assert error < 12, comparisons
                original.set(cv2.CAP_PROP_POS_FRAMES, selected)
                _, original_frame = original.read()
                cv2.imwrite(str(out / 'original-frame.png'), original_frame)
                exported.set(cv2.CAP_PROP_POS_FRAMES, selected + 31)
                ok, mid = exported.read(); assert ok
                cv2.imwrite(str(out / 'export-effect-midpoint.png'), mid)
                report['effect_midpoint_mae'] = float(np.abs(original_frame.astype(float) - mid.astype(float)).mean())
                assert report['effect_midpoint_mae'] > .5
                original.release(); exported.release()
                report['source_resume_comparisons'] = comparisons
                probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(export_path)], capture_output=True, text=True)
                report['ffmpeg_stream_info'] = probe.stderr
                assert 'Audio:' not in probe.stderr
            # Replacement and invalid input should recover without stale UI/resources.
            page.set_input_files('#file', str(args.clip)); ready(page, '#prepare')
            assert page.locator('#download').is_hidden()
            report['size_limit'] = page.evaluate("""async () => {
              const {openVideo}=await import('./media.js');
              try {await openVideo({size:101*1024*1024},new AbortController().signal); return null;}
              catch(e) {return e.message;}
            }""")
            assert '100 Mo' in report['size_limit']
            # Generate small vertical and overlong fixtures locally, never publish them.
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            vertical = out / 'vertical.mp4'
            overlong = out / 'overlong.mp4'
            for dest, size, length in [(vertical, '240x320', '2'), (overlong, '32x32', '61')]:
                subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i', f'testsrc2=size={size}:rate=30:duration={length}', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(dest)], check=True, capture_output=True)
            page.set_input_files('#file', str(overlong))
            page.wait_for_function("() => document.querySelector('#error').textContent.includes('60 secondes')")
            page.set_input_files('#file', str(vertical)); ready(page, '#prepare')
            assert '240 × 320' in page.locator('#metadata').inner_text()
            # A portrait export using a controlled effect exercises the real media path.
            report['vertical_export'] = page.evaluate("""async () => {
              const m=await import('./media.js');
              const file=await (await fetch('/'+%s)).blob();
              const signal=new AbortController().signal;
              const media=await m.openVideo(file,signal);
              try {
                const canvas=await m.frameAt(media,.5,signal);
                const result=await m.exportVideo(media,{time:.5,effect:{render:async()=>canvas},signal});
                const input=await m.openVideo(result.blob,signal);
                try {return {width:input.width,height:input.height,duration:input.duration};}
                finally {input.dispose();}
              } finally {media.dispose();}
            }""" % json.dumps(vertical.relative_to(ROOT).as_posix()))
            assert report['vertical_export'] == {'width':240,'height':320,'duration':4}
            report['unsupported_encoder_error'] = page.evaluate("""async () => {
              const {chooseEncoding}=await import('./media.js');
              const original=VideoEncoder.isConfigSupported;
              VideoEncoder.isConfigSupported=async config=>({supported:false,config});
              try {await chooseEncoding(242,322); return null;}
              catch(e) {return e.message;}
              finally {VideoEncoder.isConfigSupported=original;}
            }""")
            assert report['unsupported_encoder_error']
            report['webm_fallback'] = page.evaluate("""async () => {
              const {chooseEncoding}=await import('./media.js');
              const original=VideoEncoder.isConfigSupported;
              VideoEncoder.isConfigSupported=async config=>config.codec.startsWith('avc')?{supported:false,config}:original.call(VideoEncoder,config);
              try {const e=await chooseEncoding(244,324); return {codec:e.codec,extension:e.extension};}
              finally {VideoEncoder.isConfigSupported=original;}
            }""")
            assert report['webm_fallback']['extension'] == '.webm'
            report['missing_webcodecs_error'] = page.evaluate("""async () => {
              const {openVideo}=await import('./media.js');
              const original=window.VideoDecoder; window.VideoDecoder=undefined;
              try {await openVideo(new Blob(),new AbortController().signal); return null;}
              catch(e) {return e.message;}
              finally {window.VideoDecoder=original;}
            }""")
            assert report['missing_webcodecs_error']
            page.set_input_files('#file', {'name':'invalid.mp4','mimeType':'video/mp4','buffer':b'not a video'})
            page.wait_for_function("() => document.querySelector('#error').textContent.length > 0")
            report['invalid_file_error'] = page.locator('#error').inner_text()
            report['page_errors'] = errors
            report['non_get_requests'] = [r for r in requests if r['method'] not in ('GET', 'HEAD')]
            assert not errors, errors
            assert not report['non_get_requests'], report['non_get_requests']
            browser.close()
            report['passed'] = True
    except Exception as error:
        report['passed'] = False; report['failure'] = str(error)
        raise
    finally:
        server.shutdown()
        (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'artifacts': str(out), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
