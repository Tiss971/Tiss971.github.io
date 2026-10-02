"""Local browser pilot for the amplified two-layer bullet-time effect.

Run with the repository's validation environment, for example:
  python demo/bullet-time/validate-v2.py --clip C:/path/example.mp4

Artifacts and the full export stay under ignored test/out/. The sample video is
read in place and is never copied into the repository.
"""
from __future__ import annotations

import argparse
import base64
import functools
import http.server
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
INJECT_AFTER = "effect = await createEffect(frame, prediction, signal);"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def decode_data_url(value: str, path: Path):
    header, payload = value.split(',', 1)
    path.write_bytes(base64.b64decode(payload))
    return {"path": path.name, "bytes": path.stat().st_size, "mime": header.split(';')[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--timestamp', type=float, default=6.233333)
    parser.add_argument('--pivot-u', type=float, default=.61)
    parser.add_argument('--pivot-v', type=float, default=.47)
    parser.add_argument('--export', action='store_true', help='Export the full clip only after the visual pilot is accepted.')
    args = parser.parse_args()
    clip = args.clip.resolve(strict=True)
    out = ROOT / 'test' / 'out' / ('bullet-time-v2-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'clip': str(clip), 'timestamp': args.timestamp, 'pivot': {'u': args.pivot_u, 'v': args.pivot_v},
        'real_phone_tested': False, 'pilot_passed': False,
    }
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, requests = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome', headless=True, args=['--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1200, 'height': 1000}, accept_downloads=True)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda req: requests.append({'url': req.url, 'method': req.method}))

            def expose_effect(route):
                response = route.fetch()
                body = response.text()
                if INJECT_AFTER not in body:
                    raise AssertionError('The test hook could not find the effect construction point in app.js.')
                route.fulfill(response=response, body=body.replace(INJECT_AFTER, INJECT_AFTER + '\n    window.__effect = effect;', 1))

            page.route('**/app.js', expose_effect)
            page.goto(f'http://127.0.0.1:{server.server_port}/demo/bullet-time/bullet-time.html')
            page.wait_for_function("() => document.documentElement.dataset.ready === 'true'", timeout=60_000)
            page.set_input_files('#file', str(clip))
            page.wait_for_function("() => !document.querySelector('#prepare').disabled || document.querySelector('#error').textContent", timeout=120_000)
            assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()
            report['metadata'] = page.locator('#metadata').inner_text()
            page.locator('#timestamp').evaluate("(el,t) => {el.value=t;el.dispatchEvent(new Event('input',{bubbles:true}));}", args.timestamp)
            page.wait_for_function("() => !document.querySelector('#prepare').disabled || document.querySelector('#error').textContent", timeout=60_000)
            assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()

            # Regression probe for translucent splat coverage: compare an
            # interior foreground patch at the same nearly-zero camera move in
            # V1 and V2. The V2 background must not bleed through the subject.
            report['synthetic_layer_alpha'] = page.evaluate("""async () => {
              const {createEffect}=await import('./effect.js');
              const w=160,h=100,frame=document.createElement('canvas');frame.width=w;frame.height=h;
              const ctx=frame.getContext('2d');ctx.fillStyle='rgb(25,55,210)';ctx.fillRect(0,0,w,h);
              ctx.fillStyle='rgb(235,45,30)';ctx.fillRect(52,18,56,64);
              const depth=new Float32Array(w*h);depth.fill(.1);
              for(let y=18;y<82;y++)for(let x=52;x<108;x++)depth[y*w+x]=1;
              const signal=new AbortController().signal,e=await createEffect(frame,{grid:frame,depth,width:w,height:h},signal);
              try {
                e.pivot={u:.5,v:.5};e.intensity=.00001;
                e.mode='v1';const base=await e.render(.5),a=base.getContext('2d').getImageData(0,0,w,h).data;
                const mask=new Uint8Array(w*h);for(let y=18;y<82;y++)for(let x=52;x<108;x++)mask[y*w+x]=1;
                e.mode='v2';e.amplitude=2;await e.prepareLayers(mask,signal);
                await e.validateMotion(signal);
                const moved=await e.render(.5),b=moved.getContext('2d').getImageData(0,0,w,h).data;
                let abs=0,n=0,redLoss=0,blueGain=0;
                for(let y=25;y<75;y++)for(let x=59;x<101;x++){
                  const i=(y*w+x)*4;
                  for(let c=0;c<3;c++){abs+=Math.abs(a[i+c]-b[i+c]);n++;}
                  redLoss+=Math.max(0,a[i]-b[i]);blueGain+=Math.max(0,b[i+2]-a[i+2]);
                }
                return {insideMeanAbsoluteError:abs/n,meanRedLoss:redLoss/(42*50),meanBlueGain:blueGain/(42*50),
                  insidePixels:42*50,invalidCorePixels:Array.from(e.layerData.valid).filter(v=>!v).length};
              } finally {e.dispose();}
            }""")
            assert report['synthetic_layer_alpha']['insideMeanAbsoluteError'] < 20, report['synthetic_layer_alpha']
            assert report['synthetic_layer_alpha']['meanBlueGain'] < 20, report['synthetic_layer_alpha']
            started = time.perf_counter()
            page.click('#prepare')
            page.wait_for_function("() => window.__effect || document.querySelector('#error').textContent", timeout=600_000)
            assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()
            report['prepare_wall_seconds'] = time.perf_counter() - started
            page.wait_for_function("() => !document.querySelector('#generate').disabled", timeout=120_000)
            # The app starts a two-second preview after enabling export. Let it
            # finish before calling the exposed effect methods from this probe.
            page.wait_for_timeout(2_200)
            report['effect_metrics'] = page.evaluate("""() => {
              const lines=document.querySelector('#timings').textContent.split('\\n').filter(Boolean),timings={};
              for(const line of lines){const [label,value]=line.split(' : ');const numeric=Number(value?.replace(' s','').replace(',','.'));
                timings[label]=Number.isFinite(numeric)?numeric:value;}
              return {width:__effect.source.width,height:__effect.source.height,splats:__effect.count,timings};
            }""")

            # Save exact source and refined inverse-depth view for visual review.
            source_images = page.evaluate("""() => {
              const {width:w,height:h,rgba,depth,frame}=__effect.source;
              const source=document.createElement('canvas');source.width=w;source.height=h;
              source.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba),w,h),0,0);
              const depthCanvas=document.createElement('canvas');depthCanvas.width=w;depthCanvas.height=h;
              const ctx=depthCanvas.getContext('2d'), image=ctx.createImageData(w,h);
              let lo=Infinity,hi=-Infinity;for(const d of depth){lo=Math.min(lo,d);hi=Math.max(hi,d);}
              for(let i=0;i<depth.length;i++){const v=Math.round(255*(depth[i]-lo)/(hi-lo||1));const j=i*4;image.data[j]=image.data[j+1]=image.data[j+2]=v;image.data[j+3]=255;}
              ctx.putImageData(image,0,0);
              return {source:source.toDataURL('image/png'),depth:depthCanvas.toDataURL('image/png')};
            }""")
            report['source_frame'] = decode_data_url(source_images['source'], out / 'source-frame.png')
            report['refined_depth'] = decode_data_url(source_images['depth'], out / 'refined-depth.png')

            # Pilot the visible football runner as the mask seed. Store both the
            # untouched proposal and a small manual correction as review images.
            page.evaluate("""async ({u,v}) => {
              const e=__effect,signal=new AbortController().signal;
              e.pivot={u,v};e.mode='v1';
              e.__testMask=await e.autoMask({u,v},signal);
            }""", {'u':args.pivot_u,'v':args.pivot_v})
            mask_info = page.evaluate("""() => {
              const e=__effect,{width:w,height:h}=e.source,m=e.__testMask;
              let count=0;for(const x of m)count+=x?1:0;
              const c=document.createElement('canvas');c.width=w;c.height=h;
              const im=c.getContext('2d').createImageData(w,h);
              for(let i=0;i<m.length;i++){const j=i*4;im.data[j]=255;im.data[j+1]=40;im.data[j+2]=20;im.data[j+3]=m[i]?130:0;}
              c.getContext('2d').putImageData(im,0,0);
              return {width:w,height:h,count,fraction:count/m.length,seconds:e.maskSeconds,mask:c.toDataURL('image/png')};
            }""")
            report['automatic_mask'] = {k:v for k,v in mask_info.items() if k != 'mask'}
            report['automatic_mask']['image'] = decode_data_url(mask_info['mask'], out / 'mask-proposal.png')
            assert .002 < report['automatic_mask']['fraction'] < .72, report['automatic_mask']

            # Probe one add and one remove edit without committing them, then
            # validate the displayed/manual mask itself for the layer pilot.
            edit = page.evaluate("""() => {
              const e=__effect,{width:w,height:h}=e.source,original=e.__testMask,edited=original.slice();
              const px=Math.round(e.pivot.u*w),py=Math.round(e.pivot.v*h),radius=Math.max(3,Math.round(Math.min(w,h)*.012));
              let added=0,removed=0;
              for(let y=Math.max(0,py-radius);y<Math.min(h,py+radius);y++)for(let x=Math.max(0,px-radius);x<Math.min(w,px+radius);x++){
                if((x-px)**2+(y-py)**2>radius**2)continue;const i=y*w+x;
                if(original[i]){edited[i]=0;removed++;}else{edited[i]=1;added++;}
              }
              e.__testMask=edited;
              return {added,removed};
            }""")
            report['manual_mask_edit_probe'] = edit

            # Capture the V1 reference first. Then build and capture the two
            # separated layers once; compare 1x and increased travel in-place.
            images = page.evaluate("""async ({trajectories}) => {
              const e=__effect,signal=new AbortController().signal,out={};
              const asPng=async canvas=>canvas.toDataURL('image/png');
              e.mode='v1';e.intensity=.35;e.amplitude=1;
              const v1=await e.render(.5);out.v1=await asPng(v1);
              e.mode='v2';e.amplitude=2;
              const layerStart=performance.now();
              await e.prepareLayers(e.__testMask,signal);
              out.layerBuildSeconds=(performance.now()-layerStart)/1000;
              const layer=e.layerData,{width:w,height:h}=e.source;
              const bg=document.createElement('canvas');bg.width=w;bg.height=h;
              const bgImage=new ImageData(new Uint8ClampedArray(layer.rgba),w,h);
              for(let i=0;i<layer.valid.length;i++)if(!layer.valid[i])bgImage.data[i*4+3]=0;
              bg.getContext('2d').putImageData(bgImage,0,0);
              const validity=document.createElement('canvas');validity.width=w;validity.height=h;
              const validImage=validity.getContext('2d').createImageData(w,h);
              let valid=0,estimated=0,changedKnown=0;
              for(let i=0;i<layer.valid.length;i++){
                const j=i*4;
                if(layer.valid[i]){valid++;validImage.data[j]=30;validImage.data[j+1]=220;validImage.data[j+2]=80;validImage.data[j+3]=layer.estimated?.[i]?210:80;estimated+=layer.estimated?.[i]?1:0;}
                else{validImage.data[j]=240;validImage.data[j+1]=35;validImage.data[j+2]=25;validImage.data[j+3]=125;}
                if(!e.__testMask[i]&&!layer.estimated?.[i]){
                  const a=e.source.rgba,b=layer.rgba;
                  if(a[j]!==b[j]||a[j+1]!==b[j+1]||a[j+2]!==b[j+2])changedKnown++;
                }
              }
              validity.getContext('2d').putImageData(validImage,0,0);
              out.background=await asPng(bg);out.validity=await asPng(validity);
              out.layers={validFraction:valid/layer.valid.length,estimatedPixels:estimated,changedKnownPixels:changedKnown,
                completionSeconds:layer.seconds,validPixels:valid,invalidPixels:layer.valid.length-valid};
              for(const trajectory of trajectories){
                e.trajectory=trajectory;e.amplitude=1;
                const one=await e.validateMotion(signal);out[trajectory+'Validation1x']=one;
                out[trajectory+'V2_1x']=one?.ok?await asPng(await e.render(.5)):null;
                e.amplitude=1.5;
                const amp=await e.validateMotion(signal);out[trajectory+'Validation1_5x']=amp;
                out[trajectory+'V2_1_5x']=amp?.ok?await asPng(await e.render(.5)):null;
              }
              const first=await e.render(0),last=await e.render(1);
              const reference=document.createElement('canvas');reference.width=w;reference.height=h;
              const ctx=reference.getContext('2d');ctx.drawImage(e.source.frame,0,0);
              const src=ctx.getImageData(0,0,w,h).data;
              const a=first.getContext('2d').getImageData(0,0,w,h).data,b=last.getContext('2d').getImageData(0,0,w,h).data;
              out.exactEndpoints=src.every((v,i)=>v===a[i]&&v===b[i]);
              out.stageSeconds={mask:e.maskSeconds,background:e.backgroundSeconds,layerMesh:e.layerSeconds,motion:e.motionSeconds,
                construction:e.seconds,refinement:e.refinementSeconds};
              return out;
            }""", {'trajectories':['lateral','arc','dolly']})

            for key in ('v1','background','validity'):
                report[key] = decode_data_url(images.pop(key), out / ({'v1':'v1-reference.png','background':'completed-background.png','validity':'background-validity.png'}[key]))
            for trajectory in ('lateral','arc','dolly'):
                for suffix in ('V2_1x','V2_1_5x'):
                    key=trajectory+suffix
                    image_data=images.pop(key)
                    report[key]=decode_data_url(image_data,out/(key.lower()+'.png')) if image_data else None
            report['layers']=images.pop('layers')
            report['stage_seconds']=images.pop('stageSeconds')
            report['layer_splat_build_seconds']=images.pop('layerBuildSeconds')
            report['exact_endpoints']=images.pop('exactEndpoints')
            for trajectory in ('lateral','arc','dolly'):
                report.setdefault('motion_validation',{})[trajectory]={
                    '1x':images.pop(trajectory+'Validation1x'),
                    '1.5x':images.pop(trajectory+'Validation1_5x'),
                }
            report['motion_feasibility_passed']=report['exact_endpoints'] and any(
                r['1.5x'].get('ok') and r['1.5x'].get('amplitude',0)>=1.5
                for r in report['motion_validation'].values())
            report['pilot_visual_review_required']=True
            report['pilot_accepted']=False
            report['acceptance_note']='La viabilitÃ© des trajectoires est mesurÃ©e automatiquement. Inspecter le masque et les images avant de valider le pilote visuellement.'
            report['page_errors']=errors
            report['non_get_requests']=[r for r in requests if r['method'] not in ('GET','HEAD')]
            assert not errors, errors
            assert not report['non_get_requests'], report['non_get_requests']

            if args.export and report['motion_feasibility_passed']:
                # UI export uses the already-built two-layer effect and selected
                # timestamp. Set one validated trajectory and preserve amplitude.
                trajectory = next(t for t,r in report['motion_validation'].items()
                                  if r['1.5x'].get('ok') and r['1.5x'].get('amplitude',0)>=1.5)
                report['export_validation'] = page.evaluate("""async ({trajectory})=>{
                  const e=__effect;e.mode='v2';e.trajectory=trajectory;e.amplitude=1.5;
                  return await e.validateMotion(new AbortController().signal);
                }""", {'trajectory':trajectory})
                assert report['export_validation'].get('ok') and report['export_validation'].get('amplitude',0)>=1.5
                page.click('#generate')
                page.wait_for_function("() => !document.querySelector('#download').hidden || document.querySelector('#error').textContent", timeout=600_000)
                assert not page.locator('#error').inner_text(), page.locator('#error').inner_text()
                with page.expect_download() as pending:
                    page.click('#download')
                download=pending.value; export_path=out/download.suggested_filename;download.save_as(export_path)
                cap=cv2.VideoCapture(str(export_path))
                export={'width':cap.get(cv2.CAP_PROP_FRAME_WIDTH),'height':cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        'fps':cap.get(cv2.CAP_PROP_FPS),'frames':cap.get(cv2.CAP_PROP_FRAME_COUNT)}
                cap.release();export['duration']=export['frames']/export['fps']
                report['export']=export
                report['export_path']=str(export_path)
                assert export['fps']==30, export
                assert abs(export['duration']-34.233333)<.06, export
                probe=cv2.VideoCapture(str(export_path))
                probe.release()
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
                import subprocess
                streams=subprocess.run([ffmpeg,'-hide_banner','-i',str(export_path)],capture_output=True,text=True)
                report['export_stream_info']=streams.stderr
                assert 'Audio:' not in streams.stderr
                report['passed']=True
            else:
                report['export_skipped_reason']='--export not requested' if not args.export else 'No trajectory passed the 1.5x feasibility pilot.'
                report['passed']=True
            browser.close()
    except Exception as error:
        report['passed']=False
        report['failure']=str(error)
        raise
    finally:
        server.shutdown()
        (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({'artifacts':str(out),**report},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
