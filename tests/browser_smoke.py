"""Standalone Chromium smoke test against running Operator Atlas backend.

Usage: ATLAS_SMOKE_PORT=8765 python tests/browser_smoke.py
No public networking, no test content uploaded anywhere outside local service.
"""
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent.parent
PORT=int(os.environ.get('ATLAS_SMOKE_PORT','8765'))

def bridge(path,opts):
    url=f'http://127.0.0.1:{PORT}'+path
    body=(opts.get('body') or '').encode() if opts else None
    req=Request(url,data=body,method=opts.get('method','GET') if opts else 'GET',
                headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=45) as f:return {'ok':True,'status':f.status,'data':json.loads(f.read())}
    except HTTPError as exc:
        return {'ok':False,'status':exc.code,'data':json.loads(exc.read())}

def run():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                  args=['--no-sandbox','--disable-dev-shm-usage'])
        for width,height,label in [(1440,900,'desktop'),(390,844,'mobile')]:
            page=browser.new_page(viewport={'width':width,'height':height})
            errors=[]
            page.on('pageerror',lambda err:errors.append(str(err)))
            page.set_content((ROOT/'web/index.html').read_text())
            page.add_style_tag(content=(ROOT/'web/style.css').read_text())
            page.expose_function('pyBackend',bridge)
            page.evaluate('''() => {window.fetch=async function(path,opts={}) {
                const r=await window.pyBackend(path,opts);
                return {ok:r.ok,status:r.status,json:async()=>r.data};}}''')
            page.add_script_tag(content=(ROOT/'web/app.js').read_text())
            page.evaluate("window.dispatchEvent(new Event('DOMContentLoaded'))")
            page.wait_for_function("document.querySelectorAll('.graph-node').length===16",timeout=20000)
            page.locator('#graph-run').click()
            page.wait_for_function("document.querySelector('#graph-result').textContent.includes('The mathematical atlas')",timeout=15000)
            # Exercise the actual typed response pipeline before and after import.
            page.locator('[data-tab=answer]').click()
            page.locator('#answer-run').click()
            page.wait_for_function("document.querySelector('#answer-metadata').textContent.includes('Route')",timeout=15000)
            # Existing evidence is allowed; deterministic before/after is tested in a fresh fixture DB.
            page.locator('#answer-example').click()
            page.wait_for_function("document.querySelector('#answer-import-status').textContent.includes('Imported')",timeout=15000)
            page.locator('#answer-run').click()
            page.wait_for_function("document.querySelector('#answer-output').textContent.includes('KAPPA-47')",timeout=15000)
            assert 'source_excerpt' in page.locator('#answer-metadata').inner_text()
            page.locator('[data-tab=memory]').click()
            page.locator('#document-file').set_input_files({'name':'sample_notes.md','mimeType':'text/markdown',
                'buffer':b'Copper is electrically conductive. Scientific observations require evidence and reproducible provenance.'})
            page.locator('#import-file').click()
            page.wait_for_function("document.querySelector('#file-status').textContent.includes('Preserved')",timeout=15000)
            page.locator('#search-text').fill('electrically conductive')
            page.locator('#ask-sources').click()
            page.wait_for_function('document.querySelector("#search-results").textContent.includes("Copper")',timeout=15000)
            # Public mathematical learning is genuinely backed by imported CSV source.
            page.locator('#document-file').set_input_files(str(ROOT/'examples'/'fixtures'/'calibration.csv'))
            page.locator('#import-file').click()
            page.wait_for_function("document.querySelector('#file-status').textContent.includes('Preserved')",timeout=15000)
            page.locator('[data-tab=equations]').click()
            page.locator('#field-refresh').click()
            page.wait_for_function("document.querySelector('#field-source').textContent.includes('calibration.csv')",timeout=15000)
            first_csv_id=page.locator('#field-source option').filter(has_text='calibration.csv').last.get_attribute('value')
            page.locator('#field-source').select_option(value=first_csv_id)
            page.locator('#field-columns').fill('coordinate,reading')
            page.locator('#field-learn').click()
            page.wait_for_function("document.querySelector('#field-fit-result').textContent.includes('validation RMSE')",timeout=15000)
            page.locator('#field-predict').click()
            page.wait_for_function("document.querySelector('#field-prediction').textContent.includes('Predicted')",timeout=15000)
            # Learn the second real equation and evaluate an actual composed path.
            page.locator('[data-tab=memory]').click()
            page.locator('#document-file').set_input_files(str(ROOT/'examples'/'fixtures'/'secondary_calibration.csv'))
            page.locator('#import-file').click()
            page.wait_for_function("document.querySelector('#file-status').textContent.includes('Preserved')",timeout=15000)
            page.locator('[data-tab=equations]').click()
            page.locator('#field-refresh').click()
            second_csv_id=page.locator('#field-source option').filter(has_text='secondary_calibration.csv').first.get_attribute('value')
            page.locator('#field-source').select_option(value=second_csv_id)
            page.locator('#field-columns').fill('measurement,calibrated')
            page.locator('#field-learn').click()
            page.wait_for_function("document.querySelector('#field-fit-result').textContent.includes('validation RMSE')",timeout=15000)
            page.locator('#field-refresh').click()
            page.wait_for_function("document.querySelectorAll('#compose-first option').length>=2",timeout=15000)
            opts=page.locator('#compose-first option')
            original_id=opts.filter(has_text='calibration.csv').last.get_attribute('value')
            secondary_id=opts.filter(has_text='secondary_calibration.csv').first.get_attribute('value')
            page.locator('#compose-first').select_option(value=original_id)
            page.locator('#compose-second').select_option(value=secondary_id)
            page.locator('#compose-run').click()
            page.wait_for_function("document.querySelector('#compose-output').textContent.includes('Computed')",timeout=15000)
            assert 'source #' in page.locator('#compose-output').inner_text().lower()
            page.locator('#field-x').fill('1000')
            page.locator('#field-predict').click()
            page.wait_for_function("document.querySelector('#field-prediction').textContent.includes('Refused')",timeout=15000)
            page.locator('[data-tab=diagnostics]').click();page.locator('#refresh-events').click()
            page.wait_for_function('document.querySelectorAll("#event-list .trace-row").length>0',timeout=15000)
            page.locator('[data-tab=experiments]').click();page.locator('#run-probe').click()
            page.wait_for_function('document.querySelector("#probe-results").textContent.includes("Original validation loss")',timeout=15000)
            # The GUI controls the same backend policy as the CLI. Re-run on the
            # latency-aware setting; exact outcome depends on the host CPU.
            page.locator('#probe-objective').select_option('latency_guarded')
            page.locator('#run-probe').click()
            page.wait_for_function('document.querySelector("#probe-results").textContent.includes("Selection decision")',timeout=20000)
            assert 'applied representations' in page.locator('#probe-results').inner_text().lower()
            page.locator('[data-tab=graph]').click()
            page.screenshot(path=str(ROOT/f'graph-{label}.png'),full_page=True)
            assert not errors,errors
            print(label, 'PASS (graph, neural generation, file import, evidence retrieval, CSV equation learning, range refusal, probe, diagnostics)')
            page.close()
        browser.close()

if __name__=='__main__':run()
