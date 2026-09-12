"""Run untrusted solutions in isolated Modal sandboxes, with no expected outputs inside."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging

import modal

# Only inputs and submitted source enter the sandbox. Output comparison lives outside.
RUNNER = r"""
import base64,hashlib,json,os,resource,signal,subprocess,sys,tempfile
payload=json.loads(base64.b64decode(sys.stdin.buffer.read()))
open('/tmp/solution.py','w').write(payload['code'])
os.chmod('/tmp/solution.py',0o444)
def limits():
 os.setsid()
 resource.setrlimit(resource.RLIMIT_CPU,(3,3))
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
 resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
 resource.setrlimit(resource.RLIMIT_NPROC,(32,32))
 os.setgid(65534);os.setuid(65534)
results=[]
for inp in payload['inputs']:
 with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
  p=subprocess.Popen([sys.executable,'-I','/tmp/solution.py'],stdin=subprocess.PIPE,stdout=output,stderr=error,preexec_fn=limits,cwd='/tmp',env={'PATH':'/usr/bin:/usr/local/bin'})
  try:
   p.communicate(inp.encode(),timeout=5)
   output.seek(0);out=output.read(2*1024**2+1)
   ok=p.returncode==0 and len(out)<=2*1024**2
   digest=hashlib.sha256(' '.join(out.decode('utf-8',errors='replace').split()).encode()).hexdigest() if ok else None
   results.append({'ok':ok,'digest':digest})
   if payload.get('capture'):
    error.seek(0)
    results[-1].update(stdout=out[:4096].decode('utf-8',errors='replace'),stderr=error.read(4096).decode('utf-8',errors='replace'),returncode=p.returncode,output_truncated=len(out)>4096)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.wait();results.append({'ok':False,'digest':None,'error':'Execution exceeded 5 seconds'})
  finally:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
 if not results[-1]['ok']:
  break
print(json.dumps(results))
"""


def output_digest(text: str) -> str:
    """Hash canonical whitespace-separated tokens; expected text stays outside."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def verdict(results: list[dict], tests: list[dict]) -> dict:
    if not tests:
        raise ValueError("Refusing to reward a problem without tests")
    if not results or len(results) > len(tests):
        raise RuntimeError("Judge returned invalid test results")
    if len(results) < len(tests) and results[-1]["ok"]:
        raise RuntimeError("Judge returned incomplete successful test results")
    passed = sum(
        result["ok"] and result["digest"] == output_digest(test["output"])
        for result, test in zip(results, tests[: len(results)], strict=True)
    )
    return {
        "passed": passed == len(tests),
        "tests_passed": passed,
        "tests_run": len(results),
        "tests_total": len(tests),
    }


async def judge(code: str, tests: list[dict], app: modal.App) -> dict:
    if not code.strip():
        return {
            "passed": False,
            "tests_passed": 0,
            "tests_run": 0,
            "tests_total": len(tests),
        }
    if not tests:
        raise ValueError("Refusing to reward a problem without tests")
    payload = base64.b64encode(
        json.dumps({"code": code, "inputs": [t["input"] for t in tests]}).encode()
    ).decode()
    sandbox = await modal.Sandbox.create.aio(
        "python",
        "-c",
        RUNNER,
        app=app,
        image=modal.Image.debian_slim(python_version="3.11"),
        cpu=1,
        memory=1024,
        timeout=5 * len(tests) + 60,
        block_network=True,
    )
    return await collect_result(
        sandbox, tests, timeout=5 * len(tests) + 120, payload=payload
    )


async def collect_result(sandbox, tests, *, timeout, cleanup_timeout=30, payload=None):
    """Bound remote output and cleanup waits independently of Sandbox's timeout."""
    try:
        async with asyncio.timeout(timeout):
            if payload is not None:
                sandbox.stdin.write(payload)
                sandbox.stdin.write_eof()
                await sandbox.stdin.drain.aio()
            stdout = await sandbox.stdout.read.aio()
            await sandbox.wait.aio()
            if sandbox.returncode != 0:
                raise RuntimeError(
                    f"Judge supervisor exited {sandbox.returncode}: {(await sandbox.stderr.read.aio())[-1000:]}"
                )
            results = json.loads(stdout)
            return verdict(results, tests)
    finally:
        try:
            async with asyncio.timeout(cleanup_timeout):
                await sandbox.terminate.aio()
        except TimeoutError:
            logging.getLogger(__name__).warning(
                "Judge sandbox cleanup timed out: %s", sandbox.object_id
            )
