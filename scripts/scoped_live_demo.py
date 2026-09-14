"""Run a warm scoped deployment, sample Qwen 4B, and hold it for inspection."""
import json
from pathlib import Path
import time

import modal
import tinker
from tinker import types
from transformers import AutoTokenizer

import lilo
from lilo.engines import qwen3_5_4b_full_64k

STATUS = Path('/tmp/lilo-scoped-live.json')
STOP = Path('/tmp/lilo-scoped-live.stop')


def main():
    report = {'events': []}

    def event(name, **values):
        report['events'].append({'name': name, 'at': time.time(), **values})
        STATUS.write_text(json.dumps(report, indent=2))
        print(name, values, flush=True)

    STOP.unlink(missing_ok=True)
    engine = qwen3_5_4b_full_64k()
    with modal.enable_output(), lilo.run(
        engine=engine,
        warm=True,
        max_trainers=1,
        latest=lilo.Pool(min_containers=1, max_containers=1),
    ) as (url, api_key):
        event('ready', url=url)
        service = tinker.ServiceClient(base_url=url, api_key=api_key)
        sampler = service.create_sampling_client(base_model=engine.model)
        tokenizer = AutoTokenizer.from_pretrained(engine.model)
        prompt = types.ModelInput.from_ints(tokenizer.encode('The capital of France is'))
        response = sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=32, temperature=0),
        ).result(timeout=1800)
        sequence = response.sequences[0]
        assert sequence.tokens
        event('sample', text=tokenizer.decode(sequence.tokens), tokens=len(sequence.tokens))
        deadline = time.time() + 3600
        event('holding', expires_at=deadline, stop_file=str(STOP))
        while time.time() < deadline and not STOP.exists():
            time.sleep(2)
        event('leaving_context')
    event('stopped')


if __name__ == '__main__':
    main()
