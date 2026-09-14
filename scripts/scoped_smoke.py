"""Scoped FFT train/base/latest/pinned smoke test. Allocates 4+1+1 H100s."""
import json
import math
from pathlib import Path
import time
import modal
import tinker
from tinker import types
import lilo
from lilo.engines import qwen3_5_4b_full_64k


def main():
    report = {"started_at": time.time(), "events": []}
    output = Path('/tmp/lilo-scoped-smoke.json')
    def event(name, **values):
        report['events'].append(dict(name=name, at=time.time(), **values))
        output.write_text(json.dumps(report, indent=2))
        print(name, values, flush=True)
    try:
        engine = qwen3_5_4b_full_64k()
        with modal.enable_output(), lilo.run(
            engine=engine, name='lilo-scoped-smoke', warm=True, max_trainers=1,
            latest=lilo.Pool(min_containers=1, max_containers=1, scaledown_window=60),
        ) as (url, api_key):
            event('ready', url=url)
            service = tinker.ServiceClient(base_url=url, api_key=api_key)
            trainer = lilo.create_full_training_client(service, engine.model)
            event('training_client', model_id=trainer.model_id)
            tokenizer = trainer.get_tokenizer()
            tokens = tokenizer.encode('The capital of France is Paris.')
            datum = types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={'target_tokens': tokens[1:], 'weights': [1.]*(len(tokens)-1)})
            result = trainer.forward_backward([datum], 'cross_entropy').result(timeout=1800)
            assert result.loss_fn_outputs
            assert all(math.isfinite(v) for v in result.metrics.values() if isinstance(v, (float, int)))
            event('forward_backward', metrics=result.metrics)
            result = trainer.optim_step(types.AdamParams(learning_rate=1e-5)).result(timeout=1800)
            event('optim_step', metrics=result.metrics)
            prompt = types.ModelInput.from_ints(tokenizer.encode('The capital of France is'))
            def sample(label, client):
                response = client.sample(prompt=prompt, num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=16, temperature=0)).result(timeout=1800)
                seq = response.sequences[0]
                assert seq.tokens and len(seq.logprobs) == len(seq.tokens)
                assert all(math.isfinite(p) for p in seq.logprobs)
                event(label, text=tokenizer.decode(seq.tokens), tokens=len(seq.tokens))
            peer = tinker.ServiceClient(base_url=url, api_key=api_key)
            sample('base_sample', peer.create_sampling_client(base_model=engine.model))
            sample('latest_sample', trainer.save_weights_and_get_sampling_client())
            saved = trainer.save_weights_for_sampler('smoke-pinned').result(timeout=1800)
            event('pinned_publication', path=saved.path)
            sample('pinned_sample', peer.create_sampling_client(model_path=saved.path))
            event('body_complete')
        event('context_exited')
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        event('error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at'] = time.time()
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
