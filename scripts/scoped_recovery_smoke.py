"""Exercise explicit trainer replacement and latest/pinned sampling on Modal."""
import json
import math
import os
from pathlib import Path
import time
import modal
import tinker
from tinker import types
import lilo
from lilo.engines import qwen3_5_4b_full_64k


def main():
    report = {"started_at": time.time(), "events": []}
    output = Path('/tmp/lilo-scoped-recovery.json')
    def event(name, **values):
        report['events'].append(dict(name=name, at=time.time(), **values))
        output.write_text(json.dumps(report, indent=2))
        print(name, values, flush=True)
    # Capture the parent ID to target fault injection, without changing the API.
    import lilo.providers.modal.scoped as scoped
    original_build = scoped.build_app
    resources = {}
    def capture_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        resources['app'] = result[0]
        resources['registry'] = args[2]
        resources['manage'] = result[2]
        return result
    scoped.build_app = capture_build
    try:
        engine = qwen3_5_4b_full_64k()
        with modal.enable_output(), lilo.run(
            engine=engine, name='lilo-recovery-test', warm=True,
            latest=lilo.Pool(min_containers=1, max_containers=1, scaledown_window=60),
        ) as (url, api_key):
            event('ready', url=url)
            service = tinker.ServiceClient(base_url=url, api_key=api_key)
            trainer = lilo.create_full_training_client(service, engine.model)
            event('training_client', model_id=trainer.model_id)
            tokenizer = trainer.get_tokenizer()
            initial_checkpoint = os.environ.get('LILO_RECOVERY_TEST_CHECKPOINT')
            if initial_checkpoint:
                trainer.load_state_with_optimizer(initial_checkpoint).result(timeout=1800)
                event('initial_checkpoint_loaded', path=initial_checkpoint)
            else:
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
            pinned_client = peer.create_sampling_client(model_path=saved.path)
            sample('pinned_sample', pinned_client)
            from types import SimpleNamespace
            checkpoint = (SimpleNamespace(path=initial_checkpoint) if initial_checkpoint else
                          trainer.save_state('recovery-checkpoint').result(timeout=1800))
            event('checkpoint', path=checkpoint.path)
            old_latest = trainer.save_weights_and_get_sampling_client()
            # Fault injection only: cancel the trainer's actual Modal invocation.
            app_id = resources['app'].app_id
            engines = modal.Dict.from_name(app_id + '-engines')
            calls = [value for key, value in engines.items() if key.startswith('engine_call:')]
            assert len(calls) == 1, calls
            modal.FunctionCall.from_id(calls[0]).cancel(terminate_containers=True)
            deadline = time.monotonic() + 120
            while True:
                try:
                    modal.FunctionCall.from_id(calls[0]).get(timeout=0)
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline: raise
                    time.sleep(2)
                except modal.exception.Error:
                    break
            event('trainer_cancelled')
            replacement = lilo.create_full_training_client(service, engine.model)
            assert replacement.model_id != trainer.model_id
            event('replacement_created', model_id=replacement.model_id)
            replacement.load_state_with_optimizer(checkpoint.path).result(timeout=1800)
            event('restored')
            sample('replacement_latest_sample', replacement.save_weights_and_get_sampling_client())
            try:
                sample('unexpected_old_latest', old_latest)
            except Exception as exc:
                assert 'replaced' in str(exc).lower(), str(exc)
                event('old_latest_rejected', error=str(exc)[:300])
            else:
                raise AssertionError('old latest handle unexpectedly succeeded')
            sample('old_pinned_still_works', peer.create_sampling_client(model_path=saved.path))
            registry = modal.Dict.from_name(resources['registry'])
            records = registry.get('pin_records')
            for record in records.values():
                assert not record.get('leases'), record
                record['last_used'] = time.time() - 601
            registry.put('pin_records', records)
            stopped = resources['manage'].remote('reap_pinned')
            assert stopped
            event('idle_pinned_reaped', apps=stopped)
            sample('pinned_recreated_sample', pinned_client)
            event('body_complete')
        assert report['events'][-1]['name'] == 'body_complete', 'smoke body did not complete'
        event('context_exited')
        report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        event('error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        scoped.build_app = original_build
        report['finished_at'] = time.time()
        output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
