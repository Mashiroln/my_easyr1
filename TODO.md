
1. 处理报错：
(pid=2893044) Running step 0:   0%|                                                                                                                                         | 0.00/110 [00:24<?, ?it/s]Traceback (most recent call last):
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/mnt/data/ccy/EasyR1/verl/trainer/main.py", line 126, in <module>
    main()
  File "/mnt/data/ccy/EasyR1/verl/trainer/main.py", line 118, in main
    ray.get(runner.run.remote(ppo_config))
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/site-packages/ray/_private/worker.py", line 2858, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/mnt/data/miniconda3/envs/verl/lib/python3.10/site-packages/ray/_private/worker.py", line 958, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AssertionError): ray::Runner.run() (pid=2893044, ip=10.132.9.30, actor_id=8b502afe3c725d25982278d701000000, repr=<main.Runner object at 0x7fa5398a12a0>)
  File "/mnt/data/ccy/EasyR1/verl/trainer/main.py", line 87, in run
    trainer.fit()
  File "/mnt/data/ccy/EasyR1/verl/trainer/ray_trainer.py", line 620, in fit
    batch = self._make_batch_data(metrics=metrics)
  File "/mnt/data/ccy/EasyR1/verl/trainer/ray_trainer.py", line 494, in _make_batch_data
    new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
  File "/mnt/data/ccy/EasyR1/verl/protocol.py", line 283, in from_single_dict
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
  File "/mnt/data/ccy/EasyR1/verl/protocol.py", line 325, in from_dict
    return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)
  File "<string>", line 6, in __init__
  File "/mnt/data/ccy/EasyR1/verl/protocol.py", line 179, in __post_init__
    self.check_consistency()  # perform necessary checking
  File "/mnt/data/ccy/EasyR1/verl/protocol.py", line 266, in check_consistency
    assert len(value) == batch_size, f"key {key} length {len(value)} is not equal to bsz {batch_size}."
AssertionError: key is_prefill length 209 is not equal to bsz 512.