# Prefect Cahching - Summary

You can make Prefect persist your test‐set generation for 24 hours by adding two parameters to the `@task` decorator:

1. **`cache_key_fn=task_input_hash`** — Ensures the cache key reflects your function’s inputs (and you can customize it further to include external factors like file timestamps). ([Prefect Docs][1])
2. **`cache_expiration=timedelta(days=1)`** — Tells Prefect to consider cached results valid for one day only; after that, the task will rerun on next flow run. ([Prefect Docs][2])

By default, cached states never expire unless you specify `cache_expiration` ([Prefect Docs][1]). You can control regeneration by choosing what goes into your cache key: changes to any input that `task_input_hash` sees (e.g., function arguments, code, flow parameters) will bust the cache; to watch external data, extend `cache_key_fn` to hash file modification times or directory snapshots ([Prefect Docs][3]).

---

## 1. Specifying a One-Day Cache Expiration

```python
from datetime import timedelta
from prefect import task
from prefect.tasks import task_input_hash

@task(
    cache_key_fn=task_input_hash,             # default key based on arguments
    cache_expiration=timedelta(days=1)        # expire 24 hours after caching
)
def build_testset(...):
    ...
```

* **`cache_key_fn=task_input_hash`**
  Uses Prefect’s built-in hashing of your task’s inputs (JSON- or pickle-serialized) to generate a cache key ([Prefect Docs][3]).
* **`cache_expiration=timedelta(days=1)`**
  Stores alongside the result an expiration timestamp; after 24 hours, Prefect treats the cache as stale and reruns the task ([Prefect Docs][2]).

If you omit `cache_expiration`, your cache lives **forever** (or until manually cleared) ([Prefect Docs][1]).

---

## 2. Controlling Cache Invalidation Conditions

### 2.1 Changes in Task Inputs

By default, `task_input_hash` includes **all** positional and keyword arguments in its hash. If you pass in:

* `docs`: the **list** of `Document` objects, their metadata (including load timestamps) becomes part of the hash. ([Prefect Docs][3])
* `testset_size`, `llm_model`, `embedding_model`: any change here invalidates the cache automatically ([Prefect Docs][3]).

### 2.2 Changes in Task Code or Flow Parameters

The **task source code** and **flow parameters** are also factored into the key when using the default `DEFAULT` policy (which `task_input_hash` builds upon). Any edits to your `build_testset` function or to flow arguments will generate a new cache key ([Prefect Docs][1]).

### 2.3 Observing External File Changes

To watch for changes in your **PDF source directory** (e.g. new papers added), extend your cache key function:

```python
import os, hashlib
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta

def docs_dir_hash(ctx, args, kwargs):
    docs_path = args[0]
    # take a snapshot of file mod-times
    mtimes = []
    for fn in sorted(os.listdir(docs_path)):
        full = os.path.join(docs_path, fn)
        if os.path.isfile(full):
            mtimes.append(str(os.path.getmtime(full)))
    # combine with default input hash
    base_key = task_input_hash(ctx, args, kwargs) or ""
    return hashlib.sha256((base_key + "".join(mtimes)).encode()).hexdigest()

@task(
    cache_key_fn=docs_dir_hash,
    cache_expiration=timedelta(days=1)
)
def build_testset(docs_path: str, testset_size: int, ...):
    ...
```

Here, any file addition, deletion, or update in `docs_path` changes the hash, forcing regeneration ([Prefect Docs][3], [Prefect 2][4]).

---

## 3. Summary of Cache-Control Options

| Factor                  | Controlled by                                                          |
| ----------------------- | ---------------------------------------------------------------------- |
| Task arguments          | `cache_key_fn=task_input_hash` ([Prefect Docs][3])                     |
| Task source code & tags | Default key policy (included by `task_input_hash`) ([Prefect Docs][1]) |
| Flow parameters         | Default key policy ([Prefect Docs][1])                                 |
| Time-based expiration   | `cache_expiration=timedelta(days=1)` ([Prefect Docs][2])               |
| External files or state | Custom `cache_key_fn` (e.g. hashing timestamps) ([Prefect 2][4])       |

With these controls you can ensure your expensive test-set generation only runs when truly necessary—daily at most, and whenever underlying data or parameters have changed.

[1]: https://docs.prefect.io/v3/develop/write-tasks?utm_source=chatgpt.com "Write and run tasks - Prefect Docs"
[2]: https://docs.prefect.io/v3/develop/task-caching?utm_source=chatgpt.com "Configure task caching - Prefect Docs"
[3]: https://docs-2.prefect.io/2.10.8/concepts/tasks/?utm_source=chatgpt.com "Tasks - Prefect Docs"
[4]: https://prefect-284-docs.netlify.app/api-ref/prefect/tasks/?utm_source=chatgpt.com "prefect.tasks - Prefect 2 - Coordinating the world's dataflows"
