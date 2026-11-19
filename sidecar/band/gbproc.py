"""RTX band of derpk31pro: the parent's handle on a GB300 client process (bench/gb_client.py). The benchmark and the
tests keep the two GPUs in two processes, as an engine process and a band sidecar are: this process owns the RTX
(store, service kernel), the client owns the GB300 (send, finish), and the mailbox segment is all they share."""
import json
import os
import queue
import subprocess
import sys
import threading
import time


class GbProcess:
    def __init__(self, build: str, seg: str, trace_file: str, gb300_uuid: str, start_timeout_s: float = 300.0):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "gb_client.py")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gb300_uuid)
        # Eager module loading is for the process with the persistent kernel (a lazy load would wait behind it for
        # good). Here it would only cost: on the GB300 an eager load of the framework's modules takes ~100 s.
        env.pop("CUDA_MODULE_LOADING", None)
        self.p = subprocess.Popen([sys.executable, script, "--build", build, "--seg", seg, "--trace", trace_file],
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = self._answer(start_timeout_s, None)

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)

    def _answer(self, timeout_s: float, alive):
        t0 = time.time()
        while True:
            try:
                line = self.q.get(timeout=0.05)
            except queue.Empty:
                line = ""
            if line is None:
                raise RuntimeError(f"the GB300 client left: exit code {self.p.wait()}")
            if line.startswith("RESULT "):
                return json.loads(line[7:])
            if line.strip():
                print("  [gb300] " + line.rstrip(), flush=True)
            if time.time() - t0 > timeout_s:
                self.close(kill=True)
                raise RuntimeError("the GB300 client did not answer in time")
            if alive is not None and not alive():
                self.close(kill=True)
                raise RuntimeError("the service kernel left while the GB300 client was running")

    def run(self, timeout_s: float = 120.0, alive=None, **cmd) -> dict:
        """One trace replay on the GB300 (the fields of gb_client.py's run command). `alive`: a callable checked
        while waiting (the service kernel must stay up); pass None when the run is meant to see it down."""
        self.p.stdin.write(json.dumps(dict(cmd="run", **cmd)) + "\n")
        self.p.stdin.flush()
        return self._answer(timeout_s, alive)

    def load(self, trace_file: str, timeout_s: float = 60.0) -> dict:
        """Switch the client to another trace file."""
        self.p.stdin.write(json.dumps(dict(cmd="load", trace=trace_file)) + "\n")
        self.p.stdin.flush()
        return self._answer(timeout_s, None)

    def close(self, kill: bool = False):
        if self.p.poll() is None:
            if not kill:
                try:
                    self.p.stdin.write(json.dumps(dict(cmd="quit")) + "\n")
                    self.p.stdin.flush()
                    self.p.wait(timeout=20)
                except Exception:
                    kill = True
            if kill and self.p.poll() is None:           # never leave a client on the GB300
                self.p.kill()
                self.p.wait()
