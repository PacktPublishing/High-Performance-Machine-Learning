"""Routing policies for experiment E8: round-robin, session affinity and prefix-aware (Code 12.2).

The router runs inside the load generator process and returns a base URL per model call. The prefix-aware
policy tokenizes the prompt prefix (system prompt + history, without images) with the model's tokenizer.
"""
import hashlib
import itertools
import threading

BLOCK = 16                                    # tokens per KV block in vLLM


def block_hashes(token_ids):
    h, out = b"", []
    for i in range(0, len(token_ids) - BLOCK + 1, BLOCK):
        h = hashlib.sha256(h + bytes(str(token_ids[i:i + BLOCK]), "ascii")).digest()
        out.append(h)                         # chained: one hash per prefix
    return out


class Replica:
    def __init__(self, url):
        self.url, self.cached, self.running_requests = url, set(), 0


class Router:
    def __init__(self, urls, policy="prefix", load_weight=0.5, tokenizer=None):
        self.replicas = [Replica(u) for u in urls]
        self.policy, self.load_weight, self.tok = policy, load_weight, tokenizer
        self.rr = itertools.cycle(self.replicas)
        self.sessions = {}
        self.lock = threading.Lock()

    def pick(self, task_id, prompt_text):
        with self.lock:
            if self.policy == "round_robin":
                r = next(self.rr)
            elif self.policy == "affinity":
                r = self.sessions.setdefault(task_id, min(self.replicas, key=lambda x: x.running_requests))
            else:
                hashes = block_hashes(self.tok.encode(prompt_text))

                def score(rep):
                    hit = 0
                    while hit < len(hashes) and hashes[hit] in rep.cached:
                        hit += 1
                    return (len(hashes) - hit) + self.load_weight * rep.running_requests
                r = min(self.replicas, key=score)
                r.cached.update(hashes)               # the estimate drifts when the replica evicts blocks
            r.running_requests += 1
            return r

    def done(self, r):
        with self.lock:
            r.running_requests -= 1
