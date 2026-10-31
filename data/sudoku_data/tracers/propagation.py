"""S4: place forced cells FIFO, eliminating from peers; peers forced by an
elimination are listed on the step and queued."""

from ..grid import CELL_NAMES, PEERS, candidates


def trace_propagation(puzzle):
    g = list(puzzle)
    cands = {i: candidates(g, i) for i in range(81) if g[i] == 0}
    queue = [i for i in sorted(cands) if len(cands[i]) == 1]
    queued = set(queue)
    steps = []
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        d = next(iter(cands.pop(i)))
        g[i] = d
        listed = []
        for p in PEERS[i]:
            cs = cands.get(p)
            if cs is not None and d in cs:
                cs.discard(d)
                if len(cs) == 1 and p not in queued:
                    queued.add(p)
                    queue.append(p)
                    listed.append(p)
        step = f"set {CELL_NAMES[i]} = {d}"
        if listed:
            step += " : " + " ".join(
                f"{CELL_NAMES[p]} -> {{ {next(iter(cands[p]))} }}" for p in listed
            )
        steps.append(step)
    return steps, g
