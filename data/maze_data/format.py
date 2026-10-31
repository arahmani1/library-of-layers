"""Question and sequence layout, the router-only labels, and the parser.

    idx   0 .. 195   Q : <64 cells, each its name then open sides N E S W>
                     start <cell> goal <cell>            the prompt
    idx 196 .. 203   [ <six label cells> ]               router-only label
    idx 204 .. 205   R :
    idx 206 .. a-1   trace cells                         reasoning
    idx   a .. a+1   A :
    idx a+2 .. -2    path cells, start to goal           answer
    idx        -1    <eos>

Every perfect 8x8 maze has 63 open walls, each contributing one side letter
to both of its cells, so the prompt is always 196 tokens and the label always
sits at 196..203; ``labels[:204] = -100`` masks the whole question.
"""

from .grid import CELL_INDEX, CELL_NAMES, CELLS, SIDES, from_sides, open_letters

PROMPT_LENGTH = 196
LABEL_SPAN = (196, 204)
Q_LEN = 204  # token 204 is always R
REASONING_START = 206

# Six cell tokens each, distinct even as unordered bags of tokens. The label
# is what the router reads; the attention mask hides it from everything else.
LABELS = {
    1: "r1c1 r1c2 r2c1 r1c3 r2c2 r3c1",
    2: "r1c1 r1c2 r1c3 r1c4 r1c3 r1c2",
    3: "r1c1 r1c2 r2c2 r2c3 r3c3 r3c4",
    4: "r1c1 r8c8 r1c2 r8c7 r1c3 r8c6",
    5: "r8c1 r8c2 r8c3 r1c8 r2c8 r3c8",
}
STRATEGY_OF_LABEL = {label: k for k, label in LABELS.items()}


def question(maze, s, g, k):
    tokens = ["Q", ":"]
    for i in range(CELLS):
        tokens.append(CELL_NAMES[i])
        tokens += open_letters(maze, i)
    tokens += ["start", CELL_NAMES[s], "goal", CELL_NAMES[g], "["]
    tokens += LABELS[k].split(" ")
    tokens.append("]")
    return tokens


def build_sequence(maze, s, g, k, trace, path):
    tokens = question(maze, s, g, k) + ["R", ":"]
    tokens += [CELL_NAMES[i] for i in trace]
    tokens += ["A", ":"]
    tokens += [CELL_NAMES[i] for i in path]
    tokens.append("<eos>")
    return " ".join(tokens)


def _cells(tokens):
    try:
        return [CELL_INDEX[t] for t in tokens]
    except KeyError as error:
        raise ValueError(f"{error.args[0]!r} is not a cell") from None


def parse_sequence(sequence):
    """``(maze, start, goal, strategy, trace, path)`` from a decoded sequence.

    Raises ``ValueError`` when the layout above is not met, so a generated
    sequence can be rejected rather than misread.
    """
    tokens = sequence.split(" ")
    if tokens[:2] != ["Q", ":"]:
        raise ValueError("sequence does not open with 'Q :'")
    sides = []
    n = 2
    for i in range(CELLS):
        if n >= len(tokens) or tokens[n] != CELL_NAMES[i]:
            raise ValueError(f"expected {CELL_NAMES[i]} at token {n}")
        n += 1
        letters = []
        while n < len(tokens) and tokens[n] in SIDES:
            letters.append(tokens[n])
            n += 1
        sides.append(letters)
    maze = from_sides(sides)
    if tokens[n:n + 4:2] != ["start", "goal"]:
        raise ValueError("expected 'start <cell> goal <cell>' after the grid")
    s, g = _cells(tokens[n + 1:n + 4:2])
    n += 4
    if tokens[n:n + 1] != ["["] or tokens[n + 7:n + 8] != ["]"]:
        raise ValueError("expected a bracketed six-token label")
    k = STRATEGY_OF_LABEL.get(" ".join(tokens[n + 1:n + 7]))
    if k is None:
        raise ValueError("unknown strategy label")
    n += 8
    if tokens[n:n + 2] != ["R", ":"]:
        raise ValueError(f"expected 'R :' at token {n}")
    try:
        a = tokens.index("A", n + 2)
    except ValueError:
        raise ValueError("no 'A' after the trace") from None
    if tokens[a + 1:a + 2] != [":"] or tokens[-1] != "<eos>":
        raise ValueError("expected 'A : <path> <eos>'")
    return maze, s, g, k, _cells(tokens[n + 2:a]), _cells(tokens[a + 2:-1])
