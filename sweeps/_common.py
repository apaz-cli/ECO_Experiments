# Reusable option dicts for speed/memory settings.
#
# These are "singular + monotonic" dimensions: you want to find one
# working value (usually the fastest) and use it for every run.
# Values are listed best-first (highest throughput, most likely to fail).
#
# Usage in sweep files:
#   from _common import SPEED_OPTIONS
#   OPTIONS = {
#       **SPEED_OPTIONS,
#       "lr": { ... },
#   }
# Or pick individual ones:
#   from _common import LOCAL_BATCH_SIZE
#   OPTIONS = { "local_batch_size": LOCAL_BATCH_SIZE, ... }

LOCAL_BATCH_SIZE = {
    "values": [64, 32, 16, 8, 4, 2, 1],
    "flags": "--training.local_batch_size",
    "name": None,
    "monotonic": "decreasing",
    "singular": True,
}

COMPILE = {
    "values": [True, False],
    "flags": {True: ["--compile.enable"], False: ["--compile.no-enable"]},
    "name": None,
    "singular": True,
}

AC_MODE = {
    "values": ["none", "op", "full"],
    "flags": {
        "full": ["--activation_checkpoint.mode", "full"],
        "op": ["--activation_checkpoint.mode", "selective", "--activation_checkpoint.selective_ac_option", "op"],
        "none": ["--activation_checkpoint.mode", "none"],
    },
    "name": None,
    "monotonic": "decreasing",
    "singular": True,
}

# Bundle for convenience — unpack with **SPEED_OPTIONS
SPEED_OPTIONS = {
    "local_batch_size": LOCAL_BATCH_SIZE,
    #"compile": COMPILE, # Does not fail fast enough to be faster :(
    "ac_mode": AC_MODE,
}
