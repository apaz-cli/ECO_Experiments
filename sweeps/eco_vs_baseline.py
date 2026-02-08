#!/tmp/eco/run_sweep.py
ECO_OFF = ["--eco.enabled=false", "--model.converters", ""]
ECO_ON = ["--eco.enabled=true", "--model.converters", "quantize.linear.float8,eco"]

OPTIONS = {
    "eco_enabled": {
        "values": [False, True],
        "flags": {False: ECO_OFF, True: ECO_ON},
        "name": "eco",
    },
}
