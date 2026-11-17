Junction module versions the module library (circulatory-autogen-modules) removed, copied from its
main at da64d066 as flat external modules in `modules/` (their default parameterisations left out), and
`junction_net`, a small model that uses five of them: `inlet_flow constant_2` into a
`2in2out_junction`, a `split_junction_simple`, a `flow_merge` and a `Min_junction`.
`tests/test_junction_upgrade.py` generates it as it is and after `cuflynx-upgrade junctions`, and
checks the two give the same model.
