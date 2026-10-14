The junction module types that ordinary vessels with summing ports replaced, copied from
circulatory-autogen-modules before they were removed there (with the junction fixes of
`fix/consistent-junctions`: every version has the same equations as its ordinary twin).

`tests/test_vessel_configurations.py` builds each topology twice, with these junctions and with
the ordinary vessels, and requires the same results. That checks the node mapping
(`libcuflynx.generators.port_nodes`) against the generator's own junction handling, which these
types still go through, for as long as that handling exists.
