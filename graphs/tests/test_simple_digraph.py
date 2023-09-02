from ..simple_digraph import SimpleDiGraph


def test_add_node():
    g = SimpleDiGraph()

    assert g.add_node(0)
    assert not g.add_node(0)
    assert g.add_node(1)
