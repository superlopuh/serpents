class SimpleDiGraph:
    _nodes: set[int]
    _targets_by_sources: dict[int, set[int]]
    _sources_by_targets: dict[int, set[int]]

    def __init__(self):
        self._nodes = set()
        self._targets_by_sources = dict()
        self._sources_by_targets = dict()

    def add_node(self, node: int) -> bool:
        """
        Add a node to the graph, and return True if it was not already in the graph.
        """
        result = node not in self._nodes
        self._nodes.add(node)
        return result
