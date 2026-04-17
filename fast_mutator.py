import copy
import ast


class FastMutator:
    def __init__(self, tree):
        self.tree = tree
        self.mutants = []

    def generate(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp):
                repl = {
                    ast.Add: ast.Sub,
                    ast.Sub: ast.Add,
                    ast.Mult: ast.Div,
                    ast.Div: ast.Mult,
                    ast.FloorDiv: ast.Mult
                }
                self._mutate(node, repl, "AOR")

            if isinstance(node, ast.Compare):
                repl = {
                    ast.Gt: ast.GtE,
                    ast.GtE: ast.Gt,
                    ast.Lt: ast.LtE,
                    ast.LtE: ast.Lt,
                    ast.Eq: ast.NotEq,
                    ast.NotEq: ast.Eq
                }
                for i, op in enumerate(node.ops):
                    for src, dst in repl.items():
                        if isinstance(op, src):
                            new = copy.deepcopy(self.tree)
                            t = self._find(new, node)
                            if t and i < len(t.ops):
                                t.ops[i] = dst()
                                self._add(new, "ROR", getattr(node, "lineno", None))

            if isinstance(node, ast.BoolOp):
                repl = {
                    ast.And: ast.Or,
                    ast.Or: ast.And
                }
                self._mutate(node, repl, "LCR")

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                new = copy.deepcopy(self.tree)
                t = self._find(new, node)
                if t:
                    t.op = ast.UAdd()
                    self._add(new, "UOI", getattr(node, "lineno", None))

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)) and node.value != 0:
                    new = copy.deepcopy(self.tree)
                    t = self._find(new, node)
                    if t:
                        t.value = 0
                        self._add(new, "CRP", getattr(node, "lineno", None))

                if isinstance(node.value, bool):
                    new = copy.deepcopy(self.tree)
                    t = self._find(new, node)
                    if t:
                        t.value = not node.value
                        self._add(new, "BLR", getattr(node, "lineno", None))

            if isinstance(node, ast.If):
                new = copy.deepcopy(self.tree)
                t = self._find(new, node)
                if t:
                    t.test = ast.UnaryOp(op=ast.Not(), operand=t.test)
                    self._add(new, "NEG", getattr(node, "lineno", None))

        return self.mutants

    def _mutate(self, node, repl, label):
        for src, dst in repl.items():
            if hasattr(node, "op") and isinstance(node.op, src):
                new = copy.deepcopy(self.tree)
                t = self._find(new, node)
                if t:
                    t.op = dst()
                    self._add(new, label, getattr(node, "lineno", None))

    def _add(self, tree, label, lineno=None):
        ast.fix_missing_locations(tree)
        self.mutants.append((label, lineno, tree))

    @staticmethod
    def _find(tree, original):
        for n in ast.walk(tree):
            if (
                type(n) == type(original)
                and getattr(n, "lineno", None) == getattr(original, "lineno", None)
                and getattr(n, "col_offset", None) == getattr(original, "col_offset", None)
            ):
                return n
        return None
