import csv
import ast
import sys
import copy
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, TimeoutError

EXEC_TIMEOUT = 60


# =========================================================
# FAST AST MUTATOR
# =========================================================
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
                                self._add(new, "ROR")

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
                    self._add(new, "UOI")

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)) and node.value != 0:
                    new = copy.deepcopy(self.tree)
                    t = self._find(new, node)
                    if t:
                        t.value = 0
                        self._add(new, "CRP")

                if isinstance(node.value, bool):
                    new = copy.deepcopy(self.tree)
                    t = self._find(new, node)
                    if t:
                        t.value = not node.value
                        self._add(new, "BLR")

            if isinstance(node, ast.If):
                new = copy.deepcopy(self.tree)
                t = self._find(new, node)
                if t:
                    t.test = ast.UnaryOp(op=ast.Not(), operand=t.test)
                    self._add(new, "NEG")
        return self.mutants

    def _mutate(self, node, repl, label):
        for src, dst in repl.items():
            if hasattr(node, "op") and isinstance(node.op, src):
                new = copy.deepcopy(self.tree)
                t = self._find(new, node)
                if t:
                    t.op = dst()
                    self._add(new, label)

    def _add(self, tree, label):
        ast.fix_missing_locations(tree)
        self.mutants.append((label, tree))

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


# =========================================================
# ASSERT UTILS
# =========================================================
def _parse_asserts(a):
    if not isinstance(a, str) or not a.strip():
        return []
    return [x.strip() for x in a.splitlines() if x.strip()]


def _safe_compile(code):
    try:
        return compile(code, "<string>", "exec")
    except Exception:
        return None


# =========================================================
# EXECUTION ENVIRONMENT  (fix 7)
# =========================================================
_BASE_ENV = None


def _env():
    global _BASE_ENV
    if _BASE_ENV is None:
        import collections, math, typing, heapq, itertools
        import functools, fractions, bisect, operator
        import numpy as np
        fractions.gcd = math.gcd
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix
        g = {
            "__builtins__": __builtins__,
            "collections": collections,
            "math": math,
            "heapq": heapq,
            "itertools": itertools,
            "fractions": fractions,
            "bisect": bisect,
            "operator": operator,
            "np": np,
            "inf": math.inf,
            "gcd": math.gcd,
            "deque": collections.deque,
            "defaultdict": collections.defaultdict,
            "Counter": collections.Counter,
            "lru_cache": functools.lru_cache,
        }
        g.update(vars(typing))
        _BASE_ENV = g
    return _BASE_ENV.copy()


# =========================================================
# EXECUTION WITH TIMEOUT
# =========================================================
def _execute(compiled, asserts, extra=None):
    def run():
        env = _env()
        if extra:
            env.update(extra)
        try:
            exec(compiled, env)
            for a in asserts:
                exec(a, env)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(1) as ex:
        f = ex.submit(run)
        try:
            return f.result(timeout=EXEC_TIMEOUT)
        except TimeoutError:
            return False


# =========================================================
# ASSERT FILTERING
# =========================================================
def run_assert_block(code, asserts):
    asserts = _parse_asserts(asserts)
    compiled = _safe_compile(code)
    correct = [a for a in asserts if _execute(compiled, [a])]
    total = len(asserts)
    ok = len(correct)
    return {
        "Correct": "\n".join(correct),
        "CorrectCount": ok,
        "PassPercentage": round((ok / total) * 100, 2) if total else 0
    }


# =========================================================
# LINE COVERAGE
# =========================================================
def calculate_line_coverage(code, asserts):
    asserts = _parse_asserts(asserts)
    compiled = _safe_compile(code)
    if not compiled:
        return 0
    executed = set()
    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == "<string>":
            executed.add(frame.f_lineno)
        return tracer

    env = _env()
    sys.settrace(tracer)
    try:
        exec(compiled, env)
        for a in asserts:
            try:
                exec(a, env)
            except:
                pass
    finally:
        sys.settrace(None)
    lines = {
        i + 1
        for i, l in enumerate(code.splitlines())
        if l.strip() and not l.strip().startswith("#")
    }
    if not lines:
        return 0
    return round(len(lines & executed) / len(lines) * 100, 2)


# =========================================================
# COVERAGE TRANSFORMER
# =========================================================
class CoverageTransformer(ast.NodeTransformer):

    def __init__(self, logger, hits):
        self.logger = logger
        self.hits = hits
        self.counter = 0

    def _instrument(self, node):
        cid = self.counter
        self.counter += 1
        self.hits[cid] = {"true": False, "false": False}
        node.test = ast.Call(
            func=ast.Name(id=self.logger, ctx=ast.Load()),
            args=[ast.Constant(cid), node.test],
            keywords=[]
        )
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        return self._instrument(node)

    def visit_While(self, node):
        self.generic_visit(node)
        return self._instrument(node)

    def visit_For(self, node):
        self.generic_visit(node)
        cid = self.counter
        self.counter += 1
        self.hits[cid] = {"true": False, "false": False}
        node.iter = ast.Call(
            func=ast.Name(id="__for_logger", ctx=ast.Load()),
            args=[ast.Constant(cid), node.iter],
            keywords=[]
        )
        return node

class ConditionCoverageTransformer(ast.NodeTransformer):

    def __init__(self, logger, hits):
        self.logger = logger
        self.hits = hits
        self.counter = 0

    def _instrument(self, node):

        cid = self.counter
        self.counter += 1

        self.hits[cid] = {"true": False, "false": False}

        return ast.Call(
            func=ast.Name(id=self.logger, ctx=ast.Load()),
            args=[ast.Constant(cid), node],
            keywords=[]
        )

    def visit_Compare(self, node):
        self.generic_visit(node)
        return self._instrument(node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        new_values = []
        for v in node.values:
            new_values.append(self._instrument(v))

        node.values = new_values
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.Not):
            return self._instrument(node)

        return node

    def visit_Name(self, node):

        if isinstance(node.ctx, ast.Load):
            return self._instrument(node)

        return node

    def visit_Constant(self, node):

        if isinstance(node.value, bool):
            return self._instrument(node)

        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)

        node.test = self._instrument(node.test)

        return node


# =========================================================
# GENERIC COVERAGE (fix 8)
# =========================================================
def _calculate_coverage(code, asserts, logger_name):
    asserts = _parse_asserts(asserts)
    try:
        tree = ast.parse(code)
    except:
        return 0
    hits = {}
    t = CoverageTransformer(logger_name, hits)
    tree = t.visit(tree)
    ast.fix_missing_locations(tree)

    def logger(cid, cond):
        hits[cid]["true" if cond else "false"] = True
        return cond

    def for_logger(cid, iterable):
        used = False
        for x in iterable:
            hits[cid]["true"] = True
            used = True
            yield x
        if not used:
            hits[cid]["false"] = True

    compiled = compile(tree, "<string>", "exec")
    _execute(
        compiled,
        asserts,
        {
            logger_name: logger,
            "__for_logger": for_logger
        }
    )

    total = len(hits) * 2
    covered = sum(v for h in hits.values() for v in h.values())
    return 100 if total == 0 else round(covered / total * 100, 2)


def calculate_branch_coverage(code, asserts):
    return _calculate_coverage(code, asserts, "__branch_logger")


def calculate_condition_coverage(code, asserts):
    asserts = _parse_asserts(asserts)
    try:
        tree = ast.parse(code)
    except:
        return 0
    hits = {}
    transformer = ConditionCoverageTransformer("__cond_logger", hits)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    def cond_logger(cid, cond):
        hits[cid]["true" if cond else "false"] = True
        return cond

    compiled = compile(tree, "<string>", "exec")
    _execute(
        compiled,
        asserts,
        {
            "__cond_logger": cond_logger
        }
    )
    total = len(hits) * 2
    covered = sum(v for h in hits.values() for v in h.values())
    if total == 0:
        return 100

    return round(covered / total * 100, 2)


# =========================================================
# MUTATION TESTING (fix 6)
# =========================================================
def ast_mutation_testing(code, asserts):
    asserts = _parse_asserts(asserts)
    try:
        tree = ast.parse(code)
    except:
        return 0, 0, 0, []

    mutants = FastMutator(tree).generate()
    killed = survived = 0
    types = []
    for t, m in mutants:
        try:
            compiled = compile(m, "<string>", "exec")
        except:
            continue
        dead = not _execute(compiled, asserts)
        if dead:
            killed += 1
        else:
            survived += 1
            types.append(t)

    total = killed + survived
    score = round(killed / total * 100, 2) if total else 0
    return score, killed, survived, list(set(types))


# =========================================================
# DATASET PROCESSING
# =========================================================
def evaluate_row(args):
    row, cols = args
    code = row["code"].replace("from fractions import gcd", "from math import gcd")
    result = {"code": code}
    for col in cols:
        asserts = row.get(col)
        model = col.replace(" asserts", "").replace("-", "_")
        r = run_assert_block(code, asserts)
        result[f"{model}_Correct"] = r["Correct"]
        result[f"{model}_CorrectCount"] = r["CorrectCount"]
        result[f"{model}_PassPercentage"] = r["PassPercentage"]
        if r["CorrectCount"]:
            result[f"Line_coverage_{model}"] = calculate_line_coverage(code, r["Correct"])
            result[f"Branch_coverage_{model}"] = calculate_branch_coverage(code, r["Correct"])
            result[f"Condition_coverage_{model}"] = calculate_condition_coverage(code, r["Correct"])
            s, k, sv, t = ast_mutation_testing(code, r["Correct"])
            result[f"Mutation_score_{model}"] = s
            result[f"Mutants_killed_{model}"] = k
            result[f"Mutants_survived_{model}"] = sv
            result[f"Survived_mutant_types_{model}"] = ";".join(t)
        else:
            for m in ["Line", "Branch", "Condition"]:
                result[f"{m}_coverage_{model}"] = 0
            result[f"Mutation_score_{model}"] = 0
            result[f"Mutants_killed_{model}"] = 0
            result[f"Mutants_survived_{model}"] = 0
            result[f"Survived_mutant_types_{model}"] = ""
    return result


def process_dataset(file_to_load, columns, save_csv):
    csv.field_size_limit(10**8)
    with open(file_to_load, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with Pool(cpu_count()) as pool:
        results = pool.map(evaluate_row, [(r, columns) for r in rows])
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    # Test 1 settings
    input_csv = "Test1/My_func_with_asserts.csv"
    output_csv = "Test1/Test_results.csv"
    assert_columns = [
        "Claude_Sonnet_4_6_asserts",
        "ChatGPT_5_3_asserts",
        "Gemini_3_asserts",
        "PyTester_0_examples_asserts",
        "PyTester_1_examples_asserts",
        "PyTester_2_examples_asserts",
        "PyTester_3_examples_asserts"
    ]


    process_dataset(input_csv, assert_columns, output_csv)
