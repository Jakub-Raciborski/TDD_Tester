import csv
import ast
import sys
import copy
import os
import gc
from tqdm import tqdm
import multiprocessing

EXEC_TIMEOUT = 10
BATCH_SIZE = 5
_BASE_ENV = None
_COVERAGE_HITS = None


def __branch_logger(cid, cond):
    global _COVERAGE_HITS
    _COVERAGE_HITS[cid]["true" if cond else "false"] = True
    return cond


def __for_logger(cid, iterable):
    global _COVERAGE_HITS
    used = False
    for x in iterable:
        _COVERAGE_HITS[cid]["true"] = True
        used = True
        yield x
    if not used:
        _COVERAGE_HITS[cid]["false"] = True


def __cond_logger(cid, cond):
    global _COVERAGE_HITS
    _COVERAGE_HITS[cid]["true" if cond else "false"] = True
    return cond

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

def filter_asserts(asserts):
    valid = []
    for line in asserts:
        try:
            tree = ast.parse(line)
            if (
                len(tree.body) == 1 and
                isinstance(tree.body[0], ast.Assert)
            ):
                valid.append(line)
        except:
            continue
    return valid


def _safe_compile(code):
    try:
        return compile(code, "<string>", "exec")
    except Exception:
        return None


# =========================================================
# EXECUTION ENVIRONMENT  (fix 7)
# =========================================================
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
# EXECUTION WITH TIMEOUT (PROCESS-BASED)
# =========================================================
def _run_in_process(source_code, asserts, extra, result_queue):
    global _COVERAGE_HITS

    try:
        compiled = _safe_compile(source_code)
        if compiled is None:
            result_queue.put(False)
            return

        env = _env()
        env["__branch_logger"] = __branch_logger
        env["__for_logger"] = __for_logger
        env["__cond_logger"] = __cond_logger

        if extra:
            if "hits" in extra:
                _COVERAGE_HITS = extra["hits"]

        try:
            exec(compiled, env)
            for a in asserts:
                exec(a, env)

            if extra and "hits" in extra:
                result_queue.put(_COVERAGE_HITS)
            else:
                result_queue.put(True)

        except Exception:
            if extra and "hits" in extra:
                result_queue.put(_COVERAGE_HITS)
            else:
                result_queue.put(False)

    except Exception:
        result_queue.put(False)

def _line_worker(source_code, asserts, result_queue):
    try:
        compiled_local = _safe_compile(source_code)
        if compiled_local is None:
            result_queue.put(set())
            return

        executed = set()

        def tracer(frame, event, arg):
            if event == "line" and frame.f_code.co_filename == "<string>":
                executed.add(frame.f_lineno)
            return tracer

        env = _env()
        sys.settrace(tracer)
        try:
            exec(compiled_local, env)
            for a in asserts:
                try:
                    exec(a, env)
                except Exception:
                    pass
        finally:
            sys.settrace(None)

        result_queue.put(executed)

    except Exception:
        result_queue.put(set())


def _execute(source_code, asserts, extra=None):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_run_in_process,
        args=(source_code, asserts, extra, result_queue)
    )

    p.start()
    p.join(EXEC_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    try:
        return result_queue.get_nowait()
    except Exception:
        return False


# =========================================================
# ASSERT FILTERING
# =========================================================
def run_assert_block(code, asserts):
    asserts = filter_asserts(_parse_asserts(asserts))
    compiled = _safe_compile(code)

    correct = []
    incorrect = []

    if not compiled:
        return {
            "Correct": "",
            "Incorrect": "\n".join(asserts),
            "CorrectCount": 0,
            "IncorrectCount": len(asserts),
            "PassPercentage": 0
        }

    for a in asserts:
        if _execute(code, [a]):
            correct.append(a)
        else:
            incorrect.append(a)

    total = len(asserts)
    correct_count = len(correct)
    incorrect_count = len(incorrect)

    return {
        "Correct": "\n".join(correct),
        "Incorrect": "\n".join(incorrect),
        "CorrectCount": correct_count,
        "IncorrectCount": incorrect_count,
        "PassPercentage": round((correct_count / total) * 100, 2) if total else 0
    }


# =========================================================
# LINE COVERAGE
# =========================================================
def calculate_line_coverage(code, asserts):
    asserts = _parse_asserts(asserts)
    compiled = _safe_compile(code)
    if not compiled:
        return 0

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_line_worker,
        args=(code, asserts, result_queue)
    )

    p.start()
    p.join(EXEC_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return 0

    try:
        executed = result_queue.get_nowait()
    except Exception:
        return 0

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

    source = ast.unparse(tree)

    result = _execute(
        source,
        asserts,
        {
            "hits": hits
        }
    )

    if result is False:
        return 0

    hits = result

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

    source = ast.unparse(tree)
    result = _execute(
        source,
        asserts,
        {
            "hits": hits
        }
    )
    if result is False:
        return 0

    hits = result
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
            mutant_source = ast.unparse(m)
        except:
            continue

        dead = not _execute(mutant_source, asserts)

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
        result[f"{model}_Incorrect"] = r["Incorrect"]
        result[f"{model}_CorrectCount"] = r["CorrectCount"]
        result[f"{model}_IncorrectCount"] = r["IncorrectCount"]
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


def process_dataset(
    file_to_load,
    columns,
    save_csv,
    start_index=0
):
    csv.field_size_limit(10**8)

    with open(file_to_load, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    # =========================
    # RESUME SUPPORT
    # =========================
    if os.path.exists(save_csv):
        existing = []
        with open(save_csv, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

        if existing:
            start_index = len(existing)
            print(f"[RESUME] Starting from index {start_index}")

    results_buffer = []

    # =========================
    # MAIN LOOP + PROGRESS BAR
    # =========================
    for i in tqdm(range(start_index, total), desc="Processing", unit="row"):
        row = rows[i]

        result = evaluate_row((row, columns))
        results_buffer.append(result)

        # =========================
        # BATCH SAVE
        # =========================
        if len(results_buffer) >= BATCH_SIZE or i == total - 1:

            write_header = not os.path.exists(save_csv)

            with open(save_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=results_buffer[0].keys()
                )

                if write_header:
                    writer.writeheader()

                writer.writerows(results_buffer)

            results_buffer.clear()

            # =========================
            # MEMORY CLEANUP
            # =========================
            gc.collect()


def first_main():
    # Test 4 settings
    input_csv = "Data/case_study_4/Test_4_data.csv"
    output_csv = "Data/case_study_4/Test_4_results.csv"
    assert_columns = [
        "Claude_Sonnet_4_6_asserts",
        "ChatGPT_5_4_asserts",
        "Gemini_3_asserts",
    ]

    process_dataset(input_csv, assert_columns, output_csv, start_index=0)


if __name__ == "__main__":
    # Test 1 settings
    input_csv = "Data/case_study_1/Test_1_data.csv"
    output_csv = "Data/case_study_1/Test_1_results.csv"
    assert_columns = [
        "Claude_Sonnet_4_6_asserts",
        "ChatGPT_5_4_asserts",
        "Gemini_3_asserts",
        "PyTester_0_examples_asserts",
        "PyTester_1_examples_asserts",
        "PyTester_2_examples_asserts",
        "PyTester_3_examples_asserts"
    ]

    # # Test 4 settings
    # input_csv = "Data/case_study_4/Test_4_data.csv"
    # output_csv = "Data/case_study_4/Test_4_results.csv"
    # assert_columns = [
    #     "Claude_Sonnet_4_6_asserts",
    #     "ChatGPT_5_4_asserts",
    #     "Gemini_3_asserts",
    # ]
    process_dataset(input_csv, assert_columns, output_csv)
