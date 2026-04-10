import csv
import ast
import multiprocessing
import io
import contextlib
import sys
import copy
import pandas as pd
import re
from tqdm import tqdm


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
                    ast.FloorDiv: ast.Mult,
                }
                self._mutate(node, repl, "AOR")

            if isinstance(node, ast.Compare):
                repl = {
                    ast.Gt: ast.GtE,
                    ast.GtE: ast.Gt,
                    ast.Lt: ast.LtE,
                    ast.LtE: ast.Lt,
                    ast.Eq: ast.NotEq,
                    ast.NotEq: ast.Eq,
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
                    ast.Or: ast.And,
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
# ASSERT / PARSING
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


def _solution_has_call_solution(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    return any(
        isinstance(n, ast.FunctionDef) and n.name == "call_solution"
        for n in ast.walk(tree)
    )


def _clean_assert_block(asserts_raw):
    lines = _parse_asserts(asserts_raw)
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line.startswith("assert"):
            continue

        if "==" not in line:
            continue

        if line.startswith("assert call_solution():"):
            continue

        if len(line) > 300:
            continue

        cleaned.append(line)

    return cleaned


def _make_io_wrapper(code: str) -> str:
    if not _solution_has_call_solution(code):
        return ""

    return """
import sys
import io
import ast as _ast

def test_call_solution(input_string):
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    sys.stdin = io.StringIO("" if input_string is None else str(input_string))
    captured_output = io.StringIO()
    sys.stdout = captured_output
    try:
        call_solution()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout

    out = captured_output.getvalue().strip()
    try:
        return _ast.literal_eval(out)
    except Exception:
        return out
"""


def _rewrite_call_solution_assert(line):
    try:
        tree = ast.parse(line)
    except Exception:
        return line

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return line

    node = tree.body[0]
    test = node.test

    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Call)
        and isinstance(test.left.func, ast.Name)
        and test.left.func.id == "call_solution"
    ):
        return line

    try:
        input_value = ast.literal_eval(test.left.args[0])
        expected_value = ast.literal_eval(test.comparators[0])
    except Exception:
        return line

    return f"assert test_call_solution({repr(input_value)}) == {repr(expected_value)}"


def _prepare_asserts(code, asserts):
    lines = _clean_assert_block(asserts)
    if not lines:
        return []

    prepared = []
    use_wrapper = _solution_has_call_solution(code)

    for line in lines:
        if use_wrapper:
            line = _rewrite_call_solution_assert(line)

        try:
            tree = ast.parse(line)
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
                prepared.append(line)
        except Exception:
            continue

    return prepared


# =========================================================
# EXECUTION ENVIRONMENT
# =========================================================
_BASE_ENV = None


def _env():
    global _BASE_ENV
    if _BASE_ENV is None:
        import collections
        import math
        import typing
        import heapq
        import itertools
        import functools
        import fractions
        import bisect
        import operator
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
# COVERAGE TRANSFORMERS
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
            keywords=[],
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
            keywords=[],
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
            keywords=[],
        )

    def visit_Compare(self, node):
        self.generic_visit(node)
        return self._instrument(node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        node.values = [self._instrument(v) for v in node.values]
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
# PROCESS EXECUTION
# =========================================================
def _run_in_process(prelude_source, source_code, asserts, extra, mode, result_queue):
    try:
        env = _env()
        if extra:
            env.update(extra)

        fake_out = io.StringIO()

        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            if prelude_source:
                exec(prelude_source, env)

            if mode == "line":
                compiled = _safe_compile(source_code)
                if compiled is None:
                    result_queue.put(set())
                    return

                executed = set()

                def tracer(frame, event, arg):
                    if event == "line" and frame.f_code.co_filename == "<string>":
                        executed.add(frame.f_lineno)
                    return tracer

                try:
                    sys.settrace(tracer)
                    exec(compiled, env)
                    for a in asserts:
                        try:
                            exec(a, env)
                        except Exception:
                            pass
                except Exception:
                    pass
                finally:
                    sys.settrace(None)

                result_queue.put(executed)
                return

            if mode == "branch":
                try:
                    tree = ast.parse(source_code)
                except Exception:
                    result_queue.put({})
                    return

                hits = {}
                transformer = CoverageTransformer("__branch_logger", hits)
                tree = transformer.visit(tree)
                ast.fix_missing_locations(tree)
                compiled = compile(tree, "<string>", "exec")

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

                env["__branch_logger"] = logger
                env["__for_logger"] = for_logger

                try:
                    exec(compiled, env)
                    for a in asserts:
                        try:
                            exec(a, env)
                        except Exception:
                            pass
                except Exception:
                    pass

                result_queue.put(hits)
                return

            if mode == "condition":
                try:
                    tree = ast.parse(source_code)
                except Exception:
                    result_queue.put({})
                    return

                hits = {}
                transformer = ConditionCoverageTransformer("__cond_logger", hits)
                tree = transformer.visit(tree)
                ast.fix_missing_locations(tree)
                compiled = compile(tree, "<string>", "exec")

                def cond_logger(cid, cond):
                    hits[cid]["true" if cond else "false"] = True
                    return cond

                env["__cond_logger"] = cond_logger

                try:
                    exec(compiled, env)
                    for a in asserts:
                        try:
                            exec(a, env)
                        except Exception:
                            pass
                except Exception:
                    pass

                result_queue.put(hits)
                return

            compiled = _safe_compile(source_code)
            if compiled is None:
                result_queue.put(False)
                return

            try:
                exec(compiled, env)
                for a in asserts:
                    exec(a, env)
                result_queue.put(True)
            except Exception:
                result_queue.put(False)

    except Exception:
        if mode == "line":
            result_queue.put(set())
        elif mode in {"branch", "condition"}:
            result_queue.put({})
        else:
            result_queue.put(False)


def _execute(source_code, asserts, extra=None, prelude=None):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_run_in_process,
        args=(prelude or "", source_code, asserts, extra, "bool", result_queue),
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


def _collect_coverage(source_code, asserts, mode, prelude=None, extra=None):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_run_in_process,
        args=(prelude or "", source_code, asserts, extra, mode, result_queue),
    )

    p.start()
    p.join(EXEC_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return set() if mode == "line" else {}

    try:
        return result_queue.get_nowait()
    except Exception:
        return set() if mode == "line" else {}


# =========================================================
# ASSERT BLOCK
# =========================================================
def run_assert_block(code, asserts):
    prelude = _make_io_wrapper(code)
    asserts = _prepare_asserts(code, asserts)

    correct, incorrect = [], []
    compiled = _safe_compile(code)
    if not compiled:
        return {
            "Correct": "",
            "Incorrect": "\n".join(asserts),
            "CorrectCount": 0,
            "IncorrectCount": len(asserts),
            "PassPercentage": 0,
        }

    for a in asserts:
        if _execute(code, [a], prelude=prelude):
            correct.append(a)
        else:
            incorrect.append(a)

    total = len(asserts)
    correct_count = len(correct)

    return {
        "Correct": "\n".join(correct),
        "Incorrect": "\n".join(incorrect),
        "CorrectCount": correct_count,
        "IncorrectCount": len(incorrect),
        "PassPercentage": round((correct_count / total) * 100, 2) if total else 0,
    }


# =========================================================
# LINE / BRANCH / CONDITION COVERAGE
# =========================================================
def calculate_line_coverage(code, asserts):
    prelude = _make_io_wrapper(code)
    asserts = _prepare_asserts(code, asserts)

    compiled = _safe_compile(code)
    if not compiled:
        return 0

    executed = _collect_coverage(code, asserts, "line", prelude=prelude)

    lines = {
        i + 1
        for i, l in enumerate(code.splitlines())
        if l.strip() and not l.strip().startswith("#")
    }

    if not lines:
        return 0

    return round(len(lines & executed) / len(lines) * 100, 2)


def calculate_branch_coverage(code, asserts):
    prelude = _make_io_wrapper(code)
    asserts = _prepare_asserts(code, asserts)

    try:
        ast.parse(code)
    except Exception:
        return 0

    hits = _collect_coverage(code, asserts, "branch", prelude=prelude)
    total = len(hits) * 2
    covered = sum(v for h in hits.values() for v in h.values())

    return 100 if total == 0 else round(covered / total * 100, 2)


def calculate_condition_coverage(code, asserts):
    prelude = _make_io_wrapper(code)
    asserts = _prepare_asserts(code, asserts)

    try:
        ast.parse(code)
    except Exception:
        return 0

    hits = _collect_coverage(code, asserts, "condition", prelude=prelude)
    total = len(hits) * 2
    covered = sum(v for h in hits.values() for v in h.values())

    return 100 if total == 0 else round(covered / total * 100, 2)


# =========================================================
# MUTATION TESTING
# =========================================================
def ast_mutation_testing(code, asserts):
    prelude = _make_io_wrapper(code)
    asserts = _prepare_asserts(code, asserts)

    try:
        tree = ast.parse(code)
    except Exception:
        return 0, 0, 0, []

    mutants = FastMutator(tree).generate()
    killed = survived = 0
    types = []

    for t, m in mutants:
        try:
            mutant_source = ast.unparse(m)
        except Exception:
            continue

        dead = not _execute(mutant_source, asserts, prelude=prelude)
        if dead:
            killed += 1
        else:
            survived += 1
            types.append(t)

    total = killed + survived
    score = round(killed / total * 100, 2) if total else 0
    return score, killed, survived, list(set(types))


# =========================================================
# DATASET
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


def process_dataset(file_to_load, columns, save_csv):
    csv.field_size_limit(10**8)
    with open(file_to_load, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    results = [
        evaluate_row((r, columns))
        for r in tqdm(rows, desc="Processing dataset", unit="row")
    ]

    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


input_csv = "Data/case_study_3/Test_3_data.csv"
output_csv = "Data/case_study_3/Test_3_data_cleaned.csv"
cleaned_input_csv = "Data/case_study_3/Test_3_data_cleaned.csv"
results_output_csv = "Data/case_study_3/Test_3_results.csv"

assert_columns = [
    "Claude_Sonnet_4_6_assert",
    "ChatGPT_5_4_asserts",
    "Gemini_3_asserts",
    "PyTester_asserts",
]

def normalize_assert_calls(text):
    if pd.isna(text):
        return text

    text = str(text)

    def replace_func(match):
        return f"assert call_solution({match.group(1)}"

    pattern = r'assert\s+\w+\((.*)'

    lines = text.split("\n")
    new_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("assert"):
            # zamień tylko nazwę funkcji
            line = re.sub(pattern, replace_func, line)
        new_lines.append(line)

    return "\n".join(new_lines)


def extract_asserts(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r"```.*?```", lambda m: m.group(0).replace("```python", "").replace("```", ""), text, flags=re.DOTALL)
    matches = re.findall(r'(?m)^\s*assert[^\n]*', text)
    matches = [m.strip() for m in matches if m.strip()]

    return "\n".join(matches)


def main():
    df = pd.read_csv(input_csv)
    for col in assert_columns:
        if col in df.columns:
            df[col] = df[col].apply(extract_asserts)
            df[col] = df[col].apply(normalize_assert_calls)
        else:
            print(f"Kolumna nie istnieje: {col}")

    df.to_csv(output_csv, index=False)
    print(f"Zapisano: {output_csv}")

    process_dataset(cleaned_input_csv, assert_columns, results_output_csv)


if __name__ == "__main__":
    main()