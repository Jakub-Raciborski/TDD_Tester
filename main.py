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
class FastMutator(ast.NodeTransformer):

    def __init__(self, original_tree):
        self.original_tree = original_tree
        self.mutants = []

    def generate(self):
        for node in ast.walk(self.original_tree):
            # Arithmetic operators
            if isinstance(node, ast.BinOp):
                replacements = {
                    ast.Add: ast.Sub,
                    ast.Sub: ast.Add,
                    ast.Mult: ast.Div,
                    ast.Div: ast.Mult,
                    ast.FloorDiv: ast.Mult,
                }
                for src, dst in replacements.items():
                    if isinstance(node.op, src):
                        new_tree = copy.deepcopy(self.original_tree)
                        target = self._find_equivalent(new_tree, node)
                        if target:
                            target.op = dst()
                            ast.fix_missing_locations(new_tree)
                            self.mutants.append(("AOR", new_tree))

            # Relational operators
            if isinstance(node, ast.Compare):
                replacements = {
                    ast.Gt: ast.GtE,
                    ast.GtE: ast.Gt,
                    ast.Lt: ast.LtE,
                    ast.LtE: ast.Lt,
                    ast.Eq: ast.NotEq,
                    ast.NotEq: ast.Eq
                }
                for i, op in enumerate(node.ops):
                    for src, dst in replacements.items():
                        if isinstance(op, src):
                            new_tree = copy.deepcopy(self.original_tree)
                            target = self._find_equivalent(new_tree, node)
                            if target:
                                target.ops[i] = dst()
                                ast.fix_missing_locations(new_tree)
                                self.mutants.append(("ROR", new_tree))

            # Logical operators
            if isinstance(node, ast.BoolOp):
                replacements = {
                    ast.And: ast.Or,
                    ast.Or: ast.And
                }
                for src, dst in replacements.items():
                    if isinstance(node.op, src):
                        new_tree = copy.deepcopy(self.original_tree)
                        target = self._find_equivalent(new_tree, node)
                        if target:
                            target.op = dst()
                            ast.fix_missing_locations(new_tree)
                            self.mutants.append(("LCR", new_tree))

            # Unary operators
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    new_tree = copy.deepcopy(self.original_tree)
                    target = self._find_equivalent(new_tree, node)
                    if target:
                        target.op = ast.UAdd()
                        ast.fix_missing_locations(new_tree)
                        self.mutants.append(("UOI", new_tree))

            # Constant replacement
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    new_tree = copy.deepcopy(self.original_tree)
                    target = self._find_equivalent(new_tree, node)
                    if target:
                        target.value = 0
                        ast.fix_missing_locations(new_tree)
                        self.mutants.append(("CRP", new_tree))

                if isinstance(node.value, bool):
                    new_tree = copy.deepcopy(self.original_tree)
                    target = self._find_equivalent(new_tree, node)
                    if target:
                        target.value = not node.value
                        ast.fix_missing_locations(new_tree)
                        self.mutants.append(("BLR", new_tree))

            # Negate condition
            if isinstance(node, ast.If):
                new_tree = copy.deepcopy(self.original_tree)
                target = self._find_equivalent(new_tree, node)
                if target:
                    target.test = ast.UnaryOp(op=ast.Not(), operand=target.test)
                    ast.fix_missing_locations(new_tree)
                    self.mutants.append(("NEG", new_tree))

        return self.mutants

    @staticmethod
    def _find_equivalent(tree, original):
        for node in ast.walk(tree):
            if (
                type(node) == type(original)
                and getattr(node, "lineno", None) == getattr(original, "lineno", None)
                and getattr(node, "col_offset", None) == getattr(original, "col_offset", None)
                and getattr(node, "end_lineno", None) == getattr(original, "end_lineno", None)
                and getattr(node, "end_col_offset", None) == getattr(original, "end_col_offset", None)
            ):
                return node
        return None


# =========================================================
# ASSERT PARSING
# =========================================================
def _parse_asserts(asserts_raw):
    if not isinstance(asserts_raw, str) or not asserts_raw.strip():
        return []
    return [a.strip() for a in asserts_raw.splitlines() if a.strip()]

def _safe_compile(code: str):
    try:
        return compile(code, "<string>", "exec")
    except Exception:
        return None


# =========================================================
# EXECUTION ENVIRONMENT
# =========================================================
def _create_execution_environment():
    import collections, math, typing, heapq, itertools
    import functools, fractions, bisect, copy
    import statistics, operator
    import numpy as np

    fractions.gcd = math.gcd
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix

    safe_globals = {
        "__builtins__": __builtins__,
        "collections": collections,
        "math": math,
        "heapq": heapq,
        "itertools": itertools,
        "fractions": fractions,
        "bisect": bisect,
        "copy": copy,
        "operator": operator,
        "np": np,
        "inf": math.inf,
        "log": math.log,
        "ceil": math.ceil,
        "gcd": math.gcd,
        "mean": statistics.mean,
        "heapify": heapq.heapify,
        "heappush": heapq.heappush,
        "heappop": heapq.heappop,
        "mul": operator.mul,
        "product": itertools.product,
        "starmap": itertools.starmap,
        "deque": collections.deque,
        "defaultdict": collections.defaultdict,
        "Counter": collections.Counter,
        "lru_cache": functools.lru_cache,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "range": range,
    }
    safe_globals.update(vars(typing))
    return safe_globals.copy()


# =========================================================
# TIMEOUT EXECUTION
# =========================================================
def _worker(compiled_code, asserts_list, extra_globals, queue):
    exec_env = _create_execution_environment()
    if extra_globals:
        exec_env.update(extra_globals)
    try:
        exec(compiled_code, exec_env)
        for assertion in asserts_list:
            exec(assertion, exec_env)
        queue.put(True)
    except Exception:
        queue.put(False)


def _execute_code(compiled_obj, asserts_list, extra_globals=None):
    def runner():
        exec_env = _create_execution_environment()
        if extra_globals:
            exec_env.update(extra_globals)
        try:
            exec(compiled_obj, exec_env)
            for assertion in asserts_list:
                exec(assertion, exec_env)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner)
        try:
            return future.result(timeout=EXEC_TIMEOUT)
        except TimeoutError:
            return False


# =========================================================
# ASSERT EVALUATION
# =========================================================
def run_assert_block(code, asserts_raw):
    asserts_list = _parse_asserts(asserts_raw)
    compiled_code = _safe_compile(code)
    correct_asserts = []
    for assertion in asserts_list:
        success = _execute_code(compiled_code, [assertion])
        if success:
            correct_asserts.append(assertion)
    correct_count = len(correct_asserts)
    total_tests = len(asserts_list)
    pass_percentage = round((correct_count / total_tests) * 100, 2) if total_tests else 0.0
    return {
        "Correct": "\n".join(correct_asserts),
        "CorrectCount": correct_count,
        "PassPercentage": pass_percentage
    }


# =========================================================
# LINE COVERAGE
# =========================================================
def calculate_line_coverage(code, asserts_raw):
    asserts_list = _parse_asserts(asserts_raw)
    compiled_code = _safe_compile(code)
    if not compiled_code:
        return 0.0
    executed_lines = set()
    def tracer(frame, event, arg):
        if event == "line":
            if frame.f_code.co_filename == "<string>":
                executed_lines.add(frame.f_lineno)
        return tracer
    exec_env = _create_execution_environment()
    sys.settrace(tracer)
    try:
        exec(compiled_code, exec_env)
        for assertion in asserts_list:
            try:
                exec(assertion, exec_env)
            except Exception:
                pass
    finally:
        sys.settrace(None)
    lines = code.splitlines()
    total_lines = {
        i + 1
        for i, line in enumerate(lines)
        if line.strip() and not line.strip().startswith("#")
    }
    if not total_lines:
        return 0.0
    covered = executed_lines & total_lines
    return round((len(covered) / len(total_lines)) * 100, 2)


# =========================================================
# BRANCH COVERAGE
# =========================================================
def calculate_branch_coverage(code, asserts_raw):
    asserts_list = _parse_asserts(asserts_raw)
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0
    branch_hits = {}
    branch_id = 0

    class Transformer(ast.NodeTransformer):
        def visit_If(self, node):
            nonlocal branch_id
            self.generic_visit(node)
            bid = branch_id
            branch_id += 1
            branch_hits[bid] = {"true": False, "false": False}
            node.test = ast.Call(
                func=ast.Name(id="__branch_logger", ctx=ast.Load()),
                args=[ast.Constant(bid), node.test],
                keywords=[]
            )
            return node

    transformer = Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    def branch_logger(bid, cond):
        branch_hits[bid]["true" if cond else "false"] = True
        return cond

    compiled = compile(tree, "<string>", "exec")
    _execute_code(
        compiled,
        asserts_list,
        {"__branch_logger": branch_logger}
    )

    total = len(branch_hits) * 2
    covered = sum(v for b in branch_hits.values() for v in b.values())

    if total == 0:
        return 100.0

    return round((covered / total) * 100, 2)


# =========================================================
# CONDITION COVERAGE
# =========================================================
def calculate_condition_coverage(code, asserts_raw):
    asserts_list = _parse_asserts(asserts_raw)
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0

    cond_hits = {}
    cond_id = 0

    class Transformer(ast.NodeTransformer):
        def visit_If(self, node):
            nonlocal cond_id
            self.generic_visit(node)
            cid = cond_id
            cond_id += 1
            cond_hits[cid] = {"true": False, "false": False}
            node.test = ast.Call(
                func=ast.Name(id="__cond_logger", ctx=ast.Load()),
                args=[ast.Constant(cid), node.test],
                keywords=[]
            )
            return node

    transformer = Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    def cond_logger(cid, val):
        cond_hits[cid]["true" if val else "false"] = True
        return val

    compiled = compile(tree, "<string>", "exec")

    _execute_code(
        compiled,
        asserts_list,
        {"__cond_logger": cond_logger}
    )

    total = len(cond_hits) * 2
    covered = sum(v for c in cond_hits.values() for v in c.values())

    if total == 0:
        return 100.0
    return round((covered / total) * 100, 2)


# =========================================================
# MUTATION TESTING
# =========================================================
def ast_mutation_testing(code, asserts_raw):
    asserts_list = _parse_asserts(asserts_raw)
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0, 0, 0, []

    mutator = FastMutator(tree)
    mutants = mutator.generate()
    killed = 0
    survived = 0
    survived_types = []

    for mtype, mutant in mutants:
        try:
            compiled = compile(mutant, "<string>", "exec")
        except Exception:
            continue
        dead = False
        for assertion in asserts_list:
            success = _execute_code(compiled, [assertion])
            if not success:
                dead = True
                break

        if dead:
            killed += 1
        else:
            survived += 1
            survived_types.append(mtype)

    total = killed + survived
    score = round((killed / total) * 100, 2) if total else 0.0
    return score, killed, survived, list(set(survived_types))


# =========================================================
# DATASET PROCESSING
# =========================================================
def evaluate_row(args):
    row, columns_with_asserts = args
    code = row["code"].replace("from fractions import gcd", "from math import gcd")
    result = {"code": code}
    for col in columns_with_asserts:
        asserts = row.get(col)
        model = col.replace(" asserts", "").replace("-", "_")
        r = run_assert_block(code, asserts)
        result[f"{model}_Correct"] = r["Correct"]
        result[f"{model}_CorrectCount"] = r["CorrectCount"]
        result[f"{model}_PassPercentage"] = r["PassPercentage"]

        if r["CorrectCount"] > 0:
            result[f"Line_coverage_{model}"] = calculate_line_coverage(code, r["Correct"])
            result[f"Branch_coverage_{model}"] = calculate_branch_coverage(code, r["Correct"])
            result[f"Condition_coverage_{model}"] = calculate_condition_coverage(code, r["Correct"])
            mutation_score, killed, survived, types = ast_mutation_testing(code, r["Correct"])
            result[f"Mutation_score_{model}"] = mutation_score
            result[f"Mutants_killed_{model}"] = killed
            result[f"Mutants_survived_{model}"] = survived
            result[f"Survived_mutant_types_{model}"] = ";".join(types)
        else:
            result[f"Line_coverage_{model}"] = 0.0
            result[f"Branch_coverage_{model}"] = 0.0
            result[f"Condition_coverage_{model}"] = 0.0
            result[f"Mutation_score_{model}"] = 0.0
            result[f"Mutants_killed_{model}"] = 0
            result[f"Mutants_survived_{model}"] = 0
            result[f"Survived_mutant_types_{model}"] = ""
    return result


def process_dataset(file_csv, columns_to_check, results_file):
    csv.field_size_limit(10**8)
    with open(file_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with Pool(cpu_count()) as pool:
        results = pool.map(evaluate_row, [(r, columns_to_check) for r in rows])

    fieldnames = list(results[0].keys())
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    input_csv = "Test1/My_func_with_asserts.csv"
    output_csv = "Test1/Test_results.csv"
    assert_columns = [
        "ChatGPT5_3_asserts",
        "Gemini_3_asserts",
        "PyTester_0_examples_asserts",
        "PyTester_1_examples_asserts",
        "PyTester_2_examples_asserts",
        "PyTester_3_examples_asserts"
    ]

    process_dataset(input_csv, assert_columns, output_csv)