import csv
import ast
import sys
from multiprocessing import Pool, cpu_count

class FastMutator(ast.NodeTransformer):

    def __init__(self, original_tree):
        self.original_tree = original_tree
        self.mutants = []

    def generate(self):

        for node in ast.walk(self.original_tree):

            # Arithmetic
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

                        new_tree = ast.parse(ast.unparse(self.original_tree))
                        target = self._find_equivalent(new_tree, node)

                        if target:
                            target.op = dst()
                            self.mutants.append(("AOR", new_tree))

            # Relational
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

                            new_tree = ast.parse(ast.unparse(self.original_tree))
                            target = self._find_equivalent(new_tree, node)

                            if target:
                                target.ops[i] = dst()
                                self.mutants.append(("ROR", new_tree))

            # Logical
            if isinstance(node, ast.BoolOp):

                if isinstance(node.op, ast.And):

                    new_tree = ast.parse(ast.unparse(self.original_tree))
                    target = self._find_equivalent(new_tree, node)

                    if target:
                        target.op = ast.Or()
                        self.mutants.append(("LCR", new_tree))

                if isinstance(node.op, ast.Or):

                    new_tree = ast.parse(ast.unparse(self.original_tree))
                    target = self._find_equivalent(new_tree, node)

                    if target:
                        target.op = ast.And()
                        self.mutants.append(("LCR", new_tree))

            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    new_tree = ast.parse(ast.unparse(self.original_tree))
                    target = self._find_equivalent(new_tree, node)
                    if target:
                        target.op = ast.UAdd()
                        self.mutants.append(("UOI", new_tree))
        return self.mutants

    @staticmethod
    def _find_equivalent(tree, original):
        for node in ast.walk(tree):
            if (
                    type(node) == type(original)
                    and getattr(node, "lineno", None) == getattr(original, "lineno", None)
                    and getattr(node, "col_offset", None) == getattr(original, "col_offset", None)
            ):
                return node
        return None


# ASSERT PARSING
def _parse_asserts(asserts_raw):
    if not isinstance(asserts_raw, str) or not asserts_raw.strip():
        return []
    return [a.strip() for a in asserts_raw.splitlines() if a.strip()]


def _safe_compile(code: str):
    try:
        return compile(code, "<string>", "exec")
    except Exception:
        return None


# EXECUTION ENVIRONMENT
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


def _execute_code(compiled_obj, asserts_list, extra_globals=None):
    exec_env = _create_execution_environment()

    if extra_globals:
        exec_env.update(extra_globals)

    try:
        exec(compiled_obj, exec_env)
        for assertion in asserts_list:
            try:
                exec(assertion, exec_env)
            except Exception:
                pass
    except Exception:
        pass


# ASSERT EVALUATION
def run_assert_block(code, asserts_raw):
    correct_asserts = []
    exception_map = {}
    all_exception_types = set()

    asserts_list = _parse_asserts(asserts_raw)
    total_tests = len(asserts_list)

    compiled_code = _safe_compile(code)
    if not compiled_code:
        return {
            "Correct": "",
            "exceptions": {},
            "CorrectCount": 0,
            "PassPercentage": 0.0,
            "exception_types": set()
        }

    for assertion in asserts_list:
        exec_env = _create_execution_environment()

        try:
            exec(compiled_code, exec_env)
            exec(assertion, exec_env)
            correct_asserts.append(assertion)

        except AssertionError:
            exc_type = "AssertionError"
            all_exception_types.add(exc_type)
            exception_map.setdefault(exc_type, []).append(assertion)

        except Exception as e:
            exc_type = type(e).__name__
            all_exception_types.add(exc_type)
            exception_map.setdefault(exc_type, []).append(f"{assertion} --> {str(e)}")

    correct_count = len(correct_asserts)
    pass_percentage = round((correct_count / total_tests) * 100, 2) if total_tests else 0.0

    return {
        "Correct": "\n".join(correct_asserts),
        "exceptions": exception_map,
        "CorrectCount": correct_count,
        "PassPercentage": pass_percentage,
        "exception_types": all_exception_types
    }


# LINE COVERAGE (sys.settrace)
def calculate_line_coverage(code: str, asserts_raw: str) -> float:

    asserts_list = _parse_asserts(asserts_raw)
    if not asserts_list:
        return 0.0

    compiled_code = _safe_compile(code)
    if not compiled_code:
        return 0.0

    executed_lines = set()

    def tracer(frame, event, arg):
        if event == "line":
            if frame.f_code.co_filename == "<string>":
                executed_lines.add(frame.f_lineno)
        return tracer

    sys.settrace(tracer)

    try:
        _execute_code(compiled_code, asserts_list)
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


# BRANCH COVERAGE
def calculate_branch_coverage(code: str, asserts_raw: str):

    asserts_list = _parse_asserts(asserts_raw)
    if not asserts_list:
        return 0.0

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

        def visit_While(self, node):
            return self.visit_If(node)

        def visit_For(self, node):
            nonlocal branch_id
            self.generic_visit(node)

            bid = branch_id
            branch_id += 1

            branch_hits[bid] = {"true": False, "false": False}

            node.iter = ast.Call(
                func=ast.Name(id="__for_logger", ctx=ast.Load()),
                args=[ast.Constant(bid), node.iter],
                keywords=[]
            )

            return node

    transformer = Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if not branch_hits:
        return 100.0

    def branch_logger(bid, cond):
        branch_hits[bid]["true" if cond else "false"] = True
        return cond

    def for_logger(bid, iterable):
        used = False
        for x in iterable:
            branch_hits[bid]["true"] = True
            used = True
            yield x
        if not used:
            branch_hits[bid]["false"] = True

    compiled = compile(tree, "<string>", "exec")

    _execute_code(
        compiled,
        asserts_list,
        extra_globals={
            "__branch_logger": branch_logger,
            "__for_logger": for_logger
        }
    )

    total = len(branch_hits) * 2
    covered = sum(v for b in branch_hits.values() for v in b.values())

    return round((covered / total) * 100, 2)


# CONDITION COVERAGE
def calculate_condition_coverage(code: str, asserts_raw: str):

    asserts_list = _parse_asserts(asserts_raw)
    if not asserts_list:
        return 0.0

    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0

    cond_hits = {}
    cond_id = 0

    class Transformer(ast.NodeTransformer):

        def visit_If(self, node):
            self.generic_visit(node)
            node.test = wrap(node.test)
            return node

        def visit_While(self, node):
            return self.visit_If(node)

    def wrap(expr):
        nonlocal cond_id

        cid = cond_id
        cond_id += 1

        cond_hits[cid] = {"true": False, "false": False}

        return ast.Call(
            func=ast.Name(id="__cond_logger", ctx=ast.Load()),
            args=[ast.Constant(cid), expr],
            keywords=[]
        )

    transformer = Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if not cond_hits:
        return 100.0

    def cond_logger(cid, val):
        cond_hits[cid]["true" if val else "false"] = True
        return val

    compiled = compile(tree, "<string>", "exec")

    _execute_code(
        compiled,
        asserts_list,
        extra_globals={"__cond_logger": cond_logger}
    )

    total = len(cond_hits) * 2
    covered = sum(v for c in cond_hits.values() for v in c.values())

    return round((covered / total) * 100, 2)

# AST Mutator
def ast_mutation_testing(code: str, asserts_raw: str):

    asserts_list = _parse_asserts(asserts_raw)

    if not asserts_list:
        return 0.0, 0, 0, []

    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0, 0, 0, []

    mutator = FastMutator(tree)

    mutants = mutator.generate()

    mutants_killed = 0
    mutants_survived = 0
    survived_types = []

    for mtype, mutant_tree in mutants:

        try:
            compiled = compile(mutant_tree, "<string>", "exec")
        except Exception:
            continue

        killed = False

        for assertion in asserts_list:

            exec_env = _create_execution_environment()

            try:
                exec(compiled, exec_env)
                exec(assertion, exec_env)

            except AssertionError:
                killed = True
                break

            except Exception:
                killed = True
                break

        if killed:
            mutants_killed += 1
        else:
            mutants_survived += 1
            survived_types.append(mtype)

    total = mutants_killed + mutants_survived
    score = round((mutants_killed / total) * 100, 2) if total else 0.0
    return score, mutants_killed, mutants_survived, list(set(survived_types))

# DATASET PROCESSING
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
            #Coverage
            result[f"Line_coverage_{model}"] = calculate_line_coverage(code, r["Correct"])
            result[f"Branch_coverage_{model}"] = calculate_branch_coverage(code, r["Correct"])
            result[f"Condition_coverage_{model}"] = calculate_condition_coverage(code, r["Correct"])

            # MutPy
            mutation_score, killed, survived, survived_types = ast_mutation_testing(code, r["Correct"])
            result[f"Mutation_score_{model}"] = mutation_score
            result[f"Mutants_killed_{model}"] = killed
            result[f"Mutants_survived_{model}"] = survived
            result[f"Survived_mutant_types_{model}"] = ";".join(survived_types)

        else:
            # Coverage
            result[f"Line_coverage_{model}"] = 0.0
            result[f"Branch_coverage_{model}"] = 0.0
            result[f"Condition_coverage_{model}"] = 0.0

            # MutPy
            result[f"Mutation_score_{model}"] = 0.0
            result[f"Mutants_killed_{model}"] = 0
            result[f"Mutants_survived_{model}"] = 0
            result[f"Survived_mutant_types_{model}"] = ""

    return result


def process_dataset(file_csv: str, columns_to_check: list[str], results_file: str):

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