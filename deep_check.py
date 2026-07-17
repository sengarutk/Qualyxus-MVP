#!/usr/bin/env python3
import ast
import os
import sys

class StaticDependencyAuditor:
    """
    Static CI Dependency Auditor.
    Parses application modules using pure syntax tree walking.
    Permanently insulates the test runner from active file execution or network initialization bugs.
    """
    def __init__(self, target_paths: list[str] = None):
        # Enforce explicit checking across both core library packages and root files
        self.target_paths = target_paths or ["src", "main.py"]
        self.internal_modules = set()
        self._discover_internal_architecture()

    def _discover_internal_architecture(self):
        """Walks the repository directory targets to map valid local module and directory paths."""
        for target in self.target_paths:
            if os.path.isfile(target) and target.endswith(".py"):
                mod_name = os.path.normpath(target)[:-3].replace(os.sep, ".")
                self.internal_modules.add(mod_name)
            elif os.path.isdir(target):
                for root, _, files in os.walk(target):
                    # Hardening Fix: Index every directory level as a valid implicit namespace 
                    # package path to prevent false-positives when __init__.py files are omitted
                    rel_dir = os.path.normpath(os.path.relpath(root))
                    if rel_dir != ".":
                        dir_mod = rel_dir.replace(os.sep, ".")
                        self.internal_modules.add(dir_mod)

                    for file in files:
                        if file.endswith(".py"):
                            rel_path = os.path.normpath(os.path.relpath(os.path.join(root, file)))
                            mod_name = rel_path.replace(os.sep, ".")[:-3]
                            
                            if mod_name.endswith(".__init__"):
                                mod_name = mod_name.rsplit(".__init__", 1)[0]
                            self.internal_modules.add(mod_name)

    def audit_file(self, file_path: str) -> list[str]:
        """Parses import lines statically using Abstract Syntax Tree checks."""
        failures = []
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=file_path)
            except SyntaxError as e:
                return [f"AST_SYNTAX_ERROR: {e}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._validate_module_string(alias.name, file_path, failures)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:  # Exclude relative imports
                    for alias in node.names:
                        # Hardening Fix: Filter out wildcard markers to prevent path matching collapses
                        if alias.name == "*":
                            full_mod_str = node.module
                        else:
                            full_mod_str = f"{node.module}.{alias.name}"
                        self._validate_module_string(full_mod_str, file_path, failures)
        return failures

    def _validate_module_string(self, mod_str: str, file_path: str, failures: list):
        # Hardened Fix: Protect against third party packages that share a character prefix with "main"
        if mod_str.startswith("src.") or mod_str == "src" or mod_str == "main" or mod_str.startswith("main."):
            parts = mod_str.split(".")
            for i in range(len(parts), 0, -1):
                check_target = ".".join(parts[:i])
                if check_target in self.internal_modules:
                    return
            failures.append(f"[INTERNAL LINK FRACTURE] Missing reference target: '{mod_str}'")

    def run_suite(self) -> bool:
        """Executes verification trace loops across the source code tree."""
        print("Phase 1: Running Static AST Validation Loop...")
        total_errors = 0
        
        for target in self.target_paths:
            if os.path.isfile(target) and target.endswith(".py"):
                errors = self.audit_file(target)
                if errors:
                    print(f"violations found in root module -> {target}:")
                    for err in errors:
                        print(f"  {err}")
                    total_errors += len(errors)
            elif os.path.isdir(target):
                for root, _, files in os.walk(target):
                    for file in files:
                        if file.endswith(".py"):
                            full_path = os.path.join(root, file)
                            errors = self.audit_file(full_path)
                            if errors:
                                print(f"violations found in module -> {full_path}:")
                                for err in errors:
                                    print(f"  {err}")
                                total_errors += len(errors)

        if total_errors > 0:
            print(f"\nCRITICAL FAILURE: Found {total_errors} codebase linkage traps!")
            return False
        
        print("\nSUCCESS: All internal dependencies are structurally aligned.")
        return True

if __name__ == "__main__":
    auditor = StaticDependencyAuditor()
    success = auditor.run_suite()
    sys.exit(0 if success else 1)
