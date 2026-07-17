import os
import ast
import sys
import importlib.util

stdlib = sys.stdlib_module_names

def get_imports(path):
    imports = set()
    for root, dirs, files in os.walk(path):
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.add(alias.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.level == 0 and node.module:
                                    imports.add(node.module.split('.')[0])
                    except Exception as e:
                        pass
    return imports

project_imports = get_imports('.')

# remove stdlib
project_imports = {i for i in project_imports if i not in stdlib}

# remove local folders
local_modules = set()
for root, dirs, files in os.walk('.'):
    if '.venv' in dirs: dirs.remove('.venv')
    if '__pycache__' in dirs: dirs.remove('__pycache__')
    for d in dirs:
        if os.path.isfile(os.path.join(root, d, '__init__.py')):
            local_modules.add(d)
    for f in files:
        if f.endswith('.py'):
            local_modules.add(f[:-3])

project_imports = project_imports - local_modules
project_imports.discard('src')

missing = []
installed = []

for mod in project_imports:
    # Handle known aliases
    try:
        if importlib.util.find_spec(mod) is not None:
            installed.append(mod)
        else:
            missing.append(mod)
    except ModuleNotFoundError:
        missing.append(mod)
    except Exception:
        missing.append(mod)

print("MISSING_MODULES:", sorted(missing))
print("INSTALLED_MODULES:", sorted(installed))
