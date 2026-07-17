import os
import ast
import sys

stdlib = sys.stdlib_module_names

def get_imports(path):
    imports = set()
    for root, dirs, files in os.walk(path):
        if '.venv' in dirs:
            dirs.remove('.venv')
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

# remove local
project_imports.discard('src')

print(sorted(list(project_imports)))
