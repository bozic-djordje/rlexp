import os

def setup_artefact_paths(script_path:str):
    script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)

    yaml_path = os.path.join(script_dir, 'configs', script_name.replace('.py', '.yaml'))
    store_path = os.path.join(script_dir, 'artefacts')
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    return store_path, yaml_path