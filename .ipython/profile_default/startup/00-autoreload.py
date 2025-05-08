import builtins
from IPython.lib import deepreload

builtins.reload = deepreload.reload

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "3")


# Add this to a cell at the start of your notebook
def deep_reload():
    """Reset and reload all data_science_utils modules"""
    import sys
    import importlib

    # Remove all data_science_utils modules from sys.modules
    to_remove = [name for name in sys.modules if name.startswith("data_science_utils")]
    for name in to_remove:
        del sys.modules[name]

    # Import a key module to trigger fresh imports
    try:
        import data_science_utils
    except Exception as e:
        print(f"Error reloading modules: {e}")


# Register with IPython's post_execute event
from IPython import get_ipython

ip = get_ipython()
ip.events.register("post_execute", deep_reload)
# print("Auto deep reload enabled for data_science_utils")
