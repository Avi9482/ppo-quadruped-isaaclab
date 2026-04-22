
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import gymnasium as gym
import textwrap
from prettytable import PrettyTable

import rl_training.tasks  # noqa: F401


def main():
   
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    table.hrules = 1

    # set max width for text wrapping
    max_width = 50

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "Deeprobotics" in task_spec.id:
            # wrap long text in each column before adding it to the table
            task_name = textwrap.fill(task_spec.id, max_width)
            entry_point = textwrap.fill(task_spec.entry_point, max_width)
            config = textwrap.fill(task_spec.kwargs["env_cfg_entry_point"], max_width)

            # add details to table
            table.add_row([index + 1, task_name, entry_point, config])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
