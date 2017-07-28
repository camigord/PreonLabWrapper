import os
import sys
from six.moves import shlex_quote
import numpy as np

VISDOM_PORT = 8098

def new_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)

    return "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))

def create_commands(session, shell='bash'):
    cmds = [
    "tmux kill-session -t {}".format(session),
    "tmux new-session -s {} -n {} -d {}".format(session, "Window0", shell)
    ]
    cmds += [new_cmd(session, "Window0", ["source", "activate", "pytorchenv"])]
    cmds += [new_cmd(session, "Window0", ["python", "main.py"])]
    cmds += ["tmux split-window -h"]
    cmds += [new_cmd(session, "Window0", ["source", "activate", "pytorchenv"])]
    cmds += [new_cmd(session, "Window0", ["python", "-m", "visdom.server", "-port", str(VISDOM_PORT)])]

    cmds += ["sleep 1"]

    notes = []
    notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
    notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    notes += ["Point your browser to http://localhost:{} to open Visdom server".format(VISDOM_PORT)]

    return cmds, notes

def run():
    cmds, notes = create_commands('DDPG')

    print("Executing the following commands:")
    print("\n".join(cmds))
    print("")

    os.environ["TMUX"] = ""
    os.system("\n".join(cmds))
    print('\n'.join(notes))

if __name__ == "__main__":
    run()
