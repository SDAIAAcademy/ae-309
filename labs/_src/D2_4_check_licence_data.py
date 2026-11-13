# %% [markdown]
# # Lab 2.4 · Check the licence and data terms of three backbones and record which the department may deploy
#
# Day 2 · Session 4 (Licences of open-weight backbones) · 25 min.
#
# REPLACE: what participants build in this lab, in two or three lines.
#
# **Tasks**
# - [ ] REPLACE: the tasks, each with its minutes; they add up to 25
# - [ ] ✱ Stretch · REPLACE: an extension for those who finish early

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/computer-vision.git"
    if not Path("/content/computer-vision").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/computer-vision"], check=True)
    os.chdir("/content/computer-vision/labs")
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

# %% [markdown]
# ## 1 · REPLACE: the first task
#
# REPLACE: what to do and why, in two sentences.

# %%
# ✏️ Exercise
def first_task():
    # BEGIN SOLUTION
    return "the reference result"
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    result = first_task()
except NotImplementedError:
    result = "the reference result"
print(result)
