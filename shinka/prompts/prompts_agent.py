ENV_TO_AGENT_SYS_MSG = """
{task_sys_msg}

You are in a multi-attempt pipeline. After each evaluation you receive feedback; use it to improve the next program.

Action space (start each action with `/execute_action{{action_name}}` on its own line, followed immediately by that action's fields):
- modify_full: rewrite TARGET_PROGRAM with a full program. Provide <NAME>, <DESCRIPTION>, <CODE> (```{language} ...```), plus any reasoning text. Use program ids from the leaderboard (names are shown for readability).
- modify_diff: apply a diff to TARGET_PROGRAM. Provide <NAME>, <DESCRIPTION>, and a diff block inside <CODE>. Use program ids.
- retrieve: fetch programs. Provide TARGET_PROGRAM as a comma-separated list of program ids.
- reflect: optional thoughts. Include your reflection inline; the environment will not echo back extra guidance.
- summarize: replace the running context with a comprehensive summary of the task, progress, plan, and any details you deem important to keep (no code). If summarize is present, it is the only action executed for that turn and the conversation context will be replaced by your summary.

Example actions:
/execute_action{{modify_full}}
TARGET_PROGRAM: program_id

<NAME>better_layout</NAME>
<DESCRIPTION>Improve packing heuristic</DESCRIPTION>
<CODE>```{language}
...full code here...
```</CODE>

/execute_action{{retrieve}}
TARGET_PROGRAM: best_program_id, baseline_program_id

/execute_action{{reflect}}

/execute_action{{summarize}}
Summarize the task, current scores, key code ideas, risks, and next plan to carry forward. Include any details you consider important to preserve because the context will be replaced by this summary.

Execution order: if summarize is present, only summarize executes (context is replaced). Otherwise modify_* (if present) → retrieve → reflect. If a modify_* action is provided, it will be applied and the new program will be evaluated automatically next step. Modify action can only be one per submission. If many modify_* actions are provided, only the first will be executed.

modify_full action requirements:

Include TARGET_PROGRAM (program id), <NAME>, <DESCRIPTION>, and a full program fenced as ```{language} ...```.
Use this structure:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here.
``` 
</CODE>

Full code rewrite summary instructions:
* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
* Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

modify_diff action requirements:

- For diff edits (`modify_diff`): include TARGET_PROGRAM (program id), <NAME>, <DESCRIPTION>, and SEARCH/REPLACE blocks in this exact format:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing. Discuss the high-level changes made to the given code and how these changes achieve the objectives.
</DESCRIPTION>

<CODE>
```diff
<<<<<<< SEARCH
<Exact code block to be replaced>
=======
<Exact replacement code block>
>>>>>>> REPLACE
```
</CODE>

Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

Diff code rewrite summary instructions:
* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file.  Including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.

Leaderboard (top programs):
{leaderboard_block}
"""
