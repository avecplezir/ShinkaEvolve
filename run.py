# standard shinka imports
import os
import datetime as dt
from time import perf_counter
import sys
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

from pathlib import Path

repo_root = Path.cwd()
if not (repo_root / "shinka").exists():
    for parent in Path.cwd().resolve().parents:
        if (parent / "shinka").exists():
            repo_root = parent
            break

sys.path.insert(0, str(repo_root))
print("repo_root:", repo_root)


# default circle packing message - can be customized to your liking!
search_task_sys_msg = (
    "You are an expert mathematician specializing in circle packing problems "
    "and computational geometry. The best known result for the sum of radii "
    "when packing 26 circles in a unit square is 2.635.\n\n"
    "Key directions to explore:\n"
    "1. The optimal arrangement likely involves variable-sized circles\n"
    "2. A pure hexagonal arrangement may not be optimal due to edge effects\n"
    "3. The densest known circle packings often use a hybrid approach\n"
    "4. The optimization routine is critically important - simple physics-"
    "based models with carefully tuned parameters\n"
    "5. Consider strategic placement of circles at square corners and edges\n"
    "6. Place larger circles near the center and smaller near the edges\n"
    "7. Math literature suggests special arrangements for specific n\n"
    "8. You can use scipy.optimize to refine radii given fixed centers and "
    "constraints\n\n"
    "Be creative and try to find a new solution."
)

# pick llms based on available keys
llm_models = []
if os.getenv("GEMINI_API_KEY"):
    llm_models.append("gemini-2.5-flash")
if os.getenv("OPENAI_API_KEY"):
    llm_models.append("gpt-5-mini")
if os.getenv("ANTHROPIC_API_KEY"):
    llm_models.append("claude-3-7-sonnet")
elif os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_REGION"):
    llm_models.append("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")
if not llm_models:
    llm_models = ["gpt-5-mini"]  # fallback if no keys detected

# pick embedding model based on available keys
embedding_model_name = ""
if os.getenv("GEMINI_API_KEY"):
    embedding_model_name = "gemini-embedding-001"
elif os.getenv("OPENAI_API_KEY"):
    embedding_model_name = "text-embedding-3-small"
else:
    embedding_model_name = "text-embedding-3-small"
print(f"✅ Embedding model selected: {embedding_model_name}")

# unique experiment directory
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_tag = f"{timestamp}_weighted_fast"

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    # use all three mutation patch types and set prob for all
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    # runs for 20 generations in sequence
    num_generations=20,
    max_parallel_jobs=1,  # only one job at a time
    max_patch_resamples=3,  # resample 3 times if patch fails
    max_patch_attempts=3,  # try 3 times to fix patch via reflection
    # runs locally using the local environment (no loading of conda/docker)
    job_type="local",
    language="python",
    # set LLMs for ensemble
    llm_models=llm_models,
    llm_kwargs=dict(
        temperatures=[0.0, 0.5],  # uniform temperature sampling
        max_tokens=16384,
    ),
    # no meta scratchpad
    meta_rec_interval=None,  # e.g. every 5 generations
    meta_llm_models=None,  # e.g. ["gpt-4.1"]
    meta_llm_kwargs={},  # same as above
    # Set path to initial program relative to repo root
    init_program_path="initial.py",
    results_dir=f"results/circle_packing/{run_tag}",
    # each mutation has three chances of providing a novel solution
    max_novelty_attempts=3,
    # ensemble llm selection among candidates based on past performance
    llm_dynamic_selection=None,  # e.g. "ucb1"
    # set embedding model
    embedding_model=embedding_model_name,
)

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=20,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    enforce_island_separation=True,
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,

)

job_config = LocalJobConfig(eval_program_path="evaluate.py")

print("llm_models:", llm_models)
print("embedding_model:", embedding_model_name)
print("results_dir:", evo_config.results_dir)


if __name__ == "__main__":
    circle_packing_path = repo_root / "examples" / "circle_packing"
    if os.getcwd() != str(circle_packing_path):
        os.chdir(circle_packing_path)
        print("changed working dir to:", circle_packing_path)

    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )

    tic = perf_counter()
    runner.run()
    toc = perf_counter()

    print("completed in", round(toc - tic, 2), "s")
