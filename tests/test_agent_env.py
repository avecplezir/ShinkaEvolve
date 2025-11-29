from shinka.agent_env import ActionEngine


def test_parse_modify_full_and_retrieve_actions():
    content = (
        "/execute_action{modify_full}\n"
        "TARGET_PROGRAM: prog_a\n\n"
        "<NAME>rewrite</NAME>\n"
        "<DESCRIPTION>desc</DESCRIPTION>\n"
        "<CODE>```python\n"
        "pass\n"
        "```\n"
        "</CODE>\n"
        "/execute_action{retrieve}\n"
        "TARGET_PROGRAM: prog_a, prog_b\n"
    )
    engine = ActionEngine()
    actions, err = engine.parse_actions(content)

    assert err is None
    assert len(actions) == 2

    modify, retrieve = actions
    assert modify["type"] == "modify"
    assert modify["mode"] == "full"
    assert modify["target"] == "prog_a"
    assert "<NAME>rewrite</NAME>" in modify["patch"]
    assert retrieve["targets"] == ["prog_a", "prog_b"]


def test_parse_modify_diff_sets_mode():
    content = (
        "/execute_action{modify_diff}\n"
        "TARGET_PROGRAM: prog_c\n\n"
        "<NAME>swap</NAME>\n"
        "<DESCRIPTION>switch order</DESCRIPTION>\n"
        "<CODE>```diff\n"
        "<<<<<<< SEARCH\n"
        "a\n"
        "=======\n"
        "b\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "</CODE>\n"
    )
    actions, err = ActionEngine().parse_actions(content)

    assert err is None
    assert len(actions) == 1

    modify = actions[0]
    assert modify["type"] == "modify"
    assert modify["mode"] == "diff"
    assert modify["target"] == "prog_c"
    assert "<<<<<<< SEARCH" in modify["patch"]


def test_parse_modify_missing_target_returns_error():
    actions, err = ActionEngine().parse_actions(
        "/execute_action{modify_full}\n<NAME>x</NAME>"
    )

    assert actions == []
    assert err is not None
    assert "TARGET_PROGRAM" in err


def test_parse_reflect_action():
    content = "/execute_action{reflect}\nThese are my thoughts."
    actions, err = ActionEngine().parse_actions(content)

    assert err is None
    assert actions == [{"type": "reflect"}]


def test_parse_summarize_action():
    summary = "Keep context about metrics and failures."
    content = f"/execute_action{{summarize}}\n{summary}"
    actions, err = ActionEngine().parse_actions(content)

    assert err is None
    assert actions == [{"type": "summarize", "summary": summary}]
