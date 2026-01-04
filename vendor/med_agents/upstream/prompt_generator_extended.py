"""
POC extensions to upstream MedAgents prompts.

This file intentionally keeps all "evidence-aware" prompt variants separate so
`prompt_generator.py` can stay close to upstream.
"""

from prompt_generator import (  # type: ignore
    get_options_analysis_prompt,
    get_question_analysis_prompt,
)


def get_question_analysis_prompt_with_evidence(
    question, question_domain, evidence_context
):
    """
    Stage 2 extension: question analysis prompt with optional evidence context.
    """
    question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(
        question, question_domain
    )
    if evidence_context:
        prompt_get_question_analysis += (
            "\n\nYou are given optional external evidence excerpts. "
            "Use them if helpful. If you use evidence, cite it by its id like [E1]. "
            "If evidence is insufficient, say so and rely on your own knowledge.\n"
            f"Evidence:\n{evidence_context}\n"
        )
    return question_analyzer, prompt_get_question_analysis


def get_options_analysis_prompt_with_evidence(
    question, options, op_domain, question_analysis, evidence_context
):
    """
    Stage 2 extension: options analysis prompt with optional evidence context.
    """
    option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(
        question, options, op_domain, question_analysis
    )
    if evidence_context:
        prompt_get_options_analyses += (
            "\n\nYou are given optional external evidence excerpts. "
            "Use them if helpful. If you use evidence, cite it by its id like [E1]. "
            "If evidence is insufficient, say so and rely on your own knowledge.\n"
            f"Evidence:\n{evidence_context}\n"
        )
    return option_analyzer, prompt_get_options_analyses
