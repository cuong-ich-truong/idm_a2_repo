from prompt_generator import *
from prompt_generator_extended import *
from data_utils import *
from data_utils_extended import cleansing_syn_report_extend


def _set_llm_stage(handler, stage: str, **meta) -> None:
    """
    Best-effort stage tagging for detailed LLM-call logs.
    Our wrapper handler reads `current_stage` / `current_meta` if present.
    """
    try:
        setattr(handler, "current_stage", stage)
        setattr(handler, "current_meta", meta)
    except Exception:
        # keep upstream logic robust even if handler doesn't support attrs
        return


def fully_decode(
    qid,
    realqid,
    question,
    options,
    gold_answer,
    handler,
    args,
    dataobj,
    evidence_context=None,
):
    """
    NOTE: This is based on the original MedAgents `fully_decode` implementation.
    For this project we removed all shortcut modes (e.g., baselines / partial flows)
    and keep only the single full pipeline that runs all 5 stages (1→2→3→4→5).
    """

    (
        question_domains,
        options_domains,
        question_analyses,
        option_analyses,
        syn_report,
        output,
    ) = ("", "", "", "", "", "")
    vote_history, revision_history, syn_repo_history = [], [], []

    # Full pipeline consists of 5 stages (Stage 1→2→3→4→5).
    # Stage 1: Expert Gathering (domain routing)
    _set_llm_stage(handler, "S1_question_domain", qid=qid, realqid=realqid)
    question_classifier, prompt_get_question_domain = get_question_domains_prompt(
        question
    )
    raw_question_domain = handler.get_output_multiagent(
        user_input=prompt_get_question_domain,
        temperature=0,
        max_tokens=50,
        system_role=question_classifier,
    )
    if raw_question_domain == "ERROR.":
        raw_question_domain = "Medical Field: " + " | ".join(
            ["General Medicine" for _ in range(NUM_QD)]
        )
    question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")

    _set_llm_stage(handler, "S1_options_domain", qid=qid, realqid=realqid)
    options_classifier, prompt_get_options_domain = get_options_domains_prompt(
        question, options
    )
    raw_option_domain = handler.get_output_multiagent(
        user_input=prompt_get_options_domain,
        temperature=0,
        max_tokens=50,
        system_role=options_classifier,
    )
    if raw_option_domain == "ERROR.":
        raw_option_domain = "Medical Field: " + " | ".join(
            ["General Medicine" for _ in range(NUM_OD)]
        )
    options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")

    # Stage 2: Analysis Proposition (evidence injected here)
    tmp_question_analysis = []
    for _domain in question_domains:
        _set_llm_stage(
            handler, "S2_question_analysis", qid=qid, realqid=realqid, domain=_domain
        )
        question_analyzer, prompt_get_question_analysis = (
            get_question_analysis_prompt_with_evidence(
                question, _domain, evidence_context
            )
        )
        raw_question_analysis = handler.get_output_multiagent(
            user_input=prompt_get_question_analysis,
            temperature=0,
            max_tokens=300,
            system_role=question_analyzer,
        )
        tmp_question_analysis.append(raw_question_analysis)
    question_analyses = cleansing_analysis(
        tmp_question_analysis, question_domains, "question"
    )

    tmp_option_analysis = []
    for _domain in options_domains:
        _set_llm_stage(
            handler, "S2_options_analysis", qid=qid, realqid=realqid, domain=_domain
        )
        option_analyzer, prompt_get_options_analyses = (
            get_options_analysis_prompt_with_evidence(
                question, options, _domain, question_analyses, evidence_context
            )
        )
        raw_option_analysis = handler.get_output_multiagent(
            user_input=prompt_get_options_analyses,
            temperature=0,
            max_tokens=300,
            system_role=option_analyzer,
        )
        tmp_option_analysis.append(raw_option_analysis)
    option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, "option")

    # Stage 3: Report Summarization
    q_analyses_text = transform_dict2text(question_analyses, "question", question)
    o_analyses_text = transform_dict2text(option_analyses, "options", options)

    _set_llm_stage(handler, "S3_synth_report", qid=qid, realqid=realqid)
    synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(
        q_analyses_text, o_analyses_text
    )
    raw_synthesized_report = handler.get_output_multiagent(
        user_input=prompt_get_synthesized_report,
        temperature=0,
        max_tokens=2500,
        system_role=synthesizer,
    )
    syn_report = cleansing_syn_report_extend(question, options, raw_synthesized_report)

    # Stage 4: Collaborative Consultation (consensus loop)
    all_domains = question_domains + options_domains
    syn_repo_history = [syn_report]

    hasno_flag = True  # default value: in order to get into the while loop
    num_try = 0

    while num_try < args.max_attempt_vote and hasno_flag:
        domain_opinions = {}  # 'domain' : 'yes' / 'no'
        revision_advice = {}
        num_try += 1
        hasno_flag = False
        for domain in all_domains:
            voter, cons_prompt = get_consensus_prompt(domain, syn_report)
            _set_llm_stage(
                handler,
                "S4_vote",
                qid=qid,
                realqid=realqid,
                domain=domain,
                round=num_try,
            )
            raw_domain_opi = handler.get_output_multiagent(
                user_input=cons_prompt,
                temperature=0,
                max_tokens=30,
                system_role=voter,
            )
            domain_opinion = cleansing_voting(raw_domain_opi)  # "yes" / "no"
            domain_opinions[domain] = domain_opinion
            if domain_opinion == "no":
                advice_prompt = get_consensus_opinion_prompt(domain, syn_report)
                _set_llm_stage(
                    handler,
                    "S4_advice",
                    qid=qid,
                    realqid=realqid,
                    domain=domain,
                    round=num_try,
                )
                advice_output = handler.get_output_multiagent(
                    user_input=advice_prompt,
                    temperature=0,
                    max_tokens=500,
                    system_role=voter,
                )
                revision_advice[domain] = advice_output
                hasno_flag = True
        if hasno_flag:
            revision_prompt = get_revision_prompt(syn_report, revision_advice)
            _set_llm_stage(
                handler, "S4_revision", qid=qid, realqid=realqid, round=num_try
            )
            revised_analysis = handler.get_output_multiagent(
                user_input=revision_prompt,
                temperature=0,
                max_tokens=2500,
                system_role="",
            )
            syn_report = cleansing_syn_report_extend(
                question, options, revised_analysis
            )
            revision_history.append(revision_advice)
            syn_repo_history.append(syn_report)
        vote_history.append(domain_opinions)

    # Stage 5: Decision Making (final answer derivation)
    answer_prompt = get_final_answer_prompt_wsyn(syn_report)
    _set_llm_stage(handler, "S5_final", qid=qid, realqid=realqid)
    output = handler.get_output_multiagent(
        user_input=answer_prompt,
        temperature=0,
        max_tokens=2500,
        system_role="",
    )
    ans, output = cleansing_final_output(output)

    data_info = {
        "question": question,
        "options": options,
        "pred_answer": ans,
        "gold_answer": gold_answer,
        "question_domains": question_domains,
        "option_domains": options_domains,
        "question_analyses": question_analyses,
        "option_analyses": option_analyses,
        "syn_report": syn_report,
        "vote_history": vote_history,
        "revision_history": revision_history,
        "syn_repo_history": syn_repo_history,
        "raw_output": output,
    }

    return data_info
