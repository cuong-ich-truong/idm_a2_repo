import re


def cleansing_syn_report_extend(question, options, raw_synthesized_report):
    """
    Extended (tolerant) version of upstream `cleansing_syn_report`.

    Upstream assumes the LLM always returns the exact substring "Total Analysis:",
    which can crash with IndexError if the model drifts. This clone parses
    defensively and falls back to raw text.
    """

    raw = raw_synthesized_report or ""
    if raw == "ERROR.":
        total_analysis_text = "There is no synthesized report."
        key_knowledge_text = ""
    else:
        key_knowledge_text = ""
        total_analysis_text = ""

        key_match = re.search(
            r"key\s*knowledge\s*:\s*(.*?)(?:\n\s*total\s*analysis\s*:|$)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if key_match:
            key_knowledge_text = key_match.group(1).strip()

        total_match = re.search(
            r"total\s*analysis\s*:\s*(.*)$",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if total_match:
            total_analysis_text = total_match.group(1).strip()
        else:
            total_analysis_text = raw.strip()

    if key_knowledge_text:
        final_syn_repo = (
            f"Question: {question} \n"
            f"Options: {options} \n"
            f"Key Knowledge: {key_knowledge_text} \n"
            f"Total Analysis: {total_analysis_text} \n"
        )
    else:
        final_syn_repo = (
            f"Question: {question} \n"
            f"Options: {options} \n"
            f"Total Analysis: {total_analysis_text} \n"
        )

    return final_syn_repo
