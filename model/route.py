"""Shared development and inference route for high-confidence beam disagreements."""


def apply_route(row, policy):
    """Return a separate prediction row, preserving both component outputs."""
    cutoff = policy.get('double_keep_threshold')
    if cutoff is None:
        return row
    if not .5 <= cutoff <= 1.:
        raise ValueError('Invalid double keep cutoff')
    ranked = row.get('ranked_peptides', row['peptides'])
    beam = row['beam_peptides']
    use_beam = len(beam)==2 and len(ranked)==1 and row['chimeric_probability'] >= cutoff
    peptides = beam if use_beam else ranked
    return dict(row,peptides=list(peptides),ranked_peptides=list(ranked),
                predicted_count=len(peptides),count_constraint_satisfied=len(peptides) in (1,2),
                route='high_confidence_double_beam' if use_beam else 'candidate_ranker')
