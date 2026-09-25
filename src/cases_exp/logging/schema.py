"""Run log schemas (spec §27). Field-set mirrors for artifact validation."""

COMMON_METHOD_OUTPUT_FIELDS = {
    "task_id", "method_id", "final_answer", "representation_artifact",
    "diagnosis", "repair", "budget",
}
CASES_STATE_FIELDS = {
    "version", "claims", "source_presentation", "source_compilation",
    "active_commitments", "semantic_frame", "computational_grounding",
    "frontier_certificates", "provenance", "migration_manifest",
}
E2B_DIAGNOSIS_FIELDS = {"predicted_loci", "recommended_min_depth", "certificate_refs", "notes"}
E3_REPAIR_FIELDS = {
    "candidate_Q", "prediction", "support_refs", "authorization_valid",
    "protected_constraints_preserved", "dependency_closure_satisfied",
    "distinction_restored", "gvr", "repair_cost",
}
DEPTH_ENUM = {"GROUNDING_ONLY", "COMMITMENT_EDIT", "SOURCE_EDIT"}
PREDICTION_ENUM = {"CERTIFIED_SUCCESS", "CERTIFIED_FAIL", "UNRESOLVED", "PREDICT_SUCCESS", "PREDICT_FAIL"}
