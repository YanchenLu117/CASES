"""P0-d tests: certification primitives (Detail §4.1/§4.2/§4.3/§4.4)."""

import pytest

from cases.certification import (
    CERTIFICATION_INCOMPLETE,
    LEVEL_1,
    LEVEL_2,
    DeclaredRelations,
    EvaluatorUniverse,
    FCRScorer,
    HiddenRelationCase,
    antichain,
    build_image_frame,
    certify_level2,
    enumerate_terms,
    family_separation,
    held_out_state_query_accuracy,
    relation_precision_recall,
    semantic_round_trip_failure_rate,
    term_mask,
    validate_hidden_relation_case,
)
from cases.certification.masking import ImageFrame


def world_predicate(worlds: set[str]):
    return lambda w: w in worlds


# ------------------------------------------------------------- term domain


def test_enumerate_terms_matches_free_distributive_lattice_sizes() -> None:
    assert len(enumerate_terms(1, max_terms=100)[0]) == 3  # 0, x, 1
    assert len(enumerate_terms(2, max_terms=100)[0]) == 6
    assert len(enumerate_terms(3, max_terms=1000)[0]) == 20


def test_enumerate_terms_includes_bottom_and_top() -> None:
    terms, complete = enumerate_terms(2, max_terms=100)
    assert complete
    bottom = frozenset()
    top = frozenset({frozenset()})
    assert bottom in terms and top in terms


def test_enumerate_terms_limit_raises_incomplete() -> None:
    from cases.certification import CertificationIncomplete

    with pytest.raises(CertificationIncomplete):
        enumerate_terms(3, max_terms=10)


def test_term_mask_semantics() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    gen_masks = {0: universe.sigma(world_predicate({"w0", "w1"})), 1: universe.sigma(world_predicate({"w1", "w2"}))}
    assert term_mask(frozenset(), gen_masks, universe.size) == 0  # bottom
    assert term_mask(frozenset({frozenset()}), gen_masks, universe.size) == universe.full_mask()  # top
    assert term_mask(frozenset({frozenset({0})}), gen_masks, universe.size) == gen_masks[0]
    # meet of generators x0∧x1 = {w1}
    assert term_mask(frozenset({frozenset({0, 1})}), gen_masks, universe.size) == universe.sigma(world_predicate({"w1"}))


def test_image_frame_closure_and_ops() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    gen_masks = {0: 0b0011, 1: 0b0110}
    frame = build_image_frame(universe, gen_masks)
    # closure of {0, 15, 3, 6} under AND/OR: {0, 2, 3, 6, 7, 15}
    assert set(frame.masks) == {0b0000, 0b0010, 0b0011, 0b0110, 0b0111, 0b1111}
    assert frame.meet(0b0011, 0b0110) == 0b0010
    assert frame.join(0b0011, 0b0110) == 0b0111
    assert frame.is_refinement(0b0010, 0b0011)
    assert not frame.is_refinement(0b0011, 0b0010)
    assert frame.covers(0b0111, [0b0011, 0b0110])
    assert not frame.covers(0b0111, [0b0011])
    assert frame.image_id(0b0011) == "0011"


# ------------------------------------------------------------- Level-2


class IdentitySubframe:
    """Condition-2/3 helper: the frame itself as the computational subframe."""

    def __init__(self, frame: ImageFrame):
        self._frame = frame

    def meet(self, a: int, b: int) -> int:
        return self._frame.meet(a, b)

    def join(self, a: int, b: int) -> int:
        return self._frame.join(a, b)

    def map_term_mask(self, m: int) -> int:
        return m


def _two_identical_generators() -> tuple[EvaluatorUniverse, dict[str, object]]:
    universe = EvaluatorUniverse(("w0", "w1", "w2"))
    preds = {"g0": world_predicate({"w0", "w1"}), "g1": world_predicate({"w0", "w1"})}
    return universe, preds


def test_level2_passes_with_fully_declared_structure() -> None:
    # three generators: g0≡g1 (identical masks), g2 disjoint — the frame closes
    # to exactly 4 elements ({0, 3, 12, 15}), satisfying the nontriviality floor.
    # Full presentation: generator equality, the term identity x0∧x2 ≡ ⊥ (the
    # generators are disjoint), and the cover of top by {x0, x2}.
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    preds = {
        "g0": world_predicate({"w0", "w1"}),
        "g1": world_predicate({"w0", "w1"}),
        "g2": world_predicate({"w2", "w3"}),
    }
    declared = DeclaredRelations(
        equalities=((0, 1),),
        term_equalities=(
            (frozenset({frozenset({0, 2})}), frozenset()),  # x0∧x2 ≡ ⊥
        ),
        covers=(
            # top = x0 ∨ x2  (parent is the TOP term; members are terms)
            (frozenset({frozenset()}), (frozenset({frozenset({0})}), frozenset({frozenset({2})}))),
        ),
    )
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    assert frame.size == 4
    report = certify_level2(
        universe,
        preds,
        declared,
        IdentitySubframe(frame),
        grounding_pairs=[("h1", 0, True), ("h2", 1, True), ("h3", 2, True)],
    )
    assert report.achieved_level == LEVEL_2
    assert all(report.conditions.values())
    assert report.problems == ()


def test_level2_passes_with_distinct_generators_no_declarations() -> None:
    # all six term masks distinct -> K_eval is the identity partition, and with
    # no declarations K_cert is too; frame closes to 6 elements
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    preds = {"g0": world_predicate({"w0", "w1"}), "g1": world_predicate({"w1", "w2"})}
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    assert frame.size == 6
    report = certify_level2(
        universe,
        preds,
        DeclaredRelations(),
        IdentitySubframe(frame),
        grounding_pairs=[("h1", 0, True)],
    )
    assert report.achieved_level == LEVEL_2
    assert all(report.conditions.values())


def test_level2_fails_on_false_refinement_declaration() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2"))
    preds = {"g0": world_predicate({"w0", "w1"}), "g1": world_predicate({"w1", "w2"})}
    declared = DeclaredRelations(refinements=((0, 1),))  # g0 ⊑ g1 is FALSE
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    report = certify_level2(
        universe,
        preds,
        declared,
        IdentitySubframe(frame),
        grounding_pairs=[("h1", 0, True)],
    )
    assert report.achieved_level != LEVEL_2
    assert report.conditions["congruence_equality"] is False


def test_level2_incomplete_when_term_limit_exceeded() -> None:
    universe, preds = _two_identical_generators()
    report = certify_level2(
        universe,
        preds,
        DeclaredRelations(),
        None,
        [],
        max_terms=3,
    )
    assert report.achieved_level == LEVEL_1
    assert any(CERTIFICATION_INCOMPLETE in p for p in report.problems)


def test_level2_fails_nontriviality_floor_with_single_generator() -> None:
    universe = EvaluatorUniverse(("w0", "w1"))
    preds = {"g0": world_predicate({"w0"})}
    frame = build_image_frame(universe, {0: universe.sigma(preds["g0"])})
    report = certify_level2(universe, preds, DeclaredRelations(), IdentitySubframe(frame), [])
    assert report.achieved_level != LEVEL_2
    assert report.conditions["nontriviality_floor"] is False


def test_level2_fails_on_uncovered_grounding() -> None:
    universe, preds = _two_identical_generators()
    frame = build_image_frame(universe, {i: universe.sigma(p) for i, p in enumerate(preds.values())})
    report = certify_level2(
        universe,
        preds,
        DeclaredRelations(equalities=((0, 1),)),
        IdentitySubframe(frame),
        grounding_pairs=[("h1", 0, True), ("h2", 1, False)],
    )
    assert report.achieved_level != LEVEL_2
    assert report.conditions["grounding_coverage"] is False


def test_p_rel_r_rel_counts() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2"))
    preds = {"g0": world_predicate({"w0", "w1"}), "g1": world_predicate({"w0", "w1"}), "g2": world_predicate({"w2"})}
    # g0≡g1 true; g0⊑g2 false
    declared = DeclaredRelations(equalities=((0, 1),), refinements=((0, 2),))
    p_rel, r_rel, true_count, relevant = relation_precision_recall(universe, preds, declared)
    assert true_count == 1  # only the equality holds
    assert p_rel == pytest.approx(0.5)
    # relevant pairs: (g0,g1) eq; (g0,g2)/(g1,g2) no relation -> 1
    assert relevant == 1
    assert r_rel == pytest.approx(1.0)


def test_family_separation() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2", "w3"))
    preds = {"ga": world_predicate({"w0", "w1"}), "gb": world_predicate({"w2", "w3"})}
    fam_all = world_predicate({"w0", "w1", "w2", "w3"})
    fam_ga = world_predicate({"w0", "w1"})
    fam_none = world_predicate(set())
    sep_all, sep_count, total = family_separation(universe, preds, [fam_all, fam_ga])
    assert (sep_all, sep_count, total) == (1.0, 2, 2)
    sep_none, sep_count2, _ = family_separation(universe, preds, [fam_none])
    assert sep_none == 0.0 and sep_count2 == 0


# ----------------------------------------------------------- interventions


def test_hidden_relation_case_validation() -> None:
    universe = EvaluatorUniverse(("w0", "w1", "w2"))
    preds = {0: world_predicate({"w0", "w1"}), 1: world_predicate({"w1", "w2"})}
    bad = HiddenRelationCase("c1", true_relation=(0, 1), removed_from_proposal=True,
                             inconsistent_grounding=True, presented=True, admitted=True)
    # g0 ⊑ g1 does NOT hold (w0 in g0 only) -> flagged
    assert any("does NOT hold" in p for p in validate_hidden_relation_case(bad, universe, preds))
    good_true = HiddenRelationCase("c2", true_relation=(0, 0), removed_from_proposal=True,
                                   inconsistent_grounding=True, presented=True, admitted=True)
    assert validate_hidden_relation_case(good_true, universe, {0: preds[0], 1: preds[1]}) == []


def test_fcr_scorer_counts() -> None:
    scorer = FCRScorer()
    scorer.add(HiddenRelationCase("u1", (0, 0), True, True, presented=True, admitted=True))
    scorer.add(HiddenRelationCase("u2", (0, 0), True, True, presented=True, admitted=False))
    scorer.add(HiddenRelationCase("u3", (0, 0), True, True, presented=True, admitted=True))
    fcr, num, den = scorer.score()
    assert (fcr, num, den) == (pytest.approx(2 / 3), 2, 3)


def test_fcr_method_attributable_failure_is_worst_case() -> None:
    scorer = FCRScorer()
    scorer.add(HiddenRelationCase("u1", (0, 0), True, True, presented=False, admitted=False,
                                  method_attributable_failure=True))
    scorer.add(HiddenRelationCase("u2", (0, 0), True, True, presented=True, admitted=False))
    fcr, num, den = scorer.score()
    assert (fcr, num, den) == (pytest.approx(0.5), 1, 2)


def test_fcr_empty_is_rejected() -> None:
    with pytest.raises(ValueError):
        FCRScorer().score()


def test_round_trip_failure_rate() -> None:
    rate, failed, total = semantic_round_trip_failure_rate(
        [("p1", True), ("p2", False), ("p3", False), ("p4", True)]
    )
    assert (rate, failed, total) == (0.5, 2, 4)
    with pytest.raises(ValueError):
        semantic_round_trip_failure_rate([])


def test_held_out_state_query_accuracy_macro() -> None:
    # three classes: families 2/2=1.0, boundaries 0/1=0.0, gaps 1/1=1.0
    # macro = (1.0 + 0.0 + 1.0)/3 = 2/3
    macro, correct, total = held_out_state_query_accuracy(
        [
            ("families", "f1", "f1"),
            ("families", "f2", "f2"),
            ("boundaries", "b9", "b1"),
            ("gaps", "g1", "g1"),
        ]
    )
    assert total == 4 and correct == 3
    assert macro == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        held_out_state_query_accuracy([])
