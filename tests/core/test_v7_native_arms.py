"""this native AI-scientist / host arms tests (§5.2)."""

from cases.baselines import Host, arm_injection, host_arms, native_baselines_index, native_hosts


def test_native_hosts_and_arms():
    assert set(native_hosts()) == {Host.SCIEXPLORER, Host.MADE, Host.PIEVO}
    # four arms for SciExplorer/MADE, three for PiEvo
    assert tuple(host_arms(Host.SCIEXPLORER)) == ("native", "summary", "cases_state", "full")
    assert tuple(host_arms(Host.MADE)) == ("native", "summary", "cases_state", "full")
    assert tuple(host_arms(Host.PIEVO)) == ("native", "summary", "cases_plugin")


def test_injection_described_for_every_arm():
    for host in native_hosts():
        for arm in host_arms(host):
            assert arm_injection(host, arm) and "unspecified" not in arm_injection(host, arm)


def test_native_index_covers_plan_s52():
    idx = native_baselines_index()
    assert set(idx) == {"sciexplorer", "made", "pievo"}
    assert "official_repo" in idx["made"]
    assert "cases_plugin" in idx["pievo"]["arms"]


def test_tier_b_ai_scientist_systems():
    from cases.baselines import ai_scientist_keys, ai_scientist_systems
    assert set(ai_scientist_keys()) == {"S1", "S2", "S3", "S4"}
    idx = ai_scientist_systems()
    assert "AI Scientist-v2-Discovery" in idx["S2"]["reported_name"]
    assert "AI-Researcher-Discovery" in idx["S3"]["reported_name"]
    assert idx["S2"]["frozen_placement"] == "GB1 + MADE"
    assert "campaign API" in idx["S2"]["tool_boundary"]


def test_sciexplorer_repo_consistent():
    """host native-arm repo must equal the S1 AI-scientist index source (no drift)."""
    from cases.baselines import spec, Host
    from cases.baselines.native_arms import AI_SCIENTIST_SYSTEMS
    assert "RyanSkraba" not in spec(Host.SCIEXPLORER).official_repo
    assert spec(Host.SCIEXPLORER).official_repo == AI_SCIENTIST_SYSTEMS["S1"].source
